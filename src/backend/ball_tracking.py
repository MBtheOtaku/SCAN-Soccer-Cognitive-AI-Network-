from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


SPORTS_BALL_CLASS_ID = 32


# =========================================================
# DATA TYPES
# =========================================================


@dataclass
class BallDetection:
    frame_idx: int
    timestamp_s: float

    center_x: float
    center_y: float

    center_x_norm: float
    center_y_norm: float

    bbox_width: float
    bbox_height: float

    confidence: float
    track_id: Optional[int]

    # detected
    # tentative
    # reacquired
    # predicted
    source: str

    # How strongly this candidate agrees with the
    # existing motion trajectory.
    #
    # This is intentionally separate from YOLO confidence.
    association_score: Optional[float] = None


@dataclass
class BallTrackSummary:
    detected: bool

    total_frames: int

    # Only trusted real observations:
    # detected + confirmed reacquired.
    detection_count: int
    detection_rate: float

    reacquired_count: int
    tentative_count: int
    predicted_count: int

    mean_confidence: Optional[float]
    peak_confidence: Optional[float]

    first_frame: Optional[int]
    last_frame: Optional[int]

    first_x_norm: Optional[float]
    first_y_norm: Optional[float]

    final_x_norm: Optional[float]
    final_y_norm: Optional[float]

    tracking_end_frame: Optional[int]

    longest_prediction_gap: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BallTrackResult:
    summary: BallTrackSummary
    detections: list[BallDetection]


# =========================================================
# MOTION MODEL
# =========================================================


class BallMotionModel:
    """
    Short-horizon motion model based only on trusted,
    real ball observations.

    IMPORTANT:
    predicted and tentative points are never fed back
    into this model.

    This prevents prediction drift from becoming
    self-reinforcing.
    """

    def __init__(
        self,
        history_size: int = 8,
    ) -> None:

        self.history_size = history_size

        self.history: list[
            BallDetection
        ] = []

        self.missed_frames = 0

    @property
    def initialized(self) -> bool:
        return bool(self.history)

    @property
    def last_observation(
        self,
    ) -> Optional[BallDetection]:

        if not self.history:
            return None

        return self.history[-1]

    @property
    def last_size(self) -> Optional[float]:

        last = self.last_observation

        if last is None:
            return None

        return max(
            last.bbox_width,
            last.bbox_height,
        )

    @property
    def last_confidence(
        self,
    ) -> Optional[float]:

        last = self.last_observation

        if last is None:
            return None

        return last.confidence

    def add_observation(
        self,
        detection: BallDetection,
    ) -> None:

        self.history.append(
            detection
        )

        self.history = self.history[
            -self.history_size:
        ]

        self.missed_frames = 0

    def mark_missed(
        self,
    ) -> None:

        self.missed_frames += 1

    def begin_shot(
        self,
    ) -> None:
        """
        Before the kick the ball is mostly stationary.

        At the dynamically detected strike frame,
        discard the old stationary history so it does
        not suppress the post-strike velocity estimate.
        """

        if self.history:
            self.history = [
                self.history[-1]
            ]

        self.missed_frames = 0

    def predict(
        self,
        target_frame: int,
    ) -> Optional[
        tuple[float, float]
    ]:

        if not self.history:
            return None

        if len(self.history) == 1:

            last = self.history[-1]

            return (
                last.center_x,
                last.center_y,
            )

        samples = self.history[-6:]

        frames = np.array(
            [
                detection.frame_idx
                for detection in samples
            ],
            dtype=np.float64,
        )

        xs = np.array(
            [
                detection.center_x
                for detection in samples
            ],
            dtype=np.float64,
        )

        ys = np.array(
            [
                detection.center_y
                for detection in samples
            ],
            dtype=np.float64,
        )

        base_frame = frames[0]

        times = (
            frames - base_frame
        )

        target_time = (
            target_frame - base_frame
        )

        try:

            # Short-horizon horizontal motion.
            x_coeff = np.polyfit(
                times,
                xs,
                1,
            )

            predicted_x = float(
                np.polyval(
                    x_coeff,
                    target_time,
                )
            )

            # Allow mild vertical curvature.
            y_degree = (
                2
                if len(samples) >= 4
                else 1
            )

            y_coeff = np.polyfit(
                times,
                ys,
                y_degree,
            )

            predicted_y = float(
                np.polyval(
                    y_coeff,
                    target_time,
                )
            )

            return (
                predicted_x,
                predicted_y,
            )

        except (
            np.linalg.LinAlgError,
            ValueError,
        ):

            previous = samples[-2]
            current = samples[-1]

            frame_delta = max(
                1,
                current.frame_idx
                - previous.frame_idx,
            )

            vx = (
                current.center_x
                - previous.center_x
            ) / frame_delta

            vy = (
                current.center_y
                - previous.center_y
            ) / frame_delta

            future_delta = (
                target_frame
                - current.frame_idx
            )

            return (
                current.center_x
                + vx * future_delta,

                current.center_y
                + vy * future_delta,
            )

    def estimated_speed(
        self,
        target_frame: int,
    ) -> float:

        if len(self.history) < 2:
            return 0.0

        current = self.predict(
            target_frame
        )

        previous = self.predict(
            target_frame - 1
        )

        if (
            current is None
            or previous is None
        ):
            return 0.0

        return math.hypot(
            current[0] - previous[0],
            current[1] - previous[1],
        )

    def search_radius(
        self,
        target_frame: int,
        frame_width: int,
        frame_height: int,
    ) -> float:

        minimum_dimension = min(
            frame_width,
            frame_height,
        )

        maximum_dimension = max(
            frame_width,
            frame_height,
        )

        ball_size = (
            self.last_size
            or minimum_dimension * 0.02
        )

        speed = self.estimated_speed(
            target_frame
        )

        base_radius = max(
            ball_size * 3.5,
            minimum_dimension * 0.025,
        )

        motion_radius = (
            speed * 2.0
        )

        uncertainty_radius = (
            minimum_dimension
            * 0.015
            * self.missed_frames
        )

        radius = (
            base_radius
            + motion_radius
            + uncertainty_radius
        )

        maximum_radius = (
            maximum_dimension * 0.30
        )

        return min(
            radius,
            maximum_radius,
        )

    def predicted_confidence(
        self,
    ) -> float:

        starting_confidence = (
            self.last_confidence
            or 0.25
        )

        confidence = (
            starting_confidence
            * (
                0.65
                ** self.missed_frames
            )
        )

        return max(
            0.01,
            float(confidence),
        )

class OpticalBallTracker:
    """
    Tracks the ball frame-to-frame using pyramidal
    Lucas-Kanade optical flow.

    YOLO initializes/corrects this tracker.

    The motion model is only used as a spatial sanity
    check -- it does NOT determine the optical-flow
    ball position.
    """

    def __init__(self) -> None:
        self.prev_gray: Optional[np.ndarray] = None

        self.points: Optional[np.ndarray] = None

        self.center: Optional[
            tuple[float, float]
        ] = None

        self.bbox_width: float = 0.0
        self.bbox_height: float = 0.0

        self.last_frame: Optional[int] = None

    @property
    def active(self) -> bool:
        return (
            self.prev_gray is not None
            and self.points is not None
            and self.center is not None
            and self.last_frame is not None
        )

    def deactivate(self) -> None:
        self.prev_gray = None
        self.points = None
        self.center = None
        self.last_frame = None

    def _seed_points(
        self,
        gray: np.ndarray,
        detection: BallDetection,
    ) -> np.ndarray:
        """
        Find trackable image features around the
        currently observed ball.

        If the tiny ball has too little texture for
        Shi-Tomasi features, fall back to a small grid
        of points around the ball center.
        """

        height, width = gray.shape[:2]

        cx = float(detection.center_x)
        cy = float(detection.center_y)

        ball_size = max(
            detection.bbox_width,
            detection.bbox_height,
            6.0,
        )

        radius = max(
            7,
            int(round(ball_size * 0.8)),
        )

        x1 = max(
            0,
            int(round(cx - radius)),
        )

        y1 = max(
            0,
            int(round(cy - radius)),
        )

        x2 = min(
            width - 1,
            int(round(cx + radius)),
        )

        y2 = min(
            height - 1,
            int(round(cy + radius)),
        )

        mask = np.zeros_like(
            gray,
            dtype=np.uint8,
        )

        cv2.rectangle(
            mask,
            (x1, y1),
            (x2, y2),
            255,
            -1,
        )

        features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=16,
            qualityLevel=0.01,
            minDistance=2,
            mask=mask,
            blockSize=3,
        )

        points: list[
            tuple[float, float]
        ] = []

        if features is not None:
            for feature in features:
                x, y = feature.ravel()

                points.append(
                    (
                        float(x),
                        float(y),
                    )
                )

        # Tiny / blurred soccer balls may not have
        # enough strong corners. Add points around
        # the ball center as fallback tracking probes.
        if len(points) < 5:

            offset = max(
                2.0,
                ball_size * 0.25,
            )

            for dx in (
                -offset,
                0.0,
                offset,
            ):
                for dy in (
                    -offset,
                    0.0,
                    offset,
                ):

                    x = float(
                        np.clip(
                            cx + dx,
                            0,
                            width - 1,
                        )
                    )

                    y = float(
                        np.clip(
                            cy + dy,
                            0,
                            height - 1,
                        )
                    )

                    points.append(
                        (x, y)
                    )

        return np.array(
            points,
            dtype=np.float32,
        ).reshape(
            -1,
            1,
            2,
        )

    def reset(
        self,
        gray: np.ndarray,
        detection: BallDetection,
    ) -> None:
        """
        Initialize/correct optical flow from a trusted
        YOLO observation.
        """

        self.prev_gray = gray.copy()

        self.center = (
            float(detection.center_x),
            float(detection.center_y),
        )

        self.bbox_width = max(
            float(detection.bbox_width),
            4.0,
        )

        self.bbox_height = max(
            float(detection.bbox_height),
            4.0,
        )

        self.last_frame = (
            detection.frame_idx
        )

        self.points = self._seed_points(
            gray,
            detection,
        )

    def track(
        self,
        gray: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        frame_width: int,
        frame_height: int,
        predicted_center: Optional[
            tuple[float, float]
        ],
        gate_radius: Optional[float],
    ) -> Optional[BallDetection]:
        """
        Track the actual image content from the previous
        frame into this frame.

        Uses forward/backward optical-flow consistency to
        reject unstable points.
        """

        if not self.active:
            return None

        assert self.prev_gray is not None
        assert self.points is not None
        assert self.center is not None
        assert self.last_frame is not None

        # Optical flow should only bridge consecutive frames.
        if frame_idx != self.last_frame + 1:
            self.deactivate()
            return None

        next_points, status_forward, _ = (
            cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS
                    | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
        )

        if (
            next_points is None
            or status_forward is None
        ):
            self.deactivate()
            return None

        back_points, status_backward, _ = (
            cv2.calcOpticalFlowPyrLK(
                gray,
                self.prev_gray,
                next_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS
                    | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
        )

        if (
            back_points is None
            or status_backward is None
        ):
            self.deactivate()
            return None

        previous = self.points.reshape(
            -1,
            2,
        )

        current = next_points.reshape(
            -1,
            2,
        )

        backward = back_points.reshape(
            -1,
            2,
        )

        forward_ok = (
            status_forward.reshape(-1)
            == 1
        )

        backward_ok = (
            status_backward.reshape(-1)
            == 1
        )

        fb_error = np.linalg.norm(
            previous - backward,
            axis=1,
        )

        valid = (
            forward_ok
            & backward_ok
            & (fb_error < 2.0)
        )

        if np.count_nonzero(valid) < 3:
            self.deactivate()
            return None

        previous_valid = previous[
            valid
        ]

        current_valid = current[
            valid
        ]

        fb_valid = fb_error[
            valid
        ]

        displacement = (
            current_valid
            - previous_valid
        )

        median_displacement = (
            np.median(
                displacement,
                axis=0,
            )
        )

        residuals = np.linalg.norm(
            displacement
            - median_displacement,
            axis=1,
        )

        median_residual = float(
            np.median(
                residuals
            )
        )

        residual_limit = max(
            2.0,
            median_residual * 2.5 + 1.0,
        )

        inliers = (
            residuals
            <= residual_limit
        )

        if np.count_nonzero(inliers) < 3:
            self.deactivate()
            return None

        good_current = current_valid[
            inliers
        ]

        good_fb = fb_valid[
            inliers
        ]

        dx = float(
            np.median(
                displacement[
                    inliers,
                    0,
                ]
            )
        )

        dy = float(
            np.median(
                displacement[
                    inliers,
                    1,
                ]
            )
        )

        new_x = (
            self.center[0] + dx
        )

        new_y = (
            self.center[1] + dy
        )

        if not (
            0.0 <= new_x < frame_width
            and
            0.0 <= new_y < frame_height
        ):
            self.deactivate()
            return None

        # -------------------------------------------------
        # Motion model is a SANITY CHECK only.
        # -------------------------------------------------

        motion_score = 1.0

        if (
            predicted_center is not None
            and gate_radius is not None
        ):

            distance_to_prediction = (
                math.hypot(
                    new_x
                    - predicted_center[0],

                    new_y
                    - predicted_center[1],
                )
            )

            allowed_distance = max(
                gate_radius,
                max(
                    self.bbox_width,
                    self.bbox_height,
                )
                * 4.0,
            )

            if (
                distance_to_prediction
                > allowed_distance
            ):
                self.deactivate()
                return None

            sigma = max(
                allowed_distance * 0.5,
                1.0,
            )

            motion_score = math.exp(
                -0.5
                * (
                    distance_to_prediction
                    / sigma
                )
                ** 2
            )

        # -------------------------------------------------
        # Optical-flow confidence
        # -------------------------------------------------

        original_count = len(
            self.points
        )

        inlier_ratio = (
            len(good_current)
            / max(
                original_count,
                1,
            )
        )

        median_fb_error = float(
            np.median(
                good_fb
            )
        )

        fb_score = math.exp(
            -median_fb_error / 2.0
        )

        confidence = (
            0.50 * inlier_ratio
            + 0.30 * fb_score
            + 0.20 * motion_score
        )

        # If the optical-flow evidence itself is poor,
        # do not pretend we have an observed ball.
        if confidence < 0.40:
            self.deactivate()
            return None

        detection = BallDetection(
            frame_idx=frame_idx,

            timestamp_s=timestamp_s,

            center_x=new_x,
            center_y=new_y,

            center_x_norm=(
                new_x / frame_width
            ),

            center_y_norm=(
                new_y / frame_height
            ),

            bbox_width=self.bbox_width,
            bbox_height=self.bbox_height,

            # For source="visual", this represents
            # optical-flow tracking confidence rather
            # than YOLO classification confidence.
            confidence=float(
                confidence
            ),

            track_id=1,

            source="visual",

            association_score=float(
                motion_score
            ),
        )

        # -------------------------------------------------
        # Advance visual tracker state
        # -------------------------------------------------

        self.prev_gray = (
            gray.copy()
        )

        self.center = (
            new_x,
            new_y,
        )

        self.last_frame = (
            frame_idx
        )

        self.points = (
            good_current
            .astype(
                np.float32
            )
            .reshape(
                -1,
                1,
                2,
            )
        )

        # If too many points have disappeared,
        # reseed around the new visual ball location.
        if len(self.points) < 5:

            self.points = (
                self._seed_points(
                    gray,
                    detection,
                )
            )

        return detection


# =========================================================
# YOLO
# =========================================================


def _run_yolo(
    model: YOLO,
    image: np.ndarray,
    confidence: float,
    image_size: int,
):
    results = model.predict(
        image,
        classes=[
            SPORTS_BALL_CLASS_ID
        ],
        conf=confidence,
        imgsz=image_size,
        max_det=10,
        verbose=False,
    )

    if not results:
        return None

    return results[0]


def _detections_from_result(
    result,
    frame_idx: int,
    timestamp_s: float,
    frame_width: int,
    frame_height: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[BallDetection]:

    if result is None:
        return []

    boxes = result.boxes

    if (
        boxes is None
        or len(boxes) == 0
    ):
        return []

    detections: list[
        BallDetection
    ] = []

    for index in range(
        len(boxes)
    ):

        confidence = float(
            boxes.conf[index].item()
        )

        x1, y1, x2, y2 = (
            boxes.xyxy[index]
            .cpu()
            .tolist()
        )

        x1 += offset_x
        x2 += offset_x

        y1 += offset_y
        y2 += offset_y

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        detections.append(
            BallDetection(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,

                center_x=center_x,
                center_y=center_y,

                center_x_norm=(
                    center_x
                    / frame_width
                ),

                center_y_norm=(
                    center_y
                    / frame_height
                ),

                bbox_width=(
                    x2 - x1
                ),

                bbox_height=(
                    y2 - y1
                ),

                confidence=confidence,

                track_id=1,

                source="detected",

                association_score=None,
            )
        )

    return detections


# =========================================================
# CANDIDATE ASSOCIATION
# =========================================================


def _candidate_score(
    candidate: BallDetection,
    predicted_center: tuple[
        float,
        float,
    ],
    expected_size: Optional[float],
    gate_radius: float,
) -> Optional[float]:

    dx = (
        candidate.center_x
        - predicted_center[0]
    )

    dy = (
        candidate.center_y
        - predicted_center[1]
    )

    distance = math.hypot(
        dx,
        dy,
    )

    # Reject candidates far outside the
    # predicted motion region.
    if distance > (
        gate_radius * 1.5
    ):
        return None

    sigma = max(
        gate_radius * 0.55,
        1.0,
    )

    distance_score = math.exp(
        -0.5
        * (
            distance / sigma
        )
        ** 2
    )

    candidate_size = max(
        candidate.bbox_width,
        candidate.bbox_height,
        1.0,
    )

    if (
        expected_size is not None
        and expected_size > 0
    ):

        size_ratio = (
            candidate_size
            / expected_size
        )

        size_score = math.exp(
            -abs(
                math.log(
                    max(
                        size_ratio,
                        1e-6,
                    )
                )
            )
        )

    else:
        size_score = 1.0

    # Detector confidence is only one source
    # of evidence.
    #
    # A tiny soccer ball can have low YOLO
    # confidence but still agree extremely
    # strongly with the expected motion.
    score = (
        0.45
        * candidate.confidence

        + 0.45
        * distance_score

        + 0.10
        * size_score
    )

    return float(score)


def _select_best_candidate(
    candidates: list[BallDetection],
    predicted_center: Optional[
        tuple[float, float]
    ],
    expected_size: Optional[float],
    gate_radius: Optional[float],
) -> Optional[BallDetection]:

    if not candidates:
        return None

    # No trajectory yet.
    if (
        predicted_center is None
        or gate_radius is None
    ):

        best = max(
            candidates,
            key=lambda detection:
            detection.confidence,
        )

        best.association_score = (
            best.confidence
        )

        return best

    best_detection = None
    best_score = float("-inf")

    for candidate in candidates:

        score = _candidate_score(
            candidate=candidate,
            predicted_center=(
                predicted_center
            ),
            expected_size=(
                expected_size
            ),
            gate_radius=(
                gate_radius
            ),
        )

        if score is None:
            continue

        if score > best_score:
            best_score = score
            best_detection = candidate

    # Association must have meaningful support.
    if (
        best_detection is None
        or best_score < 0.35
    ):
        return None

    best_detection.association_score = (
        best_score
    )

    return best_detection


def _tentatives_consistent(
    previous: BallDetection,
    current: BallDetection,
    gate_radius: float,
) -> bool:
    """
    Require consecutive weak candidates to behave
    like the same moving object before promoting
    them to confirmed reacquisitions.
    """

    frame_gap = (
        current.frame_idx
        - previous.frame_idx
    )

    if frame_gap != 1:
        return False

    distance = math.hypot(
        current.center_x
        - previous.center_x,

        current.center_y
        - previous.center_y,
    )

    previous_size = max(
        previous.bbox_width,
        previous.bbox_height,
        1.0,
    )

    current_size = max(
        current.bbox_width,
        current.bbox_height,
        1.0,
    )

    size_ratio = (
        current_size
        / previous_size
    )

    # Sudden huge size changes are unlikely to
    # represent the same soccer ball.
    if not (
        0.35
        <= size_ratio
        <= 2.85
    ):
        return False

    allowed_distance = max(
        gate_radius * 0.60,
        previous_size * 4.0,
    )

    return (
        distance
        <= allowed_distance
    )


def _make_local_crop(
    frame: np.ndarray,
    predicted_center: tuple[
        float,
        float,
    ],
    radius: float,
) -> Optional[
    tuple[
        np.ndarray,
        int,
        int,
    ]
]:

    height, width = (
        frame.shape[:2]
    )

    cx, cy = predicted_center

    x1 = max(
        0,
        int(
            math.floor(
                cx - radius
            )
        ),
    )

    y1 = max(
        0,
        int(
            math.floor(
                cy - radius
            )
        ),
    )

    x2 = min(
        width,
        int(
            math.ceil(
                cx + radius
            )
        ),
    )

    y2 = min(
        height,
        int(
            math.ceil(
                cy + radius
            )
        ),
    )

    if (
        x2 - x1 < 16
        or y2 - y1 < 16
    ):
        return None

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    return (
        crop,
        x1,
        y1,
    )


def _longest_prediction_gap(
    trajectory: list[BallDetection],
) -> int:

    longest = 0
    current = 0
    previous_frame = None

    for detection in trajectory:

        if detection.source == "predicted":

            if (
                previous_frame is not None
                and detection.frame_idx
                == previous_frame + 1
            ):
                current += 1

            else:
                current = 1

            longest = max(
                longest,
                current,
            )

        else:
            current = 0

        previous_frame = (
            detection.frame_idx
        )

    return longest


# =========================================================
# MAIN TRACKER
# =========================================================


def track_ball(
    video_path: Path,
    strike_frame: Optional[int] = None,
    model_name: str = "yolo26n.pt",

    full_frame_confidence: float = 0.10,

    # A weak candidate may be useful locally,
    # but this does NOT automatically make it
    # a trusted observation.
    reacquire_confidence: float = 0.02,

    full_frame_image_size: int = 1280,
    local_image_size: int = 960,

    max_prediction_frames: int = 3,

    # A reacquisition this strong can be trusted
    # immediately.
    direct_reacquire_confidence: float = 0.18,
    direct_reacquire_score: float = 0.60,

    # Otherwise require consecutive agreement.
    tentative_confirm_frames: int = 2,
) -> BallTrackResult:

    model = YOLO(
        model_name
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 30.0
    )

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    motion = BallMotionModel()
    optical = OpticalBallTracker()

    trajectory: list[
        BallDetection
    ] = []

    # Weak candidates waiting for temporal
    # confirmation.
    pending: list[
        BallDetection
    ] = []

    frame_idx = 0

    try:
        while True:

            ok, frame = cap.read()

            if not ok:
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            timestamp_s = (
                frame_idx / fps
            )

            post_strike = (
                strike_frame is not None
                and frame_idx
                >= strike_frame
            )

            if (
                strike_frame is not None
                and frame_idx
                == strike_frame
            ):
                motion.begin_shot()

                pending.clear()

            predicted_center = (
                motion.predict(
                    frame_idx
                )
            )

            selected: Optional[
                BallDetection
            ] = None

            radius: Optional[
                float
            ] = None

            # -----------------------------------------
            # LOCAL SEARCH
            # -----------------------------------------

            if (
                post_strike
                and motion.initialized
                and predicted_center
                is not None
            ):

                radius = (
                    motion.search_radius(
                        target_frame=(
                            frame_idx
                        ),
                        frame_width=(
                            frame_width
                        ),
                        frame_height=(
                            frame_height
                        ),
                    )
                )

                local_data = (
                    _make_local_crop(
                        frame=frame,

                        predicted_center=(
                            predicted_center
                        ),

                        radius=radius,
                    )
                )

                if (
                    local_data
                    is not None
                ):

                    (
                        local_crop,
                        offset_x,
                        offset_y,
                    ) = local_data

                    local_result = (
                        _run_yolo(
                            model=model,

                            image=local_crop,

                            confidence=(
                                reacquire_confidence
                            ),

                            image_size=(
                                local_image_size
                            ),
                        )
                    )

                    local_candidates = (
                        _detections_from_result(
                            result=(
                                local_result
                            ),

                            frame_idx=(
                                frame_idx
                            ),

                            timestamp_s=(
                                timestamp_s
                            ),

                            frame_width=(
                                frame_width
                            ),

                            frame_height=(
                                frame_height
                            ),

                            offset_x=(
                                offset_x
                            ),

                            offset_y=(
                                offset_y
                            ),
                        )
                    )

                    selected = (
                        _select_best_candidate(
                            candidates=(
                                local_candidates
                            ),

                            predicted_center=(
                                predicted_center
                            ),

                            expected_size=(
                                motion.last_size
                            ),

                            gate_radius=(
                                radius
                            ),
                        )
                    )

            # -----------------------------------------
            # FULL-FRAME FALLBACK
            # -----------------------------------------

            if selected is None:

                confidence_threshold = (
                    reacquire_confidence

                    if (
                        post_strike
                        and motion.initialized
                    )

                    else
                    full_frame_confidence
                )

                full_result = (
                    _run_yolo(
                        model=model,
                        image=frame,

                        confidence=(
                            confidence_threshold
                        ),

                        image_size=(
                            full_frame_image_size
                        ),
                    )
                )

                full_candidates = (
                    _detections_from_result(
                        result=full_result,

                        frame_idx=(
                            frame_idx
                        ),

                        timestamp_s=(
                            timestamp_s
                        ),

                        frame_width=(
                            frame_width
                        ),

                        frame_height=(
                            frame_height
                        ),
                    )
                )

                if (
                    motion.initialized
                    and predicted_center
                    is not None
                ):

                    if radius is None:

                        radius = (
                            motion.search_radius(
                                target_frame=(
                                    frame_idx
                                ),

                                frame_width=(
                                    frame_width
                                ),

                                frame_height=(
                                    frame_height
                                ),
                            )
                        )

                    selected = (
                        _select_best_candidate(
                            candidates=(
                                full_candidates
                            ),

                            predicted_center=(
                                predicted_center
                            ),

                            expected_size=(
                                motion.last_size
                            ),

                            gate_radius=(
                                radius * 1.75
                            ),
                        )
                    )

                else:

                    selected = (
                        _select_best_candidate(
                            candidates=(
                                full_candidates
                            ),

                            predicted_center=None,
                            expected_size=None,
                            gate_radius=None,
                        )
                    )

            # =========================================================
            # VISUAL / OPTICAL-FLOW TRACKING
            # =========================================================

            visual_candidate: Optional[
                BallDetection
            ] = None

            if (
                post_strike
                and optical.active
            ):

                optical_radius = radius

                if (
                    optical_radius is None
                    and motion.initialized
                ):

                    optical_radius = (
                        motion.search_radius(
                            target_frame=(
                                frame_idx
                            ),

                            frame_width=(
                                frame_width
                            ),

                            frame_height=(
                                frame_height
                            ),
                        )
                    )

                visual_candidate = (
                    optical.track(
                        gray=gray,

                        frame_idx=(
                            frame_idx
                        ),

                        timestamp_s=(
                            timestamp_s
                        ),

                        frame_width=(
                            frame_width
                        ),

                        frame_height=(
                            frame_height
                        ),

                        predicted_center=(
                            predicted_center
                        ),

                        gate_radius=(
                            optical_radius
                        ),
                    )
                )

            # =========================================================
            # CLASSIFY / FUSE THE EVIDENCE
            # =========================================================

            if selected is not None:

                # -----------------------------------------------------
                # Normal continuous YOLO observation
                # -----------------------------------------------------

                if (
                    not post_strike
                    or motion.missed_frames == 0
                ):

                    selected.source = (
                        "detected"
                    )

                    pending.clear()

                    motion.add_observation(
                        selected
                    )

                    optical.reset(
                        gray,
                        selected,
                    )

                    trajectory.append(
                        selected
                    )

                # -----------------------------------------------------
                # YOLO found something after a tracking gap
                # -----------------------------------------------------

                else:

                    association = (
                        selected.association_score
                        or 0.0
                    )

                    strong_reacquisition = (
                        selected.confidence
                        >= direct_reacquire_confidence

                        and

                        association
                        >= direct_reacquire_score
                    )

                    # ---------------------------------------------
                    # Optical flow independently agrees with YOLO.
                    #
                    # Two different evidence sources finding nearly
                    # the same object is strong reacquisition evidence.
                    # ---------------------------------------------

                    visual_agreement = False

                    if (
                        visual_candidate
                        is not None
                    ):

                        agreement_distance = (
                            math.hypot(
                                selected.center_x
                                - visual_candidate.center_x,

                                selected.center_y
                                - visual_candidate.center_y,
                            )
                        )

                        ball_scale = max(
                            selected.bbox_width,
                            selected.bbox_height,

                            visual_candidate.bbox_width,
                            visual_candidate.bbox_height,

                            4.0,
                        )

                        visual_agreement = (
                            agreement_distance
                            <= ball_scale * 2.5
                        )

                    if (
                        strong_reacquisition
                        or visual_agreement
                    ):

                        selected.source = (
                            "reacquired"
                        )

                        pending.clear()

                        motion.add_observation(
                            selected
                        )

                        optical.reset(
                            gray,
                            selected,
                        )

                        trajectory.append(
                            selected
                        )

                    # ---------------------------------------------
                    # YOLO is weak, but optical flow has a coherent
                    # actual-pixel track.
                    #
                    # Prefer the visual observation rather than
                    # letting the polynomial predictor determine
                    # ball position.
                    # ---------------------------------------------

                    elif (
                        visual_candidate
                        is not None
                    ):

                        pending.clear()

                        motion.add_observation(
                            visual_candidate
                        )

                        trajectory.append(
                            visual_candidate
                        )

                    # ---------------------------------------------
                    # Weak YOLO-only candidate
                    # ---------------------------------------------

                    else:

                        selected.source = (
                            "tentative"
                        )

                        trajectory.append(
                            selected
                        )

                        if (
                            pending
                            and radius
                            is not None
                            and _tentatives_consistent(
                                previous=(
                                    pending[-1]
                                ),

                                current=(
                                    selected
                                ),

                                gate_radius=(
                                    radius
                                ),
                            )
                        ):

                            pending.append(
                                selected
                            )

                        else:

                            pending = [
                                selected
                            ]

                        motion.mark_missed()

                        if (
                            len(pending)
                            >= tentative_confirm_frames
                        ):

                            for confirmed in pending:

                                confirmed.source = (
                                    "reacquired"
                                )

                                motion.add_observation(
                                    confirmed
                                )

                            # Initialize optical flow from the
                            # newest confirmed observation.
                            optical.reset(
                                gray,
                                pending[-1],
                            )

                            pending.clear()


            # =========================================================
            # NO YOLO DETECTION
            # =========================================================

            else:

                # -----------------------------------------------------
                # Optical flow still sees coherent pixel motion.
                #
                # This is the key new behavior.
                # -----------------------------------------------------

                if (
                    visual_candidate
                    is not None
                ):

                    pending.clear()

                    motion.add_observation(
                        visual_candidate
                    )

                    trajectory.append(
                        visual_candidate
                    )

                # -----------------------------------------------------
                # Neither YOLO nor optical flow can see the ball.
                # Only NOW do we use motion prediction.
                # -----------------------------------------------------

                else:

                    pending.clear()

                    if motion.initialized:
                        motion.mark_missed()

                    if (
                        post_strike
                        and motion.initialized
                        and predicted_center
                        is not None
                        and motion.missed_frames
                        <= max_prediction_frames
                    ):

                        px, py = (
                            predicted_center
                        )

                        if (
                            0.0
                            <= px
                            < frame_width

                            and

                            0.0
                            <= py
                            < frame_height
                        ):

                            previous_size = (
                                motion.last_size
                                or 1.0
                            )

                            trajectory.append(
                                BallDetection(
                                    frame_idx=(
                                        frame_idx
                                    ),

                                    timestamp_s=(
                                        timestamp_s
                                    ),

                                    center_x=px,
                                    center_y=py,

                                    center_x_norm=(
                                        px
                                        / frame_width
                                    ),

                                    center_y_norm=(
                                        py
                                        / frame_height
                                    ),

                                    bbox_width=(
                                        previous_size
                                    ),

                                    bbox_height=(
                                        previous_size
                                    ),

                                    confidence=(
                                        motion
                                        .predicted_confidence()
                                    ),

                                    track_id=1,

                                    source=(
                                        "predicted"
                                    ),

                                    association_score=None,
                                )
                            )

            frame_idx += 1

    finally:
        cap.release()

    # =====================================================
    # SUMMARY
    # =====================================================

    trusted = [
        detection
        for detection in trajectory
        if detection.source
        in {
            "detected",
            "reacquired",
            "visual",
        }
    ]

    reacquired_count = sum(
        1
        for detection in trajectory
        if detection.source
        == "reacquired"
    )

    tentative_count = sum(
        1
        for detection in trajectory
        if detection.source
        == "tentative"
    )

    predicted_count = sum(
        1
        for detection in trajectory
        if detection.source
        == "predicted"
    )

    longest_prediction_gap = (
        _longest_prediction_gap(
            trajectory
        )
    )

    if not trusted:

        summary = BallTrackSummary(
            detected=False,

            total_frames=(
                total_frames
            ),

            detection_count=0,
            detection_rate=0.0,

            reacquired_count=(
                reacquired_count
            ),

            tentative_count=(
                tentative_count
            ),

            predicted_count=(
                predicted_count
            ),

            mean_confidence=None,
            peak_confidence=None,

            first_frame=None,
            last_frame=None,

            first_x_norm=None,
            first_y_norm=None,

            final_x_norm=None,
            final_y_norm=None,

            tracking_end_frame=(
                trajectory[-1].frame_idx
                if trajectory
                else None
            ),

            longest_prediction_gap=(
                longest_prediction_gap
            ),
        )

        return BallTrackResult(
            summary=summary,
            detections=trajectory,
        )

    confidences = [
        detection.confidence
        for detection in trusted
    ]

    first = trusted[0]
    last = trusted[-1]

    summary = BallTrackSummary(
        detected=True,

        total_frames=(
            total_frames
        ),

        detection_count=(
            len(trusted)
        ),

        detection_rate=(
            len(trusted)
            / total_frames

            if total_frames
            else 0.0
        ),

        reacquired_count=(
            reacquired_count
        ),

        tentative_count=(
            tentative_count
        ),

        predicted_count=(
            predicted_count
        ),

        mean_confidence=(
            sum(confidences)
            / len(confidences)
        ),

        peak_confidence=(
            max(confidences)
        ),

        first_frame=(
            first.frame_idx
        ),

        last_frame=(
            last.frame_idx
        ),

        first_x_norm=(
            first.center_x_norm
        ),

        first_y_norm=(
            first.center_y_norm
        ),

        final_x_norm=(
            last.center_x_norm
        ),

        final_y_norm=(
            last.center_y_norm
        ),

        tracking_end_frame=(
            trajectory[-1].frame_idx
            if trajectory
            else last.frame_idx
        ),

        longest_prediction_gap=(
            longest_prediction_gap
        ),
    )

    return BallTrackResult(
        summary=summary,
        detections=trajectory,
    )


# =========================================================
# TRAJECTORY VISUALIZATION
# =========================================================


def _draw_dashed_line(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 10,
) -> None:

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    distance = math.hypot(
        dx,
        dy,
    )

    if distance < 1.0:
        return

    ux = dx / distance
    uy = dy / distance

    position = 0.0
    draw_segment = True

    while position < distance:

        next_position = min(
            distance,
            position + dash_length,
        )

        if draw_segment:

            p1 = (
                int(
                    round(
                        start[0]
                        + ux * position
                    )
                ),
                int(
                    round(
                        start[1]
                        + uy * position
                    )
                ),
            )

            p2 = (
                int(
                    round(
                        start[0]
                        + ux
                        * next_position
                    )
                ),
                int(
                    round(
                        start[1]
                        + uy
                        * next_position
                    )
                ),
            )

            cv2.line(
                frame,
                p1,
                p2,
                color,
                thickness,
                cv2.LINE_AA,
            )

        draw_segment = (
            not draw_segment
        )

        position = next_position


def draw_ball_trail(
    video_path: Path,
    detections: list[BallDetection],
    strike_frame: Optional[int],
) -> None:

    if strike_frame is None:
        return

    post_strike = [
        detection
        for detection in detections
        if detection.frame_idx
        >= strike_frame
    ]

    if not post_strike:
        return

    detections_by_frame = {
        detection.frame_idx:
        detection

        for detection
        in post_strike
    }

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video for "
            f"trail rendering: {video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 30.0
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    temp_output = (
        video_path.with_name(
            f"{video_path.stem}"
            "_trail_temp.mp4"
        )
    )

    writer = cv2.VideoWriter(
        str(temp_output),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (
            width,
            height,
        ),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create temporary "
            "ball-trail video."
        )

    accumulated: list[
        BallDetection
    ] = []

    frame_idx = 0

    observed_color = (
        0,
        255,
        255,
    )

    predicted_color = (
        0,
        165,
        255,
    )

    tentative_color = (
        255,
        255,
        0,
    )

    visual_color = (
        255,
        255,
        0,
    )

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            current = (
                detections_by_frame.get(
                    frame_idx
                )
            )

            if current is not None:

                accumulated.append(
                    current
                )

            # -----------------------------------------
            # Draw accumulated trajectory
            # -----------------------------------------

            for previous, next_point in zip(
                accumulated,
                accumulated[1:],
            ):

                p1 = (
                    int(
                        round(
                            previous.center_x
                        )
                    ),
                    int(
                        round(
                            previous.center_y
                        )
                    ),
                )

                p2 = (
                    int(
                        round(
                            next_point.center_x
                        )
                    ),
                    int(
                        round(
                            next_point.center_y
                        )
                    ),
                )

                sources = {
                    previous.source,
                    next_point.source,
                }

                if "predicted" in sources:

                    _draw_dashed_line(
                        frame=frame,
                        start=p1,
                        end=p2,
                        color=(
                            predicted_color
                        ),
                        thickness=2,
                    )

                elif "tentative" in sources:

                    _draw_dashed_line(
                        frame=frame,
                        start=p1,
                        end=p2,
                        color=(
                            tentative_color
                        ),
                        thickness=2,
                        dash_length=6,
                    )

                elif "visual" in sources:

                    cv2.line(
                        frame,
                        p1,
                        p2,
                        visual_color,
                        3,
                        cv2.LINE_AA,
                    )

                else:

                    cv2.line(
                        frame,
                        p1,
                        p2,
                        observed_color,
                        3,
                        cv2.LINE_AA,
                    )

            # -----------------------------------------
            # Current point
            # -----------------------------------------

            if current is not None:

                center = (
                    int(
                        round(
                            current.center_x
                        )
                    ),
                    int(
                        round(
                            current.center_y
                        )
                    ),
                )

                if current.source == "predicted":

                    color = (
                        predicted_color
                    )

                    label = (
                        "BALL EST"
                    )

                    cv2.circle(
                        frame,
                        center,
                        5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                elif current.source == "tentative":

                    color = (
                        tentative_color
                    )

                    label = (
                        "BALL?"
                    )

                    cv2.circle(
                        frame,
                        center,
                        6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                else:

                    if current.source == "predicted":

                        color = (
                            predicted_color
                        )

                        label = (
                            "BALL EST"
                        )

                        cv2.circle(
                            frame,
                            center,
                            5,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                    elif current.source == "tentative":

                        color = (
                            tentative_color
                        )

                        label = (
                            "BALL?"
                        )

                        cv2.circle(
                            frame,
                            center,
                            6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                    elif current.source == "visual":

                        color = (
                            visual_color
                        )

                        label = (
                            "BALL VISUAL"
                        )

                        radius = max(
                            7,
                            int(
                                round(
                                    max(
                                        current.bbox_width,
                                        current.bbox_height,
                                    )
                                    / 2
                                )
                            ),
                        )

                        cv2.circle(
                            frame,
                            center,
                            radius,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                        cv2.circle(
                            frame,
                            center,
                            3,
                            color,
                            -1,
                            cv2.LINE_AA,
                        )

                    else:

                        color = (
                            observed_color
                        )

                        label = (
                            "BALL REACQUIRED"
                            if current.source
                            == "reacquired"

                            else
                            "BALL"
                        )

                        radius = max(
                            8,
                            int(
                                round(
                                    max(
                                        current.bbox_width,
                                        current.bbox_height,
                                    )
                                    / 2
                                )
                            ),
                        )

                        cv2.circle(
                            frame,
                            center,
                            radius,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                        cv2.circle(
                            frame,
                            center,
                            4,
                            color,
                            -1,
                            cv2.LINE_AA,
                        )

                cv2.putText(
                    frame,
                    label,

                    (
                        center[0] + 12,
                        center[1] - 12,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            if (
                frame_idx >= strike_frame
                and len(accumulated) >= 2
            ):

                cv2.putText(
                    frame,
                    "SHOT TRAJECTORY",

                    (
                        18,
                        height - 25,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    observed_color,
                    2,
                    cv2.LINE_AA,
                )

            writer.write(
                frame
            )

            frame_idx += 1

    finally:

        cap.release()
        writer.release()

    subprocess.run(
        [
            "ffmpeg",
            "-y",

            "-i",
            str(temp_output),

            "-c:v",
            "libx264",

            "-pix_fmt",
            "yuv420p",

            "-movflags",
            "+faststart",

            "-an",

            str(video_path),
        ],
        check=True,
    )

    temp_output.unlink(
        missing_ok=True
    )


# =========================================================
# CLI
# =========================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "SCAN validated soccer-ball tracker"
        )
    )

    parser.add_argument(
        "video",
        type=Path,
    )

    parser.add_argument(
        "--model",
        default="yolo26n.pt",
    )

    parser.add_argument(
        "--strike-frame",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if not args.video.exists():

        raise FileNotFoundError(
            args.video
        )

    result = track_ball(
        video_path=args.video,

        strike_frame=(
            args.strike_frame
        ),

        model_name=args.model,
    )

    print(
        json.dumps(
            result.summary.to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()