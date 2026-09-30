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


# COCO class 32 = sports ball
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

    # SCAN currently maintains one active ball track
    # during a shooting rep.
    track_id: Optional[int]

    # detected   = observed normally
    # reacquired = lost previously, then observed again
    # predicted  = short-horizon motion estimate only
    source: str


@dataclass
class BallTrackSummary:
    detected: bool

    total_frames: int

    # Observed + reacquired detections.
    # Predicted points do NOT count as detections.
    detection_count: int
    detection_rate: float

    reacquired_count: int
    predicted_count: int

    mean_confidence: Optional[float]
    peak_confidence: Optional[float]

    # First / last REAL observation.
    first_frame: Optional[int]
    last_frame: Optional[int]

    first_x_norm: Optional[float]
    first_y_norm: Optional[float]

    final_x_norm: Optional[float]
    final_y_norm: Optional[float]

    # Last frame represented by the complete trajectory,
    # which may include short predicted gaps.
    tracking_end_frame: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BallTrackResult:
    summary: BallTrackSummary
    detections: list[BallDetection]


# =========================================================
# SHORT-HORIZON MOTION MODEL
# =========================================================


class BallMotionModel:
    """
    Short-horizon ball motion model.

    Only REAL observations are stored in history.
    Predicted points never feed back into the model.

    That prevents a bad prediction from recursively
    drifting farther and farther away.
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
        The pre-strike ball is usually stationary.

        Once the strike begins, old stationary samples
        should not dominate the motion estimate.

        Keep only the latest real observation.
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
        """
        Predict ball center at target_frame.

        X is modeled linearly.

        Y uses a short quadratic fit once enough
        observations are available, allowing mild
        curvature / gravity in the image trajectory.

        With fewer observations it falls back to
        linear motion.
        """

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

        # Shift frame numbers close to zero for
        # better numerical stability.
        base_frame = frames[0]

        times = (
            frames - base_frame
        )

        target_time = (
            target_frame - base_frame
        )

        try:
            # Horizontal motion is normally well
            # approximated over this short window
            # by a straight line.
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

            # Allow mild vertical curvature if
            # enough real observations exist.
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
            # Safe fallback: use the last two
            # observations and constant velocity.

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
        """
        Dynamically expand the search region using:

        - previous ball size
        - current estimated speed
        - number of missed frames

        No clip-specific coordinates are used.
        """

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

        # Prevent an uncertainty explosion from
        # effectively turning the local search
        # back into the entire image.
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

        # Confidence decays each frame that the
        # ball is not actually observed.
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


# =========================================================
# YOLO HELPERS
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

        # Local-crop coordinates must be
        # converted back into full-frame space.
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

                # SCAN currently assumes one active
                # soccer ball in a shooting rep.
                track_id=1,

                source="detected",
            )
        )

    return detections


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

    # Hard spatial gate.
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

    # Confidence matters, but when the ball
    # becomes tiny we intentionally allow
    # spatial continuity to provide evidence.
    score = (
        0.50
        * candidate.confidence

        + 0.40
        * distance_score

        + 0.10
        * size_score
    )

    return score


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

    # Before a trajectory exists, use the
    # highest-confidence sports-ball candidate.
    if (
        predicted_center is None
        or gate_radius is None
    ):
        return max(
            candidates,
            key=lambda detection:
            detection.confidence,
        )

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
            gate_radius=gate_radius,
        )

        if score is None:
            continue

        if score > best_score:
            best_score = score
            best_detection = candidate

    # Generic evidence threshold.
    #
    # A weak detector result can still pass if
    # it is very consistent with the trajectory.
    if best_score < 0.25:
        return None

    return best_detection


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


# =========================================================
# BALL TRACKER
# =========================================================


def track_ball(
    video_path: Path,
    strike_frame: Optional[int] = None,
    model_name: str = "yolo26n.pt",

    # Initial / ordinary whole-frame detection.
    full_frame_confidence: float = 0.10,

    # Used only when we already know where the
    # ball is expected to be.
    reacquire_confidence: float = 0.02,

    full_frame_image_size: int = 1280,

    # The small predicted crop is enlarged by
    # YOLO to this input size.
    local_image_size: int = 960,

    # Predictions are allowed only briefly.
    # Real observations are still required for
    # confirmed goal entry.
    max_prediction_frames: int = 8,
) -> BallTrackResult:
    """
    Motion-aware SCAN soccer-ball tracker.

    Strategy:

    PRE-STRIKE
        Full-frame YOLO establishes the ball.

    POST-STRIKE
        1. Predict next position from recent real observations.
        2. Search an adaptive local crop at high effective resolution.
        3. If local search fails, perform full-frame reacquisition.
        4. If both fail, emit a short-lived predicted point.
        5. Never feed predicted points back into the motion model.
    """

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

    trajectory: list[
        BallDetection
    ] = []

    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            timestamp_s = (
                frame_idx / fps
            )

            post_strike = (
                strike_frame is not None
                and frame_idx
                >= strike_frame
            )

            # -----------------------------------------
            # At the strike, remove old stationary
            # history so the sudden ball acceleration
            # can be learned quickly.
            # -----------------------------------------

            if (
                strike_frame is not None
                and frame_idx
                == strike_frame
            ):
                motion.begin_shot()

            predicted_center = (
                motion.predict(
                    frame_idx
                )
            )

            selected: Optional[
                BallDetection
            ] = None

            # -----------------------------------------
            # POST-STRIKE:
            # Search locally FIRST.
            #
            # This is what helps with a ball that is
            # rapidly shrinking in the full image.
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
            # FULL-FRAME SEARCH
            #
            # Used normally before the shot.
            #
            # After the shot it acts as a fallback
            # reacquisition mechanism.
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
                        frame_idx=frame_idx,
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

                    # Full-frame fallback gets a
                    # slightly wider spatial gate.
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

            # -----------------------------------------
            # REAL OBSERVATION
            # -----------------------------------------

            if selected is not None:

                if (
                    motion.missed_frames
                    > 0
                ):
                    selected.source = (
                        "reacquired"
                    )

                else:
                    selected.source = (
                        "detected"
                    )

                motion.add_observation(
                    selected
                )

                trajectory.append(
                    selected
                )

            # -----------------------------------------
            # TEMPORARY PREDICTION
            # -----------------------------------------

            else:

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

                    # Do not emit nonsense outside
                    # the actual frame.
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

                                source="predicted",
                            )
                        )

            frame_idx += 1

    finally:
        cap.release()

    # =====================================================
    # SUMMARY
    # =====================================================

    observed = [
        detection
        for detection in trajectory
        if detection.source
        != "predicted"
    ]

    reacquired_count = sum(
        1
        for detection in trajectory
        if detection.source
        == "reacquired"
    )

    predicted_count = sum(
        1
        for detection in trajectory
        if detection.source
        == "predicted"
    )

    if not observed:

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
        )

        return BallTrackResult(
            summary=summary,
            detections=trajectory,
        )

    confidences = [
        detection.confidence
        for detection in observed
    ]

    first = observed[0]
    last = observed[-1]

    summary = BallTrackSummary(
        detected=True,

        total_frames=(
            total_frames
        ),

        detection_count=(
            len(observed)
        ),

        detection_rate=(
            len(observed)
            / total_frames

            if total_frames
            else 0.0
        ),

        reacquired_count=(
            reacquired_count
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
    )

    return BallTrackResult(
        summary=summary,
        detections=trajectory,
    )


# =========================================================
# TRAIL VISUALIZATION
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
                        + ux
                        * position
                    )
                ),
                int(
                    round(
                        start[1]
                        + uy
                        * position
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
    """
    Add the post-strike ball trajectory to the
    already-generated SCAN annotated video.

    Solid yellow:
        observed / reacquired ball

    Dashed orange:
        short-horizon prediction only
    """

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
            # Draw all trajectory segments accumulated
            # up to this point.
            # -----------------------------------------

            for previous, current_point in zip(
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
                            current_point.center_x
                        )
                    ),
                    int(
                        round(
                            current_point.center_y
                        )
                    ),
                )

                predicted_segment = (
                    previous.source
                    == "predicted"

                    or

                    current_point.source
                    == "predicted"
                )

                if predicted_segment:

                    _draw_dashed_line(
                        frame=frame,
                        start=p1,
                        end=p2,
                        color=(
                            predicted_color
                        ),
                        thickness=2,
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
            # Current point marker
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

                if (
                    current.source
                    == "predicted"
                ):

                    cv2.circle(
                        frame,
                        center,
                        5,
                        predicted_color,
                        2,
                        cv2.LINE_AA,
                    )

                    label = "BALL EST"

                    color = (
                        predicted_color
                    )

                else:

                    radius = max(
                        8,
                        int(
                            round(
                                max(
                                    current
                                    .bbox_width,

                                    current
                                    .bbox_height,
                                )
                                / 2
                            )
                        ),
                    )

                    cv2.circle(
                        frame,
                        center,
                        radius,
                        observed_color,
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.circle(
                        frame,
                        center,
                        4,
                        observed_color,
                        -1,
                        cv2.LINE_AA,
                    )

                    label = (
                        "BALL"
                        if current.source
                        == "detected"

                        else
                        "BALL REACQUIRED"
                    )

                    color = (
                        observed_color
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
                frame_idx
                >= strike_frame

                and len(
                    accumulated
                ) >= 2
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

    # Browser-compatible H.264.
    subprocess.run(
        [
            "ffmpeg",
            "-y",

            "-i",
            str(
                temp_output
            ),

            "-c:v",
            "libx264",

            "-pix_fmt",
            "yuv420p",

            "-movflags",
            "+faststart",

            "-an",

            str(
                video_path
            ),
        ],
        check=True,
    )

    temp_output.unlink(
        missing_ok=True
    )


# =========================================================
# CLI TEST
# =========================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "SCAN motion-aware "
                "soccer-ball tracker"
            )
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