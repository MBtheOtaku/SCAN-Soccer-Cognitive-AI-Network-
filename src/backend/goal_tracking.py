from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import math

from src.backend.ball_tracking import BallDetection


# ---------------------------------------------------------
# Goal calibration
# ---------------------------------------------------------


@dataclass
class GoalCalibration:
    """
    Four goal corners stored as normalized image coordinates.

    Point order:
        top_left
        top_right
        bottom_right
        bottom_left
    """

    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_right: tuple[float, float]
    bottom_left: tuple[float, float]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ) -> "GoalCalibration":
        return cls(
            top_left=tuple(data["top_left"]),
            top_right=tuple(data["top_right"]),
            bottom_right=tuple(data["bottom_right"]),
            bottom_left=tuple(data["bottom_left"]),
        )

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> "GoalCalibration":
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return cls.from_dict(data)

    def save(
        self,
        path: Path,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
            ),
            encoding="utf-8",
        )

    def pixel_points(
        self,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Convert normalized calibration points back into
        pixel coordinates for the current video.
        """

        normalized = [
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
        ]

        return np.array(
            [
                [
                    x * width,
                    y * height,
                ]
                for x, y in normalized
            ],
            dtype=np.float32,
        )


# ---------------------------------------------------------
# Goal outcome
# ---------------------------------------------------------


@dataclass
class GoalOutcome:
    goal_calibrated: bool

    # True only when SCAN actually observes the ball
    # entering the calibrated goal polygon.
    goal_entry_detected: Optional[bool]

    # True when an observed goal entry is found.
    # Later this will also support observed misses.
    shot_on_target: Optional[bool]

    # Goal-relative coordinates:
    # x: 0 = left post, 1 = right post
    # y: 0 = crossbar, 1 = ground
    entry_x_norm: Optional[float]
    entry_y_norm: Optional[float]

    entry_frame: Optional[int]
    entry_time_s: Optional[float]

    # Examples:
    # upper_left
    # upper_center
    # upper_right
    # lower_left
    # lower_center
    # lower_right
    goal_zone: Optional[str]

    confidence: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------
# Goal zone classification
# ---------------------------------------------------------


def classify_goal_zone(
    x_norm: float,
    y_norm: float,
) -> str:
    """
    Split the goal into 6 zones:

        upper_left | upper_center | upper_right
        lower_left | lower_center | lower_right
    """

    if x_norm < 1.0 / 3.0:
        horizontal = "left"

    elif x_norm > 2.0 / 3.0:
        horizontal = "right"

    else:
        horizontal = "center"

    if y_norm < 0.5:
        vertical = "upper"

    else:
        vertical = "lower"

    return f"{vertical}_{horizontal}"

def _goal_coordinates(
    detection: BallDetection,
    transform: np.ndarray,
) -> tuple[float, float]:
    """
    Convert an image-space ball center into
    normalized goal coordinates.
    """

    point = np.array(
        [
            [
                [
                    detection.center_x,
                    detection.center_y,
                ]
            ]
        ],
        dtype=np.float32,
    )

    mapped = cv2.perspectiveTransform(
        point,
        transform,
    )[0][0]

    return (
        float(mapped[0]),
        float(mapped[1]),
    )


def _inside_unit_goal(
    point: tuple[float, float],
) -> bool:

    x, y = point

    return (
        0.0 <= x <= 1.0
        and
        0.0 <= y <= 1.0
    )


def _segment_goal_entry_t(
    start: tuple[float, float],
    end: tuple[float, float],
) -> Optional[float]:
    """
    Liang-Barsky line clipping against the
    normalized goal rectangle:

        0 <= x <= 1
        0 <= y <= 1

    Returns the trajectory fraction t at which
    the segment first enters the goal.

    Returns None if no entry occurs.
    """

    x0, y0 = start
    x1, y1 = end

    # If we already started inside the goal,
    # this segment cannot establish the original
    # entry event.
    if _inside_unit_goal(
        start
    ):
        return None

    dx = x1 - x0
    dy = y1 - y0

    p = [
        -dx,
        dx,
        -dy,
        dy,
    ]

    q = [
        x0,
        1.0 - x0,
        y0,
        1.0 - y0,
    ]

    t_enter = 0.0
    t_exit = 1.0

    for pi, qi in zip(
        p,
        q,
    ):

        if abs(pi) < 1e-9:

            if qi < 0:
                return None

            continue

        ratio = qi / pi

        if pi < 0:

            t_enter = max(
                t_enter,
                ratio,
            )

        else:

            t_exit = min(
                t_exit,
                ratio,
            )

        if t_enter > t_exit:
            return None

    if not (
        0.0
        <= t_enter
        <= 1.0
    ):
        return None

    return float(
        t_enter
    )


def _detection_evidence(
    detection: BallDetection,
) -> float:
    """
    Combine YOLO confidence with trajectory
    association quality.

    A tiny ball may have low detector confidence
    but still strongly agree with the motion model.
    """

    association = getattr(
        detection,
        "association_score",
        None,
    )

    if association is None:
        association = (
            detection.confidence
        )

    evidence = (
        0.35
        * detection.confidence

        + 0.65
        * association
    )

    return float(
        np.clip(
            evidence,
            0.0,
            1.0,
        )
    )

# ---------------------------------------------------------
# Goal entry detection
# ---------------------------------------------------------


def analyze_goal_outcome(
    detections: list[BallDetection],
    strike_frame: Optional[int],
    calibration: GoalCalibration,
    frame_width: int,
    frame_height: int,

    # Provisional validation thresholds.
    # These are general evidence thresholds,
    # not coordinates or values specific to
    # one video.
    max_observation_gap_frames: int = 4,
    min_entry_confidence: float = 0.35,
) -> GoalOutcome:
    """
    Detect an OBSERVED trajectory crossing into
    the calibrated goal.

    Predicted and tentative points may help the
    ball tracker search, but they cannot prove
    a goal.

    A goal entry requires a credible trajectory
    segment supported by trusted observations.
    """

    if strike_frame is None:

        return GoalOutcome(
            goal_calibrated=True,

            goal_entry_detected=None,
            shot_on_target=None,

            entry_x_norm=None,
            entry_y_norm=None,

            entry_frame=None,
            entry_time_s=None,

            goal_zone=None,
            confidence=None,
        )

    # -----------------------------------------------------
    # TRUSTED post-strike observations only
    # -----------------------------------------------------

    trusted = [
        detection

        for detection in detections

        if (
            detection.frame_idx
            >= strike_frame

            and

            detection.source
            in {
                "detected",
                "reacquired",
                "visual",
            }
        )
    ]

    trusted.sort(
        key=lambda detection:
        detection.frame_idx
    )

    if len(trusted) < 2:

        return GoalOutcome(
            goal_calibrated=True,

            goal_entry_detected=None,
            shot_on_target=None,

            entry_x_norm=None,
            entry_y_norm=None,

            entry_frame=None,
            entry_time_s=None,

            goal_zone=None,
            confidence=None,
        )

    # -----------------------------------------------------
    # Map image-space goal into unit square
    # -----------------------------------------------------

    goal_points = (
        calibration.pixel_points(
            frame_width,
            frame_height,
        )
    )

    destination = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    transform = (
        cv2.getPerspectiveTransform(
            goal_points,
            destination,
        )
    )

    # -----------------------------------------------------
    # Look for the FIRST credible outside -> goal crossing
    # -----------------------------------------------------

    for previous, current in zip(
        trusted,
        trusted[1:],
    ):

        frame_gap = (
            current.frame_idx
            - previous.frame_idx
        )

        # Too much unobserved time exists between
        # these points to claim a precise crossing.
        if (
            frame_gap <= 0
            or frame_gap
            > max_observation_gap_frames
        ):
            continue

        previous_goal = (
            _goal_coordinates(
                previous,
                transform,
            )
        )

        current_goal = (
            _goal_coordinates(
                current,
                transform,
            )
        )

        entry_t = (
            _segment_goal_entry_t(
                previous_goal,
                current_goal,
            )
        )

        if entry_t is None:
            continue

        entry_x = (
            previous_goal[0]
            + entry_t
            * (
                current_goal[0]
                - previous_goal[0]
            )
        )

        entry_y = (
            previous_goal[1]
            + entry_t
            * (
                current_goal[1]
                - previous_goal[1]
            )
        )

        entry_x = float(
            np.clip(
                entry_x,
                0.0,
                1.0,
            )
        )

        entry_y = float(
            np.clip(
                entry_y,
                0.0,
                1.0,
            )
        )

        previous_evidence = (
            _detection_evidence(
                previous
            )
        )

        current_evidence = (
            _detection_evidence(
                current
            )
        )

        # Geometric mean means both endpoints
        # need reasonable support.
        evidence = math.sqrt(
            previous_evidence
            * current_evidence
        )

        # Penalize larger observational gaps.
        gap_penalty = math.exp(
            -0.20
            * max(
                0,
                frame_gap - 1,
            )
        )

        entry_confidence = (
            evidence
            * gap_penalty
        )

        # The segment geometrically crosses the
        # goal, but visual evidence is not strong
        # enough to call it a confirmed goal.
        if (
            entry_confidence
            < min_entry_confidence
        ):
            continue

        entry_frame_float = (
            previous.frame_idx
            + entry_t
            * frame_gap
        )

        entry_time_s = (
            previous.timestamp_s
            + entry_t
            * (
                current.timestamp_s
                - previous.timestamp_s
            )
        )

        zone = classify_goal_zone(
            entry_x,
            entry_y,
        )

        return GoalOutcome(
            goal_calibrated=True,

            goal_entry_detected=True,
            shot_on_target=True,

            entry_x_norm=(
                entry_x
            ),

            entry_y_norm=(
                entry_y
            ),

            entry_frame=int(
                round(
                    entry_frame_float
                )
            ),

            entry_time_s=float(
                entry_time_s
            ),

            goal_zone=zone,

            confidence=float(
                entry_confidence
            ),
        )

    # -----------------------------------------------------
    # No sufficiently supported crossing observed.
    #
    # IMPORTANT:
    # this does NOT mean "miss".
    # -----------------------------------------------------

    return GoalOutcome(
        goal_calibrated=True,

        goal_entry_detected=False,

        shot_on_target=None,

        entry_x_norm=None,
        entry_y_norm=None,

        entry_frame=None,
        entry_time_s=None,

        goal_zone=None,

        confidence=None,
    )

# ---------------------------------------------------------
# Interactive calibration
# ---------------------------------------------------------


def calibrate_goal(
    video_path: Path,
    output_path: Path,
    frame_index: int = 0,
) -> GoalCalibration:
    """
    Open a frame and let the user click the four goal corners.

    Click order:

        1. Top-left
        2. Top-right
        3. Bottom-right
        4. Bottom-left

    Press ENTER after all four points are selected.
    Press R to reset.
    Press ESC or Q to cancel.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    ok, frame = cap.read()

    cap.release()

    if not ok:
        raise RuntimeError(
            f"Could not read frame {frame_index}"
        )

    height, width = frame.shape[:2]

    clicked: list[
        tuple[int, int]
    ] = []

    labels = [
        "TOP LEFT",
        "TOP RIGHT",
        "BOTTOM RIGHT",
        "BOTTOM LEFT",
    ]

    window_name = "SCAN Goal Calibration"

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param,
    ):
        if (
            event
            == cv2.EVENT_LBUTTONDOWN
            and len(clicked) < 4
        ):
            clicked.append(
                (x, y)
            )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    print()
    print("SCAN GOAL CALIBRATION")
    print("---------------------")
    print("Click the goal corners in this order:")
    print("1. TOP LEFT")
    print("2. TOP RIGHT")
    print("3. BOTTOM RIGHT")
    print("4. BOTTOM LEFT")
    print()
    print("ENTER = save")
    print("R     = reset")
    print("Q/ESC = cancel")
    print()

    while True:
        display = frame.copy()

        cv2.putText(
            display,
            (
                "Click: TL -> TR -> BR -> BL"
            ),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for index, point in enumerate(
            clicked
        ):
            cv2.circle(
                display,
                point,
                7,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                str(index + 1),
                (
                    point[0] + 10,
                    point[1] - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if index > 0:
                cv2.line(
                    display,
                    clicked[index - 1],
                    point,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        if len(clicked) == 4:
            cv2.line(
                display,
                clicked[3],
                clicked[0],
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "Press ENTER to save",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(
            window_name,
            display,
        )

        key = (
            cv2.waitKey(20)
            & 0xFF
        )

        if key in (
            ord("r"),
            ord("R"),
        ):
            clicked.clear()

        elif key in (
            ord("q"),
            ord("Q"),
            27,
        ):
            cv2.destroyAllWindows()

            raise RuntimeError(
                "Goal calibration cancelled."
            )

        elif key in (
            13,
            10,
        ):
            if len(clicked) == 4:
                break

    cv2.destroyAllWindows()

    normalized = [
        (
            x / width,
            y / height,
        )
        for x, y in clicked
    ]

    calibration = GoalCalibration(
        top_left=normalized[0],
        top_right=normalized[1],
        bottom_right=normalized[2],
        bottom_left=normalized[3],
    )

    calibration.save(
        output_path
    )

    print(
        f"Goal calibration saved to: "
        f"{output_path}"
    )

    return calibration

def _lerp_point(
    p1: tuple[int, int],
    p2: tuple[int, int],
    t: float,
) -> tuple[int, int]:
    """
    Linear interpolation between two 2D points.
    """
    x = int(round(p1[0] + (p2[0] - p1[0]) * t))
    y = int(round(p1[1] + (p2[1] - p1[1]) * t))
    return (x, y)


def draw_goal_overlay(
    frame: np.ndarray,
    calibration: GoalCalibration,
) -> None:
    """
    Draw the calibrated goal boundary and 6-zone grid
    onto the current video frame.

    GoalCalibration stores normalized coordinates,
    so they are converted back to integer pixel
    coordinates before drawing.
    """

    height, width = frame.shape[:2]

    # -----------------------------------------------------
    # Convert normalized calibration coordinates
    # back into frame pixel coordinates.
    #
    # pixel_points() returns:
    # TL, TR, BR, BL
    # -----------------------------------------------------

    goal_points = calibration.pixel_points(
        width,
        height,
    )

    top_left = (
        int(round(goal_points[0][0])),
        int(round(goal_points[0][1])),
    )

    top_right = (
        int(round(goal_points[1][0])),
        int(round(goal_points[1][1])),
    )

    bottom_right = (
        int(round(goal_points[2][0])),
        int(round(goal_points[2][1])),
    )

    bottom_left = (
        int(round(goal_points[3][0])),
        int(round(goal_points[3][1])),
    )

    goal_polygon = np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        ],
        dtype=np.int32,
    )

    # -----------------------------------------------------
    # Transparent goal-region fill
    # -----------------------------------------------------

    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        [goal_polygon],
        (40, 120, 40),
    )

    cv2.addWeighted(
        overlay,
        0.10,
        frame,
        0.90,
        0,
        frame,
    )

    # -----------------------------------------------------
    # Outer calibrated goal boundary
    # -----------------------------------------------------

    cv2.polylines(
        frame,
        [goal_polygon],
        isClosed=True,
        color=(80, 255, 180),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    # -----------------------------------------------------
    # Corner markers
    # -----------------------------------------------------

    for point in (
        top_left,
        top_right,
        bottom_right,
        bottom_left,
    ):
        cv2.circle(
            frame,
            point,
            5,
            (80, 255, 180),
            -1,
            cv2.LINE_AA,
        )

    # -----------------------------------------------------
    # Vertical thirds
    # -----------------------------------------------------

    for t in (
        1.0 / 3.0,
        2.0 / 3.0,
    ):

        top_point = _lerp_point(
            top_left,
            top_right,
            t,
        )

        bottom_point = _lerp_point(
            bottom_left,
            bottom_right,
            t,
        )

        cv2.line(
            frame,
            top_point,
            bottom_point,
            (120, 220, 120),
            1,
            cv2.LINE_AA,
        )

    # -----------------------------------------------------
    # Horizontal halfway line
    # -----------------------------------------------------

    left_mid = _lerp_point(
        top_left,
        bottom_left,
        0.5,
    )

    right_mid = _lerp_point(
        top_right,
        bottom_right,
        0.5,
    )

    cv2.line(
        frame,
        left_mid,
        right_mid,
        (120, 220, 120),
        1,
        cv2.LINE_AA,
    )

    # -----------------------------------------------------
    # Goal label
    # -----------------------------------------------------

    label_x = max(
        5,
        top_left[0],
    )

    label_y = max(
        22,
        top_left[1] - 10,
    )

    cv2.putText(
        frame,
        "CALIBRATED GOAL",
        (
            label_x,
            label_y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (80, 255, 180),
        2,
        cv2.LINE_AA,
    )

    # -----------------------------------------------------
    # Zone labels
    # -----------------------------------------------------

    def zone_center(
        x_t: float,
        y_t: float,
    ) -> tuple[int, int]:

        top_position = _lerp_point(
            top_left,
            top_right,
            x_t,
        )

        bottom_position = _lerp_point(
            bottom_left,
            bottom_right,
            x_t,
        )

        return _lerp_point(
            top_position,
            bottom_position,
            y_t,
        )

    zones = [
        (
            "UL",
            1.0 / 6.0,
            0.25,
        ),
        (
            "UC",
            0.5,
            0.25,
        ),
        (
            "UR",
            5.0 / 6.0,
            0.25,
        ),
        (
            "LL",
            1.0 / 6.0,
            0.75,
        ),
        (
            "LC",
            0.5,
            0.75,
        ),
        (
            "LR",
            5.0 / 6.0,
            0.75,
        ),
    ]

    for (
        label,
        x_t,
        y_t,
    ) in zones:

        center_x, center_y = (
            zone_center(
                x_t,
                y_t,
            )
        )

        cv2.putText(
            frame,
            label,
            (
                center_x - 10,
                center_y + 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SCAN goal calibration"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    calibrate_parser = (
        subparsers.add_parser(
            "calibrate"
        )
    )

    calibrate_parser.add_argument(
        "video",
        type=Path,
    )

    calibrate_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/goal_calibration.json"
        ),
    )

    calibrate_parser.add_argument(
        "--frame",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if args.command == "calibrate":
        if not args.video.exists():
            raise FileNotFoundError(
                args.video
            )

        calibrate_goal(
            video_path=args.video,
            output_path=args.output,
            frame_index=args.frame,
        )


if __name__ == "__main__":
    main()