from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

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


# ---------------------------------------------------------
# Goal entry detection
# ---------------------------------------------------------


def analyze_goal_outcome(
    detections: list[BallDetection],
    strike_frame: Optional[int],
    calibration: GoalCalibration,
    frame_width: int,
    frame_height: int,
) -> GoalOutcome:
    """
    Check whether the observed post-strike ball trajectory
    enters the calibrated goal.

    IMPORTANT:
    This currently reports only OBSERVED goal entries.

    If SCAN loses the ball before it reaches the goal,
    it returns goal_entry_detected=False but does NOT
    automatically call the shot a miss.
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

    post_strike = [
        detection
        for detection in detections
        if detection.frame_idx >= strike_frame
    ]

    if not post_strike:
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
    # Build goal polygon
    # -----------------------------------------------------

    goal_points = calibration.pixel_points(
        frame_width,
        frame_height,
    )

    goal_polygon = goal_points.astype(
        np.float32
    )

    # -----------------------------------------------------
    # Perspective transform
    #
    # Maps the actual goal shape in the video into:
    #
    # (0,0) ------------ (1,0)
    #   |                  |
    #   |                  |
    # (0,1) ------------ (1,1)
    # -----------------------------------------------------

    destination = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(
        goal_points,
        destination,
    )

    # -----------------------------------------------------
    # Find first observed point inside goal
    # -----------------------------------------------------

    for detection in post_strike:
        point = (
            float(detection.center_x),
            float(detection.center_y),
        )

        inside = cv2.pointPolygonTest(
            goal_polygon,
            point,
            False,
        )

        if inside < 0:
            continue

        point_array = np.array(
            [[[point[0], point[1]]]],
            dtype=np.float32,
        )

        mapped = cv2.perspectiveTransform(
            point_array,
            transform,
        )[0][0]

        goal_x = float(
            np.clip(
                mapped[0],
                0.0,
                1.0,
            )
        )

        goal_y = float(
            np.clip(
                mapped[1],
                0.0,
                1.0,
            )
        )

        zone = classify_goal_zone(
            goal_x,
            goal_y,
        )

        return GoalOutcome(
            goal_calibrated=True,

            goal_entry_detected=True,
            shot_on_target=True,

            entry_x_norm=goal_x,
            entry_y_norm=goal_y,

            entry_frame=detection.frame_idx,
            entry_time_s=detection.timestamp_s,

            goal_zone=zone,

            confidence=detection.confidence,
        )

    # -----------------------------------------------------
    # Ball was tracked after the strike, but was never
    # actually observed inside the goal.
    #
    # We do NOT call this a miss yet, because the detector
    # may simply have lost the ball before it reached goal.
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