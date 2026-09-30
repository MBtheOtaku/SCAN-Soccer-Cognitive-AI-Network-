from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import subprocess
import cv2


# COCO class 32 = sports ball
SPORTS_BALL_CLASS_ID = 32


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


@dataclass
class BallTrackSummary:
    detected: bool

    total_frames: int
    detection_count: int
    detection_rate: float

    mean_confidence: Optional[float]
    peak_confidence: Optional[float]

    first_frame: Optional[int]
    last_frame: Optional[int]

    first_x_norm: Optional[float]
    first_y_norm: Optional[float]

    final_x_norm: Optional[float]
    final_y_norm: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BallTrackResult:
    summary: BallTrackSummary
    detections: list[BallDetection]


def track_ball(
    video_path: Path,
    model_name: str = "yolo26n.pt",
    confidence_threshold: float = 0.10,
    image_size: int = 1280,
) -> BallTrackResult:
    """
    Detect and track the soccer ball through a video.

    This is the first SCAN ball-tracking MVP.

    It uses the COCO 'sports ball' class and keeps one
    highest-confidence ball detection per frame.

    The returned per-frame detections will later be used
    for goal-entry and shot-outcome classification.
    """

    try:
        ultralytics_module = importlib.import_module(
            "ultralytics"
        )

        YOLO = getattr(
            ultralytics_module,
            "YOLO",
        )

    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for ball tracking. "
            "Install it with: pip install ultralytics"
        ) from exc

    model = YOLO(model_name)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS) or 30.0
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    detections: list[BallDetection] = []

    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            # persist=True tells the tracker that this frame
            # follows the previous frame in the same video.
            results = model.track(
                frame,
                persist=True,
                classes=[SPORTS_BALL_CLASS_ID],
                conf=confidence_threshold,
                imgsz=image_size,
                verbose=False,
            )

            if not results:
                frame_idx += 1
                continue

            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                frame_idx += 1
                continue

            # There should normally only be one soccer ball.
            # If YOLO returns multiple sports-ball detections,
            # keep the most confident one for this MVP.
            best_index = int(
                boxes.conf.argmax().item()
            )

            confidence = float(
                boxes.conf[best_index].item()
            )

            x1, y1, x2, y2 = (
                boxes.xyxy[best_index]
                .cpu()
                .tolist()
            )

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            track_id: Optional[int] = None

            if boxes.id is not None:
                try:
                    track_id = int(
                        boxes.id[best_index].item()
                    )
                except (ValueError, TypeError):
                    track_id = None

            detections.append(
                BallDetection(
                    frame_idx=frame_idx,
                    timestamp_s=frame_idx / fps,

                    center_x=center_x,
                    center_y=center_y,

                    center_x_norm=(
                        center_x / width
                        if width
                        else 0.0
                    ),

                    center_y_norm=(
                        center_y / height
                        if height
                        else 0.0
                    ),

                    bbox_width=x2 - x1,
                    bbox_height=y2 - y1,

                    confidence=confidence,
                    track_id=track_id,
                )
            )

            frame_idx += 1

    finally:
        cap.release()

    # -----------------------------------------------------
    # No ball found
    # -----------------------------------------------------

    if not detections:
        summary = BallTrackSummary(
            detected=False,

            total_frames=total_frames,
            detection_count=0,
            detection_rate=0.0,

            mean_confidence=None,
            peak_confidence=None,

            first_frame=None,
            last_frame=None,

            first_x_norm=None,
            first_y_norm=None,

            final_x_norm=None,
            final_y_norm=None,
        )

        return BallTrackResult(
            summary=summary,
            detections=[],
        )

    # -----------------------------------------------------
    # Ball found
    # -----------------------------------------------------

    first = detections[0]
    last = detections[-1]

    confidences = [
        detection.confidence
        for detection in detections
    ]

    mean_confidence = (
        sum(confidences) / len(confidences)
    )

    summary = BallTrackSummary(
        detected=True,

        total_frames=total_frames,

        detection_count=len(detections),

        detection_rate=(
            len(detections) / total_frames
            if total_frames
            else 0.0
        ),

        mean_confidence=mean_confidence,
        peak_confidence=max(confidences),

        first_frame=first.frame_idx,
        last_frame=last.frame_idx,

        first_x_norm=first.center_x_norm,
        first_y_norm=first.center_y_norm,

        final_x_norm=last.center_x_norm,
        final_y_norm=last.center_y_norm,
    )

    return BallTrackResult(
        summary=summary,
        detections=detections,
    )

def draw_ball_trail(
    video_path: Path,
    detections: list[BallDetection],
    strike_frame: Optional[int],
    max_gap_frames: int = 3,
) -> None:
    """
    Draw the observed post-strike ball trajectory onto
    SCAN's already-annotated video.

    The trail begins at the estimated strike frame and
    persists for the remainder of the video.

    It only draws observed detections. It does not invent
    or extrapolate a trajectory when the ball is lost.
    """

    if strike_frame is None:
        return

    post_strike = [
        detection
        for detection in detections
        if detection.frame_idx >= strike_frame
    ]

    if not post_strike:
        return

    detections_by_frame = {
        detection.frame_idx: detection
        for detection in post_strike
    }

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video for trail rendering: "
            f"{video_path}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS) or 30.0
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    temp_output = video_path.with_name(
        f"{video_path.stem}_trail_temp.mp4"
    )

    writer = cv2.VideoWriter(
        str(temp_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            "Could not create temporary "
            "ball-trail video."
        )

    # Stores:
    # (frame index, (x, y))
    trail_points: list[
        tuple[int, tuple[int, int]]
    ] = []

    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            detection = detections_by_frame.get(
                frame_idx
            )

            if detection is not None:
                point = (
                    int(round(detection.center_x)),
                    int(round(detection.center_y)),
                )

                trail_points.append(
                    (
                        frame_idx,
                        point,
                    )
                )

            # ---------------------------------------------
            # Draw accumulated trajectory
            # ---------------------------------------------

            for (
                previous,
                current,
            ) in zip(
                trail_points,
                trail_points[1:],
            ):
                previous_frame, previous_point = (
                    previous
                )

                current_frame, current_point = (
                    current
                )

                # Don't connect large gaps where the ball
                # was not actually observed.
                if (
                    current_frame
                    - previous_frame
                    <= max_gap_frames
                ):
                    cv2.line(
                        frame,
                        previous_point,
                        current_point,
                        (0, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )

            # ---------------------------------------------
            # Highlight current observed ball
            # ---------------------------------------------

            if detection is not None:
                current_point = (
                    int(round(detection.center_x)),
                    int(round(detection.center_y)),
                )

                radius = max(
                    8,
                    int(
                        round(
                            max(
                                detection.bbox_width,
                                detection.bbox_height,
                            )
                            / 2
                        )
                    ),
                )

                cv2.circle(
                    frame,
                    current_point,
                    radius,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.circle(
                    frame,
                    current_point,
                    4,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "BALL",
                    (
                        current_point[0] + 12,
                        current_point[1] - 12,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # ---------------------------------------------
            # Label trajectory after the strike
            # ---------------------------------------------

            if (
                frame_idx >= strike_frame
                and len(trail_points) >= 2
            ):
                cv2.putText(
                    frame,
                    "SHOT TRAJECTORY",
                    (18, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
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

    # Convert back to browser-compatible H.264 and
    # replace the original SCAN annotated video.
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SCAN soccer-ball tracker"
    )

    parser.add_argument(
        "video",
        type=Path,
    )

    parser.add_argument(
        "--model",
        default="yolo26n.pt",
    )

    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(
            args.video
        )

    result = track_ball(
        args.video,
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