from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

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