"""Locate a single player before asking MediaPipe for detailed landmarks.

The fixed crop is a padded movement envelope, in decoded video coordinates.
It is not an athlete-identity classifier or a substitute for pose confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class PersonCrop:
    bounds: tuple[int, int, int, int]
    sampled_frames: int
    person_frames: int
    edge_frames: int
    background_candidate_frames: int = 0


class AmbiguousPlayerCrop(RuntimeError):
    """More than one person prevents automatic target selection."""


def person_envelope(
    samples: Sequence[Sequence[Sequence[float]]],
    width: int,
    height: int,
    padding: float = .30,
) -> PersonCrop:
    """Build a crop for a single, clearly dominant foreground player.

    Boxes contain x1, y1, x2, y2, confidence. Empty samples remain empty; no
    interpolation, pose-confidence changes or synthetic measurements occur.
    Much smaller background proposals may be excluded by a provisional scale
    check. Comparable foreground people remain ambiguous; identity is unverified.
    """
    if width < 50 or height < 50 or not math.isfinite(padding) or padding < 0:
        raise ValueError("Invalid player-crop dimensions or padding.")
    selected = []
    edge_frames = 0
    background_candidate_frames = 0
    for candidates in samples:
        valid = []
        for candidate in candidates:
            if len(candidate) != 5 or not all(math.isfinite(v) for v in candidate):
                continue
            x1, y1, x2, y2, score = candidate
            if not .35 <= score <= 1.0:
                continue
            x1, x2 = max(0., x1), min(float(width), x2)
            y1, y2 = max(0., y1), min(float(height), y2)
            if x2 - x1 >= 12 and y2 - y1 >= 24:
                valid.append((x1, y1, x2, y2))
        if len(valid) > 1:
            primary = max(valid, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            primary_height = primary[3] - primary[1]
            primary_area = (primary[2] - primary[0]) * primary_height
            foreground = [primary]
            for candidate in valid:
                if candidate is primary:
                    continue
                candidate_height = candidate[3] - candidate[1]
                candidate_area = (candidate[2] - candidate[0]) * candidate_height
                # Goal structures and distant people can produce small person
                # proposals. Require both large scale separations rather than
                # simply picking the largest of two plausible athletes.
                if candidate_height <= .35 * primary_height and candidate_area <= .15 * primary_area:
                    continue
                foreground.append(candidate)
            if len(foreground) > 1:
                raise AmbiguousPlayerCrop("Multiple foreground people detected; automatic player selection is uncertain.")
            valid = foreground
            background_candidate_frames += 1
        if valid:
            box = valid[0]
            selected.append(box)
            # A detector box at an image edge can indicate a clipped body. This
            # is a framing diagnostic, not proof that every landmark is visible.
            if box[0] <= 2 or box[1] <= 2 or box[2] >= width - 2 or box[3] >= height - 2:
                edge_frames += 1
    if len(selected) < 3:
        raise RuntimeError("Too few repeated person detections for a player crop.")
    margin = max(8., padding * max(b[3] - b[1] for b in selected))
    bounds = (
        max(0, math.floor(min(b[0] for b in selected) - margin)),
        max(0, math.floor(min(b[1] for b in selected) - margin)),
        min(width, math.ceil(max(b[2] for b in selected) + margin)),
        min(height, math.ceil(max(b[3] for b in selected) + margin)),
    )
    if bounds[2] - bounds[0] < 50 or bounds[3] - bounds[1] < 50:
        raise RuntimeError("Detected player crop is too small.")
    return PersonCrop(bounds, len(samples), len(selected), edge_frames, background_candidate_frames)


def sample_frames_sequential(capture, indices):
    """Decode each requested frame once, avoiding repeated GOP seeking."""
    indices = list(indices)
    if any(type(i) is not int or i < 0 for i in indices) or indices != sorted(set(indices)):
        raise ValueError("Expected distinct increasing nonnegative frame indices")
    current, exhausted = 0, False
    for index in indices:
        frame = None
        while not exhausted and current <= index:
            ok, decoded = capture.read()
            current += 1
            if not ok:
                exhausted = True
            elif current - 1 == index:
                frame = decoded
        yield index, frame


def detect_person_crop(
    video_path: Path,
    weights_path: Path | None = None,
    samples: int = 12,
    padding: float = .30,
) -> PersonCrop:
    """Sample a local person detector; never download weights during analysis."""
    weights_path = weights_path or Path(__file__).resolve().parents[3] / "data/models/yolo11n.pt"
    if not weights_path.is_file():
        raise RuntimeError("Local person detector weights are unavailable.")
    if samples < 3:
        raise ValueError("Player calibration requires at least three samples.")
    from ultralytics import YOLO

    detector = YOLO(str(weights_path))
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open video for player calibration.")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 3:
            raise RuntimeError("Too few video frames for player calibration.")
        observations = []
        shape = None
        indices = [int(i) for i in np.linspace(0, count - 1, min(samples, count), dtype=int)]
        for index, frame in sample_frames_sequential(cap, indices):
            if frame is None:
                observations.append([])
                continue
            # Read orientation from decoded pixels, including phone rotation.
            if shape is not None and shape != frame.shape[:2]:
                raise RuntimeError("Video dimensions changed during calibration.")
            shape = frame.shape[:2]
            result = detector.predict(
                frame, classes=[0], imgsz=1280, conf=.35, device="cpu", verbose=False,
            )[0]
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            scores = result.boxes.conf.cpu().numpy().tolist()
            observations.append([box + [score] for box, score in zip(boxes, scores)])
        if shape is None:
            raise RuntimeError("No video frames could be decoded for player calibration.")
        return person_envelope(observations, shape[1], shape[0], padding)
    finally:
        cap.release()
