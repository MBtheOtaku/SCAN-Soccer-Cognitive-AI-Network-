"""Dedicated-model adapter and simple overlapping tiles; no new SAHI dependency."""
import math
import cv2
import numpy as np
from ultralytics import YOLO
from scripts.ball_temporal_v2 import Candidate, deduplicate


def tile_starts(length, size=640, overlap=0.25):
    if length <= 0 or size <= 0 or not 0 <= overlap < 1:
        raise ValueError("Invalid tile dimensions or overlap.")
    final = max(0, length - size)
    starts = list(range(0, final + 1, max(1, round(size * (1 - overlap)))))
    return sorted(set(starts + [final]))


class BallDetector:
    def __init__(self, weights, mode="tiled", ball_class_id=None, device="cpu"):
        if mode not in {"single", "tiled"}:
            raise ValueError("Mode must be single or tiled.")
        self.model = YOLO(str(weights))
        names = self.model.names
        names = dict(enumerate(names)) if isinstance(names, list) else names
        if ball_class_id is None:
            matches = [int(k) for k, v in names.items()
                       if str(v).strip().lower().replace("_", " ")
                       in {"ball", "sports ball", "football", "soccer ball"}]
            if len(matches) != 1:
                raise ValueError(f"Specify --ball-class-id after checking model classes: {names}")
            ball_class_id = matches[0]
        if ball_class_id not in names:
            raise ValueError(f"Class {ball_class_id} is not in model metadata: {names}")
        self.ball_class_id, self.mode, self.device = ball_class_id, mode, device
        self.names = names

    def detect(self, frame):
        height, width = frame.shape[:2]
        origins = [(0, 0)] if self.mode == "single" else [
            (x, y) for y in tile_starts(height) for x in tile_starts(width)]
        candidates = []
        for x, y in origins:
            crop = frame if self.mode == "single" else frame[y:y + 640, x:x + 640]
            results = self.model.predict(crop, classes=[self.ball_class_id], conf=0.10,
                                         imgsz=1280 if self.mode == "single" else 640,
                                         max_det=24, device=self.device, verbose=False)
            if not results or results[0].boxes is None:
                continue
            boxes = results[0].boxes
            for coordinates, confidence in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()):
                x1, y1, x2, y2 = coordinates
                x1, x2 = max(0, x1 + x), min(width, x2 + x)
                y1, y2 = max(0, y1 + y), min(height, y2 + y)
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2, confidence)):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                patch = frame[max(0, int(y1)):min(height, math.ceil(y2)),
                              max(0, int(x1)):min(width, math.ceil(x2))]
                if patch.size == 0:
                    continue
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                hist, _ = np.histogramdd(hsv[:, :, :2].reshape(-1, 2),
                                         bins=(8, 4), range=((0, 180), (0, 256)))
                hist = hist.flatten() / max(1, hist.sum())
                candidate = Candidate(x1, y1, x2, y2, confidence, tuple(float(v) for v in hist))
                if candidate.valid():
                    candidates.append(candidate)
        return deduplicate(candidates)
