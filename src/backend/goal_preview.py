"""Extract a visible reference frame using the same decoder as analysis."""
from math import isfinite
from pathlib import Path
import cv2
import numpy as np


def extract_goal_preview(path: Path) -> tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Could not decode this clip for goal setup.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if isfinite(fps) and fps > 0 else 30
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        last = int(count) - 1 if isfinite(count) and count >= 1 else None
        indexes = [0, 1, *(round(fps * t) for t in (.25, .5, 1, 2))]
        visited = set()
        for index in indexes:
            index = min(index, last) if last is not None else index
            if index in visited:
                continue
            visited.add(index)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if np.mean(gray > 12) < .01:
                continue
            scale = min(1., 1280 / image.shape[1])
            if scale < 1:
                image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
            return image, index
        raise ValueError("No visible reference frame was found near the start. Try another clip or skip goal setup; technique analysis remains available.")
    finally:
        cap.release()
