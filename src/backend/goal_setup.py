"""Validation of user-confirmed, request-specific goal setup."""
import json
import cv2
import numpy as np
from src.backend.goal_tracking import GoalCalibration


def parse_goal_setup(raw: str | None) -> GoalCalibration | None:
    if raw is None or raw == "null":
        return None
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("confirmed") is not True:
        raise ValueError("Confirm the goal corners before analysis.")
    points = np.array(data["corners"], dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
        raise ValueError("Four normalized goal corners are required.")
    if not cv2.isContourConvex(points) or cv2.contourArea(points, oriented=True) < .001:
        raise ValueError("Goal corners must form a clockwise, nonempty polygon.")
    tl, tr, br, bl = points
    if not (tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]):
        raise ValueError("Corner order must be top-left, top-right, bottom-right, bottom-left.")
    return GoalCalibration(*(tuple(float(v) for v in point) for point in points))


def decode_reference(contents: bytes):
    if not contents or len(contents) > 5_000_000:
        raise ValueError("Use a goal reference image smaller than 5 MB.")
    try:
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("Invalid goal reference image.") from exc
    if image is None or min(image.shape[:2]) < 16 or image.shape[0] * image.shape[1] > 12_000_000:
        raise ValueError("Invalid or oversized goal reference image.")
    return image
