"""Reference-anchored goal calibration for camera pan, rotation and modest zoom.

Initial goal corners remain manual. Optical flow observes reference features on
the goal/background; robust fitting moves the corners. Failed frames return None,
never the last good calibration. No predicted polygon can prove a goal entry.
"""
import cv2
import numpy as np
from src.backend.goal_tracking import GoalCalibration


class GoalMotionTracker:
    def __init__(self, reference: np.ndarray, calibration: GoalCalibration, external_reference: bool = False):
        self.external_reference = external_reference
        self.calibration = calibration
        self.height, self.width = reference.shape[:2]
        self.reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.corners = calibration.pixel_points(self.width, self.height)
        if not np.isfinite(self.corners).all() or not cv2.isContourConvex(self.corners):
            raise ValueError("Goal corners must form a finite convex polygon.")
        if cv2.contourArea(self.corners) < 100:
            raise ValueError("Goal calibration is too small to track.")
        x, y, w, h = cv2.boundingRect(self.corners)
        # Include nearby static detail; exclude the foreground below the goal.
        mask = np.zeros_like(self.reference)
        mask[max(0, y - h // 2):min(self.height, y + h), max(0, x - w // 10):min(self.width, x + w + w // 10)] = 255
        self.points = cv2.goodFeaturesToTrack(self.reference, 300, .02, 8, mask=mask)
        self.last_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.valid_frames = 0
        self.lost_frames = 0

    def update(self, frame: np.ndarray, frame_index: int):
        try:
            result = self._observe(frame, frame_index)
        except (cv2.error, ValueError):
            result = None
        if result is None:
            self.lost_frames += 1
        else:
            self.valid_frames += 1
        return result

    def _observe(self, frame, frame_index):
        if frame.shape[:2] != (self.height, self.width):
            return None
        if not self.external_reference and frame_index == self.calibration.reference_frame:
            self.last_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            return self.calibration
        if self.points is None or len(self.points) < 15:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        guess = cv2.transform(self.points, self.last_matrix)
        target, forward, error = cv2.calcOpticalFlowPyrLK(
            self.reference, gray, self.points, guess, winSize=(25, 25), maxLevel=4,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        if target is None:
            return None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.reference, target, None, winSize=(25, 25), maxLevel=4)
        if backward is None:
            return None
        good = (forward.ravel() == 1) & (back_status.ravel() == 1) & (error.ravel() < 20)
        good &= np.linalg.norm(backward - self.points, axis=2).ravel() < 1
        if good.sum() < 15:
            return None
        source = self.points[good].reshape(-1, 2)
        target = target[good].reshape(-1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=2)
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            return None
        selected = inliers.ravel().astype(bool)
        if selected.sum() < 15 or selected.mean() < .65:
            return None
        # Distributed support keeps one passing player's features from steering it.
        span = np.ptp(source[selected], axis=0)
        goal_span = np.ptp(self.corners, axis=0)
        if np.any(span < goal_span * np.array([.5, .3])):
            return None
        scale = np.hypot(matrix[0, 0], matrix[1, 0])
        if not .7 < scale < 1.4:
            return None
        moved = cv2.transform(self.corners[None], matrix)[0]
        if np.any(moved < 0) or np.any(moved[:, 0] >= self.width) or np.any(moved[:, 1] >= self.height):
            return None
        self.last_matrix = matrix.astype(np.float32)
        points = moved / np.array([self.width, self.height])
        return GoalCalibration(*(tuple(point) for point in points), reference_frame=self.calibration.reference_frame)

    def summary(self):
        return {"mode": "reference_anchored_optical_flow", "reference_frame": None if self.external_reference else self.calibration.reference_frame,
                "reference_source": "captured_photo" if self.external_reference else "clip_frame",
                "valid_frames": self.valid_frames, "lost_frames": self.lost_frames,
                "confidence_note": "Initial corners must match this clip. Tracking supports pan, rotation and modest zoom; large viewpoint changes, cuts or weak visual detail can require recalibration. Lost frames do not support goal judgements."}
