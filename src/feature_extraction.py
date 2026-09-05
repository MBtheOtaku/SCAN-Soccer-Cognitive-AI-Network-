"""
feature_extraction.py
Placeholder for the computer vision pipeline: takes raw drill footage
(video frames) and produces RepFeatures objects consumed by controller.py.

TODO (Phase 1):
  - Load video / frame stream (OpenCV or similar)
  - Run pose estimation / object detection on ball + player
  - Segment footage into discrete reps
  - Derive error_type, is_hazard, mid_action, scan_rate per rep
  - Return a RepFeatures instance per rep
"""

from player_state import RepFeatures


def extract_rep_features(frame_batch, rep_id: int, timestamp: float) -> RepFeatures:
    """
    Stub function. Replace internals with real CV inference.
    frame_batch: list/array of video frames corresponding to one rep.
    """
    # Placeholder logic - replace with real detection/classification
    return RepFeatures(
        rep_id=rep_id,
        timestamp=timestamp,
        error_type=None,
        is_hazard=False,
        mid_action=False,
        scan_rate=0.0,
        confidence=0.0,
    )
