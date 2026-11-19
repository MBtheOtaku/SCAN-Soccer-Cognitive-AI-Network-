"""Geometry checks for the offline detector's overlapping image slices."""

from types import SimpleNamespace

import numpy as np

from experiments.ball_tracking.closed_loop_detector_track import detect_slices


class _FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _CropAwareModel:
    def __init__(self):
        self.seen_origins = []

    def predict(self, crops, **kwargs):
        assert kwargs == {
            "imgsz": 640, "conf": 0.05, "classes": [0],
            "device": "cpu", "verbose": False,
        }
        results = []
        for crop in crops:
            # The image encodes its own pixel coordinates, so each crop exposes
            # the origin used by detect_slices without consulting any labels.
            left, top = int(crop[0, 0, 0]), int(crop[0, 0, 1])
            self.seen_origins.append((left, top))
            boxes = [[48 - left, 48 - top, 52 - left, 52 - top]]
            confidences = [0.95 if (left, top) == (40, 40) else 0.50]
            if (left, top) == (40, 40):
                boxes.append([38, 38, 42, 42])  # Separate object at (80, 80).
                confidences.append(0.60)
            results.append(SimpleNamespace(boxes=SimpleNamespace(
                xyxy=_FakeTensor(boxes), conf=_FakeTensor(confidences))))
        return results


def test_slices_restore_full_image_coordinates_and_deduplicate_overlap():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(100)[None, :]
    image[:, :, 1] = np.arange(100)[:, None]
    model = _CropAwareModel()

    proposals, crop_count = detect_slices(
        model, image, center=(50, 50), size=60, offset=20, confidence=0.05)

    assert crop_count == 9
    assert set(model.seen_origins) == {
        (left, top) for left in (0, 20, 40) for top in (0, 20, 40)
    }
    # The same ball appears in all nine crops, but must be reported once at
    # its full-image position; a distant detection remains a separate proposal.
    assert proposals == [
        {"center": [50.0, 50.0], "confidence": 0.95},
        {"center": [80.0, 80.0], "confidence": 0.60},
    ]
