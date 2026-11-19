"""Guard the offline contrast probe's fixed-input comparison."""

import json

import numpy as np
import pytest

from experiments.ball_tracking import probe_contrast_detector as probe


def test_probe_uses_same_crop_geometry_and_preserves_original(monkeypatch):
    gradient = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    image = np.dstack((gradient, gradient, gradient))
    unchanged = image.copy()
    calls = []

    def fake_detect(_model, variant, center, size, offset, confidence):
        calls.append((variant.copy(), center, size, offset, confidence))
        return ([{"center": [20.0, 30.0], "confidence": 0.6}], 9)

    monkeypatch.setattr(probe, "detect_slices", fake_detect)
    comparison, variants = probe.probe_frame(object(), image, (120.0, 130.0))
    enhanced = variants['clahe']

    assert np.array_equal(image, unchanged)
    assert enhanced.shape == image.shape and enhanced.dtype == image.dtype
    assert not np.array_equal(enhanced, image)
    assert len(calls) == 3
    assert np.array_equal(calls[0][0], image)
    assert np.array_equal(calls[1][0], enhanced)
    assert np.array_equal(calls[2][0], variants['unsharp'])
    assert calls[0][1:] == calls[1][1:] == calls[2][1:] == ((120.0, 130.0), 384, 128, 0.05)
    assert comparison["crop_count"] == 9


def test_frozen_run_must_match_weights_detector_and_settings(tmp_path):
    report_path = tmp_path / "report.json"
    report = {"weights_sha256": "weights-a",
              "implementation_sha256": {"detector": "detector-a"},
              "settings": probe.FROZEN_SETTINGS,
              "rows": [{"frame": 67, "crop_center": [12.0, 34.0]}],
              "frame_files": [{"frame": 67, "image": "000067.jpg"}]}
    report_path.write_text(json.dumps(report), encoding="utf-8")

    _, rows, files = probe._validated_run(report_path, "weights-a", "detector-a", (67,))
    assert rows[67]["crop_center"] == [12.0, 34.0]
    assert files[67]["image"] == "000067.jpg"
    with pytest.raises(ValueError, match="Weights"):
        probe._validated_run(report_path, "weights-b", "detector-a", (67,))
    with pytest.raises(ValueError, match="Detector implementation"):
        probe._validated_run(report_path, "weights-a", "detector-b", (67,))
    with pytest.raises(ValueError, match="Requested frame"):
        probe._validated_run(report_path, "weights-a", "detector-a", (68,))


def test_motion_map_compensates_translation_and_preserves_inputs():
    import cv2
    rng = np.random.default_rng(5)
    previous = rng.integers(20, 100, (160, 200, 3), dtype=np.uint8)
    current = cv2.warpAffine(previous, np.float32([[1, 0, 4], [0, 1, -3]]), (200, 160))
    current[75:85, 95:105] = 230
    before_copy, after_copy = previous.copy(), current.copy()
    overlay, info = probe.positive_motion_overlay(previous, current)
    assert overlay is not None
    assert info['shift_px'] == pytest.approx([4, -3], abs=0.5)
    assert np.array_equal(previous, before_copy) and np.array_equal(current, after_copy)
    assert not np.array_equal(overlay[75:85, 95:105], current[75:85, 95:105])
    assert np.array_equal(overlay[40:55, 40:55], current[40:55, 40:55])


def test_motion_map_does_not_invent_movement_on_static_textured_frame():
    rng = np.random.default_rng(3)
    image = rng.integers(20, 150, (100, 120, 3), dtype=np.uint8)
    overlay, _ = probe.positive_motion_overlay(image, image.copy())
    assert np.array_equal(overlay, image)
    with pytest.raises(ValueError, match='equal-sized'):
        probe.positive_motion_overlay(image[:50], image)
