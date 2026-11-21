"""Resolution controls preserve source coordinates and disclose real resizing."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.ball_tracking import track_unlabelled_video as runner


def test_native_frame_is_not_copied_or_resized():
    image = np.zeros((1920, 1080, 3), dtype=np.uint8)

    prepared, transform = runner.prepare_frame(image, None)

    assert prepared is image
    assert transform == {
        "source_width": 1080, "source_height": 1920,
        "width": 1080, "height": 1920,
        "scale_x": 1.0, "scale_y": 1.0,
        "status": "native", "interpolation": "none",
    }


@pytest.mark.parametrize("source_shape, target_shape", [
    ((1920, 1080), (1280, 720)),
    ((3840, 2160), (1280, 720)),
    ((1080, 1920), (720, 1280)),
    ((1921, 1081), (1279, 720)),
])
def test_short_edge_resize_preserves_orientation_and_reports_actual_scale(
        source_shape, target_shape):
    image = np.zeros((*source_shape, 3), dtype=np.uint8)

    prepared, transform = runner.prepare_frame(image, 720)

    assert prepared.shape == (*target_shape, 3)
    assert transform["source_width"] == source_shape[1]
    assert transform["source_height"] == source_shape[0]
    assert transform["width"] == target_shape[1]
    assert transform["height"] == target_shape[0]
    assert transform["scale_x"] == pytest.approx(target_shape[1] / source_shape[1])
    assert transform["scale_y"] == pytest.approx(target_shape[0] / source_shape[0])
    assert transform["status"] == "downsampled"
    assert transform["interpolation"] == "INTER_AREA"


def test_resize_pixels_really_use_area_downsampling():
    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, size=(1920, 1080, 3), dtype=np.uint8)

    prepared, _ = runner.prepare_frame(image, 720)

    expected = cv2.resize(image, (720, 1280), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(prepared, expected)


def test_matching_short_edge_remains_native():
    image = np.zeros((1280, 720, 3), dtype=np.uint8)

    prepared, transform = runner.prepare_frame(image, 720)

    assert prepared is image
    assert transform["status"] == "native"
    assert transform["interpolation"] == "none"
    assert transform["scale_x"] == transform["scale_y"] == 1.0


@pytest.mark.parametrize("target", [True, False, 0, -720, 720.0, 1081])
def test_invalid_target_or_upscaling_is_rejected(target):
    image = np.zeros((1920, 1080, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        runner.prepare_frame(image, target)


def install_capture(monkeypatch, shapes):
    """Decode synthetic source frames while retaining the real resize function."""
    class Capture:
        position = 0
        released = False

        def isOpened(self):
            return True

        def get(self, key):
            if key == cv2.CAP_PROP_FPS:
                return 30.0
            if key == cv2.CAP_PROP_FRAME_COUNT:
                return len(shapes)
            return 0.0

        def read(self):
            if self.position >= len(shapes):
                return False, None
            image = np.full((*shapes[self.position], 3), self.position, dtype=np.uint8)
            self.position += 1
            return True, image

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    def write_frame(path, image, settings):
        Path(path).write_bytes(image[:1, :1].tobytes())
        return True

    monkeypatch.setattr(cv2, "imwrite", write_frame)
    return capture


@pytest.mark.parametrize("target, expected_seed, expected_shape", [
    (None, (900.0, 1500.0), (1920, 1080)),
    (720, (600.0, 1000.0), (1280, 720)),
])
def test_runner_maps_source_seed_once_and_saves_both_coordinate_spaces(
        tmp_path, monkeypatch, target, expected_seed, expected_shape):
    source, weights = tmp_path / "original.mov", tmp_path / "frozen.pt"
    source.write_bytes(b"unmodified source bytes")
    weights.write_bytes(b"frozen weights")
    capture = install_capture(monkeypatch, [(1920, 1080)] * 3)
    observed = []

    def detect(model, image, predicted, size, offset, confidence):
        observed.append((tuple(image.shape[:2]), tuple(predicted)))
        return [{"center": list(expected_seed), "confidence": 0.9}], 9

    monkeypatch.setattr(runner, "detect_slices", detect)
    report = runner.run_video(
        {"source": str(source), "seed": [900, 1500], "start_frame": 1},
        object(), weights, tmp_path / "result", threads=1,
        processing_short_edge=target,
    )

    assert capture.released
    assert observed == [(expected_shape, expected_seed)] * 2
    assert tuple(report["source_seed"]) == (900.0, 1500.0)
    assert tuple(report["seed"]) == expected_seed
    assert report["width"] == expected_shape[1]
    assert report["height"] == expected_shape[0]
    assert report["frame_transform"]["source_width"] == 1080
    assert report["frame_transform"]["source_height"] == 1920
    assert report["frame_transform"]["width"] == expected_shape[1]
    assert report["frame_transform"]["height"] == expected_shape[0]
    assert [frame["frame"] for frame in report["frame_files"]] == [1, 2]
    assert all((frame["height"], frame["width"]) == expected_shape
               for frame in report["frame_files"])
    assert all(tuple(point["center"]) == expected_seed for point in report["path"])
    saved = json.loads((tmp_path / "result/report.json").read_text(encoding="utf-8"))
    assert saved["source_seed"] == [900.0, 1500.0]
    assert saved["seed"] == list(expected_seed)
    assert saved["frame_transform"] == report["frame_transform"]
    assert saved["source_sha256"] == runner.file_sha256(source)


def test_runner_rejects_source_dimension_change_hidden_by_downsampling(tmp_path, monkeypatch):
    source, weights = tmp_path / "changing.mov", tmp_path / "frozen.pt"
    source.write_bytes(b"source")
    weights.write_bytes(b"weights")
    # Both source dimensions would independently resize to the same 720x1280.
    capture = install_capture(monkeypatch, [(1920, 1080), (3840, 2160)])
    monkeypatch.setattr(runner, "detect_slices", lambda *args: ([], 9))

    with pytest.raises(ValueError, match="dimensions changed"):
        runner.run_video({"source": str(source), "seed": [300, 600]}, object(),
                         weights, tmp_path / "result", threads=1,
                         processing_short_edge=720)

    assert capture.released
    assert not (tmp_path / "result/report.json").exists()


def test_native_fallback_supplies_original_pixels_before_unchanged_identity_checks(tmp_path, monkeypatch):
    from experiments.ball_tracking import native_detail_fallback as detail
    from experiments.ball_tracking import native_evidence_verifier as verification

    source, weights = tmp_path / "original.mov", tmp_path / "frozen.pt"
    source.write_bytes(b"source")
    weights.write_bytes(b"weights")
    install_capture(monkeypatch, [(1920, 1080)] * 2)
    monkeypatch.setattr(runner, "detect_slices", lambda *args: ([], 9))
    calls = []

    def augment(model, source_image, transform, predicted, proposals):
        calls.append((source_image.shape, transform["width"], tuple(predicted)))
        return [{"center": [600.0, 1000.0], "confidence": .8,
                 "proposal_source": "native_detail"}], {"status": "synthetic_native_test", "crop_count": 1}

    monkeypatch.setattr(detail, "augment_native_detail", augment)
    class Verification:
        def __init__(self, original, working, seed):
            assert original.shape == (1920, 1080, 3)
            assert working.shape == (1280, 720, 3)
            assert tuple(seed) == (900, 1500)

        def filter(self, frame, original, working, proposals, *, accepted_frames=()):
            assert accepted_frames == list(range(frame))
            return proposals, {"status": "synthetic_verification_test"}

        def configuration(self):
            return {"status": "synthetic_verification_test"}

    monkeypatch.setattr(verification, "NativeEvidenceVerifier", Verification)
    report = runner.run_video({"source": str(source), "seed": [900, 1500]},
                              object(), weights, tmp_path / "result", 1,
                              association="motion-recovery", processing_short_edge=720,
                              native_detail_fallback=True)

    assert calls == [((1920, 1080, 3), 720, (600.0, 1000.0))] * 2
    assert report["native_detail_fallback"] is True
    assert report["seed_kind"] == "one_manual_starting_position_no_reseeding"
    assert report["association_method"] == "motion-recovery"
    assert all(row["native_detail"]["crop_count"] == 1 for row in report["rows"])
    assert all(row["native_evidence_verification"]["status"] == "synthetic_verification_test" for row in report["rows"])
    assert all(point["detected"] and not point["identity_verified"] for point in report["path"])
    assert report["native_detail_implementation_sha256"]["fallback"] == runner.file_sha256(
        runner.Path(detail.__file__))
