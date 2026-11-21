"""Native-detail diagnostics preserve scene regions, scale and source identity."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.ball_tracking import probe_native_detail as probe


@pytest.mark.parametrize("width,height,native_width,native_height,center", [
    (720, 1280, 1080, 1920, [360, 640]),
    (720, 1280, 2160, 3840, [0, 0]),
    (1280, 720, 3840, 2160, [5000, -500]),
    (720, 1280, 720, 1280, [719, 1279]),
])
def test_exact_frozen_crop_union_and_native_tiles_cover_matched_region(
        width, height, native_width, native_height, center):
    region = probe.paired_regions(width, height, native_width, native_height, center)
    # Independently reproduce the unchanged detector's original3x3 geometry.
    origins = {(max(0, min(width - 384, round(center[0] + dx - 192))),
                max(0, min(height - 384, round(center[1] + dy - 192))))
               for dx in (-128, 0, 128) for dy in (-128, 0, 128)}
    assert set(region["frozen_origins"]) == origins
    assert region["processing_xyxy"] == [min(p[0] for p in origins), min(p[1] for p in origins),
                                          max(p[0] for p in origins) + 384, max(p[1] for p in origins) + 384]
    left, top, right, bottom = region["native_xyxy"]
    assert 0 <= left < right <= native_width
    assert 0 <= top < bottom <= native_height
    tiles = probe.native_origins(region["native_xyxy"])
    xs, ys = sorted({p[0] for p in tiles}), sorted({p[1] for p in tiles})
    assert xs[0] == left and xs[-1] + 384 == right
    assert ys[0] == top and ys[-1] + 384 == bottom
    assert all(0 < b - a <= 256 for a, b in zip(xs, xs[1:]))
    assert all(0 < b - a <= 256 for a, b in zip(ys, ys[1:]))
    assert all(left <= x <= right - 384 and top <= y <= bottom - 384 for x, y in tiles)


def test_scene_footprint_is_preserved_at4k_while384_native_tiles_retain_detail():
    region = probe.paired_regions(720, 1280, 2160, 3840, [360, 640])
    assert region["processing_xyxy"] == [40, 320, 680, 960]
    assert region["native_xyxy"] == [120, 960, 2040, 2880]
    assert len(probe.native_origins(region["native_xyxy"])) == 49
    assert region["dedup_native_px"] == 45


@pytest.mark.parametrize("native_width,native_height", [(640, 1138), (1080, 1080)])
def test_rejects_source_enlargement_or_wrong_rotation_aspect(native_width, native_height):
    with pytest.raises(ValueError):
        probe.paired_regions(720, 1280, native_width, native_height, [300, 400])


def test_proposal_coordinate_round_trip_and_scene_scaled_deduplication():
    proposals = [{"center": [300, 600], "confidence": 0.8},
                 {"center": [340, 600], "confidence": 0.9},
                 {"center": [390, 600], "confidence": 0.7}]
    kept = probe.deduplicate(proposals, 45)
    assert [p["center"] for p in kept] == [[340, 600], [390, 600]]
    normalized = probe.proposal_spaces(kept, [3, 3], native=True)
    assert normalized[0]["processing_center"] == pytest.approx([340 / 3, 200])
    remapped = probe.proposal_spaces([{"center": normalized[0]["processing_center"], "confidence": 0.9}],
                                    [3, 3], native=False)
    assert remapped[0]["native_center"] == pytest.approx([340, 600])


class Tensor:
    def __init__(self, values):
        self.values = np.array(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def test_native_inference_uses_real384_crops_bounded_batches_and_frozen_model_settings():
    calls = []

    class Model:
        def predict(self, crops, **settings):
            calls.append((len(crops), settings))
            assert all(crop.shape == (384, 384, 3) for crop in crops)
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=Tensor([[10, 20, 20, 30]]), conf=Tensor([0.2]))) for _ in crops]

    image = np.zeros((3840, 2160, 3), dtype=np.uint8)
    region = probe.paired_regions(720, 1280, 2160, 3840, [360, 640])
    proposals, crop_count = probe.detect_native(Model(), image, region)
    assert crop_count == 49
    assert [n for n, _ in calls] == [16, 16, 16, 1]
    assert all(settings == {"imgsz": 640, "conf": 0.05, "classes": [0], "device": "cpu", "verbose": False}
               for _, settings in calls)
    assert proposals[0]["center"] == [135, 985]
    assert len(proposals) == 49


@pytest.mark.parametrize("frames", [None, [], [1, 1], [True], [-1], [99], list(range(25))])
def test_probe_frame_gate_requires_predetermined_bounded_available_frames(frames):
    with pytest.raises(ValueError, match="probe frames"):
        probe.validate_frames(frames, set(range(30)))


def test_probe_frames_are_sorted_without_selecting_on_detector_results():
    assert probe.validate_frames([12, 4, 9], set(range(20))) == [4, 9, 12]


def _case_files(tmp_path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"unchanged-native-original")
    runs = tmp_path / "runs"
    directory = runs / "clip"
    directory.mkdir(parents=True)
    report = {"status": "offline_pilot_not_production", "clip": "clip",
              "source_sha256": probe.file_sha256(source), "weights_sha256": "frozen-weights",
              "implementation_sha256": {"detector": probe.file_sha256(Path(probe.__file__).with_name("closed_loop_detector_track.py"))},
              "settings": probe.SETTINGS.copy(), "rows": [{"frame": 10, "crop_center": [100, 200]}]}
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    case = {"source": str(source), "clip": "clip", "native_probe_frames": [10], "ball_colour": "yellow"}
    return case, report, runs, directory / "report.json"


def test_case_validation_preserves_source_colour_and_rejects_changed_source_weights_settings(tmp_path):
    case, report, runs, report_path = _case_files(tmp_path)
    prepared = probe.prepare_case(case, runs, "frozen-weights")
    assert prepared["source_colour"] == "yellow"
    assert prepared["frames"] == [10]
    for field, value in [("source_sha256", "different-source"), ("weights_sha256", "other-weights"),
                         ("settings", {**probe.SETTINGS, "confidence": 0.01})]:
        altered = {**report, field: value}
        report_path.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(ValueError, match="exact source and frozen detector"):
            probe.prepare_case(case, runs, "frozen-weights")


@pytest.mark.parametrize("implementation", [None, {}, {"detector": "different-code"}])
def test_case_rejects_missing_or_changed_detector_implementation(tmp_path, implementation):
    case, report, runs, report_path = _case_files(tmp_path)
    report["implementation_sha256"] = implementation
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen detector"):
        probe.prepare_case(case, runs, "frozen-weights")


def test_case_preserves_explicit_colour_metadata_without_passing_it_to_detector(tmp_path):
    case, report, runs, report_path = _case_files(tmp_path)
    del case["ball_colour"]
    case["colour_metadata_not_tracker_input"] = "neon green"
    assert probe.prepare_case(case, runs, "frozen-weights")["source_colour"] == "neon green"


@pytest.mark.parametrize("change", [{"source_width": 2160}, {"source_height": 1080},
                                    {"width": 1080}, {"scale_x": 1.0}, {"scale_y": float("nan")}])
def test_supplied_transform_must_match_native_decode_and_processing_geometry(change):
    transform = {"source_width": 1080, "source_height": 1920, "width": 720, "height": 1280,
                 "scale_x": 2 / 3, "scale_y": 2 / 3}
    report = {"width": 720, "height": 1280, "frame_transform": {**transform, **change}}
    with pytest.raises(ValueError, match="geometry|transform"):
        probe.validate_source_geometry(report, 1080, 1920)


def test_transform_validation_reports_legacy_absence_and_accepts_matching_geometry():
    report = {"width": 720, "height": 1280}
    assert probe.validate_source_geometry(report, 1080, 1920) == "legacy_report_transform_unavailable"
    report["frame_transform"] = {"source_width": 1080, "source_height": 1920, "width": 720,
                                 "height": 1280, "scale_x": 2 / 3, "scale_y": 2 / 3}
    assert probe.validate_source_geometry(report, 1080, 1920) == "matched_normalized_report_transform"


def test_rendered_review_labels_raw_proposals_and_places_centres_relative_to_crop(tmp_path):
    region = probe.paired_regions(720, 1280, 1080, 1920, [360, 640])
    proposals = [{"processing_center": [100, 400], "native_center": [150, 600], "confidence": 0.5}]
    row = {"frame": 10, "frame_over_fps_s": 1 / 3, "region": region,
           "baseline": {"image": "clip/10-720.png", "proposals": proposals, "crop_count": 9},
           "native": {"image": "clip/10-native.png", "proposals": proposals, "crop_count": 16}}
    destination = tmp_path / "index.html"
    probe.render_review({"clips": [{"clip": "clip", "ball_colour": "yellow", "frames": [row]}]}, destination)
    page = destination.read_text(encoding="utf-8")
    assert 'cx="60.000" cy="80.000"' in page
    assert 'cx="90.000" cy="120.000"' in page
    assert "raw detector proposals, not verified balls or tracker selections" in page
    assert "More circles do not mean better accuracy" in page
    assert "<script" not in page


def test_probe_decodes_original_sequentially_and_compares_identical_source_frame(tmp_path, monkeypatch):
    import cv2
    import hashlib

    images = [np.full((1920, 1080, 3), n * 20, dtype=np.uint8) for n in range(3)]
    seen, released = [], []

    class Capture:
        index = 0

        def isOpened(self):
            return True

        def get(self, key):
            return 30 if key == cv2.CAP_PROP_FPS else self.index * 1000 / 30

        def read(self):
            seen.append(self.index)
            if self.index == len(images):
                return False, None
            image = images[self.index]
            self.index += 1
            return True, image

        def release(self):
            released.append(True)

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: Capture())
    calls = []

    def baseline(model, image, center, size, offset, confidence):
        calls.append(("baseline", image.shape, int(image[0, 0, 0])))
        assert center == [360, 640]
        return [], 9

    def native(model, image, region):
        calls.append(("native", image.shape, int(image[0, 0, 0])))
        return [], len(probe.native_origins(region["native_xyxy"]))

    monkeypatch.setattr(probe, "detect_slices", baseline)
    monkeypatch.setattr(probe, "detect_native", native)
    source = tmp_path / "video.mov"
    source.write_bytes(b"original")
    report_path = tmp_path / "saved-report.json"
    report_path.write_text("saved")
    case = {"clip": "clip", "source": source, "source_sha256": probe.file_sha256(source),
            "source_colour": "neon green", "report_path": report_path,
            "report": {"width": 720, "height": 1280, "fps": 30},
            "rows": {2: {"crop_center": [360, 640]}}, "frames": [2]}
    result = probe.run_case(case, object(), tmp_path)
    assert seen == [0, 1, 2]
    assert calls == [("baseline", (1280, 720, 3), 40), ("native", (1920, 1080, 3), 40)]
    assert released == [True]
    row = result["frames"][0]
    assert row["source_decoded_pixels_sha256"] == hashlib.sha256(images[2].tobytes()).hexdigest()
    assert row["baseline"]["proposals"] == row["native"]["proposals"] == []
    for method in ("baseline", "native"):
        image_path = tmp_path / row[method]["image"]
        assert image_path.is_file()
        assert probe.file_sha256(image_path) == row[method]["image_sha256"]
