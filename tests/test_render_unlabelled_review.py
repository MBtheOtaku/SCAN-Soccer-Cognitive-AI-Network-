"""Checks for the offline, unlabelled ball-path review renderer."""

import copy
import json
import re

import pytest

from experiments.ball_tracking.render_unlabelled_review import render_review


def _report(tmp_path):
    frames = tmp_path / "source-frames"
    frames.mkdir()
    for frame in (10, 11):
        (frames / f"{frame:06d}.jpg").write_bytes(f"original frame {frame}".encode())
    return {
        "status": "offline_pilot_not_production",
        "clip": "unseen_shot_1",
        "split": "unseen_session_unlabelled",
        "source_name": "IMG_9999.mov",
        "source_sha256": "a" * 64,
        "weights_sha256": "b" * 64,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "start_frame": 10,
        "end_frame": 11,
        "seed": [200.0, 800.0],
        "frames_directory": str(frames),
        "rows": [
            {"frame": 10, "crop_center": [200.0, 800.0], "candidate_count": 1,
             "candidates": [{"center": [202.0, 802.0], "confidence": 0.8}]},
            {"frame": 11, "crop_center": [204.0, 804.0], "candidate_count": 0,
             "candidates": []},
        ],
        "path": [
            {"frame": 10, "center": [202.0, 802.0], "confidence": 0.8,
             "predicted": [200.0, 800.0], "detected": True},
            {"frame": 11, "center": None, "confidence": None,
             "predicted": [204.0, 804.0], "detected": False},
        ],
        "frame_files": [
            {"frame": frame, "image": f"{frame:06d}.jpg", "time_s": frame / 30,
             "width": 1920, "height": 1080}
            for frame in (10, 11)
        ],
    }


def _save(tmp_path, report):
    directory = tmp_path / "reports" / report["clip"]
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return directory.parent


def _inspected(report):
    report = copy.deepcopy(report)
    report.update(clip="inspected_shot_1", split="inspected_session_unlabelled",
                  association_method="temporal")
    report["path"][0].update(reason="near_prediction_provisional", missing_run=0)
    report["path"][1].update(reason="no_candidate", missing_run=1)
    return report


def test_render_unlabelled_report_without_fabricated_accuracy(tmp_path):
    reports = _save(tmp_path, _report(tmp_path))
    output = tmp_path / "review"
    target, clips, frames = render_review(reports, output)

    assert (clips, frames) == (1, 2)
    html = target.read_text(encoding="utf-8")
    assert "Frozen detector on an unseen recording session" in html
    assert "no manual labels or accuracy scores" in html
    assert '<label style="display:none"><input id="show-label"' in html
    assert "__REVIEW_DATA__" not in html
    match = re.search(r"const REVIEW_DATA = (\{.*\});", html)
    assert match
    clip = json.loads(match.group(1))["clips"][0]
    assert clip["title"] == "IMG_9999.mov"
    assert "frame_transform" not in clip
    assert "Downsampled clips show processing previews" not in html
    assert "Original image pixels" in html
    assert clip["split"] == "unseen session · unlabelled"
    assert [item["frame"] for item in clip["frames"]] == [10, 11]
    assert [item["selected"] for item in clip["frames"]] == [[202.0, 802.0], None]
    assert all(item["label"] is None and item["error_px"] is None for item in clip["frames"])
    assert (output / clip["frames"][0]["image"]).read_bytes() == b"original frame 10"


def _transform(report, source_width, source_height):
    width, height = report["width"], report["height"]
    downsampled = (width, height) != (source_width, source_height)
    report["frame_transform"] = {
        "source_width": source_width, "source_height": source_height,
        "width": width, "height": height,
        "scale_x": width / source_width, "scale_y": height / source_height,
        "status": "downsampled" if downsampled else "native",
        "interpolation": "INTER_AREA" if downsampled else "none",
    }
    return report


@pytest.mark.parametrize("source_dimensions", [(1920, 1080), (2880, 1620), (3840, 2160)])
def test_optional_frame_transform_distinguishes_native_and_processing_images(tmp_path, source_dimensions):
    report = _transform(_report(tmp_path), *source_dimensions)
    reports = _save(tmp_path, report)
    target, _, _ = render_review(reports, tmp_path / "review")
    html = target.read_text(encoding="utf-8")
    clip = json.loads(re.search(r"const REVIEW_DATA = (\{.*\});", html).group(1))["clips"][0]

    assert clip["frame_transform"] == report["frame_transform"]
    assert clip["frames"][0]["width"] == 1920
    assert clip["frames"][0]["selected"] == [202.0, 802.0]
    if source_dimensions == (1920, 1080):
        assert clip["title"] == "IMG_9999.mov"
        assert "Downsampled clips show processing previews" not in html
        assert "Original image pixels" in html
    else:
        assert clip["title"] == (f"IMG_9999.mov · source {source_dimensions[0]}×{source_dimensions[1]}; "
                                 "preview 1920×1080")
        assert "Downsampled clips show processing previews, not native-resolution original frames" in html
        assert "Processing image pixels" in html
        assert "160 × 160 processing pixels" in html
        assert "`Processing frame ${frame.frame}" in html
        assert "Original image pixels" not in html
        assert "160 × 160 source pixels" not in html


@pytest.mark.parametrize("mutation", [
    lambda report: report.update(frame_transform=None),
    lambda report: report.update(frame_transform=[]),
    lambda report: report["frame_transform"].update(source_width=True),
    lambda report: report["frame_transform"].update(source_height=0),
    lambda report: report["frame_transform"].update(width=720),
    lambda report: report["frame_transform"].update(source_width=1000),
    lambda report: report["frame_transform"].update(scale_x=1.0),
    lambda report: report["frame_transform"].update(scale_y=float("nan")),
    lambda report: report["frame_transform"].update(status="native"),
    lambda report: report["frame_transform"].update(interpolation="none"),
    lambda report: report["frame_transform"].pop("scale_y"),
])
def test_rejects_invalid_or_inconsistent_frame_transform(tmp_path, mutation):
    report = _transform(_report(tmp_path), 3840, 2160)
    mutation(report)
    reports = _save(tmp_path, report)
    with pytest.raises(ValueError, match="frame_transform"):
        render_review(reports, tmp_path / "review")


@pytest.mark.parametrize("association", ["temporal", "recovery", "motion-recovery"])
def test_render_inspected_temporal_report_as_development_not_unseen(tmp_path, association):
    report = _inspected(_report(tmp_path))
    report["association_method"] = association
    reports = _save(tmp_path, report)
    target, clips, frames = render_review(reports, tmp_path / "review")

    assert (clips, frames) == (1, 2)
    html = target.read_text(encoding="utf-8")
    assert "Previously inspected development clips using temporal association" in html
    assert "Offline development review • inspected session • temporal association • manual start" in html
    assert "Frozen detector on an unseen recording session" not in html
    assert "there are no manual labels or accuracy scores here" in html
    assert "Association: ${frame.association_reason" in html
    match = re.search(r"const REVIEW_DATA = (\{.*\});", html)
    assert match
    clip = json.loads(match.group(1))["clips"][0]
    assert clip["split"] == "inspected session · development · unlabelled"
    assert [row["association_reason"] for row in clip["frames"]] == [
        "near_prediction_provisional", "no_candidate"]
    assert [row["missing_run"] for row in clip["frames"]] == [0, 1]
    assert all(row["label"] is None and row["error_px"] is None for row in clip["frames"])


def test_mixed_review_keeps_per_clip_session_labels(tmp_path):
    unseen = _report(tmp_path)
    _save(tmp_path, unseen)
    reports = _save(tmp_path, _inspected(unseen))
    target, clips, frames = render_review(reports, tmp_path / "review")

    assert (clips, frames) == (2, 4)
    html = target.read_text(encoding="utf-8")
    assert "Offline tracks from unseen and previously inspected recording sessions" in html
    assert "Frozen detector on an unseen recording session" not in html
    match = re.search(r"const REVIEW_DATA = (\{.*\});", html)
    assert match
    by_id = {item["id"]: item for item in json.loads(match.group(1))["clips"]}
    assert by_id["unseen_shot_1"]["split"] == "unseen session · unlabelled"
    assert by_id["inspected_shot_1"]["split"] == "inspected session · development · unlabelled"
    assert "association_reason" not in by_id["unseen_shot_1"]["frames"][0]


@pytest.mark.parametrize("mutation", [
    lambda report: report.update(association_method="legacy"),
    lambda report: report["path"][0].update(reason=""),
    lambda report: report["path"][1].update(missing_run=-1),
])
def test_rejects_incomplete_inspected_temporal_provenance(tmp_path, mutation):
    report = _inspected(_report(tmp_path))
    mutation(report)
    reports = _save(tmp_path, report)
    with pytest.raises(ValueError):
        render_review(reports, tmp_path / "review")


@pytest.mark.parametrize("mutation", [
    lambda report: report.update(status="production"),
    lambda report: report.update(weights_sha256="bad"),
    lambda report: report["path"][1].update(frame=12),
    lambda report: report["path"][0].update(predicted=[float("nan"), 800]),
    lambda report: report["path"][1].update(center=[1, 2]),
    lambda report: report["frame_files"][0].update(image="../outside.jpg"),
])
def test_rejects_invalid_provenance_or_misaligned_path(tmp_path, mutation):
    report = copy.deepcopy(_report(tmp_path))
    mutation(report)
    reports = _save(tmp_path, report)
    with pytest.raises(ValueError):
        render_review(reports, tmp_path / "review")
