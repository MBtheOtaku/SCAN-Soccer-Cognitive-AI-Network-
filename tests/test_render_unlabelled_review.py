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
    assert clip["split"] == "unseen session · unlabelled"
    assert [item["frame"] for item in clip["frames"]] == [10, 11]
    assert [item["selected"] for item in clip["frames"]] == [[202.0, 802.0], None]
    assert all(item["label"] is None and item["error_px"] is None for item in clip["frames"])
    assert (output / clip["frames"][0]["image"]).read_bytes() == b"original frame 10"


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
