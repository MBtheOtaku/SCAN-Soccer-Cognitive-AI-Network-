"""Post-inference comparison must score only hash-matched, visible review frames."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.ball_tracking.review_identity_comparison import compare_runs


CLIP = "rep-synthetic"
PROVENANCE = "assistant_visual_review_provisional_needs_human_confirmation"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict, dict, dict]:
    source = tmp_path / "rep-synthetic.mov"
    source.write_bytes(b"synthetic source video")
    frame_bytes = {frame: f"decoded original frame {frame}".encode() for frame in range(4)}
    reports = []
    for name, selections, split in (
        ("baseline", [[10, 10], [55, 55], None, [10, 10]], "unseen_session_unlabelled"),
        ("revised", [[12, 10], [11, 10], [13, 10], [10, 10]], "inspected_session_unlabelled"),
    ):
        run_dir = tmp_path / name / CLIP
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(parents=True)
        for frame, content in frame_bytes.items():
            (frames_dir / f"{frame:06d}.jpg").write_bytes(content)
        report = {
            "status": "offline_pilot_not_production",
            "clip": CLIP,
            "split": split,
            "source_name": str(source),
            "source_sha256": _sha(source.read_bytes()),
            "weights_sha256": "a" * 64,
            "fps": 30.0,
            "width": 100,
            "height": 100,
            "start_frame": 0,
            "end_frame": 3,
            "seed": [10, 10],
            "seed_kind": "one_manual_starting_position_no_reseeding",
            "settings": {"crop_size": 384, "slice_offset": 128, "confidence": 0.05,
                         "class_id": 0, "inference_size": 640, "device": "cpu"},
            "implementation_sha256": {"detector": "d" * 64,
                                      "association": ("e" if name == "baseline" else "f") * 64},
            "frames_directory": str(frames_dir),
            "frame_files": [{"frame": frame, "image": f"{frame:06d}.jpg", "time_s": frame / 30,
                             "width": 100, "height": 100} for frame in range(4)],
            "rows": [{"frame": frame, "crop_center": [10, 10], "candidate_count": 1,
                      "candidates": [{"center": [10, 10], "confidence": 0.5}]}
                     for frame in range(4)],
            "path": [{"frame": frame, "predicted": [10, 10], "center": position,
                      "confidence": 0.5 if position else None, "detected": position is not None}
                     for frame, position in enumerate(selections)],
        }
        (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        reports.append(report)
    refs_dir = tmp_path / "annotations"
    refs_dir.mkdir()
    annotation = {
        "clip": CLIP,
        "source_name": source.name,
        "source_sha256": _sha(source.read_bytes()),
        "source_width": 100,
        "source_height": 100,
        "frame_range": [0, 9],
        "provenance": PROVENANCE,
        "purpose": "offline_evaluation_reference_only_not_tracker_input",
        "frames": [
            {"frame": frame, "status": "visible", "center": [10, 10], "uncertainty_px": 5,
             "frame_sha256": _sha(frame_bytes[frame]), "reviewed_original_frame": True}
            for frame in range(3)
        ] + [
            {"frame": 3, "status": "uncertain", "center": None, "uncertainty_px": None,
             "frame_sha256": _sha(frame_bytes[3]), "reviewed_original_frame": True},
            # The reference is outside both saved report windows. It must never be scored.
            {"frame": 9, "status": "visible", "center": [90, 90], "uncertainty_px": 8,
             "frame_sha256": "b" * 64, "reviewed_original_frame": True},
        ],
    }
    (refs_dir / "reference.json").write_text(json.dumps(annotation), encoding="utf-8")
    return tmp_path / "baseline", tmp_path / "revised", refs_dir, tmp_path / "comparison", reports[0], reports[1], annotation


def _write_report(directory: Path, report: dict) -> None:
    (directory / CLIP / "report.json").write_text(json.dumps(report), encoding="utf-8")


def _write_annotation(directory: Path, annotation: dict) -> None:
    (directory / "reference.json").write_text(json.dumps(annotation), encoding="utf-8")


def test_compares_only_visible_matching_frames_and_renders_originals(tmp_path: Path) -> None:
    baseline, revised, annotations, output, _, _, _ = _fixture(tmp_path)
    page, summary = compare_runs(baseline, revised, annotations, output)
    counts = summary["clips"][0]
    assert counts["window"] == [0, 3]
    assert counts["baseline"] == {
        "visible_reference_frames": 3,
        "selected_within_10_px": 1,
        "selected_within_20_px": 1,
        "wrong_selected_over_20_px": 1,
        "missing_selection": 1,
        "longest_contiguous_nonmatch_frames": 2,
        "recovered_after_nonmatch_events": 0,
    }
    assert counts["revised"]["selected_within_20_px"] == 3
    assert counts["revised"]["wrong_selected_over_20_px"] == 0
    assert counts["revised"]["missing_selection"] == 0
    assert counts["frames"][3]["reference_status"] == "uncertain"
    assert counts["frames"][3]["baseline"]["error_px"] is None
    assert "provisional" in page.read_text(encoding="utf-8").lower()
    assert "Inspected development" in page.read_text(encoding="utf-8")
    assert "Baseline" in page.read_text(encoding="utf-8")
    assert "Revised" in page.read_text(encoding="utf-8")
    assert "__REVIEW_DATA__" not in page.read_text(encoding="utf-8")
    assert (output / "frames" / CLIP / "000002.jpg").read_bytes() == b"decoded original frame 2"
    saved = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert saved["split"] == "inspected_development_with_provisional_references"


@pytest.mark.parametrize("fault, message", [
    ("source_bytes", "Source video bytes"),
    ("source_annotation", "Source SHA mismatch"),
    ("source_path", "different source file paths"),
    ("frame_annotation", "Reviewed original JPEG SHA mismatch"),
    ("frame_cross_run", "different decoded original JPEGs"),
    ("provenance", "provisional evaluation-only"),
    ("weights", "different detector weights"),
    ("seed", "different manual seed positions"),
    ("coverage", "different frame coverage"),
    ("settings", "different detector settings"),
    ("detector_impl", "different detector implementations"),
    ("seed_kind", "Expected one manual seed"),
    ("bad_reference", "Non-visible reference must not have a centre"),
])
def test_rejects_mismatches_and_invalid_references(tmp_path: Path, fault: str, message: str) -> None:
    baseline, revised, annotations, output, base, new, annotation = _fixture(tmp_path)
    base, new, annotation = copy.deepcopy((base, new, annotation))
    if fault == "source_bytes":
        Path(base["source_name"]).write_bytes(b"source was altered after inference")
    elif fault == "source_annotation":
        annotation["source_sha256"] = "b" * 64
        _write_annotation(annotations, annotation)
    elif fault == "source_path":
        copied_source = tmp_path / "copy" / Path(new["source_name"]).name
        copied_source.parent.mkdir()
        copied_source.write_bytes(Path(new["source_name"]).read_bytes())
        new["source_name"] = str(copied_source)
        _write_report(revised, new)
    elif fault == "frame_annotation":
        annotation["frames"][1]["frame_sha256"] = "b" * 64
        _write_annotation(annotations, annotation)
    elif fault == "frame_cross_run":
        (revised / CLIP / "frames" / "000001.jpg").write_bytes(b"different decoded image")
    elif fault == "provenance":
        annotation["provenance"] = "ground_truth"
        _write_annotation(annotations, annotation)
    elif fault == "weights":
        new["weights_sha256"] = "c" * 64
        _write_report(revised, new)
    elif fault == "seed":
        new["seed"] = [20, 20]
        _write_report(revised, new)
    elif fault == "coverage":
        new["start_frame"] = 1
        for key in ("frame_files", "rows", "path"):
            new[key] = new[key][1:]
        _write_report(revised, new)
    elif fault == "settings":
        new["settings"]["confidence"] = 0.10
        _write_report(revised, new)
    elif fault == "detector_impl":
        new["implementation_sha256"]["detector"] = "c" * 64
        _write_report(revised, new)
    elif fault == "seed_kind":
        new["seed_kind"] = "automatic_reseed"
        _write_report(revised, new)
    elif fault == "bad_reference":
        annotation["frames"][3]["center"] = [10, 10]
        _write_annotation(annotations, annotation)
    with pytest.raises(ValueError, match=message):
        compare_runs(baseline, revised, annotations, output)


def test_recovery_counts_only_after_visible_reference_miss(tmp_path: Path) -> None:
    baseline, revised, annotations, output, base, _, _ = _fixture(tmp_path)
    base["path"][2]["center"] = [11, 10]
    base["path"][2]["detected"] = True
    base["path"][2]["confidence"] = 0.5
    _write_report(baseline, base)
    _, summary = compare_runs(baseline, revised, annotations, output)
    counts = summary["clips"][0]["baseline"]
    assert counts["recovered_after_nonmatch_events"] == 1
    assert counts["longest_contiguous_nonmatch_frames"] == 1
    assert counts["selected_within_20_px"] == 2
