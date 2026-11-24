import copy

import pytest

from experiments.ball_tracking.score_recovery_comparison import compare, score_report


def inputs():
    m = {"clip_id": "review", "source_sha256": "source",
         "frames": [{"frame": f, "width": 300, "height": 300} for f in range(4)]}
    labels = {"schema": 1, "clip_id": "review", "source_sha256": "source", "annotations": [
        {"frame": 0, "state": "visible", "box": [144, 144, 156, 156]},
        {"frame": 1, "state": "visible", "box": [144, 144, 156, 156]},
        {"frame": 2, "state": "uncertain", "box": None},
        {"frame": 3, "state": "absent", "box": None}]}
    report = {"source_sha256": "source", "source_seed": [10, 10], "weights_sha256": "weights",
              "width": 100, "height": 100,
              "frame_transform": {"source_width": 300, "source_height": 300},
              "path": [{"frame": f, "detected": False, "center": None,
                        "predicted": [50, 50]} for f in range(4)]}
    return m, labels, report


def test_predictions_never_count_as_observations():
    m, labels, report = inputs()
    s = score_report(m, labels, report)
    assert s["correct_observations"] == 0
    assert s["longest_visible_gap_frames"] == 2
    assert s["uncertain_excluded"] == 1


def test_source_scaling_and_frozen_padding():
    m, labels, report = inputs()
    report["path"][0].update(detected=True, center=[50, 50])
    report["path"][1].update(detected=True, center=[54, 50])
    s = score_report(m, labels, report)
    assert s["correct_observations"] == 2
    assert s["strict_box_hits"] == 1
    assert s["false_acceptances"] == 0


def test_false_selection_on_absent_counts_but_uncertain_is_excluded():
    m, labels, report = inputs()
    for point in report["path"]:
        point.update(detected=True, center=[90, 90])
    s = score_report(m, labels, report)
    assert s["false_acceptances"] == 3
    assert s["false_acceptances_absent"] == 1
    assert s["selected_observation_precision"] == 0


@pytest.mark.parametrize("change", ["hash", "incomplete", "duplicate", "geometry"])
def test_rejects_wrong_or_incomplete_references(change):
    m, labels, report = inputs()
    if change == "hash":
        report["source_sha256"] = "other"
    elif change == "incomplete":
        labels["annotations"][0]["state"] = "unreviewed"
        labels["annotations"][0]["box"] = None
    elif change == "duplicate":
        report["path"].append(report["path"][0])
    else:
        m["frames"][0]["width"] = 400
    with pytest.raises(ValueError):
        score_report(m, labels, report)


def test_paired_comparison_requires_same_seed_and_reports_actual_gains():
    m, labels, baseline = inputs()
    candidate = copy.deepcopy(baseline)
    candidate["path"][1].update(detected=True, center=[50, 50])
    s = compare(m, labels, baseline, candidate)
    assert s["improved_frames"] == [1]
    assert s["regressed_frames"] == []
    candidate["source_seed"] = [99, 99]
    with pytest.raises(ValueError, match="same initial seed"):
        compare(m, labels, baseline, candidate)
