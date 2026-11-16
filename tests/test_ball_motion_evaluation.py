"""Check sparse ball review denominators and missing-path handling."""

from experiments.ball_tracking.evaluate_new_clips import score


def test_sparse_review_counts_missing_path_frame_as_missing():
    report = {"results": [{"clip": "clip", "mode": "positive", "path": [
        {"frame": 10, "center": [10, 10], "observed": True},
        {"frame": 11, "center": [30, 10], "observed": False},
    ]}]}
    labels = {"clips": {"clip": {"annotations": [
        {"frame": 10, "center": [10, 10]},
        {"frame": 11, "center": [10, 10]},
        {"frame": 12, "center": [10, 10]},
    ]}}}

    result = score(report, labels)["clip"]["positive"]

    assert result["labelled"] == 3
    assert result["tracked"] == 2
    assert result["within_10px"] == 1
    assert result["within_20px"] == 2
    assert result["rows"][2]["error_px"] is None


def test_sparse_review_reports_flagged_mark_separately_from_raw_error():
    report = {"results": [{"clip": "clip", "mode": "positive", "path": [
        {"frame": 10, "center": [50, 50], "observed": True,
         "reliability": "uncertain"},
    ]}]}
    labels = {"clips": {"clip": {"annotations": [
        {"frame": 10, "center": [50, 50]},
    ]}}}

    result = score(report, labels)["clip"]["positive"]

    assert result["within_10px"] == 1  # Raw location remains inspectable.
    assert result["flagged_uncertain"] == 1  # Reliability is separate.
