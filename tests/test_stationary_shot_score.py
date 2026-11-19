"""Dense review must expose lost and uncertain ball positions."""

import pytest

from experiments.ball_tracking.score_stationary_shot import (
    score_path,
    validate_review_window,
)


def test_score_counts_lost_and_uncertain_positions_in_visible_denominator():
    result = {"anchor": 5, "frames_attempted": 3, "path": [
        {"frame": 6, "center": [10, 10], "observed": True,
         "reliability": "provisional"},
        {"frame": 7, "center": [10, 10], "observed": False,
         "reliability": "uncertain"},
    ]}
    labels = [{"frame": frame, "state": "visible", "box": [9, 9, 11, 11]}
              for frame in (6, 7, 8)]
    score = score_path(result, labels)
    assert score["visible_reviewed"] == 3
    assert score["within_20px"] == 2
    assert score["provisional_observed_within_20px"] == 1
    assert score["provisional_observed_wrong_20px"] == 0
    assert score["rows"][2]["reliability"] == "lost"
    assert score["visible_missed_20px"] == 1
    assert score["visible_miss_runs_20px"] == [
        {"start_frame": 8, "end_frame": 8, "length_frames": 1}]
    assert score["visible_without_supported_hit_20px"] == 2
    assert score["longest_visible_unsupported_run_20px"] == 2


def test_absent_ball_does_not_count_uncertain_or_predicted_paths_as_accepted():
    result = {"anchor": 0, "frames_attempted": 4, "path": [
        {"frame": 1, "center": [12, 12], "observed": True,
         "reliability": "provisional"},
        {"frame": 2, "center": [14, 12], "observed": True,
         "reliability": "uncertain"},
        {"frame": 3, "center": [16, 12], "observed": False,
         "reliability": "provisional"},
    ]}
    labels = [{"frame": frame, "state": "absent", "box": None}
              for frame in range(1, 5)]
    score = score_path(result, labels)
    assert score["absent_reviewed"] == 4
    assert score["absent_with_path"] == 3
    assert score["false_accepted_absent_frames"] == 1
    assert [row["false_accepted"] for row in score["absent_rows"]] == [
        True, False, False, False]
    assert score["visible_missed_20px"] == 0
    assert score["longest_visible_miss_run_20px"] == 0


def test_visible_miss_runs_require_adjacent_frames_and_separate_raw_from_supported():
    result = {"anchor": 0, "frames_attempted": 8, "path": [
        {"frame": 1, "center": [10, 10], "observed": True,
         "reliability": "provisional"},
        {"frame": 2, "center": [40, 40], "observed": True,
         "reliability": "provisional"},
        {"frame": 4, "center": [40, 40], "observed": True,
         "reliability": "uncertain"},
        {"frame": 6, "center": [40, 40], "observed": True,
         "reliability": "uncertain"},
        {"frame": 7, "center": [10, 10], "observed": True,
         "reliability": "uncertain"},
        {"frame": 8, "center": [10, 10], "observed": True,
         "reliability": "provisional"},
    ]}
    labels = [{"frame": frame, "state": "visible", "box": [9, 9, 11, 11]}
              for frame in (8, 7, 6, 4, 3, 2, 1)]
    labels.append({"frame": 5, "state": "uncertain", "box": None})
    score = score_path(result, labels)
    assert score["within_20px"] == 3
    assert score["provisional_observed_wrong_20px"] == 1
    assert score["visible_missed_20px"] == 4
    assert score["visible_miss_runs_20px"] == [
        {"start_frame": 2, "end_frame": 4, "length_frames": 3},
        {"start_frame": 6, "end_frame": 6, "length_frames": 1}]
    assert score["longest_visible_miss_run_20px"] == 3
    assert score["visible_without_supported_hit_20px"] == 5
    assert score["visible_unsupported_runs_20px"] == [
        {"start_frame": 2, "end_frame": 4, "length_frames": 3},
        {"start_frame": 6, "end_frame": 7, "length_frames": 2}]
    assert score["longest_visible_unsupported_run_20px"] == 3


def test_score_requires_dense_and_completed_review():
    result = {"anchor": 5, "frames_attempted": 3}
    manifest = {"frames": [{"frame": frame} for frame in (6, 8)]}
    labels = {"annotations": [{"frame": frame, "state": "visible"}
                              for frame in (6, 8)]}
    with pytest.raises(ValueError, match="every frame"):
        validate_review_window(result, manifest, labels)
    manifest["frames"].append({"frame": 7})
    with pytest.raises(ValueError, match="Review every frame"):
        validate_review_window(result, manifest, labels)
