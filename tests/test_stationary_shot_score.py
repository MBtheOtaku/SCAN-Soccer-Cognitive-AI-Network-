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
    assert score["rows"][2]["reliability"] == "lost"


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
