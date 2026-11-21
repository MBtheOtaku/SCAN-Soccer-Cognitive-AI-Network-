import pytest

from scripts.review_contact_timing import score_interval


def reference(**kwargs):
    return dict(status="reviewed", contact_start_frame=79,
                contact_end_frame=80, first_ball_motion_frame=81, **kwargs)


@pytest.mark.parametrize("estimate,error", [(60, -19), (79, 0), (80, 0), (97, 17)])
def test_signed_distance_to_interval_not_midpoint(estimate, error):
    result = score_interval(estimate, reference(), 151, 30)
    assert result["signed_error_frames"] == error
    assert result["absolute_error_s"] == abs(error) / 30


def test_unresolved_and_missing_estimate_are_not_successes():
    assert score_interval(79, {"status": "unresolved"}, 151, 30)["status"] == "unresolved"
    assert score_interval(None, reference(), 151, 30)["status"] == "no_estimate"


@pytest.mark.parametrize("key,value", [
    ("contact_start_frame", -1), ("contact_start_frame", True),
    ("contact_start_frame", 81), ("contact_end_frame", 82),
    ("first_ball_motion_frame", 151), ("first_ball_motion_frame", 80.5),
    ("status", "predicted"),
])
def test_reject_invalid_references(key, value):
    label = reference()
    label[key] = value
    with pytest.raises(ValueError):
        score_interval(80, label, 151, 30)


@pytest.mark.parametrize("fps", [0, -30, float("nan"), float("inf")])
def test_reject_invalid_timing(fps):
    with pytest.raises(ValueError):
        score_interval(80, reference(), 151, fps)


def test_reject_out_of_clip_estimate():
    with pytest.raises(ValueError):
        score_interval(151, reference(), 151, 30)
