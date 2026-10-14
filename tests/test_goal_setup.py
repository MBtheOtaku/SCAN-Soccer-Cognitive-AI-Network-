import json
import cv2
import numpy as np
import pytest
from src.backend.goal_proposal import propose_goals
from src.backend.goal_setup import parse_goal_setup, decode_reference
from src.backend.goal_motion import GoalMotionTracker
from src.backend.goal_tracking import GoalCalibration


POINTS = [[.2, .2], [.8, .2], [.8, .6], [.2, .6]]


def test_only_confirmed_corners_are_accepted():
    assert parse_goal_setup(None) is None
    assert parse_goal_setup("null") is None
    calibration = parse_goal_setup(json.dumps({"confirmed": True, "corners": POINTS}))
    assert calibration.top_left == pytest.approx(POINTS[0])
    with pytest.raises(ValueError):
        parse_goal_setup(json.dumps({"confirmed": False, "corners": POINTS}))


@pytest.mark.parametrize("points", [POINTS[::-1], [[0, 0]] * 4, [[-.1, .2], *POINTS[1:]], [[float('nan'), .2], *POINTS[1:]], POINTS[:3], [POINTS[0], POINTS[2], POINTS[1], POINTS[3]]])
def test_invalid_geometry_is_rejected(points):
    with pytest.raises((ValueError, KeyError)):
        parse_goal_setup(json.dumps({"confirmed": True, "corners": points}))


def test_corrupt_and_oversized_photos_are_rejected():
    for contents in [b"", b"not an image", b"x" * 5_000_001]:
        with pytest.raises(ValueError):
            decode_reference(contents)


def test_tiny_photos_are_rejected():
    _, encoded = cv2.imencode(".png", np.zeros((1, 1, 3), np.uint8))
    with pytest.raises(ValueError):
        decode_reference(encoded.tobytes())


def test_blank_scene_abstains():
    assert propose_goals(np.zeros((360, 640, 3), np.uint8)) == []


def test_synthetic_front_goal_proposal():
    image = np.full((360, 640, 3), (45, 95, 45), np.uint8)
    cv2.rectangle(image, (140, 95), (480, 241), (90, 90, 90), -1)
    cv2.line(image, (140, 100), (480, 100), (245, 245, 245), 5)
    cv2.line(image, (140, 100), (140, 240), (245, 245, 245), 5)
    cv2.line(image, (480, 100), (480, 240), (245, 245, 245), 5)
    candidates = propose_goals(image)
    assert candidates
    expected = np.array([[140, 100], [480, 100], [480, 240], [140, 240]]) / [640, 360]
    np.testing.assert_allclose(candidates[0]["corners"], expected, atol=.025)
    assert "confidence" not in candidates[0]  # Rank scores are not probabilities.


def test_reference_photo_is_verified_even_at_clip_frame_zero():
    rng = np.random.default_rng(12)
    reference = cv2.cvtColor(cv2.GaussianBlur(rng.integers(0, 256, (240, 400), np.uint8), (5, 5), .9), cv2.COLOR_GRAY2BGR)
    cal = GoalCalibration((.25, .2), (.75, .2), (.75, .6), (.25, .6))
    tracker = GoalMotionTracker(reference, cal, external_reference=True)
    assert tracker.update(np.zeros_like(reference), 0) is None
    moved = cv2.warpAffine(reference, np.float32([[1, 0, 12], [0, 1, 8]]), (400, 240))
    result = tracker.update(moved, 0)
    assert result is not None
    np.testing.assert_allclose(result.pixel_points(400, 240), cal.pixel_points(400, 240) + [12, 8], atol=1)
