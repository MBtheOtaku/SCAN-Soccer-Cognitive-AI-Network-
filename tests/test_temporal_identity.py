"""Behavioural checks for causal identity, abstention and camera compensation."""

import math
import numpy as np
import pytest

from experiments.ball_tracking.temporal_identity import TemporalIdentity, estimate_camera_translation


def row(frame, *points):
    return {"clip": "synthetic", "frame": frame, "split": "test",
            "candidates": [{"center": [x, y], "confidence": c} for x, y, c in points]}


def test_weak_nearby_ball_survives_high_confidence_clothing_distractor():
    tracker = TemporalIdentity((100, 100))
    for f in range(4):
        tracker.update(row(f, (100 + 6 * f, 100, 0.8)))
    result = tracker.update(row(4, (124, 100, 0.056), (175, 126, 0.99)))
    assert result['center'] == [124, 100]
    assert result['identity_verified'] is False


def test_lone_turf_mark_does_not_rotate_velocity_and_weak_ball_can_return():
    tracker = TemporalIdentity((100, 100))
    for f in range(4):
        tracker.update(row(f, (100 + 5 * f, 100, 0.8)))
    before = tracker.velocity
    missing = tracker.update(row(4, (85, 168, 0.95)))
    assert missing['center'] is None and missing['detected'] is False
    assert tracker.velocity == before
    recovered = tracker.update(row(5, (125, 100, 0.051)))
    assert recovered['center'] == [125, 100]
    assert missing['center'] is None  # Earlier uncertainty is not rewritten.


def test_large_launch_requires_consecutive_moving_evidence():
    tracker = TemporalIdentity((100, 100))
    tracker.update(row(0, (100, 100, 0.9)))
    first = tracker.update(row(1, (150, 100, 0.95)))
    second = tracker.update(row(2, (200, 100, 0.90)))
    assert not first['detected'] and first['center'] is None
    assert second['center'] == [200, 100]
    assert second['reason'] == 'temporally_supported_provisional'


def test_static_distant_distractor_never_promotes():
    tracker = TemporalIdentity((100, 100))
    tracker.update(row(0, (100, 100, 0.9)))
    for f in range(1, 9):
        result = tracker.update(row(f, (160, 100, 0.99)))
        assert result['center'] is None


def test_launch_acceleration_can_be_confirmed_without_a_single_frame_jump():
    tracker = TemporalIdentity((663, 806))
    tracker.update(row(0, (663, 806, .4)))
    first = tracker.update(row(1, (635, 793, .36)))
    second = tracker.update(row(2, (572, 755, .43)))
    assert first['center'] is None
    assert second['center'] == [572, 755]
    assert tracker.update(row(3, (521, 725, .7)))['center'] == [521, 725]


def test_long_loss_requires_new_seed_instead_of_accepting_a_rogue_ball():
    tracker = TemporalIdentity((100, 100))
    tracker.update(row(0, (100, 100, 0.9)))
    for f in range(1, 14):
        assert tracker.update(row(f))['center'] is None
    result = tracker.update(row(14, (100, 100, 0.99)))
    assert result['center'] is None
    assert result['reason'] == 'lost_requires_new_seed'


def test_two_nearby_equal_candidates_remain_ambiguous():
    tracker = TemporalIdentity((100, 100))
    result = tracker.update(row(0, (91, 100, .8), (109, 100, .8)))
    assert result['center'] is None
    assert result['reason'] == 'ambiguous_candidates'


def test_camera_movement_is_not_ball_velocity_or_branch_motion():
    tracker = TemporalIdentity((100, 100))
    tracker.update(row(0, (100, 100, .9)))
    for f in range(1, 5):
        result = tracker.update(row(f, (100 + 5*f, 100, .9)), (5, 0))
        assert result['center'] == [100 + 5*f, 100]
        assert math.hypot(*tracker.velocity) < 1e-9
    tracker = TemporalIdentity((100, 100))
    for f in range(4):
        assert tracker.update(row(f, (160 + 5*f, 100, .99)), (5, 0))['center'] is None


def test_bad_candidate_and_noncontiguous_input_are_rejected():
    tracker = TemporalIdentity((100, 100))
    tracker.update(row(3))
    with pytest.raises(ValueError, match='contiguous'):
        tracker.update(row(5))
    with pytest.raises(ValueError, match='Invalid detector'):
        tracker.update(row(4, (float('nan'), 0, .5)))


def test_camera_translation_estimate_and_unreliable_alignment():
    import cv2
    rng = np.random.default_rng(3)
    previous = rng.integers(0, 255, (160, 200, 3), dtype=np.uint8)
    current = cv2.warpAffine(previous, np.float32([[1, 0, 4], [0, 1, -3]]), (200, 160))
    shift, info = estimate_camera_translation(previous, current)
    assert shift == pytest.approx((4, -3), abs=.5)
    assert info['status'] == 'estimated_translation_only'
    blank = np.zeros((160, 200, 3), np.uint8)
    assert estimate_camera_translation(blank, blank)[0] == (0, 0)
