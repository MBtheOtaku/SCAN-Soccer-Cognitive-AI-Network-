"""Offline recovery behaviour at short gaps, rebounds and competing objects."""

import math

import pytest

from experiments.ball_tracking.recovery_identity import RecoveryIdentity, RecoverySettings
from experiments.ball_tracking.temporal_identity import TemporalIdentity


def observation(frame, *points, camera_status="estimated_translation_only"):
    return {
        "clip": "synthetic_recovery",
        "frame": frame,
        "split": "test",
        "camera_motion": {"status": camera_status},
        "candidates": [
            {"center": [x, y], "confidence": confidence}
            for x, y, confidence in points
        ],
    }


def stationary_history(tracker):
    """Five supported observations followed by an actual detection gap."""
    for frame in range(5):
        assert tracker.update(observation(frame, (100, 100, .8)))["detected"]
    missing = [tracker.update(observation(frame)) for frame in (5, 6)]
    assert all(result["center"] is None for result in missing)
    return missing


def test_stationary_ball_recovers_only_after_three_supported_observations():
    tracker = RecoveryIdentity((100, 100))
    missing = stationary_history(tracker)
    attempted = [tracker.update(observation(frame, (104, 100, .15)))
                 for frame in (7, 8, 9)]
    assert [result["center"] for result in attempted] == [None, None, [104, 100]]
    assert attempted[-1]["reason"] == "slow_recovery_provisional"
    assert attempted[-1]["identity_verified"] is False
    assert attempted[-1]["missing_run"] == 0
    assert all(result["center"] is None for result in missing + attempted[:2])
    assert tracker.update(observation(10, (104, 100, .15)))["center"] == [104, 100]


def test_small_rebound_jitter_can_reverse_without_requiring_straight_motion():
    tracker = RecoveryIdentity((100, 100))
    stationary_history(tracker)
    results = [tracker.update(observation(frame, (x, 100, .2)))
               for frame, x in ((7, 104), (8, 100), (9, 103))]
    assert results[0]["center"] is None
    assert results[1]["center"] is None
    assert results[2]["center"] == [103, 100]
    assert results[2]["reason"] == "slow_recovery_provisional"


def test_sparse_support_recovers_without_backfilling_the_intermediate_gap():
    tracker = RecoveryIdentity((100, 100))
    earlier = stationary_history(tracker)
    sequence = [(7, 104), (8, None), (9, 105), (10, None), (11, 104)]
    results = [tracker.update(observation(frame, *((x, 100, .2),) if x is not None else ()))
               for frame, x in sequence]
    assert [result["center"] for result in results] == [None, None, None, None, [104, 100]]
    assert all(result["center"] is None for result in earlier + results[:-1])


def test_recovery_support_expires_across_a_four_frame_detection_gap():
    tracker = RecoveryIdentity((100, 100))
    stationary_history(tracker)
    results = [tracker.update(observation(7, (104, 100, .2)))]
    results += [tracker.update(observation(frame)) for frame in range(8, 12)]
    results += [tracker.update(observation(frame, (104, 100, .2))) for frame in (12, 13)]
    assert all(result["center"] is None for result in results)


def test_a_seed_with_too_little_supported_history_is_not_enough_for_recovery():
    tracker = RecoveryIdentity((100, 100))
    for frame in range(4):
        assert tracker.update(observation(frame, (100, 100, .8)))["detected"]
    for frame in (4, 5):
        assert tracker.update(observation(frame))["center"] is None
    results = [tracker.update(observation(frame, (104, 100, .2))) for frame in (6, 7, 8)]
    assert all(result["center"] is None for result in results)


@pytest.mark.parametrize("persistent", [False, True])
def test_far_turf_is_not_recovered_even_when_it_repeats(persistent):
    tracker = RecoveryIdentity((100, 100))
    stationary_history(tracker)
    results = []
    for frame in range(7, 15):
        points = ((150, 100, .99),) if persistent or frame == 7 else ()
        results.append(tracker.update(observation(frame, *points)))
    assert all(result["center"] is None for result in results)


def test_flying_track_does_not_adopt_a_stationary_object_at_its_last_position():
    tracker = RecoveryIdentity((100, 100))
    # A sustained 15 px/frame track is well above the recovery speed limit.
    for frame in range(9):
        assert tracker.update(observation(frame, (100 + 15 * frame, 100, .8)))["detected"]
    assert math.hypot(*tracker.velocity) > 12
    for frame in (9, 10):
        assert tracker.update(observation(frame))["center"] is None
    results = [tracker.update(observation(frame, (220, 100, .99))) for frame in range(11, 16)]
    assert all(result["center"] is None for result in results)


def test_two_nearby_objects_never_supply_unique_recovery_support():
    tracker = RecoveryIdentity((100, 100))
    stationary_history(tracker)
    results = [tracker.update(observation(frame, (94, 100, .9), (107, 100, .9)))
               for frame in range(7, 12)]
    assert all(result["center"] is None for result in results)


def test_recovery_does_not_bypass_the_original_loss_horizon():
    tracker = RecoveryIdentity((100, 100))
    for frame in range(5):
        assert tracker.update(observation(frame, (100, 100, .8)))["detected"]
    for frame in range(5, 18):
        assert tracker.update(observation(frame))["center"] is None
    results = [tracker.update(observation(frame, (100, 100, .99))) for frame in range(18, 23)]
    assert all(result["center"] is None for result in results)
    assert all(result["reason"] == "lost_requires_new_seed" for result in results)


@pytest.mark.parametrize("camera_status", ["unavailable_no_correction", "not_estimated", None])
def test_unreliable_camera_alignment_does_not_enable_slow_recovery(camera_status):
    tracker = RecoveryIdentity((100, 100))
    stationary_history(tracker)
    results = [tracker.update(observation(frame, (104, 100, .2), camera_status=camera_status))
               for frame in (7, 8, 9, 10)]
    assert all(result["center"] is None for result in results)


def test_camera_pan_does_not_become_a_recovered_ball_velocity():
    tracker = RecoveryIdentity((100, 100))
    stationary_history(tracker)
    results = []
    for frame in (7, 8, 9):
        x = 100 + 5 * (frame - 6)
        results.append(tracker.update(observation(frame, (x, 100, .2)), camera_shift=(5, 0)))
    assert [result["center"] for result in results] == [None, None, [115, 100]]
    assert math.hypot(*tracker.velocity) == pytest.approx(0, abs=1e-9)


def test_opt_in_recovery_does_not_change_legacy_stationary_gap_behaviour():
    legacy = TemporalIdentity((100, 100))
    recovery = RecoveryIdentity((100, 100))
    stationary_history(legacy)
    stationary_history(recovery)
    legacy_results = [legacy.update(observation(frame, (104, 100, .15))) for frame in (7, 8, 9)]
    recovery_results = [recovery.update(observation(frame, (104, 100, .15))) for frame in (7, 8, 9)]
    assert all(result["center"] is None for result in legacy_results)
    assert recovery_results[-1]["center"] == [104, 100]


def test_recovery_distance_scales_with_image_size():
    tracker = RecoveryIdentity((200, 200), scale=2)
    for frame in range(5):
        assert tracker.update(observation(frame, (200, 200, .8)))["detected"]
    for frame in (5, 6):
        assert tracker.update(observation(frame))["center"] is None
    results = [tracker.update(observation(frame, (250, 200, .2))) for frame in (7, 8, 9)]
    assert [result["center"] for result in results] == [None, None, [250, 200]]


def test_single_observation_cannot_be_configured_as_temporal_recovery():
    with pytest.raises(ValueError, match="settings"):
        RecoveryIdentity((100, 100), recovery_settings=RecoverySettings(confirmation_hits=1))


def test_recovered_ball_cannot_jump_to_a_pair_of_far_white_background_marks():
    tracker = RecoveryIdentity((323, 687))
    for frame in range(5):
        tracker.update(observation(frame, (323, 687, .2)))
    for frame in (5, 6):
        tracker.update(observation(frame))
    for frame in (7, 8, 9):
        recovered = tracker.update(observation(frame, (323, 687, .2)))
    assert recovered['reason'] == 'slow_recovery_provisional'
    for frame in (10, 11, 12):
        assert tracker.update(observation(frame))['center'] is None
    first = tracker.update(observation(13, (384, 694, .3)))
    second = tracker.update(observation(14, (405, 701, .3)))
    assert first['center'] is None and second['center'] is None
    assert first['post_recovery_local_gate'] is True
    assert second['post_recovery_local_gate'] is True


def test_recovered_ball_can_continue_moving_locally_instead_of_being_pinned():
    tracker = RecoveryIdentity((100, 100))
    stationary_history(tracker)
    for frame in (7, 8, 9):
        tracker.update(observation(frame, (104, 100, .2)))
    for frame in range(10, 15):
        center = 104 + 5 * (frame - 9)
        result = tracker.update(observation(frame, (center, 100, .2)))
        assert result['center'] == [center, 100]
