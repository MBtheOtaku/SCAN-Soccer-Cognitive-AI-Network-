"""Causal motion release after local slow-ball recovery, kept offline."""

import pytest
import copy

from experiments.ball_tracking.motion_recovery_identity import MotionRecoveryIdentity


def observation(frame, *points, camera_status="estimated_translation_only"):
    return {
        "clip": "synthetic_motion_recovery",
        "frame": frame,
        "split": "test",
        "camera_motion": {"status": camera_status},
        "candidates": [
            {"center": [x, y], "confidence": confidence}
            for x, y, confidence in points
        ],
    }


@pytest.mark.parametrize("bad_frame", [12, 14])
def test_rejected_noncontiguous_input_does_not_mutate_release_state(bad_frame):
    tracker = recovered_tracker()
    launch(tracker)
    before = copy.deepcopy(tracker.__dict__)
    with pytest.raises(ValueError, match="contiguous"):
        tracker.update(observation(bad_frame, camera_status="unavailable_no_correction"))
    assert tracker.__dict__ == before
    result = tracker.update(observation(13, (176, 100, .7)))
    assert result["center"] == [176, 100]
    assert result["motion_release_active"] is True


def recovered_tracker(*, scale=1.0):
    tracker = MotionRecoveryIdentity((100 * scale, 100 * scale), scale=scale)
    for frame in range(5):
        assert tracker.update(observation(frame, (100 * scale, 100 * scale, .8)))["detected"]
    for frame in (5, 6):
        assert tracker.update(observation(frame))["center"] is None
    for frame in (7, 8, 9):
        result = tracker.update(observation(frame, (104 * scale, 100 * scale, .3)))
    assert result["reason"] == "slow_recovery_provisional"
    assert result["motion_release_active"] is False
    return tracker


def launch(tracker, *, scale=1.0, extra=()):
    results = []
    for frame, x in ((10, 118), (11, 136), (12, 156)):
        results.append(tracker.update(observation(frame, (x * scale, 100 * scale, .7), *extra)))
    return results


def test_three_contiguous_fast_local_steps_release_the_guard_provisionally():
    tracker = recovered_tracker()
    results = launch(tracker)
    assert [result["center"] for result in results] == [[118, 100], [136, 100], [156, 100]]
    assert [result["motion_release_active"] for result in results] == [False, False, True]
    assert results[-1]["motion_release_steps"] >= 3
    assert all(result["identity_verified"] is False for result in results)
    assert all("motion_release_reason" in result for result in results)
    assert all("stationary_veto_count" in result for result in results)
    assert "provisional" in str(tracker.configuration()).lower()


def test_released_flight_reacquires_only_the_current_supported_detection():
    tracker = recovered_tracker()
    launch(tracker)
    missing = [tracker.update(observation(frame)) for frame in (13, 14)]
    first = tracker.update(observation(15, (216, 100, .7)))
    second = tracker.update(observation(16, (236, 100, .7)))
    assert [result["center"] for result in missing] == [None, None]
    assert first["center"] is None
    assert second["center"] == [236, 100]
    assert second["identity_verified"] is False
    # Earlier rows remain missing; confirmation never invents past observations.
    assert all(result["center"] is None for result in missing + [first])


def test_camera_translation_alone_does_not_release_a_stationary_ball():
    tracker = recovered_tracker()
    results = []
    for frame in (10, 11, 12, 13):
        x = 104 + 16 * (frame - 9)
        results.append(tracker.update(observation(frame, (x, 100, .7)), camera_shift=(16, 0)))
    assert all(result["detected"] for result in results)
    assert all(not result["motion_release_active"] for result in results)


def test_reversing_local_jitter_does_not_release_the_guard():
    tracker = recovered_tracker()
    results = [tracker.update(observation(frame, (x, 100, .7)))
               for frame, x in ((10, 118), (11, 104), (12, 118), (13, 104))]
    assert all(not result["motion_release_active"] for result in results)


@pytest.mark.parametrize("step_count", [1, 2])
def test_one_or_two_fast_steps_are_not_enough_to_release(step_count):
    tracker = recovered_tracker()
    for frame, x in list(((10, 118), (11, 136)))[:step_count]:
        result = tracker.update(observation(frame, (x, 100, .7)))
    assert result["motion_release_active"] is False


def test_a_missing_observation_interrupts_motion_release_support():
    tracker = recovered_tracker()
    for frame, x in ((10, 118), (11, 136)):
        assert tracker.update(observation(frame, (x, 100, .7)))["motion_release_active"] is False
    missing = tracker.update(observation(12))
    resumed = tracker.update(observation(13, (156, 100, .7)))
    assert missing["center"] is None
    assert missing["motion_release_active"] is False
    assert resumed["motion_release_active"] is False


def test_far_competing_candidates_cannot_arm_motion_release():
    tracker = recovered_tracker()
    results = [tracker.update(observation(frame, (x, 100, .99), (x + 20, 100, .95)))
               for frame, x in ((10, 170), (11, 190), (12, 210), (13, 230))]
    assert all(result["center"] is None for result in results)
    assert all(not result["motion_release_active"] for result in results)


def test_nearby_competitor_blocks_release_even_if_parent_selects_one_object():
    tracker = recovered_tracker()
    results = [tracker.update(observation(frame, (x, 100, .9), (x + 12, 100, .05)))
               for frame, x in ((10, 118), (11, 136), (12, 156))]
    assert all(result["detected"] for result in results)
    assert all(not result["motion_release_active"] for result in results)


@pytest.mark.parametrize("camera_status", ["unavailable_no_correction", None])
def test_unavailable_camera_alignment_cannot_arm_release(camera_status):
    tracker = recovered_tracker()
    results = [tracker.update(observation(frame, (x, 100, .7), camera_status=camera_status))
               for frame, x in ((10, 118), (11, 136), (12, 156))]
    assert all(not result["motion_release_active"] for result in results)


def test_camera_failure_resets_a_partial_release_chain():
    tracker = recovered_tracker()
    for frame, x in ((10, 118), (11, 136)):
        tracker.update(observation(frame, (x, 100, .7)))
    failure = tracker.update(observation(12, (156, 100, .7),
                                         camera_status="unavailable_no_correction"))
    resumed = tracker.update(observation(13, (176, 100, .7)))
    assert failure["motion_release_active"] is False
    assert resumed["motion_release_active"] is False


def test_motion_release_scales_with_image_size():
    tracker = recovered_tracker(scale=2)
    results = launch(tracker, scale=2)
    assert [result["motion_release_active"] for result in results] == [False, False, True]
    assert results[-1]["center"] == [312, 200]


def test_release_never_bypasses_the_hard_loss_horizon():
    tracker = recovered_tracker()
    launch(tracker)
    for frame in range(13, 26):
        assert tracker.update(observation(frame))["center"] is None
    returns = [tracker.update(observation(frame, (156 + 20 * (frame - 12), 100, .9)))
               for frame in (26, 27, 28)]
    assert all(result["center"] is None for result in returns)
    assert all(result["reason"] == "lost_requires_new_seed" for result in returns)


def test_another_slow_recovery_rearms_the_local_guard():
    tracker = recovered_tracker()
    assert launch(tracker)[-1]["motion_release_active"] is True
    for frame, x in enumerate((176, 190, 198, 202, 204, 205, 205, 205), start=13):
        tracker.update(observation(frame, (x, 100, .7)))
    for frame in (21, 22):
        assert tracker.update(observation(frame))["center"] is None
    results = [tracker.update(observation(frame, (205, 100, .3))) for frame in (23, 24, 25)]
    assert results[-1]["reason"] == "slow_recovery_provisional"
    assert results[-1]["motion_release_active"] is False
    assert results[-1]["post_recovery_local_gate"] is True


def test_known_stationary_background_is_vetoed_without_vetoing_the_flying_ball():
    tracker = recovered_tracker()
    # This repeated proposal is well separated from accepted ball observations.
    launch(tracker, extra=((210, 100, .99),))
    for frame in (13, 14):
        tracker.update(observation(frame, (210, 100, .99)))
    first = tracker.update(observation(15, (210, 100, .99), (216, 100, .7)))
    second = tracker.update(observation(16, (210, 100, .99), (236, 100, .7)))
    assert first["center"] != [210, 100]
    assert second["center"] == [236, 100]
    assert first["stationary_veto_count"] >= 1
    assert second["stationary_veto_count"] >= 1
