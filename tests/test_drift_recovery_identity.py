"""Adversarial synthetic checks for the opt-in prediction-drift experiment."""
import pytest

from experiments.ball_tracking.drift_recovery_identity import DriftRecoveryIdentity


def tracker():
    t = DriftRecoveryIdentity((100, 100))
    for f in range(5):
        assert t.update(row(f, [(100, 100)]))["detected"]
    # Model a flight prediction moving away from a slower observed ball.
    t.velocity = (0, -20)
    return t


def row(f, points, aligned=True):
    return {"clip": "synthetic", "split": "test", "frame": f, "camera_motion": {"status": "estimated_translation_only" if aligned else "unavailable"},
            "candidates": [{"center": list(p), "confidence": .8} for p in points]}


def test_recovers_only_current_point_after_three_moving_observations():
    t = tracker()
    results = [t.update(row(f, [(x, 100)])) for f, x in [(5, 155), (6, 159), (7, 163)]]
    assert [r["detected"] for r in results] == [False, False, True]
    assert results[-1]["reason"] == "drift_recovery_provisional"
    assert results[-1]["center"] == [163, 100]
    assert results[-1]["drift_support_frames"] == [5, 6, 7]
    assert results[-1]["identity_verified"] is False
    assert all(r["center"] is None for r in results[:2])


@pytest.mark.parametrize("points,aligned", [
    ([(155, 100)], False), ([(155, 100), (160, 105)], True),
    ([(250, 100)], True),
])
def test_withholds_unaligned_ambiguous_or_outside_anchor(points, aligned):
    t = tracker()
    results = [t.update(row(f, points, aligned)) for f in range(5, 10)]
    assert not any(r["reason"] == "drift_recovery_provisional" for r in results)


def test_stationary_false_positive_cannot_earn_recovery():
    t = tracker()
    results = [t.update(row(f, [(155, 100)])) for f in range(5, 10)]
    assert not any(r["reason"] == "drift_recovery_provisional" for r in results)


def test_no_recovery_after_lost_deadline():
    t = tracker()
    for f in range(5, 18):
        t.update(row(f, []))
    results = [t.update(row(f, [(155 + 4 * (f - 18), 100)])) for f in range(18, 21)]
    assert not any(r["detected"] for r in results)
