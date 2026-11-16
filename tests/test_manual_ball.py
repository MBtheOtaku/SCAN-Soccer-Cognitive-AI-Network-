import numpy as np
import pytest

from scripts.manual_ball_tracker import ManualBallTracker
from scripts.review_manual_ball import evaluate, validate_seed


def patch():
    return np.random.default_rng(4).integers(25, 245, (16, 16), dtype=np.uint8)


def image(*positions):
    frame = np.full((140, 220), 15, dtype=np.uint8)
    for x, y in positions:
        frame[y:y+16, x:x+16] = patch()
    return frame


def test_seed_then_track_translation_after_confirmation():
    tracker = ManualBallTracker(image((50, 50)), [50, 50, 66, 66])
    assert tracker.update(image((53, 51)), 1)["status"] == "calibrating"
    row = tracker.update(image((56, 52)), 2)
    assert row["status"] == "tracking"
    assert row["box"] == pytest.approx([56, 52, 72, 68], abs=1)


def test_competing_similar_balls_stop_and_never_reacquire():
    tracker = ManualBallTracker(image((50, 50)), [50, 50, 66, 66])
    row = tracker.update(image((53, 50), (85, 50)), 1)
    assert row["status"] == "ambiguous" and row["box"] is None
    row = tracker.update(image((56, 50)), 2)
    assert row["status"] == "ambiguous" and row["box"] is None


def test_distant_rogue_ball_does_not_trigger_global_switch():
    tracker = ManualBallTracker(image((50, 50)), [50, 50, 66, 66])
    row = tracker.update(image((53, 50), (180, 100)), 1)
    assert row["box"][0] == pytest.approx(53, abs=1)


def test_disappearance_latches_loss_even_when_target_returns():
    tracker = ManualBallTracker(image((50, 50)), [50, 50, 66, 66])
    assert tracker.update(image(), 1)["status"] == "lost"
    assert tracker.update(image((50, 50)), 2)["box"] is None


def test_fast_departure_does_not_become_a_predicted_measurement():
    tracker = ManualBallTracker(image((50, 50)), [50, 50, 66, 66])
    assert tracker.update(image((180, 100)), 1)["box"] is None


def test_single_identical_replacement_can_cause_undetected_identity_switch():
    # Counterexample, not a robustness claim: the target disappears and a second
    # visually identical ball replaces it along a plausible path. Pixels alone
    # provide no evidence that the identity changed.
    tracker = ManualBallTracker(image((50, 50)), [50, 50, 66, 66])
    tracker.update(image((53, 50)), 1)
    row = tracker.update(image((56, 50)), 2)
    assert row["status"] == "tracking"


def test_reject_low_texture_and_invalid_seed():
    with pytest.raises(ValueError, match="texture"):
        ManualBallTracker(image(), [50, 50, 66, 66])
    with pytest.raises(ValueError, match="inside"):
        ManualBallTracker(image(), [-1, 0, 16, 16])
    with pytest.raises(ValueError, match="four pixels"):
        ManualBallTracker(image(), [0, 0, 2, 2])


def test_reject_frame_gap_and_resolution_change():
    tracker = ManualBallTracker(image((50, 50)), [50, 50, 66, 66])
    with pytest.raises(ValueError, match="consecutive"):
        tracker.update(image((50, 50)), 2)
    with pytest.raises(ValueError, match="dimensions"):
        tracker.update(np.zeros((100, 100), np.uint8), 1)


def test_wrong_video_seed_rejected(tmp_path):
    source = tmp_path / "clip.mov"
    source.write_bytes(b"video")
    with pytest.raises(ValueError, match="source video"):
        validate_seed({"schema": 1, "clip_id": "13a1022c5009", "source_sha256": "wrong"}, source, "13a1022c5009")


def test_posthoc_evaluation_excludes_seed_and_counts_drift():
    manifest = {"clip_id": "13a1022c5009", "source_sha256": "example", "frames": [
        {"frame": f, "width": 220, "height": 140} for f in range(4)]}
    labels = {"schema": 1, "clip_id": manifest["clip_id"], "source_sha256": "example", "annotations": [
        {"frame": f, "state": "visible", "box": [50, 50, 66, 66]} for f in range(3)] +
        [{"frame": 3, "state": "absent", "box": None}]}
    rows = [{"frame": 0, "status": "seeded", "box": [50, 50, 66, 66]},
            {"frame": 1, "status": "tracking", "box": [50, 50, 66, 66]},
            {"frame": 2, "status": "tracking", "box": [100, 100, 116, 116]},
            {"frame": 3, "status": "tracking", "box": [100, 100, 116, 116]}]
    counts = evaluate(rows, manifest, labels, 0)["counts"]
    assert counts == {"visible": 2, "tracked_visible": 2, "center_inside_label": 1,
                      "absent": 1, "accepted_on_absent": 1}
