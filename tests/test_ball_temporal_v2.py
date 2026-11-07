import math
import pytest
from scripts.ball_temporal_v2 import Candidate, TemporalBall, deduplicate


def ball(x=100, score=0.8, appearance=(1.0, 0.0), size=10):
    return Candidate(x-size/2, 95, x+size/2, 95+size, score, appearance)


def establish(tracker):
    states = [tracker.update(i, i/30, [ball(100+i)]) for i in range(3)]
    assert [r["status"] for r in states] == ["tentative", "tentative", "observed"]
    return states


def test_consecutive_observations_required():
    establish(TemporalBall())


def test_two_startup_objects_are_not_resolved_by_highest_score():
    result = TemporalBall().update(0, 0, [ball(100, .95), ball(300, .7)])
    assert result["status"] == "uncertain"
    assert result["observation"] is None


def test_seed_guides_matching_but_is_not_itself_an_observation():
    tracker = TemporalBall(seed=(100, 100))
    result = tracker.update(0, 0, [])
    assert result["status"] == "lost" and result["observation"] is None
    result = tracker.update(1, 1/30, [ball(100, .3), ball(300, .95)])
    assert result["status"] == "tentative"
    assert result["observation"]["x1"] == 95


def test_distant_high_score_does_not_take_over():
    tracker = TemporalBall()
    establish(tracker)
    result = tracker.update(3, .1, [ball(103, .4), ball(400, .99)])
    assert result["status"] == "observed"
    assert result["observation"]["x1"] == 98


def test_appearance_mismatch_is_not_an_observation():
    tracker = TemporalBall()
    establish(tracker)
    result = tracker.update(3, .1, [ball(103, appearance=(0.0, 1.0))])
    assert result["status"] == "predicted" and result["observation"] is None


def test_misses_do_not_become_measurements_and_expire():
    tracker = TemporalBall(max_missing=2)
    establish(tracker)
    for i in (3, 4):
        result = tracker.update(i, i/30, [])
        assert result["status"] == "predicted" and result["observation"] is None
        assert result["prediction"] is not None
    result = tracker.update(5, 5/30, [])
    assert result["status"] == "lost" and result["prediction"] is None


def test_ambiguous_matches_do_not_update_history():
    tracker = TemporalBall()
    establish(tracker)
    result = tracker.update(3, .1, [ball(93), ball(113)])
    assert result["status"] == "uncertain" and result["observation"] is None
    assert tracker.history[-1][0] == 2


def test_tentative_sequence_is_broken_by_missing_frame():
    tracker = TemporalBall()
    tracker.update(0, 0, [ball()])
    tracker.update(1, 1/30, [])
    assert tracker.update(2, 2/30, [ball()])["status"] == "tentative"


def test_reacquisition_starts_a_new_segment():
    tracker = TemporalBall(max_missing=0)
    establish(tracker)
    assert tracker.update(3, .1, [])["status"] == "lost"
    results = [tracker.update(i, i/30, [ball(300)]) for i in (4, 5, 6)]
    assert results[-1]["segment"] == 2


def test_duplicate_boxes_and_invalid_candidates_are_removed():
    candidates = [ball(), ball(score=.7), Candidate(0, 0, 0, 10, .9), ball(math.nan)]
    assert len(deduplicate(candidates)) == 1


def test_bad_timestamp_or_skipped_frame_is_rejected():
    tracker = TemporalBall()
    tracker.update(0, 0, [])
    with pytest.raises(ValueError):
        tracker.update(1, 0, [])
    with pytest.raises(ValueError):
        tracker.update(2, .1, [])


def test_large_size_jump_is_rejected():
    tracker = TemporalBall()
    establish(tracker)
    result = tracker.update(3, .1, [ball(103, size=60)])
    assert result["observation"] is None
