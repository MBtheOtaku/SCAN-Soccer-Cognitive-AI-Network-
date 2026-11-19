"""Behaviour checks for the offline, manually seeded ball-association pilot."""

from experiments.ball_tracking.associate_detector_candidates import associate


def _row(frame, *candidates):
    return {
        "clip": "testclip0001",
        "frame": frame,
        "split": "test",
        "candidates": [
            {"center": [x, y], "confidence": confidence}
            for x, y, confidence in candidates
        ],
    }


def test_fast_first_frame_prefers_strong_ball_over_nearby_weak_decoy():
    path = associate(
        [
            _row(0, (8, 0, 0.10), (80, 0, 0.95)),
            _row(1, (85, 0, 0.10), (150, 0, 0.90)),
        ],
        seed=(0, 0),
    )

    assert [result["center"] for result in path] == [[80, 0], [150, 0]]
    assert all(result["detected"] for result in path)


def test_later_goal_distractor_does_not_steal_coherent_ball():
    path = associate(
        [
            _row(0, (0, 0, 0.95)),
            _row(1, (35, 0, 0.90)),
            _row(2, (70, 0, 0.75), (135, 0, 0.99)),
        ],
        seed=(0, 0),
    )

    assert path[2]["eligible_count"] == 2
    assert path[2]["center"] == [70, 0]
    assert path[2]["detected"] is True


def test_missing_proposal_is_not_a_detection_and_next_frame_reacquires():
    path = associate(
        [
            _row(0, (0, 0, 0.95)),
            _row(1, (30, 0, 0.90)),
            _row(2),
            _row(3, (90, 0, 0.90)),
        ],
        seed=(0, 0),
    )

    assert path[2]["detected"] is False
    assert path[2]["center"] is None
    assert path[2]["confidence"] is None
    assert path[2]["candidate_count"] == 0
    assert path[3]["detected"] is True
    assert path[3]["center"] == [90, 0]
