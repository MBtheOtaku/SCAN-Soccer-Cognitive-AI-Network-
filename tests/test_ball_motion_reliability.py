"""Exercise the provisional trajectory warning without claiming ball identity."""

from experiments.ball_tracking.reliability import mark_reliability


def path_with_late_jump():
    centers = [(100 + 3 * index, 100) for index in range(9)]
    centers += [(161, 101), (166, 101)]
    return [{"frame": index + 1, "center": center}
            for index, center in enumerate(centers)]


def test_late_near_goal_jump_marks_remaining_path_uncertain():
    path = path_with_late_jump()

    result = mark_reliability(path, anchor=0, goal_center=(170, 100))

    assert result["uncertain_from_frame"] == 10
    assert result["reason"] == "late_near_goal_motion_jump"
    assert all(row["reliability"] == "provisional" for row in path[:9])
    assert all(row["reliability"] == "uncertain" for row in path[9:])


def test_no_goal_or_smooth_path_remains_provisional():
    path = path_with_late_jump()
    assert mark_reliability(path, anchor=0, goal_center=None)["status"] == "provisional"
    assert all(row["reliability"] == "provisional" for row in path)

    smooth = path[:9]
    assert mark_reliability(smooth, anchor=0, goal_center=(170, 100))["status"] == "provisional"


def test_fast_launch_is_not_flagged_as_late_jump():
    path = [{"frame": index + 1, "center": (100 + 30 * index, 100)}
            for index in range(12)]

    result = mark_reliability(path, anchor=0, goal_center=(450, 100))

    assert result["status"] == "provisional"
