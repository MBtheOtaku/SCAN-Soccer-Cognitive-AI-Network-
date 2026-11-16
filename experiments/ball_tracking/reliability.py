"""Provisional reliability flags for hand-seeded offline ball paths.

These flags describe suspicious trajectory changes; they do not prove a ball
identity swap or a physical event. No flagged position should feed coaching.
"""

import math
import statistics


def mark_reliability(path, anchor, goal_center):
    """Mark the first missing detection, weak appearance, or late jump.

    Keeping the boundary sticky is deliberate: this pilot cannot independently
    re-identify the ball after an ambiguous transition.
    """
    boundary = None
    reason = None
    for row in path:
        evidence = row.get("candidate", {}).get("appearance")
        if row.get("observed") is False:
            boundary, reason = row["frame"], "no_current_frame_detection"
            break
        if evidence and (evidence["correlation"] < 0.05 and
                         evidence["hist_distance"] > 0.75):
            boundary, reason = row["frame"], "weak_current_frame_appearance"
            break
    if goal_center is not None:
        for index in range(4, len(path)):
            current = path[index]
            previous = path[index - 1]
            if current["frame"] - anchor < 8:
                continue
            prior_speeds = [math.dist(path[j]["center"], path[j-1]["center"])
                            for j in range(index - 3, index)]
            typical_speed = statistics.median(prior_speeds)
            jump = math.dist(current["center"], previous["center"])
            if (math.dist(previous["center"], goal_center) <= 120 and
                    typical_speed <= 15 and jump >= 25 and
                    jump >= 2 * typical_speed):
                if boundary is None or current["frame"] < boundary:
                    boundary = current["frame"]
                    reason = "late_near_goal_motion_jump"
                break
    for row in path:
        row["reliability"] = ("uncertain" if boundary is not None and
                              row["frame"] >= boundary else "provisional")
    return {
        "status": "uncertain" if boundary is not None else "provisional",
        "uncertain_from_frame": boundary,
        "reason": reason,
    }
