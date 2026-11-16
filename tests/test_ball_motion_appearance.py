"""Current-frame appearance is evidence only in the offline tracker."""

import cv2
import numpy as np

from experiments.ball_tracking.appearance import (
    appearance_evidence, appearance_penalty, seed_template,
)
from experiments.ball_tracking.reliability import mark_reliability


def test_current_frame_patch_favors_same_ball_pattern_over_decoy():
    image = np.zeros((100, 140, 3), np.uint8)
    cv2.circle(image, (25, 50), 10, (210, 210, 210), -1)
    cv2.circle(image, (25, 50), 4, (15, 15, 15), -1)
    template = seed_template(image, [12, 37, 38, 63])
    current = np.zeros_like(image)
    cv2.circle(current, (70, 50), 10, (210, 210, 210), -1)
    cv2.circle(current, (70, 50), 4, (15, 15, 15), -1)
    cv2.rectangle(current, (98, 38), (122, 62), (50, 80, 200), -1)

    ball = appearance_evidence(template, current, (70, 50))
    decoy = appearance_evidence(template, current, (110, 50))

    assert appearance_penalty(ball) < appearance_penalty(decoy)
    assert ball["hist_distance"] < decoy["hist_distance"]


def test_missing_detection_marks_future_path_uncertain():
    path = [
        {"frame": 1, "center": (10, 10), "observed": True},
        {"frame": 2, "center": (20, 10), "observed": False},
        {"frame": 3, "center": (30, 10), "observed": True},
    ]

    result = mark_reliability(path, anchor=0, goal_center=None)

    assert result["reason"] == "no_current_frame_detection"
    assert result["uncertain_from_frame"] == 2
    assert [row["reliability"] for row in path] == [
        "provisional", "uncertain", "uncertain",
    ]


def test_weak_current_frame_appearance_marks_uncertainty():
    path = [{"frame": 1, "center": (10, 10), "observed": True,
             "candidate": {"appearance": {"correlation": -0.2,
                                          "hist_distance": 0.92}}}]

    result = mark_reliability(path, anchor=0, goal_center=None)

    assert result["reason"] == "weak_current_frame_appearance"
    assert path[0]["reliability"] == "uncertain"
