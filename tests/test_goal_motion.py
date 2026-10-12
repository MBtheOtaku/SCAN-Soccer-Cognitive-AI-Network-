import cv2
import numpy as np
from src.backend.goal_motion import GoalMotionTracker
from src.backend.goal_tracking import GoalCalibration, analyze_goal_outcome
from src.backend.ball_tracking import BallDetection


def calibration(dx=0, dy=0):
    return GoalCalibration(((100 + dx) / 400, (50 + dy) / 240), ((200 + dx) / 400, (50 + dy) / 240),
                           ((200 + dx) / 400, (150 + dy) / 240), ((100 + dx) / 400, (150 + dy) / 240))


def texture():
    rng = np.random.default_rng(17)
    # Band-limited texture survives resampling during rotation/zoom, like video.
    gray = cv2.GaussianBlur(rng.integers(0, 256, (240, 400), dtype=np.uint8), (5, 5), .9)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def ball(frame, x):
    return BallDetection(frame, frame / 30, x, 100, x / 400, 100 / 240, 8, 8, .95, None, "detected", .95)


def test_pan_tracks_calibrated_corners():
    reference = texture()
    tracker = GoalMotionTracker(reference, calibration())
    moved = cv2.warpAffine(reference, np.float32([[1, 0, 12], [0, 1, 8]]), (400, 240))
    result = tracker.update(moved, 1)
    assert result is not None
    np.testing.assert_allclose(result.pixel_points(400, 240), calibration(12, 8).pixel_points(400, 240), atol=1)


def test_static_camera_preserves_corners():
    reference = texture()
    tracker = GoalMotionTracker(reference, calibration())
    result = tracker.update(reference, 1)
    np.testing.assert_allclose(result.pixel_points(400, 240), calibration().pixel_points(400, 240), atol=.2)


def test_lost_track_does_not_return_last_polygon():
    reference = texture()
    tracker = GoalMotionTracker(reference, calibration())
    assert tracker.update(reference, 0) is not None
    assert tracker.update(np.zeros_like(reference), 1) is None
    assert tracker.summary()["lost_frames"] == 1
    assert tracker.update(reference, 2) is not None  # Direct reference matching can recover.


def test_camera_pan_alone_cannot_prove_goal_entry():
    detections = [ball(0, 90), ball(1, 110)]
    static = analyze_goal_outcome(detections, 0, calibration(), 400, 240)
    assert static.goal_entry_detected is True  # Demonstrates the original false crossing.
    dynamic = analyze_goal_outcome(detections, 0, calibration(), 400, 240,
                                  frame_calibrations={0: calibration(), 1: calibration(20)})
    assert dynamic.goal_entry_detected is False


def test_observed_entry_uses_current_goal_coordinates():
    result = analyze_goal_outcome([ball(0, 90), ball(1, 160)], 0, calibration(), 400, 240,
                                 frame_calibrations={0: calibration(), 1: calibration(20)})
    assert result.goal_entry_detected is True


def test_lost_goal_between_ball_observations_withholds_judgement():
    result = analyze_goal_outcome([ball(0, 90), ball(2, 160)], 0, calibration(), 400, 240,
                                 frame_calibrations={0: calibration(), 1: None, 2: calibration(20)})
    assert result.goal_entry_detected is None
    assert result.shot_on_target is None


def test_old_calibration_defaults_to_frame_zero():
    data = calibration().to_dict()
    data.pop("reference_frame")
    assert GoalCalibration.from_dict(data).reference_frame == 0


def test_zoom_and_rotation_use_reference_geometry():
    reference = texture()
    tracker = GoalMotionTracker(reference, calibration())
    matrix = cv2.getRotationMatrix2D((150, 100), 2, 1.05)
    moved = cv2.warpAffine(reference, matrix, (400, 240))
    result = tracker.update(moved, 1)
    assert result is not None
    expected = cv2.transform(calibration().pixel_points(400, 240)[None], matrix)[0]
    np.testing.assert_allclose(result.pixel_points(400, 240), expected, atol=1)


def test_nonzero_reference_frame_is_preserved():
    reference = texture()
    cal = calibration()
    cal.reference_frame = 10
    tracker = GoalMotionTracker(reference, cal)
    shifted = cv2.warpAffine(reference, np.float32([[1, 0, 12], [0, 1, 8]]), (400, 240))
    assert tracker.update(shifted, 0) is not None
    result = tracker.update(reference, 10)
    assert result.reference_frame == 10
    np.testing.assert_allclose(tracker.last_matrix, [[1, 0, 0], [0, 1, 0]])
