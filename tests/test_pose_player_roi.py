"""Player crops must preserve native geometry and conservative pose checks."""
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from src.backend.pose_mvp import player_roi


def box(x=1200, y=1800, w=200, h=400, score=.9):
    return [x, y, x + w, y + h, score]


def test_native_portrait_crop_contains_the_complete_movement_envelope():
    detections = [[box(1400)], [box(1700)], [box(1900)]]
    crop = player_roi.person_envelope(detections, 2160, 3840)
    assert crop.bounds == (1280, 1680, 2160, 2320)
    assert crop.person_frames == crop.sampled_frames == 3
    assert crop.bounds[3] - crop.bounds[1] < 3840 / 2
    for candidates in detections:
        x1, y1, x2, y2, _ = candidates[0]
        assert crop.bounds[0] <= x1 < x2 <= crop.bounds[2]
        assert crop.bounds[1] <= y1 < y2 <= crop.bounds[3]


def test_landscape_coordinates_and_image_edges_are_preserved():
    crop = player_roi.person_envelope([[box(0, 20)], [box(20, 20)], [box(40, 20)]], 1280, 720)
    assert crop.bounds == (0, 0, 360, 540)
    assert crop.edge_frames == 1


def test_missing_frames_stay_missing_and_do_not_shrink_the_envelope():
    crop = player_roi.person_envelope([[box()], [], [box(1500)], [], [box(1700)]], 2160, 3840)
    assert crop.sampled_frames == 5
    assert crop.person_frames == 3
    assert crop.bounds == (1080, 1680, 2020, 2320)


def test_single_isolated_false_person_cannot_calibrate_a_crop():
    with pytest.raises(RuntimeError, match="Too few repeated"):
        player_roi.person_envelope([[], [box()], []], 2160, 3840)


@pytest.mark.parametrize("bad", [
    box(score=.34), box(score=1.1), box(w=0), box(h=0), box(x=float("nan")),
    [1, 2, 3, 4], [float("inf"), 1, 2, 3, .9],
])
def test_weak_nonfinite_and_degenerate_boxes_are_not_evidence(bad):
    with pytest.raises(RuntimeError, match="Too few repeated"):
        player_roi.person_envelope([[bad]] * 3, 2160, 3840)


def test_multiple_people_do_not_trigger_largest_person_selection():
    with pytest.raises(player_roi.AmbiguousPlayerCrop):
        player_roi.person_envelope([[box(), box(100, 200, 60, 150)]] * 3, 2160, 3840)


def test_tiny_background_proposal_does_not_block_a_clear_foreground_player():
    primary = box(430, 948, 100, 234)
    background = box(564, 933, 30, 79, .74)
    crop = player_roi.person_envelope([[primary, background]] * 3, 1080, 1920)
    reference = player_roi.person_envelope([[primary]] * 3, 1080, 1920)
    assert crop.bounds == reference.bounds
    assert crop.background_candidate_frames == 3
    assert crop.person_frames == 3


@pytest.mark.parametrize("secondary", [box(100, 200, 20, 300), box(100, 200, 200, 120), box(100, 200, 170, 350)])
def test_background_exclusion_requires_both_height_and_area_separation(secondary):
    with pytest.raises(player_roi.AmbiguousPlayerCrop):
        player_roi.person_envelope([[box(), secondary]] * 3, 2160, 3840)


@pytest.mark.parametrize("width,height,padding", [(20, 100, .3), (100, 20, .3), (100, 100, -1), (100, 100, float("nan"))])
def test_invalid_crop_parameters_are_rejected(width, height, padding):
    with pytest.raises(ValueError):
        player_roi.person_envelope([[box()]] * 3, width, height, padding)


def test_missing_weights_never_invoke_a_downloader(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Missing model must not instantiate YOLO or download weights")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=forbidden))
    with pytest.raises(RuntimeError, match="weights are unavailable"):
        player_roi.detect_person_crop(tmp_path / "video.mov", tmp_path / "missing.pt")


def fake_detector_runtime(monkeypatch, tmp_path, frames, fail_prediction=False):
    weights = tmp_path / "person.pt"
    weights.write_bytes(b"local-test-model")
    class Tensor:
        def __init__(self, values):
            self.values = values
        def cpu(self):
            return self
        def numpy(self):
            return np.array(self.values)
    calls = []
    class Detector:
        def predict(self, image, **kwargs):
            calls.append((image.shape[:2], kwargs))
            if fail_prediction:
                raise RuntimeError("inference unavailable")
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([box()[:4]]), conf=Tensor([.9])))]
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: Detector()))
    class Capture:
        released = False
        def isOpened(self):
            return True
        def get(self, _):
            return len(frames)
        def set(self, _, value):
            self.index = value
        def read(self):
            frame = frames[self.index]
            return frame is not None, frame
        def release(self):
            self.released = True
    cap = Capture()
    monkeypatch.setattr(player_roi.cv2, "VideoCapture", lambda _: cap)
    return weights, cap, calls


def test_decoded_portrait_shape_controls_crop_and_person_class_filter(monkeypatch, tmp_path):
    portrait = np.zeros((3840, 2160, 3), dtype=np.uint8)
    weights, cap, calls = fake_detector_runtime(monkeypatch, tmp_path, [portrait] * 3)
    result = player_roi.detect_person_crop(Path("phone.mov"), weights, samples=3)
    assert result.bounds == (1080, 1680, 1520, 2320)
    assert cap.released
    assert len(calls) == 3
    assert all(shape == (3840, 2160) and options["classes"] == [0] and options["imgsz"] == 1280 for shape, options in calls)


@pytest.mark.parametrize("failure", ["inference", "dimensions", "decode"])
def test_capture_is_released_on_calibration_failure(monkeypatch, tmp_path, failure):
    frames = [np.zeros((3840, 2160, 3), np.uint8)] * 3
    if failure == "dimensions":
        frames[1] = np.zeros((2160, 3840, 3), np.uint8)
    elif failure == "decode":
        frames = [None] * 3
    weights, cap, _ = fake_detector_runtime(monkeypatch, tmp_path, frames, failure == "inference")
    with pytest.raises(RuntimeError):
        player_roi.detect_person_crop(Path("phone.mov"), weights, samples=3)
    assert cap.released


@pytest.fixture
def pose_module(monkeypatch):
    # Exercise the production adapter without needing a MediaPipe native graph.
    monkeypatch.setitem(sys.modules, "mediapipe", types.ModuleType("mediapipe"))
    name = "pose_crop_adapter_test"
    source = Path(__file__).resolve().parents[1] / "src/backend/pose_mvp/scan_pose_mvp_v2.py"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_production_calibration_prefers_person_crop_without_full_frame_pose_bootstrap(monkeypatch, pose_module):
    monkeypatch.setattr(player_roi, "detect_person_crop", lambda *args, **kwargs: player_roi.PersonCrop((100, 200, 600, 900), 12, 9, 2))
    diagnostics = {}
    roi = pose_module.detect_player_roi(Path("video.mov"), Path("pose.task"), diagnostics=diagnostics)
    assert (roi.x1, roi.y1, roi.x2, roi.y2) == (100, 200, 600, 900)
    assert diagnostics["crop_source"] == "person_detector"
    assert diagnostics["person_at_image_edge_sample_frames"] == 2


def test_ambiguous_calibration_is_preserved_for_analysis_guard(monkeypatch, pose_module):
    def ambiguous(*args, **kwargs):
        raise player_roi.AmbiguousPlayerCrop("Multiple people")
    monkeypatch.setattr(player_roi, "detect_person_crop", ambiguous)
    diagnostics = {}
    with pytest.raises(player_roi.AmbiguousPlayerCrop):
        pose_module.detect_player_roi(Path("video.mov"), Path("pose.task"), diagnostics=diagnostics)
    assert diagnostics["player_identity_ambiguous"] is True


def test_optional_person_detector_failure_retains_pose_only_fallback(monkeypatch, pose_module):
    def unavailable(*args, **kwargs):
        raise RuntimeError("local model unavailable")
    monkeypatch.setattr(player_roi, "detect_person_crop", unavailable)
    landmarks = [SimpleNamespace(x=.4 + .1 * (i % 2), y=.3 + .1 * (i % 4), visibility=.9, presence=.9) for i in range(33)]
    class Landmarker:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def detect(self, image):
            return SimpleNamespace(pose_landmarks=[landmarks])
    pose_module.mp = SimpleNamespace(
        tasks=SimpleNamespace(BaseOptions=lambda **kwargs: None, vision=SimpleNamespace(
            PoseLandmarker=SimpleNamespace(create_from_options=lambda options: Landmarker()),
            PoseLandmarkerOptions=lambda **kwargs: None,
            RunningMode=SimpleNamespace(IMAGE="image"),
        )),
        Image=lambda **kwargs: None, ImageFormat=SimpleNamespace(SRGB="rgb"),
    )
    class Capture:
        released = False
        def isOpened(self):
            return True
        def get(self, field):
            return {player_roi.cv2.CAP_PROP_FRAME_COUNT: 3, player_roi.cv2.CAP_PROP_FRAME_WIDTH: 1280, player_roi.cv2.CAP_PROP_FRAME_HEIGHT: 720}[field]
        def set(self, *args):
            pass
        def read(self):
            return True, np.zeros((720, 1280, 3), np.uint8)
        def release(self):
            self.released = True
    cap = Capture()
    monkeypatch.setattr(pose_module.cv2, "VideoCapture", lambda _: cap)
    diagnostics = {}
    roi = pose_module.detect_player_roi(Path("video.mov"), Path("pose.task"), diagnostics=diagnostics)
    assert diagnostics == {"person_detector_status": "unavailable", "crop_source": "pose_calibration_fallback"}
    assert roi.x1 < .4 * 1280 and roi.x2 > .5 * 1280
    assert roi.y1 < .3 * 720 and roi.y2 > .6 * 720
    assert cap.released


def test_crop_mapping_does_not_change_landmark_confidence_threshold(pose_module):
    landmarks = [SimpleNamespace(x=.5, y=.5, presence=.34, visibility=.99) for _ in range(33)]
    assert pose_module.compute_metrics(landmarks, pose_module.ROI(100, 200, 600, 900), 0, 0) is None
    assert np.array_equal(pose_module.px(landmarks, 0, pose_module.ROI(100, 200, 600, 900)), [350, 550])


def visible_landmarks():
    points = [SimpleNamespace(x=.5, y=.5, presence=.9, visibility=.9) for _ in range(33)]
    for index, x, y in [(11, .3, .2), (12, .7, .2), (23, .4, .5), (24, .6, .5), (25, .4, .7), (26, .6, .7), (27, .4, .9), (28, .6, .9)]:
        points[index].x, points[index].y = x, y
    return points


@pytest.mark.parametrize("field,value", [("x", -.01), ("x", 1.01), ("y", -.01), ("y", 1.01), ("x", float("nan"))])
def test_confident_off_crop_ankle_is_not_a_measured_pose(pose_module, field, value):
    landmarks = visible_landmarks()
    setattr(landmarks[27], field, value)
    assert pose_module.compute_metrics(landmarks, pose_module.ROI(100, 200, 600, 900), 0, 0) is None


def test_visible_core_is_accepted_even_if_an_unused_hand_is_outside_crop(pose_module):
    landmarks = visible_landmarks()
    landmarks[15].x = 1.1
    metric = pose_module.compute_metrics(landmarks, pose_module.ROI(100, 200, 600, 900), 0, 0)
    assert metric is not None
    assert metric.left_ankle_x == 300
    assert metric.left_ankle_y == 830
