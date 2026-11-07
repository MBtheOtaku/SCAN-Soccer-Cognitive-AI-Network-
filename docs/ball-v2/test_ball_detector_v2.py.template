from types import SimpleNamespace
import numpy as np
import pytest
import scripts.ball_detector_v2 as module


class Tensor:
    def __init__(self, values):
        self.values = values
    def cpu(self):
        return self
    def tolist(self):
        return self.values


class Model:
    names = {0: "ball"}
    def __init__(self):
        self.calls = []
    def predict(self, crop, **options):
        self.calls.append((crop.shape, options))
        boxes = SimpleNamespace(xyxy=Tensor([[10, 20, 30, 40]]), conf=Tensor([.9]))
        return [SimpleNamespace(boxes=boxes)]


def test_tiles_cover_edges_and_small_images():
    assert module.tile_starts(100) == [0]
    assert module.tile_starts(720) == [0, 80]
    starts = module.tile_starts(1280)
    assert starts == [0, 480, 640]
    assert all(any(start <= pixel < start + 640 for start in starts) for pixel in range(1280))


def test_metadata_selects_ball_zero_and_maps_crop_offsets(monkeypatch):
    model = Model()
    monkeypatch.setattr(module, "YOLO", lambda _: model)
    detector = module.BallDetector("unused.pt", mode="tiled")
    candidates = detector.detect(np.zeros((1280, 720, 3), dtype=np.uint8))
    assert detector.ball_class_id == 0
    assert len(model.calls) == 6
    assert all(options["classes"] == [0] for _, options in model.calls)
    assert {(c.x1, c.y1) for c in candidates} == {
        (10, 20), (90, 20), (10, 500), (90, 500), (10, 660), (90, 660)}
    assert all(abs(sum(c.appearance) - 1) < 1e-9 for c in candidates)


def test_single_mode_does_not_tile(monkeypatch):
    model = Model()
    monkeypatch.setattr(module, "YOLO", lambda _: model)
    detector = module.BallDetector("unused.pt", mode="single")
    assert len(detector.detect(np.zeros((1280, 720, 3), dtype=np.uint8))) == 1
    assert len(model.calls) == 1


def test_unknown_model_classes_require_explicit_selection(monkeypatch):
    model = Model()
    model.names = {0: "person"}
    monkeypatch.setattr(module, "YOLO", lambda _: model)
    with pytest.raises(ValueError, match="Specify --ball-class-id"):
        module.BallDetector("unused.pt")
