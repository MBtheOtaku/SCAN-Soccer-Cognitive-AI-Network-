"""Close objects must not disappear merely because detector scores differ."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.ball_tracking.native_box_detector import box_iou, detect_native_boxes, merge_boxes


def proposal(box, confidence):
    return {"bbox_native_xyxy": box, "center": [(box[0] + box[2]) / 2,
                                               (box[1] + box[3]) / 2],
            "confidence": confidence}


def test_nearby_goal_object_cannot_suppress_nonoverlapping_ball_box():
    # Development-frame geometry: the centres are within the old 45px radius.
    ball = proposal([969.6, 1991.7, 994.5, 2013.3], .115)
    clutter = proposal([944.4, 1996.5, 961.2, 2013.6], .143)
    assert box_iou(ball["bbox_native_xyxy"], clutter["bbox_native_xyxy"]) == 0
    assert merge_boxes([ball, clutter]) == [clutter, ball]


def test_overlapping_tile_duplicates_merge_by_box_overlap():
    first = proposal([100, 200, 120, 220], .2)
    duplicate = proposal([101, 199, 121, 219], .3)
    assert merge_boxes([first, duplicate]) == [duplicate]


def test_partial_overlap_does_not_merge_distinct_candidates():
    a = proposal([100, 200, 120, 220], .2)
    b = proposal([115, 200, 135, 220], .3)
    assert len(merge_boxes([a, b])) == 2


@pytest.mark.parametrize("box", [[1, 2, 1, 3], [1, 2, 3], [1, 2, float('nan'), 3]])
def test_invalid_box_rejected(box):
    with pytest.raises(ValueError):
        merge_boxes([{"bbox_native_xyxy": box, "confidence": .3}])


@pytest.mark.parametrize("threshold", [0, 1, float('nan')])
def test_invalid_merge_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        merge_boxes([], threshold)


def tensor(values):
    return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array(values)))


def test_tile_translation_and_cross_tile_duplicate_merge_preserve_true_neighbours():
    calls = []

    class Model:
        def predict(self, crops, **kwargs):
            calls.append((len(crops), kwargs))
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=tensor([[200, 200, 220, 220]]), conf=tensor([.4])))
                    for _ in crops]

    image = np.zeros((640, 640, 3), np.uint8)
    proposals, count = detect_native_boxes(Model(), image, {"native_xyxy": [0, 0, 640, 640]})
    assert count == 4 and len(proposals) == 4
    assert {tuple(p["bbox_native_xyxy"]) for p in proposals} == {
        (200, 200, 220, 220), (200, 456, 220, 476),
        (456, 200, 476, 220), (456, 456, 476, 476),
    }
    assert calls[0][1] == {"imgsz": 640, "conf": .05, "classes": [0],
                           "device": "cpu", "verbose": False}


def test_native_inference_batches_are_bounded():
    calls = []

    class Model:
        def predict(self, crops, **kwargs):
            calls.append(len(crops))
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=tensor([]), conf=tensor([])))
                    for _ in crops]

    image = np.zeros((1920, 1920, 3), np.uint8)
    proposals, count = detect_native_boxes(Model(), image, {"native_xyxy": [0, 0, 1920, 1920]})
    assert proposals == [] and count == 49
    assert sum(calls) == 49 and max(calls) == 16


def test_local_grid_always_includes_the_causal_query_centred_crop():
    samples = []

    class Model:
        def predict(self, crops, **kwargs):
            samples.extend(crop[192, 192].tolist() for crop in crops)
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=tensor([]), conf=tensor([])))
                    for _ in crops]

    image = np.zeros((2500, 1500, 3), np.uint8)
    image[1994, 983] = [255, 10, 20]
    region = {"native_xyxy": [695, 1706, 1271, 2282],
              "query_center_native": [982.517898, 1994.123319]}
    _, count = detect_native_boxes(Model(), image, region)
    assert count == 5
    assert samples[0] == [255, 10, 20]
    assert samples[1:] == [[0, 0, 0]] * 4


def test_query_centred_crop_does_not_duplicate_an_existing_crop():
    class Model:
        def predict(self, crops, **kwargs):
            assert len(crops) == 1
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=tensor([]), conf=tensor([])))]

    _, count = detect_native_boxes(Model(), np.zeros((384, 384, 3), np.uint8),
                                   {"native_xyxy": [0, 0, 384, 384],
                                    "query_center_native": [192, 192]})
    assert count == 1


def test_invalid_query_cannot_request_a_synthetic_or_outside_crop():
    with pytest.raises(ValueError, match="finite native query"):
        detect_native_boxes(None, np.zeros((384, 384, 3), np.uint8),
                            {"native_xyxy": [0, 0, 384, 384],
                             "query_center_native": [float('nan'), 192]})
