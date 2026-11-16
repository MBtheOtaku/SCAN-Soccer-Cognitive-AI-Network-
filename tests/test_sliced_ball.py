import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.finetune_ball import score_frame
from scripts.review_sliced_ball import audit_test_independence, audit_test_inventory, chosen_images
from scripts.sliced_ball import predict_sliced, suppress_duplicates, tile_origins


def test_windows_cover_edges_and_reject_invalid_stride():
    assert tile_origins(1280, 512, 128) == [0, 384, 768]
    assert tile_origins(720, 512, 128) == [0, 208]
    assert tile_origins(300, 512, 128) == [0]
    with pytest.raises(ValueError, match="overlap"):
        tile_origins(720, 512, 512)
    for size in range(1, 1500, 37):
        origins = tile_origins(size, 512, 128)
        assert origins[0] == 0
        assert origins[-1] + min(size, 512) == size
        assert all(b <= a + 512 for a, b in zip(origins, origins[1:]))


def test_overlap_detections_return_to_full_frame_and_merge_once():
    image = np.zeros((768, 768, 3), dtype=np.uint8)
    image[400:408, 400:408, 0] = 255
    calls = []

    class Array:
        def __init__(self, data):
            self.data = data

        def cpu(self):
            return self

        def tolist(self):
            return self.data

    class MarkerDetector:
        def predict(self, crop, **options):
            calls.append(options)
            ys, xs = np.where(crop[:, :, 0] == 255)
            boxes = [[int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]] if len(xs) else []
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Array(boxes),
                                                          conf=Array([0.8] if boxes else [])))]

    result = predict_sliced(image, MarkerDetector(), tile_size=512, overlap=256)
    assert len(result["tiles"]) == len(calls) == 4
    assert len(result["raw_candidates"]) == 4
    assert len(result["predictions"]) == 1
    assert result["predictions"][0]["box"] == [400, 400, 408, 408]
    assert score_frame([400, 400, 408, 408], result["predictions"])["tp"] == 1
    assert all(call["classes"] == [0] and call["conf"] == 0.25 for call in calls)


def test_duplicate_suppression_retains_separate_ball_proposals():
    proposals = [{"box": [0, 0, 10, 10], "confidence": 0.9},
                 {"box": [1, 1, 11, 11], "confidence": 0.8},
                 {"box": [30, 30, 40, 40], "confidence": 0.7}]
    kept = suppress_duplicates(proposals, 0.5, 10)
    assert [p["confidence"] for p in kept] == [0.9, 0.7]
    assert score_frame([0, 0, 10, 10], kept)["fp"] == 1


def test_review_selects_exact_exported_frames(tmp_path):
    for name in ("a_000001.jpg", "a_000002.jpg", "a_000003.jpg"):
        (tmp_path / name).write_bytes(b"image")
    selected = chosen_images(tmp_path, ["a_000003", "a_000001.jpg"], 1)
    assert [path.name for path in selected] == ["a_000003.jpg", "a_000001.jpg"]
    with pytest.raises(ValueError, match="not in this split"):
        chosen_images(tmp_path, ["missing"], 6)


def test_separate_test_session_can_be_audited_without_train_or_val(tmp_path):
    images = tmp_path / "images/test"
    labels = tmp_path / "labels/test"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "cccccccccccc_000001.jpg").write_bytes(b"test image")
    (labels / "cccccccccccc_000001.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (tmp_path / "export_report.json").write_text(json.dumps({
        "clips": {"cccccccccccc": {"split": "test", "group": "new-session"}}}))
    records = audit_test_inventory(tmp_path)
    assert len(records) == 1 and len(records[0]["image_sha256"]) == 64
    (tmp_path / "export_report.json").write_text(json.dumps({
        "clips": {"cccccccccccc": {"split": "val", "group": "new-session"}}}))
    with pytest.raises(ValueError, match="does not match"):
        audit_test_inventory(tmp_path)


def test_test_session_must_not_overlap_training_or_development(tmp_path):
    trained, fresh = tmp_path / "trained", tmp_path / "fresh"
    trained.mkdir()
    fresh.mkdir()
    (trained / "export_report.json").write_text(json.dumps({"clips": {
        "aaaaaaaaaaaa": {"split": "train", "group": "night"},
        "bbbbbbbbbbbb": {"split": "val", "group": "day"}}}))
    (fresh / "export_report.json").write_text(json.dumps({"clips": {
        "cccccccccccc": {"split": "test", "group": "new"}}}))
    training = [{"image_sha256": "train-hash"}]
    test = [{"image": "images/test/cccccccccccc_000001.jpg",
             "image_sha256": "new-hash"}]
    audit_test_independence(trained, fresh, training, test)
    test[0]["image_sha256"] = "train-hash"
    with pytest.raises(ValueError, match="contains an image"):
        audit_test_independence(trained, fresh, training, test)
    test[0]["image_sha256"] = "new-hash"
    (fresh / "export_report.json").write_text(json.dumps({"clips": {
        "cccccccccccc": {"split": "test", "group": "day"}}}))
    with pytest.raises(ValueError, match="shares a training"):
        audit_test_independence(trained, fresh, training, test)
