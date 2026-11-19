import json
import sys
from types import SimpleNamespace

import pytest

from scripts.finetune_ball import (check_model, inventory, iou, score_frame,
                                   summarise, training_options)


def test_perfect_match_and_extra_false_object():
    truth = [10, 10, 20, 20]
    predictions = [{"box": truth}, {"box": [100, 100, 110, 110]}]
    assert score_frame(truth, predictions) == {"tp": 1, "fp": 1, "fn": 0, "best_iou": 1.0}


def test_wrong_object_is_both_miss_and_false_positive():
    row = score_frame([10, 10, 20, 20], [{"box": [100, 100, 110, 110]}])
    assert (row["tp"], row["fp"], row["fn"]) == (0, 1, 1)
    assert iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1/7)


def test_absent_ball_and_empty_predictions():
    assert score_frame(None, []) == {"tp": 0, "fp": 0, "fn": 0, "best_iou": None}
    assert score_frame(None, [{"box": [1, 1, 2, 2]}])["fp"] == 1
    assert score_frame([1, 1, 2, 2], [])["fn"] == 1


def test_duplicate_ball_predictions_only_one_true_positive():
    box = [1, 1, 2, 2]
    assert score_frame(box, [{"box": box}, {"box": box}])["fp"] == 1


def test_summary_does_not_manufacture_precision_without_predictions():
    result = summarise([score_frame([1, 1, 2, 2], [])])
    assert result["precision"] is None
    assert result["recall"] == 0


def test_training_is_bounded_and_windows_compatible(tmp_path):
    options = training_options(tmp_path / "dataset.yaml", tmp_path, 1, 640, 2, "cpu", 17)
    assert options["epochs"] == 1 and options["workers"] == 0
    assert options["seed"] == 17 and options["deterministic"]
    assert options["freeze"] == 10 and options["mosaic"] == 0
    assert options["lr0"] == 0.0001 and not options["amp"]
    assert options["nbs"] == 2 and options["warmup_epochs"] == 0


def test_coco_sports_ball_seed_requires_explicit_opt_in(tmp_path):
    model = SimpleNamespace(task="detect", names={32: "sports ball"})
    with pytest.raises(ValueError, match="COCO sports-ball"):
        check_model(model)
    assert check_model(model, allow_coco=True) == 32


def make_dataset(root):
    import cv2
    import numpy as np
    report = {"clips": {}}
    for split, clip, shade in [("train", "aaaaaaaaaaaa", 60), ("val", "bbbbbbbbbbbb", 120)]:
        images, labels = root / "images" / split, root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        cv2.imwrite(str(images / (clip + "_000001.jpg")), np.full((32, 32, 3), shade, dtype=np.uint8))
        (labels / (clip + "_000001.txt")).write_text("0 0.5 0.5 0.2 0.2\n")
        report["clips"][clip] = {"split": split, "group": split + "-session"}
    (root / "export_report.json").write_text(json.dumps(report))
    return report


def test_inventory_hashes_and_rejects_group_leakage(tmp_path):
    report = make_dataset(tmp_path)
    rows = inventory(tmp_path)
    assert len(rows) == 2 and len(rows[0]["image_sha256"]) == 64
    report["clips"]["bbbbbbbbbbbb"]["group"] = "train-session"
    (tmp_path / "export_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="group crosses"):
        inventory(tmp_path)


def test_inventory_rejects_invalid_labels(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "labels/train/aaaaaaaaaaaa_000001.txt").write_text("0 0.99 0.5 0.4 0.2")
    with pytest.raises(ValueError, match="normalised"):
        inventory(tmp_path)


def test_comparison_writes_before_after_report_with_same_prediction_settings(tmp_path, monkeypatch):
    from scripts.finetune_ball import compare

    make_dataset(tmp_path / "dataset")
    baseline, adapted = tmp_path / "baseline.pt", tmp_path / "adapted.pt"
    baseline.write_bytes(b"baseline")
    adapted.write_bytes(b"adapted")
    calls = []

    class Array:
        def __init__(self, data):
            self.data = data

        def cpu(self):
            return self

        def tolist(self):
            return self.data

    class Model:
        task, names = "detect", {0: "ball"}

        def __init__(self, path):
            self.path = path

        def predict(self, image, **options):
            calls.append(options)
            box = [13, 13, 19, 19] if self.path.endswith("adapted.pt") else [0, 0, 3, 3]
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Array([box]), conf=Array([0.8])))]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    compare(tmp_path, baseline, adapted, 640, "cpu", 0.25)
    report = json.loads((tmp_path / "comparison/report.json").read_text())
    assert report["summary"]["baseline"]["fn"] == 1
    assert report["summary"]["baseline"]["fp"] == 1
    assert report["summary"]["adapted"]["tp"] == 1
    assert calls[0] == calls[1]
    assert (tmp_path / "comparison/index.html").is_file()


def test_train_sanity_uses_training_split_and_preserves_validation_report(tmp_path, monkeypatch):
    from scripts.finetune_ball import digest, inventory, train_sanity

    dataset = tmp_path / "dataset"
    make_dataset(dataset)
    baseline, adapted = tmp_path / "baseline.pt", tmp_path / "adapted.pt"
    baseline.write_bytes(b"baseline")
    adapted.write_bytes(b"adapted")
    (tmp_path / "comparison").mkdir()
    (tmp_path / "comparison/report.json").write_text("existing validation")
    calls = []

    class Array:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class Model:
        task, names = "detect", {0: "ball"}

        def __init__(self, path):
            self.adapted = path.endswith("adapted.pt")

        def predict(self, image, **options):
            calls.append(options)
            box = [13, 13, 19, 19] if self.adapted else [0, 0, 3, 3]
            confidence = 0.12 if self.adapted else 0.9
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=Array([box]), conf=Array([confidence])))]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    metadata = {"status": "complete", "best": str(adapted),
                "best_sha256": digest(adapted), "baseline_sha256": digest(baseline),
                "dataset_inventory": inventory(dataset),
                "options": {"imgsz": 640, "device": "cpu"}}
    train_sanity(tmp_path, metadata)
    report = json.loads((tmp_path / "train-sanity/report.json").read_text())
    assert report["split"].startswith("train_seen")
    assert [row["image"] for row in report["frames"]] == ["aaaaaaaaaaaa_000001.jpg"]
    assert report["summary"]["adapted"]["0.1"]["tp"] == 1
    assert report["summary"]["adapted"]["0.25"]["fn"] == 1
    assert calls[0] == calls[1]
    assert calls[0]["conf"] == 0.01
    assert (tmp_path / "comparison/report.json").read_text() == "existing validation"
