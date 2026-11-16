import json

import cv2
import numpy as np
import pytest

from scripts.ball_crop_dataset import (centred_tile, contains_with_margin, create,
                                       intersects, positive_tiles, tile_boxes)
from scripts.finetune_ball import inventory


def test_crop_geometry_keeps_complete_ball_and_excludes_empty_tiles():
    tiles = tile_boxes(128, 128, 64, 16)
    truth = [28, 28, 36, 36]
    selected = positive_tiles(tiles, truth)
    assert selected == [[0, 0, 64, 64]]
    assert contains_with_margin(selected[0], truth)
    assert not intersects([64, 64, 128, 128], truth)
    assert centred_tile([31, 31, 39, 39], 128, 128, 64) == [3, 3, 67, 67]
    with pytest.raises(ValueError, match="No complete tile"):
        positive_tiles([[64, 64, 128, 128]], truth)


def test_export_preserves_split_and_labels_mined_context(tmp_path):
    source = tmp_path / "reviewed"
    clips = {"aaaaaaaaaaaa": {"split": "train", "group": "night"},
             "bbbbbbbbbbbb": {"split": "val", "group": "day"}}
    for clip, item in clips.items():
        split = item["split"]
        images, labels = source / "images" / split, source / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        image = np.full((128, 128, 3), 50 if split == "train" else 110, dtype=np.uint8)
        image[28:36, 28:36] = 250
        cv2.imwrite(str(images / f"{clip}_000001.jpg"), image)
        (labels / f"{clip}_000001.txt").write_text("0 0.25 0.25 0.0625 0.0625\n")
    (source / "export_report.json").write_text(json.dumps({"clips": clips}))
    mine = tmp_path / "train-sanity.json"
    mine.write_text(json.dumps({"split": "train_seen_during_adaptation_not_generalisation",
                                "frames": [{"image": "aaaaaaaaaaaa_000001.jpg",
                                            "baseline": {"predictions": []},
                                            "adapted": {"predictions": [
                                                {"box": [31, 31, 39, 39], "confidence": 0.8}]}}]}))
    output = tmp_path / "crops"
    report = create(source, output, size=64, overlap=16, mine=mine)
    assert report["counts"]["train"]["hard_context"] == 1
    assert report["counts"]["train"]["empty"] == 1
    assert report["counts"]["val"]["hard_context"] == 0
    assert len(inventory(output)) == len(report["crops"])
    context = next(row for row in report["crops"] if row["category"] == "hard_context")
    assert context["crop_box"] is not None
    assert (output / "labels/train/aaaaaaaaaaaa_000001_context0.txt").read_text().startswith("0 ")
    for row in report["crops"]:
        if row["category"] == "empty":
            assert (output / "labels" / row["image"].split("/")[1] /
                    (row["image"].split("/")[-1][:-4] + ".txt")).read_text() == ""
    with pytest.raises(ValueError, match="already exists"):
        create(source, output, size=64, overlap=16, mine=mine)
