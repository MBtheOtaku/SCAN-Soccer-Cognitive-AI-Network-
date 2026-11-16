"""Build an auditable, tile-scale YOLO ball dataset from reviewed full frames."""

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

from scripts.finetune_ball import DEFAULT_DATA, ROOT, digest, inventory, write_json
from scripts.sliced_ball import tile_origins

DEFAULT_MINE = (ROOT / "data/reference-review/ball-finetune"
                / "20261108T000537258416Z/train-sanity/report.json")
OUTPUTS = ROOT / "data/reference-review/ball-crops"


def pixel_truth(label: Path, width: int, height: int):
    values = label.read_text(encoding="utf-8").split()
    if not values:
        return None
    _, x, y, box_width, box_height = map(float, values)
    return [(x-box_width/2)*width, (y-box_height/2)*height,
            (x+box_width/2)*width, (y+box_height/2)*height]


def tile_boxes(width: int, height: int, size: int, overlap: int):
    if width < size or height < size:
        raise ValueError("Source frames must be at least as large as the tile.")
    return [[x, y, x + size, y + size]
            for y in tile_origins(height, size, overlap)
            for x in tile_origins(width, size, overlap)]


def intersects(a, b) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def positive_tiles(tiles, truth, limit=2, margin=3):
    eligible = [tile for tile in tiles if tile[0]+margin <= truth[0]
                and tile[1]+margin <= truth[1] and truth[2] <= tile[2]-margin
                and truth[3] <= tile[3]-margin]
    if not eligible:
        raise ValueError("No complete tile contains the reviewed ball box.")
    cx, cy = (truth[0]+truth[2])/2, (truth[1]+truth[3])/2
    rank = lambda tile: ((tile[0]+tile[2])/2-cx)**2 + ((tile[1]+tile[3])/2-cy)**2
    first = min(eligible, key=rank)
    if limit == 1 or len(eligible) == 1:
        return [first]
    remaining = [tile for tile in eligible if tile != first]
    second = max(remaining, key=lambda tile: (tile[0]-first[0])**2 +
                 (tile[1]-first[1])**2)
    return [first, second]


def contains_with_margin(tile, truth, margin=3):
    return (tile[0]+margin <= truth[0] and tile[1]+margin <= truth[1]
            and truth[2] <= tile[2]-margin and truth[3] <= tile[3]-margin)


def negative_tiles(image, tiles, truth, limit):
    import cv2

    choices = [tile for tile in tiles if truth is None or not intersects(tile, truth)]
    scored = []
    for tile in choices:
        x1, y1, x2, y2 = tile
        grey = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        scored.append((cv2.Laplacian(grey, cv2.CV_32F).var(), tile))
    return [tile for _, tile in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]


def centred_tile(box, width, height, size):
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    x = min(max(round(cx-size/2), 0), width-size)
    y = min(max(round(cy-size/2), 0), height-size)
    return [x, y, x+size, y+size]


def mined_negatives(report_path: Path | None, source: Path):
    if report_path is None:
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("split") != "train_seen_during_adaptation_not_generalisation":
        raise ValueError("Hard-negative proposals must come from a training-split sanity report.")
    experiment = report_path.parent.parent / "experiment.json"
    if experiment.is_file():
        saved = json.loads(experiment.read_text(encoding="utf-8"))
        if saved["dataset_inventory"] != inventory(source):
            raise ValueError("Mining report came from a different labelled dataset snapshot.")
    source_names = {image.name for image in (source / "images/train").glob("*.jpg")}
    if {row["image"] for row in report["frames"]} != source_names:
        raise ValueError("Mining report does not cover the selected training images.")
    return {row["image"]: sorted(
        (p for model in ("baseline", "adapted") for p in row[model]["predictions"]),
        key=lambda p: p["confidence"], reverse=True) for row in report["frames"]}


def create(source: Path, output: Path, size=512, overlap=128, mine: Path | None = DEFAULT_MINE):
    import cv2

    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a fresh folder.")
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid tile size or overlap.")
    source_inventory = inventory(source)
    source_report = json.loads((source / "export_report.json").read_text(encoding="utf-8"))
    proposals = mined_negatives(mine, source)
    plan = []
    for split in ("train", "val"):
        for image_path in sorted((source / "images" / split).glob("*.jpg")):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Unreadable source image: {image_path}")
            height, width = image.shape[:2]
            truth = pixel_truth(source / "labels" / split / (image_path.stem + ".txt"), width, height)
            tiles = tile_boxes(width, height, size, overlap)
            for index, tile in enumerate(positive_tiles(tiles, truth) if truth else []):
                plan.append((split, image_path, tile, truth, f"ball{index}", None))
            for index, tile in enumerate(negative_tiles(image, tiles, truth, 1 if truth else 2)):
                plan.append((split, image_path, tile, None, f"empty{index}", None))
            if split == "train":
                for candidate in proposals.get(image_path.name, []):
                    tile = centred_tile(candidate["box"], width, height, size)
                    if truth is not None and intersects(tile, truth) and not contains_with_margin(tile, truth):
                        continue
                    if any(item[0] == split and item[1] == image_path and item[2] == tile
                           for item in plan):
                        continue
                    crop_truth = truth if truth is not None and contains_with_margin(tile, truth) else None
                    kind = "context0" if crop_truth is not None else "hard0"
                    plan.append((split, image_path, tile, crop_truth, kind, candidate))
                    break
    if not any(item[3] is not None and item[0] == "train" for item in plan):
        raise ValueError("Training split has no positive crops.")
    records, counts = [], {split: {"ball": 0, "empty": 0, "hard": 0, "hard_context": 0}
                           for split in ("train", "val")}
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    cached_path, cached_image = None, None
    for split, source_path, tile, truth, kind, candidate in plan:
        if source_path != cached_path:
            cached_image = cv2.imread(str(source_path))
            if cached_image is None:
                raise ValueError(f"Unreadable source image: {source_path}")
            cached_path = source_path
        x1, y1, x2, y2 = tile
        name = f"{source_path.stem}_{kind}.jpg"
        crop = cached_image[y1:y2, x1:x2]
        image_path = output / "images" / split / name
        if not cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Cannot write crop: {image_path}")
        line = ""
        crop_truth = None
        if truth is not None:
            crop_truth = [truth[0]-x1, truth[1]-y1, truth[2]-x1, truth[3]-y1]
            left, top, right, bottom = crop_truth
            line = (f"0 {(left+right)/(2*size):.8f} {(top+bottom)/(2*size):.8f} "
                    f"{(right-left)/size:.8f} {(bottom-top)/size:.8f}\n")
        label_path = output / "labels" / split / (Path(name).stem + ".txt")
        label_path.write_text(line, encoding="utf-8")
        category = ("hard_context" if candidate is not None and truth is not None else
                    "hard" if candidate is not None else
                    "ball" if truth is not None else "empty")
        counts[split][category] += 1
        records.append({"image": f"images/{split}/{name}", "source": source_path.name,
                        "source_box": truth, "crop": tile, "crop_box": crop_truth,
                        "category": category, "proposal": candidate,
                        "image_sha256": digest(image_path), "label_sha256": digest(label_path)})
    (output / "dataset.yaml").write_text(
        "path: " + json.dumps(output.as_posix()) +
        "\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n", encoding="utf-8")
    write_json(output / "export_report.json", source_report)
    report = {"schema": 1, "source_dataset": str(source), "tile_size": size,
              "overlap": overlap, "source_inventory": source_inventory,
              "hard_negative_report": str(mine.resolve()) if mine else None,
              "hard_negative_report_sha256": digest(mine) if mine else None,
              "counts": counts, "crops": records,
              "limitation": "Mined crops contain model-selected distractors. If the reviewed ball "
                            "is in the crop it remains labelled. Inspect mined crops for an "
                            "unlabelled second ball before trusting them."}
    write_json(output / "crop_report.json", report)
    if len(inventory(output)) != len(records):
        raise RuntimeError("Exported crop inventory is incomplete.")
    figures = []
    for record in records:
        if record["category"] in {"hard", "hard_context"}:
            figures.append(f'<figure><img src="{html.escape(record["image"])}"><figcaption>'
                           f'{html.escape(record["source"])}; {record["category"]}</figcaption></figure>')
    page = ('<!doctype html><meta charset="utf-8"><title>SCAN mined crops</title>'
            '<style>body{background:#09120c;color:#edf5ed;font:16px system-ui}'
            'main{display:flex;flex-wrap:wrap}figure{width:260px;margin:10px}'
            'img{width:100%}</style><h1>Mined distractor crops</h1>'
            '<p>Review for another unlabelled ball. Context crops still label the reviewed ball.</p>'
            '<main>' + ''.join(figures) + '</main>')
    (output / "negative-review.html").write_text(page, encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print("Export: " + str(output))
    print("Negative review: " + str(output / "negative-review.html"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--mine", type=Path, default=DEFAULT_MINE)
    parser.add_argument("--no-mine", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or OUTPUTS / stamp
    create(args.source, output, args.tile_size, args.overlap,
           None if args.no_mine else args.mine)


if __name__ == "__main__":
    main()
