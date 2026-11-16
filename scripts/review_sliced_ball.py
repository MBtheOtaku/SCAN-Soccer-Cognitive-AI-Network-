"""Review offline sliced ball detections against saved, session-separated labels."""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path

from scripts.finetune_ball import check_model, digest, inventory, score_frame, summarise, write_json
from scripts.sliced_ball import predict_sliced


def chosen_images(folder: Path, requested: list[str], max_frames: int) -> list[Path]:
    images = sorted(folder.glob("*.jpg"))
    if not images:
        raise ValueError("No images in the requested split.")
    if requested:
        selected = []
        for value in requested:
            name = Path(value).name
            if Path(name).suffix.lower() != ".jpg":
                name += ".jpg"
            path = folder / name
            if not path.is_file():
                raise ValueError(f"Frame is not in this split: {value}")
            if path not in selected:
                selected.append(path)
        return selected
    if max_frames <= 0:
        raise ValueError("Use --max-frames greater than zero or explicit --frame values.")
    count = min(max_frames, len(images))
    if count == 1:
        return [images[len(images) // 2]]
    return [images[round(i * (len(images) - 1) / (count - 1))] for i in range(count)]


def audit_test_inventory(dataset: Path) -> list[dict]:
    """Allow a separately exported, test-only recording without training data."""
    report = json.loads((dataset / "export_report.json").read_text(encoding="utf-8"))["clips"]
    images = sorted((dataset / "images/test").glob("*.jpg"))
    labels = sorted((dataset / "labels/test").glob("*.txt"))
    if not images or {p.stem for p in images} != {p.stem for p in labels}:
        raise ValueError("Missing or mismatched test images and labels.")
    records = []
    for image in images:
        clip = image.stem.split("_")[0]
        if report[clip]["split"] != "test" or not report[clip]["group"]:
            raise ValueError("Test image does not match the export report.")
        label = dataset / "labels/test" / (image.stem + ".txt")
        parts = label.read_text(encoding="utf-8").split()
        if parts:
            if len(parts) != 5 or parts[0] != "0":
                raise ValueError("Expected one class-0 test ball box per image.")
            x, y, width, height = map(float, parts[1:])
            if not all(math.isfinite(value) for value in (x, y, width, height)) or not (
                0 < width <= 1 and 0 < height <= 1 and width/2-1e-7 <= x <= 1-width/2+1e-7
                and height/2-1e-7 <= y <= 1-height/2+1e-7
            ):
                raise ValueError("Invalid normalised test ball box.")
        records.append({"image": image.relative_to(dataset).as_posix(),
                        "image_sha256": digest(image), "label_sha256": digest(label)})
    return records


def audit_test_independence(training_dataset: Path, test_dataset: Path,
                            training_inventory: list[dict], test_inventory: list[dict]):
    """Keep a future test recording outside every training/development session."""
    training_clips = json.loads((training_dataset / "export_report.json").read_text(
        encoding="utf-8"))["clips"]
    test_clips = json.loads((test_dataset / "export_report.json").read_text(
        encoding="utf-8"))["clips"]
    training_groups = {clip["group"] for clip in training_clips.values()}
    test_groups = {test_clips[row["image"].split("/")[-1].split("_")[0]]["group"]
                   for row in test_inventory}
    if training_groups & test_groups:
        raise ValueError("Test recording shares a training or development session.")
    trained_hashes = {row["image_sha256"] for row in training_inventory}
    if trained_hashes & {row["image_sha256"] for row in test_inventory}:
        raise ValueError("Test recording contains an image from training or development.")


def review(args):
    import cv2
    from ultralytics import YOLO

    run = args.run.resolve()
    metadata = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete":
        raise ValueError("The saved training experiment must be complete.")
    weights = run / "baseline.pt" if args.checkpoint == "baseline" else Path(metadata["best"])
    expected = metadata["baseline_sha256"] if args.checkpoint == "baseline" else metadata["best_sha256"]
    if not weights.is_file() or digest(weights) != expected:
        raise ValueError("Checkpoint is missing or differs from the saved experiment.")
    if inventory(run / "dataset") != metadata["dataset_inventory"]:
        raise ValueError("Dataset snapshot differs from the saved experiment.")
    dataset = (args.source_dataset or run / "dataset").resolve()
    if dataset == (run / "dataset").resolve():
        evaluated_inventory = metadata["dataset_inventory"]
    else:
        evaluated_inventory = audit_test_inventory(dataset) if args.split == "test" else inventory(dataset)
        if args.split == "test":
            seen_inventory = list(metadata["dataset_inventory"])
            source_record = run / "dataset/crop_report.json"
            if source_record.is_file():
                seen_inventory.extend(json.loads(source_record.read_text(
                    encoding="utf-8"))["source_inventory"])
            audit_test_independence(run / "dataset", dataset,
                                    seen_inventory, evaluated_inventory)
        crop_report = run / "dataset/crop_report.json"
        if crop_report.is_file():
            origin = json.loads(crop_report.read_text(encoding="utf-8"))
            if Path(origin["source_dataset"]).resolve() == dataset and origin["source_inventory"] != evaluated_inventory:
                raise ValueError("Original frame dataset changed since the crop experiment.")
    images = chosen_images(dataset / "images" / args.split, args.frame, args.max_frames)
    model = YOLO(str(weights))
    check_model(model)
    previous = None
    if args.confidence == 0.25 and dataset == (run / "dataset").resolve():
        reference = run / ("train-sanity" if args.split == "train" else "comparison") / "report.json"
        if reference.is_file():
            saved = json.loads(reference.read_text(encoding="utf-8"))
            if saved.get(f"{args.checkpoint}_sha256") == expected:
                previous = {row["image"]: row[args.checkpoint] for row in saved["frames"]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    source_label = "snapshot" if dataset == (run / "dataset").resolve() else "source"
    output = run / "sliced-review" / f"{source_label}-{args.split}-{args.checkpoint}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    rows, cards = [], []
    for number, source in enumerate(images, 1):
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Unreadable image: {source}")
        height, width = image.shape[:2]
        label = (dataset / "labels" / args.split / (source.stem + ".txt")).read_text().strip()
        truth = None
        if label:
            _, x, y, box_width, box_height = map(float, label.split())
            truth = [(x-box_width/2)*width, (y-box_height/2)*height,
                     (x+box_width/2)*width, (y+box_height/2)*height]
        result = predict_sliced(image, model, tile_size=args.tile_size, overlap=args.overlap,
                                imgsz=args.imgsz, confidence=args.confidence, device=args.device)
        score = score_frame(truth, result["predictions"])
        row = {"image": source.name, "truth": truth, "score": score, **result}
        if previous is not None and source.name in previous:
            saved = previous[source.name]
            row["full_frame_score"] = saved["scores"]["0.25"] if args.split == "train" else {
                key: saved[key] for key in ("tp", "fp", "fn", "best_iou")}
        overlay = image.copy()
        for tile in result["tiles"]:
            cv2.rectangle(overlay, tuple(tile[:2]), tuple(tile[2:]), (130, 130, 130), 1)
        if truth is not None:
            cv2.rectangle(overlay, tuple(round(v) for v in truth[:2]),
                          tuple(round(v) for v in truth[2:]), (70, 255, 70), 3)
        for prediction in result["predictions"]:
            box = prediction["box"]
            cv2.rectangle(overlay, tuple(round(v) for v in box[:2]),
                          tuple(round(v) for v in box[2:]), (0, 220, 255), 2)
        if not cv2.imwrite(str(output / source.name), overlay):
            raise OSError("Could not write sliced review image.")
        rows.append(row)
        description = (f'TP {score["tp"]}, FP {score["fp"]}, FN {score["fn"]}; '
                       f'{len(result["tiles"])} tiles; {result["elapsed_seconds"]:.1f}s')
        if "full_frame_score" in row:
            full = row["full_frame_score"]
            description += f'; full frame TP {full["tp"]}, FP {full["fp"]}, FN {full["fn"]}'
        cards.append(f'<figure><img loading="lazy" src="{html.escape(source.name)}"><figcaption>'
                     f'{html.escape(source.name)}<br>{html.escape(description)}</figcaption></figure>')
        print(f"Reviewed {number}/{len(images)}: {source.name}: {description}", flush=True)
    report = {"split": args.split, "checkpoint": args.checkpoint, "checkpoint_sha256": expected,
              "evaluation_dataset": str(dataset),
              "evaluation_inventory_sha256": hashlib.sha256(
                  json.dumps(evaluated_inventory, sort_keys=True).encode("utf-8")).hexdigest(),
              "tile_size": args.tile_size, "overlap": args.overlap, "imgsz": args.imgsz,
              "confidence": args.confidence, "matching_iou": 0.5, "nms_iou": 0.5,
              "summary": summarise([row["score"] for row in rows]),
              "total_inference_seconds": sum(row["elapsed_seconds"] for row in rows),
              "frames": rows,
              "limitation": "Offline detector evaluation; no temporal identity, contact or shot outcome. "
                            + ("This test session was checked for distinct group IDs and image hashes."
                               if args.split == "test" else
                               "The saved daylight session informed checkpoint selection and is not "
                               "an untouched final test set.")}
    write_json(output / "report.json", report)
    page = ('<!doctype html><meta charset="utf-8"><title>SCAN sliced ball review</title>'
            '<style>body{background:#09120c;color:#edf5ed;font:16px system-ui}'
            'main{display:flex;flex-wrap:wrap}figure{width:300px;margin:10px}'
            'img{width:100%}pre{white-space:pre-wrap}</style>'
            '<h1>Sliced ball detection</h1><p>Green: reviewed ball. Yellow: merged ball proposals. '
            'Grey: tile boundaries. A proposal is not confirmed ball identity. '
            'This detector has box labels, not segmentation masks.</p><pre>'
            + html.escape(json.dumps(report["summary"], indent=2)) + '</pre><main>'
            + ''.join(cards) + '</main>')
    (output / "index.html").write_text(page, encoding="utf-8")
    print("Review: " + str(output / "index.html"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source-dataset", type=Path,
                        help="Evaluate original labelled frames instead of this run's crop snapshot.")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--checkpoint", choices=("baseline", "adapted"), default="adapted")
    parser.add_argument("--frame", action="append", default=[], help="Exact exported frame name; repeat for several.")
    parser.add_argument("--max-frames", type=int, default=6, help="Evenly spaced frames if --frame is omitted.")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.tile_size <= 0 or not 0 <= args.overlap < args.tile_size or not 0 < args.confidence < 1:
        parser.error("Use positive tile size, overlap smaller than tile size, and confidence in (0,1).")
    review(args)


if __name__ == "__main__":
    main()
