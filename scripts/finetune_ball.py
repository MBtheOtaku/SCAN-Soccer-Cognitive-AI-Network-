"""Offline SCAN ball adaptation experiment; never changes production weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/reference-review/ball-labels/export-v1"
RUNS = ROOT / "data/reference-review/ball-finetune"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def inventory(folder):
    """Validate our one-ball exports before any costly training starts."""
    import cv2

    report = json.loads((folder / "export_report.json").read_text(encoding="utf-8"))["clips"]
    seen_groups, seen_sources, records = {}, {}, []
    for split in ("train", "val"):
        images = sorted((folder / "images" / split).glob("*.jpg"))
        labels = sorted((folder / "labels" / split).glob("*.txt"))
        if not images or {p.stem for p in images} != {p.stem for p in labels}:
            raise ValueError(f"Missing or mismatched images/labels in {split}.")
        positive = 0
        for image in images:
            clip = image.stem.split("_")[0]
            item = report[clip]
            if item["split"] != split or not item["group"]:
                raise ValueError("Export report does not match image split.")
            group = item["group"]
            if group in seen_groups and seen_groups[group] != split:
                raise ValueError("Recording group crosses train/val split.")
            seen_groups[group] = split
            sha = digest(image)
            if sha in seen_sources and seen_sources[sha] != split:
                raise ValueError("Identical image appears across train/val splits.")
            seen_sources[sha] = split
            decoded = cv2.imread(str(image))
            if decoded is None:
                raise ValueError(f"Unreadable image: {image}")
            label = folder / "labels" / split / (image.stem + ".txt")
            text = label.read_text(encoding="utf-8").strip()
            if text:
                values = text.split()
                if len(values) != 5 or values[0] != "0":
                    raise ValueError("Expected at most one class-0 ball box per image.")
                x, y, w, h = map(float, values[1:])
                if not all(math.isfinite(v) for v in (x, y, w, h)) or not (
                    0 < w <= 1 and 0 < h <= 1 and w/2-1e-7 <= x <= 1-w/2+1e-7
                    and h/2-1e-7 <= y <= 1-h/2+1e-7
                ):
                    raise ValueError("Invalid normalised ball box.")
                positive += 1
            records.append({"image": image.relative_to(folder).as_posix(),
                            "image_sha256": sha, "label_sha256": digest(label)})
        if not positive:
            raise ValueError(f"No positive ball examples in {split}.")
    return records


def training_options(dataset, run, epochs, imgsz, batch, device, seed):
    return dict(data=str(dataset), project=str(run), name="train", exist_ok=False,
                epochs=epochs, imgsz=imgsz, batch=batch, device=device, workers=0,
                seed=seed, deterministic=True, optimizer="AdamW", lr0=0.0001,
                lrf=0.1, freeze=10, patience=5, pretrained=True, amp=False,
                nbs=batch, warmup_epochs=0.0, warmup_bias_lr=0.0,
                cache=False, plots=False, save=True, val=True,
                mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0,
                perspective=0.0, shear=0.0, scale=0.15, translate=0.05,
                fliplr=0.5, flipud=0.0, hsv_h=0.01, hsv_s=0.25, hsv_v=0.25)


def iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area / union if union > 0 else 0.0


def score_frame(truth, predictions, threshold=0.5):
    # One labelled target per image: at most one prediction can be a true positive.
    overlaps = [iou(truth, p["box"]) for p in predictions] if truth is not None else []
    hit = truth is not None and any(v >= threshold for v in overlaps)
    return {"tp": int(hit), "fp": len(predictions)-int(hit),
            "fn": int(truth is not None and not hit),
            "best_iou": max(overlaps, default=0.0) if truth is not None else None}


def summarise(rows):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "frames_with_false_detections": sum(r["fp"] > 0 for r in rows),
            "frames": len(rows)}


def check_model(model, allow_coco=False):
    names = model.names
    if model.task == "detect" and names == {0: "ball"}:
        return 0
    if allow_coco and model.task == "detect" and names.get(32) == "sports ball":
        return 32
    raise ValueError(f"Expected a one-class ball model or approved COCO sports-ball seed: {names}")


def compare(run, baseline, adapted, imgsz, device, confidence):
    import cv2
    from ultralytics import YOLO

    out = run / "comparison"
    out.mkdir(exist_ok=False)
    models = {"baseline": YOLO(str(baseline)), "adapted": YOLO(str(adapted))}
    classes = {name: check_model(model, allow_coco=True)
               for name, model in models.items()}
    frames, cards = [], []
    images = sorted((run / "dataset/images/val").glob("*.jpg"))
    if not images:
        raise ValueError("No validation images.")
    for number, source in enumerate(images):
        image = cv2.imread(str(source))
        h, w = image.shape[:2]
        text = (run / "dataset/labels/val" / (source.stem + ".txt")).read_text().strip()
        truth = None
        if text:
            _, x, y, bw, bh = map(float, text.split())
            truth = [(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h]
        row = {"image": source.name, "truth": truth}
        overlay = image.copy()
        if truth:
            cv2.rectangle(overlay, tuple(round(v) for v in truth[:2]),
                          tuple(round(v) for v in truth[2:]), (80, 255, 80), 2)
        for name, model in models.items():
            result = model.predict(image, device=device, imgsz=imgsz, conf=confidence,
                                   iou=0.5, max_det=24, classes=[classes[name]], rect=False,
                                   verbose=False)[0]
            predictions = [{"box": box, "confidence": conf} for box, conf in
                           zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())]
            row[name] = {**score_frame(truth, predictions), "predictions": predictions}
            colour = (220, 80, 240) if name == "baseline" else (255, 220, 0)
            for prediction in predictions:
                box = prediction["box"]
                cv2.rectangle(overlay, tuple(round(v) for v in box[:2]),
                              tuple(round(v) for v in box[2:]), colour, 2)
        if not cv2.imwrite(str(out / source.name), overlay):
            raise OSError("Could not write comparison image; check free disk space.")
        frames.append(row)
        details = " · ".join(f'{n}: TP {row[n]["tp"]}, FP {row[n]["fp"]}, FN {row[n]["fn"]}'
                             for n in models)
        cards.append(f'<figure><img loading="lazy" src="{source.name}"><figcaption>{html.escape(source.name)}<br>{details}</figcaption></figure>')
        if number % 10 == 0:
            print(f"Compared {number+1}/{len(images)} validation images", flush=True)
    report = {"split": "development_validation_used_for_checkpoint_selection_not_final_test",
              "mode": "full_frame_no_tracking", "imgsz": imgsz, "confidence": confidence,
              "matching_iou": 0.5, "nms_iou": 0.5, "max_det": 24,
              "baseline_sha256": digest(baseline), "adapted_sha256": digest(adapted),
              "summary": {n: summarise([r[n] for r in frames]) for n in models}, "frames": frames}
    write_json(out / "report.json", report)
    page = ('<!doctype html><meta charset="utf-8"><title>SCAN detector comparison</title>'
            '<style>body{background:#09120c;color:#eef5ee;font:16px system-ui}main{display:flex;flex-wrap:wrap}figure{width:300px;margin:10px}img{width:100%}pre{white-space:pre-wrap}</style>'
            '<h1>Development validation comparison</h1><p>Green: manual label. Pink: baseline. Cyan: adapted.</p>'
            '<p>Fixed confidence and IoU thresholds; full-frame detections without temporal tracking. '
            'These validation clips selected the checkpoint; this is not an untouched test set. '
            'Small boxes make IoU sensitive to a few pixels. Inspect the overlays too.</p><pre>'
            + html.escape(json.dumps(report["summary"], indent=2)) + '</pre><main>' + ''.join(cards) + '</main>')
    (out / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("Review: " + str(out / "index.html"))


def train_sanity(run, metadata):
    """Check whether a saved adaptation can locate its own training examples."""
    import cv2
    from ultralytics import YOLO

    baseline = run / "baseline.pt"
    adapted = Path(metadata["best"])
    if not baseline.is_file() or not adapted.is_file():
        raise ValueError("A saved baseline and adapted checkpoint are required.")
    if digest(baseline) != metadata["baseline_sha256"] or digest(adapted) != metadata["best_sha256"]:
        raise ValueError("Checkpoint differs from the saved experiment.")
    if inventory(run / "dataset") != metadata["dataset_inventory"]:
        raise ValueError("Dataset snapshot differs from the saved experiment.")
    images = sorted((run / "dataset/images/train").glob("*.jpg"))
    if not images:
        raise ValueError("No training images in the experiment snapshot.")
    out = run / "train-sanity"
    out.mkdir(exist_ok=False)
    models = {"baseline": YOLO(str(baseline)), "adapted": YOLO(str(adapted))}
    classes = {name: check_model(model, allow_coco=True)
               for name, model in models.items()}
    thresholds = (0.01, 0.10, 0.25)
    rows, cards = [], []
    for number, source in enumerate(images, 1):
        image = cv2.imread(str(source))
        h, w = image.shape[:2]
        label = (run / "dataset/labels/train" / (source.stem + ".txt")).read_text().strip()
        truth = None
        if label:
            _, x, y, bw, bh = map(float, label.split())
            truth = [(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h]
        row = {"image": source.name, "truth": truth}
        overlay = image.copy()
        if truth is not None:
            cv2.rectangle(overlay, tuple(round(v) for v in truth[:2]),
                          tuple(round(v) for v in truth[2:]), (80, 255, 80), 2)
        for name, model in models.items():
            result = model.predict(image, device=metadata["options"]["device"],
                                   imgsz=metadata["options"]["imgsz"], conf=thresholds[0],
                                   iou=0.5, max_det=100, classes=[classes[name]], rect=False,
                                   verbose=False)[0]
            predictions = [{"box": box, "confidence": confidence} for box, confidence in
                           zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())]
            scores = {str(t): score_frame(truth, [p for p in predictions if p["confidence"] >= t])
                      for t in thresholds}
            row[name] = {"scores": scores, "predictions": predictions}
            # Show the highest-overlap candidate even when it is below the normal cutoff.
            if truth is not None and predictions:
                candidate = max(predictions, key=lambda p: iou(truth, p["box"]))
                box = candidate["box"]
                colour = (220, 80, 240) if name == "baseline" else (255, 220, 0)
                cv2.rectangle(overlay, tuple(round(v) for v in box[:2]),
                              tuple(round(v) for v in box[2:]), colour, 2)
        if not cv2.imwrite(str(out / source.name), overlay):
            raise OSError("Could not write training review image.")
        rows.append(row)
        detail = "; ".join(f'{name} hits at 0.25: {row[name]["scores"]["0.25"]["tp"]}'
                           for name in models)
        cards.append(f'<figure><img loading="lazy" src="{source.name}"><figcaption>'
                     f'{html.escape(source.name)}<br>{html.escape(detail)}</figcaption></figure>')
        if number % 10 == 0 or number == len(images):
            print(f"Checked {number}/{len(images)} training images", flush=True)
    summary = {name: {str(t): summarise([row[name]["scores"][str(t)] for row in rows])
                      for t in thresholds} for name in models}
    report = {"split": "train_seen_during_adaptation_not_generalisation",
              "mode": "full_frame_no_tracking", "imgsz": metadata["options"]["imgsz"],
              "candidate_confidence_floor": thresholds[0], "score_confidences": thresholds,
              "matching_iou": 0.5, "nms_iou": 0.5, "max_det": 100,
              "baseline_sha256": digest(baseline), "adapted_sha256": digest(adapted),
              "summary": summary, "frames": rows}
    write_json(out / "report.json", report)
    page = ('<!doctype html><meta charset="utf-8"><title>SCAN train sanity</title>'
            '<style>body{background:#09120c;color:#eef5ee;font:16px system-ui}'
            'main{display:flex;flex-wrap:wrap}figure{width:300px;margin:10px}'
            'img{width:100%}pre{white-space:pre-wrap}</style>'
            '<h1>Training image sanity check</h1>'
            '<p>These images were used to train the adapted model. This is a memorisation check, '
            'not an estimate of performance on new footage. Green: manual label; pink: baseline '
            'candidate with highest overlap; cyan: adapted candidate with highest overlap. '
            'Displayed candidates may be below 0.25 confidence. Tiny boxes make IoU sensitive '
            'to a few pixels. See report.json for all proposals and scores.</p><pre>'
            + html.escape(json.dumps(summary, indent=2)) + '</pre><main>' + ''.join(cards) + '</main>')
    (out / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Review: " + str(out / "index.html"))


def train(args):
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not args.weights.is_file():
        raise ValueError("Local baseline weights missing; no model download will be requested.")
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz < 320 or args.imgsz % 32:
        raise ValueError("Use positive epochs/batch and imgsz >=320 divisible by 32.")
    records = inventory(args.dataset)
    clips = json.loads((args.dataset / "export_report.json").read_text(encoding="utf-8"))["clips"]
    groups = {split: sorted({item["group"] for item in clips.values() if item["split"] == split})
              for split in ("train", "val")}
    model = YOLO(str(args.weights))
    baseline_class = check_model(model, allow_coco=args.coco_sports_ball)
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = RUNS / stamp
    run.mkdir()
    # Snapshot both data and initial checkpoint; never train against mutable labels.
    shutil.copytree(args.dataset, run / "dataset", ignore=shutil.ignore_patterns("*.cache"))
    shutil.copy2(args.weights, run / "baseline.pt")
    dataset_yaml = run / "dataset/dataset.yaml"
    dataset_yaml.write_text('path: ' + json.dumps((run / 'dataset').as_posix()) +
                            '\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n', encoding="utf-8")
    options = training_options(dataset_yaml, run, args.epochs, args.imgsz, args.batch,
                               args.device, args.seed)
    metadata = {"status": "running", "python": sys.version, "torch": torch.__version__,
                "ultralytics": ultralytics.__version__, "cuda_available": torch.cuda.is_available(),
                "model_parameters": sum(p.numel() for p in model.model.parameters()),
                "baseline_class_id": baseline_class,
                "baseline_sha256": digest(run / "baseline.pt"), "dataset_inventory": records,
                "options": options, "comparison_confidence": args.confidence,
                "recording_groups": groups,
                "limitation": "Validation recordings were used for checkpoint selection and are not an "
                              "independent test set; inspect full clips before deployment."}
    write_json(run / "experiment.json", metadata)
    print("Experiment: " + str(run), flush=True)
    started = time.perf_counter()
    try:
        model.train(**options)
        best = Path(model.trainer.best)
        if not best.is_file():
            raise RuntimeError("Training ended without best.pt.")
        # Compare to the original snapshot on exactly the same validation images.
        if not args.skip_compare:
            compare(run, run / "baseline.pt", best, args.imgsz, args.device, args.confidence)
        metadata.update(status="complete", best=str(best), best_sha256=digest(best))
    except Exception as exc:
        metadata.update(status="failed", error_type=type(exc).__name__)
        raise
    except KeyboardInterrupt:
        metadata.update(status="interrupted")
        raise
    finally:
        metadata["elapsed_seconds"] = time.perf_counter() - started
        write_json(run / "experiment.json", metadata)
    print("Saved adapted checkpoint: " + str(best))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    training.add_argument("--weights", type=Path, default=ROOT / "data/models/football-ball-detection.pt")
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--imgsz", type=int, default=960)
    training.add_argument("--batch", type=int, default=2)
    training.add_argument("--device", default="cpu")
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--coco-sports-ball", action="store_true",
                          help="Allow a COCO-pretrained sports-ball seed instead of a one-class seed")
    training.add_argument("--confidence", type=float, default=0.25)
    training.add_argument("--skip-compare", action="store_true")
    evaluation = sub.add_parser("compare")
    evaluation.add_argument("run", type=Path)
    sanity = sub.add_parser("train-sanity")
    sanity.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        if not 0 < args.confidence < 1:
            parser.error("Confidence must be between zero and one.")
        train(args)
    elif args.command == "compare":
        metadata = json.loads((args.run / "experiment.json").read_text(encoding="utf-8"))
        compare(args.run, args.run / "baseline.pt", Path(metadata["best"]),
                metadata["options"]["imgsz"], metadata["options"]["device"], metadata["comparison_confidence"])
    else:
        metadata = json.loads((args.run / "experiment.json").read_text(encoding="utf-8"))
        if metadata["status"] != "complete":
            parser.error("The experiment must be complete before its training check.")
        train_sanity(args.run, metadata)


if __name__ == "__main__":
    main()
