"""Offline confidence sweep on a fixed, labelled ball-crop sample."""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from scripts.finetune_ball import digest, iou, score_frame, summarise, write_json


def truth_box(image, label):
    values = label.read_text(encoding="utf-8").split()
    if not values:
        return None
    height, width = image.shape[:2]
    _, x, y, box_width, box_height = map(float, values)
    return [(x-box_width/2)*width, (y-box_height/2)*height,
            (x+box_width/2)*width, (y+box_height/2)*height]


def probe(dataset, weights, output, *, stride=10, imgsz=960):
    images = sorted((dataset / "images/val").glob("*.jpg"))[::stride]
    if not images:
        raise ValueError("No validation images to probe.")
    models = {name: YOLO(str(path)) for name, path in weights.items()}
    thresholds = (0.01, 0.03, 0.05, 0.10, 0.25)
    rows = []
    for number, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        truth = truth_box(image, dataset / "labels/val" / f"{path.stem}.txt")
        row = {"image": path.name, "truth": truth}
        for name, model in models.items():
            result = model.predict(image, device="cpu", imgsz=imgsz, conf=thresholds[0],
                                   iou=0.5, max_det=300, classes=[0], rect=False,
                                   verbose=False)[0]
            predictions = [{"box": box, "confidence": confidence} for box, confidence in
                           zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())]
            nearest = max(predictions, key=lambda p: iou(truth, p["box"])) if truth and predictions else None
            row[name] = {
                "predictions": predictions,
                "closest": {**nearest, "iou": iou(truth, nearest["box"])} if nearest else None,
                "scores": {str(t): score_frame(truth, [p for p in predictions
                                                       if p["confidence"] >= t]) for t in thresholds},
            }
        rows.append(row)
        print(f"Probed {number}/{len(images)}: {path.name}", flush=True)
    report = {
        "dataset": str(dataset.resolve()), "stride": stride, "imgsz": imgsz,
        "matching_iou": 0.5, "thresholds": thresholds,
        "weights": {name: {"path": str(path.resolve()), "sha256": digest(path)}
                    for name, path in weights.items()},
        "summary": {name: {str(t): summarise([row[name]["scores"][str(t)] for row in rows])
                           for t in thresholds} for name in models},
        "rows": rows,
    }
    write_json(output, report)
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    if args.stride < 1 or args.imgsz < 320 or args.imgsz % 32:
        parser.error("Stride must be positive; image size at least 320 and divisible by 32.")
    if args.output.exists():
        parser.error("Choose a fresh output path.")
    probe(args.dataset, {"baseline": args.baseline, "adapted": args.adapted},
          args.output, stride=args.stride, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
