"""Offline false-proposal check on frames labelled target-ball absent.

All 384-pixel image tiles are searched. A proposal is not automatically a
false soccer-ball detection: another ball may be visible. The annotation only
says the ball of interest is absent and cannot support its identity.
"""

import argparse
import json
import math
from pathlib import Path

from scripts.ball_dataset import file_sha256, validate_labels

ROOT = Path(__file__).resolve().parents[2]


def origins_for_image(width: int, height: int, size: int, stride: int) -> list[tuple[int, int]]:
    if min(width, height) < size or stride <= 0:
        raise ValueError("Invalid tile size or stride")
    xs = sorted(set([*range(0, width - size + 1, stride), width - size]))
    ys = sorted(set([*range(0, height - size + 1, stride), height - size]))
    return [(x, y) for y in ys for x in xs]


def run(weights: Path, size: int, stride: int, confidence: float) -> dict:
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if model.task != "detect" or model.names.get(0) != "ball":
        raise ValueError("Expected a one-class ball detector")
    rows = []
    for label_path in sorted((ROOT / "data/reference-review/ball-labels").glob("*/labels.json")):
        clip = label_path.parent.name
        manifest = json.loads((label_path.parent / "manifest.json").read_text(encoding="utf-8"))
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        validate_labels(manifest, labels)
        absent = {item["frame"] for item in labels["annotations"] if item["state"] == "absent"}
        if not absent:
            continue
        source = ROOT / "data/input" / manifest["source_name"]
        if file_sha256(source) != manifest["source_sha256"]:
            raise ValueError(f"Source video changed: {clip}")
        for item in manifest["frames"]:
            frame = item["frame"]
            if frame not in absent:
                continue
            image = cv2.imread(str(label_path.parent / item["image"]))
            if image is None:
                raise ValueError(f"Missing reviewed image: {clip} {frame}")
            tile_origins = origins_for_image(image.shape[1], image.shape[0], size, stride)
            detections = []
            for start in range(0, len(tile_origins), 8):
                batch_origins = tile_origins[start:start + 8]
                crops = [image[y:y + size, x:x + size] for x, y in batch_origins]
                results = model.predict(crops, imgsz=640, conf=confidence,
                                        classes=[0], device="cpu", verbose=False)
                for (left, top), result in zip(batch_origins, results, strict=True):
                    for box, likelihood in zip(result.boxes.xyxy.cpu().numpy().tolist(),
                                               result.boxes.conf.cpu().numpy().tolist(), strict=True):
                        detections.append({"center": [left + (box[0] + box[2]) / 2,
                                                       top + (box[1] + box[3]) / 2],
                                           "confidence": likelihood})
            proposals = []
            for detection in sorted(detections, key=lambda entry: -entry["confidence"]):
                if all(math.dist(detection["center"], other["center"]) > 15 for other in proposals):
                    proposals.append(detection)
            rows.append({"clip": clip, "frame": frame, "tiles": len(tile_origins),
                         "candidate_count": len(proposals), "candidates": proposals})
            print(f"{clip} frame {frame}: {len(proposals)} proposals", flush=True)
    return {"status": "offline_pilot_not_production", "weights": str(weights.resolve()),
            "size": size, "stride": stride, "confidence": confidence,
            "frames": len(rows), "frames_with_proposals": sum(row["candidate_count"] > 0 for row in rows),
            "limitation": "Target ball absent does not imply no other ball is visible. These full-image detections are candidates, not accepted target identities.",
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.weights, args.size, args.stride, args.confidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": report["frames"],
                      "frames_with_proposals": report["frames_with_proposals"]}, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
