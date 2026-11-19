"""Offline ball search centred on this tracker's own preceding prediction.

Manual contact seed starts a single clip. Labels are read only for validation
and after every detection/association step has completed, for scoring.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

from experiments.ball_tracking.associate_detector_candidates import associate_next
from experiments.ball_tracking.score_detector_candidates import score
from scripts.ball_dataset import file_sha256, validate_labels

ROOT = Path(__file__).resolve().parents[2]


def detect_slices(model, image, center: tuple[float, float], size: int,
                  offset: int, confidence: float) -> tuple[list[dict], int]:
    height, width = image.shape[:2]
    if min(width, height) < size:
        raise ValueError("Crop size exceeds reference image")
    offsets = (-offset, 0, offset) if offset else (0,)
    origins = list(dict.fromkeys(
        (max(0, min(width - size, round(center[0] + dx - size / 2))),
         max(0, min(height - size, round(center[1] + dy - size / 2))))
        for dx, dy in itertools.product(offsets, repeat=2)))
    crops = [image[top:top + size, left:left + size] for left, top in origins]
    results = model.predict(crops, imgsz=640, conf=confidence,
                            classes=[0], device="cpu", verbose=False)
    detections = []
    for (left, top), result in zip(origins, results, strict=True):
        for box, likelihood in zip(result.boxes.xyxy.cpu().numpy().tolist(),
                                   result.boxes.conf.cpu().numpy().tolist(), strict=True):
            detections.append({"center": [left + (box[0] + box[2]) / 2,
                                           top + (box[1] + box[3]) / 2],
                               "confidence": likelihood})
    proposals = []
    for detection in sorted(detections, key=lambda item: -item["confidence"]):
        if all(math.dist(detection["center"], item["center"]) > 15 for item in proposals):
            proposals.append(detection)
    return proposals, len(origins)


def run(clip: str, start: int, end: int, seed: tuple[float, float],
        weights: Path, size: int, offset: int, confidence: float) -> dict:
    import cv2
    from ultralytics import YOLO

    if start > end or size <= 0 or offset < 0 or not 0 < confidence < 1:
        raise ValueError("Invalid frame range or detector settings")
    metadata = json.loads((ROOT / "experiments/ball_tracking/splits-dense-pilot.json")
                          .read_text(encoding="utf-8"))
    split_by_clip = {item["clip_id"]: item["split"] for item in metadata["clips"]}
    if clip not in split_by_clip:
        raise ValueError("Clip is not in the reviewed dense pilot")
    labels_dir = ROOT / "experiments/ball_tracking/reviewed_labels" / clip
    if not labels_dir.is_dir():
        labels_dir = ROOT / "data/reference-review/ball-labels" / clip
    manifest = json.loads((labels_dir / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((labels_dir / "labels.json").read_text(encoding="utf-8"))
    validate_labels(manifest, labels)
    source = ROOT / "data/input" / manifest["source_name"]
    if file_sha256(source) != manifest["source_sha256"]:
        raise ValueError("Source video does not match reviewed labels")
    frames = {item["frame"]: ROOT / "data/reference-review/ball-labels" / clip / item["image"]
              for item in manifest["frames"]}
    annotations = {item["frame"]: item for item in labels["annotations"]}
    for frame in range(start, end + 1):
        if frame not in frames or annotations.get(frame, {}).get("state") != "visible":
            raise ValueError(f"Missing reviewed visible frame: {clip} {frame}")
    model = YOLO(str(weights))
    if model.task != "detect" or model.names.get(0) != "ball":
        raise ValueError("Expected a one-class ball detector")

    last, velocity = tuple(seed), (0.0, 0.0)
    rows, path = [], []
    for frame in range(start, end + 1):
        predicted = (last[0] + velocity[0], last[1] + velocity[1])
        image = cv2.imread(str(frames[frame]))
        if image is None:
            raise ValueError(f"Unreadable reference frame: {frames[frame]}")
        proposals, crops = detect_slices(model, image, predicted, size, offset, confidence)
        row = {"clip": clip, "frame": frame, "split": split_by_clip[clip],
               "crop_center": predicted, "crop_count": crops,
               "candidate_count": len(proposals), "candidates": proposals}
        selected, last, velocity = associate_next(row, last, velocity)
        rows.append(row)
        path.append(selected)
        print(f"Frame {frame}: {len(proposals)} candidates, "
              f"{'selected' if selected['detected'] else 'missing'}", flush=True)

    candidate_score = score({"rows": rows})["summary"]
    selected_rows = [{**row, "candidates":
                      [{"center": selected["center"], "confidence": selected["confidence"]}]
                      if selected["detected"] else [],
                      "candidate_count": int(selected["detected"])}
                     for row, selected in zip(rows, path, strict=True)]
    selected_score = score({"rows": selected_rows})["summary"]
    return {"status": "offline_pilot_not_production", "clip": clip,
            "split": split_by_clip[clip], "source_sha256": manifest["source_sha256"],
            "weights": str(weights.resolve()), "seed": seed,
            "start_frame": start, "end_frame": end,
            "crop_size": size, "slice_offset": offset, "confidence": confidence,
            "candidate_assessment": candidate_score,
            "selected_assessment": selected_score,
            "limitation": "Search is causal and self-centred, but the start is manually seeded, development clips informed method choices, and no ball identity is verified for coaching or shot outcome.",
            "rows": rows, "path": path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--seed", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--slice-offset", type=int, default=128)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.clip, args.start_frame, args.end_frame, tuple(args.seed),
                 args.weights, args.size, args.slice_offset, args.confidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": result["candidate_assessment"],
                      "selected": result["selected_assessment"]}, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
