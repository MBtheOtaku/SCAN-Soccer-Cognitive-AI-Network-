"""Offline ball-detector gate on fixed, reviewed SCAN frames.

The crop is centred on the labelled ball, so this measures detection only,
not localization, association, or a deployable search strategy.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from scripts.ball_dataset import file_sha256, validate_labels

FRAMES = {
    "6292a1b2b194": [80, 90, 100],
    "560706a458b1": [80, 90],
    "13a1022c5009": [76, 80],
    "e9935750581e": [118, 120],
}
ROOT = Path(__file__).resolve().parents[2]


def evaluate(weights: Path, size: int, confidence: float, frame_set: dict | None = None,
             class_id: int = 0, center_mode: str = "oracle", slice_offset: int = 0,
             only_split: str | None = None):
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(weights))
    expected_name = "ball" if class_id == 0 else "sports ball" if class_id == 32 else None
    if model.task != "detect" or expected_name is None or model.names.get(class_id) != expected_name:
        raise ValueError(f"Expected class {class_id} to be {expected_name}: {model.names}")
    rows = []
    cases = (frame_set["clips"] if frame_set is not None else
             [{"clip_id": clip, "frames": numbers,
               "split": "val" if clip == "e9935750581e" else "train"}
              for clip, numbers in FRAMES.items()])
    if center_mode not in ("oracle", "trajectory") or slice_offset < 0:
        raise ValueError(f"Unknown crop centre mode: {center_mode}")
    seen = set()
    for case in cases:
        clip, numbers, split = case["clip_id"], case["frames"], case["split"]
        if only_split is not None and split != only_split:
            continue
        if split not in ("train", "val") or not numbers or len(numbers) != len(set(numbers)):
            raise ValueError(f"Invalid frame selection for {clip}")
        if any((clip, number) in seen for number in numbers):
            raise ValueError(f"Duplicate frame selection for {clip}")
        seen.update((clip, number) for number in numbers)
        tracked_folder = ROOT / "experiments/ball_tracking/reviewed_labels" / clip
        folder = tracked_folder if tracked_folder.is_dir() else ROOT / "data/reference-review/ball-labels" / clip
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        label_data = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
        validate_labels(manifest, label_data)
        source = ROOT / "data/input" / manifest["source_name"]
        if file_sha256(source) != manifest["source_sha256"]:
            raise ValueError(f"Source video changed: {clip}")
        path_centers = None
        if center_mode == "trajectory":
            if "trajectory" not in case:
                raise ValueError(f"Trajectory is required for {clip}")
            trajectory = json.loads((ROOT / case["trajectory"]).read_text(encoding="utf-8"))
            result = trajectory["result"]
            if (trajectory["status"] != "offline_pilot_not_production" or
                    Path(result["source_name"]).name != manifest["source_name"]):
                raise ValueError(f"Trajectory does not belong to {clip}")
            path_centers = {row["frame"]: row["center"] for row in result["path"]}
        image_folder = ROOT / "data/reference-review/ball-labels" / clip
        images = {number: image_folder / item["image"]
                  for number, item in ((item["frame"], item) for item in manifest["frames"])}
        annotations = label_data["annotations"]
        labels = {item["frame"]: item for item in annotations}
        for frame in numbers:
            label = labels.get(frame)
            if frame not in images or label is None or label.get("state") != "visible":
                raise ValueError(f"Missing visible reference ball: {clip} frame {frame}")
            image = cv2.imread(str(images[frame]))
            if image is None:
                raise ValueError(f"Unreadable reference image: {images[frame]}")
            box = label["box"]
            truth = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            crop_center = truth if path_centers is None else path_centers.get(frame)
            if crop_center is None:
                rows.append({"clip": clip, "frame": frame, "split": split,
                             "crop_center": None, "candidate_count": 0,
                             "other_proposals": 0,
                             "nearest_px": None, "within_20_px": False})
                continue
            if image.shape[1] < size or image.shape[0] < size:
                raise ValueError(f"Crop exceeds reference image: {images[frame]}")
            offsets = (0,) if slice_offset == 0 else (-slice_offset, 0, slice_offset)
            origins = list(dict.fromkeys(
                (max(0, min(image.shape[1] - size, round(crop_center[0] + dx - size / 2))),
                 max(0, min(image.shape[0] - size, round(crop_center[1] + dy - size / 2))))
                for dx, dy in itertools.product(offsets, repeat=2)))
            crops = [image[top:top + size, left:left + size] for left, top in origins]
            results = model.predict(crops, imgsz=640, conf=confidence,
                                    classes=[class_id], device="cpu", verbose=False)
            detections = []
            for (left, top), result in zip(origins, results, strict=True):
                for box, score in zip(result.boxes.xyxy.cpu().numpy().tolist(),
                                      result.boxes.conf.cpu().numpy().tolist(), strict=True):
                    detections.append({"center": (left + (box[0] + box[2]) / 2,
                                                  top + (box[1] + box[3]) / 2),
                                       "confidence": score})
            # The same ball may appear in overlapping slices. Merge close centres
            # without looking at the reviewed ball location.
            proposals = []
            for detection in sorted(detections, key=lambda item: -item["confidence"]):
                if all(math.dist(detection["center"], item["center"]) > 15
                       for item in proposals):
                    proposals.append(detection)
            distances = [math.dist(truth, item["center"]) for item in proposals]
            rows.append({"clip": clip, "frame": frame, "split": split,
                         "crop_center": crop_center,
                         "crop_count": len(origins),
                         "candidate_count": len(proposals), "nearest_px": min(distances, default=None),
                         "within_20_px": any(d <= 20 for d in distances),
                         "other_proposals": sum(d > 20 for d in distances),
                         "candidates": proposals})
    return {"weights": str(weights.resolve()), "crop_size": size, "confidence": confidence,
            "class_id": class_id, "class_name": expected_name,
            "center_mode": center_mode,
            "slice_offset": slice_offset,
            "limitation": ("Oracle-centred crops use the ground-truth location and cannot test search or identity."
                           if center_mode == "oracle" else
                           "Path-centred crops use an unverified, manually seeded offline trajectory; "
                           "nearby detections do not establish ball identity."),
            "rows": rows,
            "summary": {split: {"hits": sum(r["within_20_px"] for r in rows if r["split"] == split),
                                "frames": sum(r["split"] == split for r in rows),
                                "other_proposals": sum(r["other_proposals"]
                                                       for r in rows if r["split"] == split)}
                        for split in ("train", "val")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--class-id", type=int, choices=(0, 32), default=0,
                        help="0 for a SCAN ball model; 32 for a COCO sports-ball model")
    parser.add_argument("--frames-file", type=Path,
                        help="Reviewed frame selection JSON; omitted for the original nine-frame gate")
    parser.add_argument("--frame-range", nargs=2, type=int, metavar=("FIRST", "LAST"),
                        help="For a single configured clip, test every frame in this inclusive range")
    parser.add_argument("--center-mode", choices=("oracle", "trajectory"), default="oracle")
    parser.add_argument("--slice-offset", type=int, default=0,
                        help="Add a 3x3 grid of overlapping crops around each centre")
    parser.add_argument("--only-split", choices=("train", "val"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame_set = json.loads(args.frames_file.read_text(encoding="utf-8")) if args.frames_file else None
    if args.frame_range:
        if (frame_set is None or len(frame_set["clips"]) != 1 or
                args.frame_range[0] > args.frame_range[1]):
            parser.error("--frame-range requires one configured clip and FIRST <= LAST")
        frame_set["clips"][0]["frames"] = list(range(args.frame_range[0], args.frame_range[1] + 1))
    if args.center_mode == "trajectory" and frame_set is None:
        parser.error("--center-mode trajectory requires --frames-file with trajectory paths")
    report = evaluate(args.weights, args.size, args.confidence, frame_set,
                      args.class_id, args.center_mode, args.slice_offset,
                      args.only_split)
    report["frame_set_sha256"] = (hashlib.sha256(args.frames_file.read_bytes()).hexdigest()
                                   if args.frames_file else None)
    report["requested_frame_range"] = args.frame_range
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
