"""Offline sweep of motion-proposal sensitivity on labelled failure frames.

This diagnoses candidate availability only. It never uses ball labels to make
proposals, and it does not change the tracker or its reliability decisions.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from scripts.ball_dataset import file_sha256, validate_labels

from .pilot_motion_track import align_lk, label_root, output, root


def proposals(previous, current, *, threshold: int, min_area: int, positive: bool) -> list[dict]:
    prior = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    now = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    delta = cv2.subtract(now, prior) if positive else cv2.subtract(prior, now)
    mask = cv2.morphologyEx((delta >= threshold).astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    found = []
    for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if min_area <= area <= 2000 and 3 <= w <= 100 and 3 <= h <= 100:
            found.append({"center": centers[index].tolist(), "area": int(area),
                          "box": [int(x), int(y), int(x + w), int(y + h)]})
    return found


def read_pair(clip_id: str, frame: int):
    folder = label_root / clip_id
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    validate_labels(manifest, labels)
    source = root / "data" / "input" / manifest["source_name"]
    if file_sha256(source) != manifest["source_sha256"]:
        raise ValueError(f"Source hash changed for {clip_id}")
    match = next((row for row in labels["annotations"] if row["frame"] == frame), None)
    if match is None or match["state"] != "visible" or frame < 1:
        raise ValueError(f"{clip_id} frame {frame} needs a visible ball label")
    cap = cv2.VideoCapture(str(source))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
        ok_previous, previous = cap.read()
        ok_current, current = cap.read()
        if not ok_previous or not ok_current:
            raise ValueError(f"Cannot decode {clip_id} frames {frame - 1}:{frame}")
    finally:
        cap.release()
    x1, y1, x2, y2 = match["box"]
    return align_lk(previous, current), current, [(x1 + x2) / 2, (y1 + y2) / 2]


def audit_frame(clip_id: str, frame: int) -> dict:
    previous, current, truth = read_pair(clip_id, frame)
    rows = []
    for threshold in (6, 8, 10, 12, 15, 18):
        for min_area in (3, 6, 10):
            both = proposals(previous, current, threshold=threshold,
                             min_area=min_area, positive=True)
            both += proposals(previous, current, threshold=threshold,
                              min_area=min_area, positive=False)
            near = sorted(((math.dist(item["center"], truth), item) for item in both),
                          key=lambda entry: entry[0])
            rows.append({"threshold": threshold, "min_area": min_area,
                         "proposal_count": len(both),
                         "within_20px": sum(distance <= 20 for distance, _ in near),
                         "within_80px": sum(distance <= 80 for distance, _ in near),
                         "nearest_distance_px": near[0][0] if near else None,
                         "nearest_proposal": near[0][1] if near else None})
    return {"clip_id": clip_id, "frame": frame, "truth_center": truth, "settings": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", metavar="CLIP_ID:FRAME")
    parser.add_argument("--output", type=Path,
                        default=output / "motion-threshold-sweep.json")
    args = parser.parse_args()
    targets = []
    for target in args.targets:
        clip_id, separator, frame = target.partition(":")
        if not separator or not frame.isdecimal():
            parser.error(f"Invalid target {target}; use CLIP_ID:FRAME")
        targets.append((clip_id, int(frame)))
    report = {"status": "offline_development_diagnostic_not_production",
              "note": "Truth is used only to evaluate raw proposals. More proposals can mean more false targets.",
              "frames": [audit_frame(clip_id, frame) for clip_id, frame in targets]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for item in report["frames"]:
        print(item["clip_id"], item["frame"], flush=True)
        for setting in item["settings"]:
            if setting["min_area"] == 10:
                print("  delta", setting["threshold"], "proposals", setting["proposal_count"],
                      "hits", setting["within_20px"], "nearest",
                      round(setting["nearest_distance_px"], 1)
                      if setting["nearest_distance_px"] is not None else None, flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
