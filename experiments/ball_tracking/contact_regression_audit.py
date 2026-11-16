"""Audit proposal recall and selection at the 13a1022c5009 contact transition."""

import json
import math

import cv2

from .appearance import appearance_evidence, seed_template
from .pilot_motion_track import align_lk, anchors, candidates, label_root, output, root, track


CLIP = "13a1022c5009"


def main():
    anchor, goal = anchors[CLIP]
    folder = label_root / CLIP
    manifest = json.loads((folder / "manifest.json").read_text())
    labels = {entry["frame"]: entry for entry in
              json.loads((folder / "labels.json").read_text())["annotations"]}
    source = root / "data/input" / manifest["source_name"]
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, anchor)
    ok, previous = cap.read()
    if not ok:
        raise ValueError(source)
    seed = seed_template(previous, labels[anchor]["box"])
    paths = {}
    for mode, kwargs in {
        "positive": {"positive": True},
        "negative": {"negative": True},
        "both_appearance": {"both_signed": True, "appearance_weight": 1,
                            "appearance_max_hist_distance": .85,
                            "appearance_gate_frames": 6},
    }.items():
        result = track(CLIP, anchor, goal, length=30,
                       launch_progress_px_per_frame=12, launch_soft_weight=.1,
                       **kwargs)
        paths[mode] = {row["frame"]: row for row in result["path"]}
    rows = []
    for frame in range(anchor + 1, anchor + 31):
        ok, current = cap.read()
        if not ok:
            break
        label = labels.get(frame)
        if label and label["state"] == "visible":
            box = label["box"]
            truth = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            aligned = align_lk(previous, current)
            row = {"frame": frame, "truth": truth,
                   "seed_to_truth": appearance_evidence(seed, current, truth)}
            for mode in ("positive", "negative"):
                proposals = candidates(aligned, current,
                                       positive=mode == "positive",
                                       negative=mode == "negative")
                nearest = min(proposals, key=lambda p: math.dist(p["center"], truth),
                              default=None)
                row[mode] = {
                    "count": len(proposals),
                    "nearest_distance_px": math.dist(nearest["center"], truth) if nearest else None,
                    "nearest": nearest,
                    "appearance": appearance_evidence(seed, current, nearest["center"])
                    if nearest else None,
                }
            row["selected"] = {mode: {"error_px": math.dist(path[frame]["center"], truth),
                                      "observed": path[frame]["observed"],
                                      "center": path[frame]["center"]}
                               for mode, path in paths.items() if frame in path}
            rows.append(row)
        previous = current
    cap.release()
    destination = output / "identity-pilot" / "contact-regression-audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"clip": CLIP, "rows": rows}, indent=2))
    for row in rows:
        print(row["frame"],
              'positive', round(row["positive"]["nearest_distance_px"])
              if row["positive"]["nearest_distance_px"] is not None else 'none',
              'negative', round(row["negative"]["nearest_distance_px"])
              if row["negative"]["nearest_distance_px"] is not None else 'none',
              'selected', {key: round(value["error_px"])
                           for key, value in row["selected"].items()}, flush=True)
    print(destination)


if __name__ == "__main__":
    main()
