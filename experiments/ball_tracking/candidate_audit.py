"""Audit raw motion proposals against independently marked original frames."""

import json
import math

import cv2

from .holdout_shot_check import CASES
from .pilot_motion_track import align_lk, candidates, output, root


def audit(clip, case):
    source = root / "data/input" / f"{clip}.mov"
    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_FRAMES, case["anchor"])
    ok, previous = capture.read()
    if not ok:
        raise ValueError(source)
    rows = []
    for frame in range(case["anchor"] + 1, max(case["centers"]) + 1):
        ok, current = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode {clip} frame {frame}")
        if frame in case["centers"]:
            aligned = align_lk(previous, current)
            truth = case["centers"][frame]
            row = {"frame": frame, "truth": truth}
            for mode in ("positive", "absolute", "negative"):
                proposals = candidates(aligned, current,
                                       positive=mode == "positive",
                                       negative=mode == "negative")
                nearest = min(proposals, key=lambda candidate:
                              math.dist(candidate["center"], truth), default=None)
                row[mode] = {"count": len(proposals),
                             "nearest_px": math.dist(nearest["center"], truth)
                             if nearest else None,
                             "nearest": nearest}
            rows.append(row)
        previous = current
    capture.release()
    return rows


def main():
    results = {clip: audit(clip, case) for clip, case in CASES.items()}
    dest = output / "holdout-additional-clips" / "candidate-audit.json"
    dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for clip, rows in results.items():
        print(clip)
        for mode in ("positive", "absolute", "negative"):
            distances = [row[mode]["nearest_px"] for row in rows]
            print(mode, f'within20 {sum(d is not None and d <= 20 for d in distances)}/{len(rows)}',
                  [None if d is None else round(d, 1) for d in distances])
    print(dest)


if __name__ == "__main__":
    main()
