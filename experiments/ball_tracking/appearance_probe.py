"""Inspect seed-to-current-frame patch similarity for true and selected candidates."""

import json
import math

import cv2
import numpy as np

from .holdout_shot_check import CASES
from .pilot_motion_track import output, root


def patch(frame, center, width, height):
    image = cv2.getRectSubPix(frame, (width, height),
                              (float(center[0]), float(center[1])))
    return image


def features(seed, current):
    seed_gray = cv2.cvtColor(seed, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    correlation = float(cv2.matchTemplate(current_gray, seed_gray,
                                          cv2.TM_CCOEFF_NORMED)[0, 0])
    a = cv2.cvtColor(seed, cv2.COLOR_BGR2HSV)
    b = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([a], [0, 1], None, [18, 8], [0, 180, 0, 256])
    hist_b = cv2.calcHist([b], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a, alpha=1, norm_type=cv2.NORM_L1)
    cv2.normalize(hist_b, hist_b, alpha=1, norm_type=cv2.NORM_L1)
    distance = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))
    return {"correlation": correlation, "hist_distance": distance,
            "brightness": float(current_gray.mean()),
            "edge_fraction": float(np.mean(cv2.Canny(current_gray, 50, 120) > 0))}


def main():
    audit = json.loads((output / "holdout-additional-clips" / "candidate-audit.json").read_text())
    report = json.loads((output / "holdout-additional-clips" / "report.json").read_text())
    for clip, case in CASES.items():
        capture = cv2.VideoCapture(str(root / "data/input" / f"{clip}.mov"))
        capture.set(cv2.CAP_PROP_POS_FRAMES, case["anchor"])
        ok, frame = capture.read()
        if not ok:
            raise ValueError(clip)
        box = case["seed_box"]
        width, height = box[2]-box[0], box[3]-box[1]
        seed = patch(frame, ((box[0]+box[2])/2, (box[1]+box[3])/2), width, height)
        chosen_path = {row["frame"]: row for result in report["results"]
                       if result["clip"] == clip and result["mode"] == "positive"
                       for row in result["path"]}
        for row in audit[clip]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
            ok, frame = capture.read()
            if not ok:
                raise ValueError((clip, row["frame"]))
            truth = row["truth"]
            chosen = chosen_path[row["frame"]]["center"]
            near = row["positive"]["nearest"]
            print(clip, row["frame"], 'error', round(math.dist(truth, chosen)),
                  'truth', features(seed, patch(frame, truth, width, height)),
                  'chosen', features(seed, patch(frame, chosen, width, height)),
                  'near', features(seed, patch(frame, near["center"], width, height))
                  if near else None)
        capture.release()


if __name__ == "__main__":
    main()
