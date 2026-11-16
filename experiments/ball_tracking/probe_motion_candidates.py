"""Offline upper-bound probe: does simple frame differencing propose the moving ball?"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parents[2]
folder = root / "data/reference-review/ball-labels"
output = root / "data/reference-review/motion-candidates"
output.mkdir(exist_ok=True)
start_frames = {"560706a458b1": 47, "6292a1b2b194": 77,
                "ee73dffd61b5": 120, "e9935750581e": 118,
                "96b382d82d92": 136}
thresholds = (15, 25, 35)


def align(previous, current):
    h, w = previous.shape[:2]
    scale = 0.25
    shape = (round(w*scale), round(h*scale))
    a = cv2.resize(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), shape).astype(np.float32)
    b = cv2.resize(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), shape).astype(np.float32)
    shift, response = cv2.phaseCorrelate(a, b)
    dx, dy = shift[0]/scale, shift[1]/scale
    if response < 0.05 or max(abs(dx), abs(dy)) > 30:
        return previous, [0, 0], response
    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(previous, transform, (w, h)), [dx, dy], response


def align_lk(previous, current):
    h, w = previous.shape[:2]
    a = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(a, maxCorners=450, qualityLevel=0.02,
                                     minDistance=14, blockSize=7)
    if points is None or len(points) < 12:
        return previous, 0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None,
                                                   winSize=(21, 21), maxLevel=3)
    good = status.reshape(-1).astype(bool)
    if sum(good) < 12:
        return previous, 0
    transform, inliers = cv2.estimateAffinePartial2D(points[good], tracked[good],
                                                      method=cv2.RANSAC,
                                                      ransacReprojThreshold=2.5)
    if transform is None or inliers.sum() < 10 or max(abs(transform[0, 2]),
                                                       abs(transform[1, 2])) > 40:
        return previous, 0
    return cv2.warpAffine(previous, transform, (w, h)), int(inliers.sum())


def candidates(previous, current, threshold, positive=False):
    prior = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    now = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    delta = cv2.subtract(now, prior) if positive else cv2.absdiff(prior, now)
    mask = (delta >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    return [{"box": [int(x), int(y), int(x+w), int(y+h)],
             "center": [float(centroids[i, 0]), float(centroids[i, 1])], "area": int(area)}
            for i, (x, y, w, h, area) in enumerate(stats[1:], 1)
            if 10 <= area <= 2000 and 3 <= w <= 100 and 3 <= h <= 100]


records = []
for clip_id, start in start_frames.items():
    manifest = json.loads((folder / clip_id / "manifest.json").read_text())
    labels = {a["frame"]: a for a in json.loads((folder / clip_id / "labels.json").read_text())["annotations"]}
    last = max(labels)
    cap = cv2.VideoCapture(str(root / "data/input" / manifest["source_name"]))
    ok, previous = cap.read()
    if not ok:
        raise ValueError(manifest["source_name"])
    for frame in range(1, last+1):
        ok, current = cap.read()
        if not ok:
            break
        label = labels.get(frame)
        if frame >= start and label and label["state"] == "visible":
            truth = label["box"]
            center = [(truth[0]+truth[2])/2, (truth[1]+truth[3])/2]
            aligned, shift, response = align(previous, current)
            lk_aligned, lk_inliers = align_lk(previous, current)
            row = {"clip": clip_id, "frame": frame, "truth": truth,
                   "shift": shift, "phase_response": response,
                   "lk_inliers": lk_inliers, "methods": {}}
            for method, prior in (("raw", previous), ("phase", aligned),
                                  ("lk", lk_aligned), ("lk_positive", lk_aligned)):
                row["methods"][method] = {}
                for threshold in thresholds:
                    proposals = candidates(prior, current, threshold,
                                           positive=method == "lk_positive")
                    nearest = min(proposals, key=lambda c: math.dist(center, c["center"])) if proposals else None
                    row["methods"][method][str(threshold)] = {
                        "count": len(proposals),
                        "nearest_px": math.dist(center, nearest["center"]) if nearest else None,
                        "nearest": nearest,
                        "within_20px": sum(math.dist(center, c["center"]) <= 20 for c in proposals),
                        "within_40px": sum(math.dist(center, c["center"]) <= 40 for c in proposals),
                    }
            records.append(row)
        previous = current
    cap.release()
    print(clip_id, sum(r["clip"] == clip_id for r in records), flush=True)
summary = {}
for clip_id in start_frames:
    subset = [r for r in records if r["clip"] == clip_id]
    summary[clip_id] = {method: {str(t): {
        "frames": len(subset),
        "candidate_within_20px": sum(r["methods"][method][str(t)]["within_20px"] > 0 for r in subset),
        "candidate_within_40px": sum(r["methods"][method][str(t)]["within_40px"] > 0 for r in subset),
        "median_candidates": sorted(r["methods"][method][str(t)]["count"] for r in subset)[len(subset)//2]
        if subset else None,
    } for t in thresholds} for method in ("raw", "phase", "lk", "lk_positive")}
(output / "report.json").write_text(json.dumps({"summary": summary, "frames": records}, indent=2))
print(json.dumps(summary, indent=2))
