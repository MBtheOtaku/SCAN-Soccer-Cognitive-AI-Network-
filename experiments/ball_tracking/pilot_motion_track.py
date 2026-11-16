"""Offline beam-search probe after one manually supplied contact-frame seed.

This pilot uses reviewed contact frames and hand-read goal centres. Its output
is a candidate trajectory, not verified ball identity, contact, or outcome.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .appearance import appearance_evidence, appearance_penalty, seed_template
from .reliability import mark_reliability

root = Path(__file__).resolve().parents[2]
label_root = root / "data/reference-review/ball-labels"
output = root / "data/reference-review/beam-motion-track"
output.mkdir(exist_ok=True)
# Goal centres were visually read from the contact frame, independently of
# the held-out ball trajectory. Production must use calibrated goal geometry.
anchors = {"13a1022c5009": (78, (260, 610)),
           "560706a458b1": (46, (275, 465)),
           "6292a1b2b194": (50, (275, 465)),
           "e9935750581e": (117, (300, 540)),
           "96b382d82d92": (136, (430, 585)),
           "ee73dffd61b5": (120, None)}


def align_lk(previous, current):
    h, w = previous.shape[:2]
    a = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(a, maxCorners=450, qualityLevel=0.02,
                                     minDistance=14, blockSize=7)
    if points is None or len(points) < 12:
        return previous
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None,
                                                   winSize=(21, 21), maxLevel=3)
    good = status.reshape(-1).astype(bool)
    if sum(good) < 12:
        return previous
    transform, inliers = cv2.estimateAffinePartial2D(points[good], tracked[good],
                                                      method=cv2.RANSAC,
                                                      ransacReprojThreshold=2.5)
    if transform is None or inliers.sum() < 10 or max(abs(transform[0, 2]),
                                                       abs(transform[1, 2])) > 40:
        return previous
    return cv2.warpAffine(previous, transform, (w, h))


def candidates(previous, current, positive=False, negative=False):
    if positive and negative:
        raise ValueError("Choose one signed motion cue")
    prior = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    now = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    delta = (cv2.subtract(now, prior) if positive else
             cv2.subtract(prior, now) if negative else cv2.absdiff(prior, now))
    mask = cv2.morphologyEx((delta >= 15).astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    found = []
    for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if not (10 <= area <= 2000 and 3 <= w <= 100 and 3 <= h <= 100):
            continue
        cx, cy = centers[index]
        patch = gray[max(0, round(cy)-3):min(gray.shape[0], round(cy)+4),
                     max(0, round(cx)-3):min(gray.shape[1], round(cx)+4)]
        found.append({"center": (float(cx), float(cy)), "area": int(area),
                      "brightness": float(np.mean(patch)),
                      "box": [int(x), int(y), int(x+w), int(y+h)]})
    return found


def track(clip_id, anchor, goal, length=50, positive=False, negative=False,
          launch_progress_px_per_frame=0, launch_soft_weight=0,
          seed_box=None, source_name=None, seed_kind=None,
          both_signed=False, appearance_weight=0,
          appearance_max_hist_distance=None, appearance_gate_frames=0):
    if both_signed and (positive or negative):
        raise ValueError("Combined signed motion replaces positive/negative selection")
    if appearance_weight < 0:
        raise ValueError("Appearance weight must be nonnegative")
    if appearance_max_hist_distance is not None and appearance_weight == 0:
        raise ValueError("Appearance gate requires a positive appearance weight")
    if appearance_max_hist_distance is not None and not 0 <= appearance_max_hist_distance <= 1:
        raise ValueError("Appearance histogram distance must be between 0 and 1")
    if appearance_gate_frames < 0:
        raise ValueError("Appearance gate duration must be nonnegative")
    if seed_box is None:
        folder = label_root / clip_id
        manifest = json.loads((folder / "manifest.json").read_text())
        labels = {a["frame"]: a for a in json.loads(
            (folder / "labels.json").read_text())["annotations"]}
        start_box = labels[anchor]["box"]
        source_name = manifest["source_name"]
    else:
        if source_name is None:
            source_name = f"{clip_id}.mov"
        labels = {}
        start_box = seed_box
    seed = ((start_box[0]+start_box[2])/2, (start_box[1]+start_box[3])/2)
    cap = cv2.VideoCapture(str(root / "data/input" / source_name))
    cap.set(cv2.CAP_PROP_POS_FRAMES, anchor)
    ok, previous = cap.read()
    if not ok:
        raise ValueError(clip_id)
    template = seed_template(previous, start_box) if appearance_weight else None
    beams = [{"cost": 0, "previous": None, "center": seed, "missing": 0, "path": []}]
    all_candidates = []
    for frame in range(anchor+1, anchor+length+1):
        ok, current = cap.read()
        if not ok:
            break
        aligned = align_lk(previous, current)
        if both_signed:
            proposals = candidates(aligned, current, positive=True)
            for proposal in proposals:
                proposal["polarity"] = "positive"
            darkening = candidates(aligned, current, negative=True)
            for proposal in darkening:
                proposal["polarity"] = "negative"
            proposals.extend(darkening)
        else:
            proposals = candidates(aligned, current, positive=positive, negative=negative)
        if template is not None:
            for proposal in proposals:
                proposal["appearance"] = appearance_evidence(
                    template, current, proposal["center"])
        all_candidates.append({"frame": frame, "count": len(proposals)})
        next_beams = []
        for beam in beams:
            center = beam["center"]
            velocity = (0, 0) if beam["previous"] is None else (
                center[0]-beam["previous"][0], center[1]-beam["previous"][1])
            predicted = (center[0]+velocity[0], center[1]+velocity[1])
            for proposal in proposals:
                if (appearance_max_hist_distance is not None and
                        frame - anchor <= appearance_gate_frames and
                        proposal["appearance"]["hist_distance"] > appearance_max_hist_distance):
                    continue
                target = proposal["center"]
                # Only constrain the short launch phase. This is a directional
                # prior from goal geometry, not evidence of entering the goal.
                goal_distance = math.dist(goal, target) if goal else None
                seed_goal_distance = math.dist(goal, seed) if goal else None
                if goal and goal_distance > seed_goal_distance + 12:
                    continue
                # A launch candidate must separate from the contact location.
                # This is a short, optional experiment, not verified ball motion.
                launch_shortfall = (max(0, launch_progress_px_per_frame*(frame-anchor) -
                                        (seed_goal_distance-goal_distance))
                                    if goal and frame-anchor <= 6 else 0)
                if launch_shortfall > 0 and launch_soft_weight == 0:
                    continue
                if goal and frame-anchor <= 30 and goal_distance > math.dist(goal, center) + 15:
                    continue
                displacement = math.dist(center, target)
                residual = math.dist(predicted, target)
                if displacement > 110 or residual > (85 if beam["previous"] is None or beam["missing"] else 65):
                    continue
                if beam["previous"] is None and displacement < 10:
                    continue
                # Low cost = smooth trajectory, moving bright object, moderate component area.
                cost = (beam["cost"] + residual/20 +
                        max(0, 90-proposal["brightness"])/70 +
                        max(0, math.log(max(1, proposal["area"])/500))/3)
                if template is not None:
                    cost += appearance_weight * appearance_penalty(proposal["appearance"])
                if beam["previous"] is None:
                    cost += abs(displacement-30)/35
                cost += launch_soft_weight*launch_shortfall
                if goal:
                    cost += max(0, goal_distance-math.dist(goal, center))/8
                    if frame-anchor <= 10:
                        cost -= 0.06*min(50, max(0, math.dist(goal, center)-goal_distance))
                next_beams.append({"cost": cost, "previous": center, "center": target,
                                   "missing": 0, "path": beam["path"] +
                                   [{"frame": frame, "center": target, "observed": True,
                                     "candidate": proposal}]})
            if beam["missing"] < 2:
                next_beams.append({"cost": beam["cost"]+3.5, "previous": center,
                                   "center": predicted if beam["previous"] is not None else center,
                                   "missing": beam["missing"]+1,
                                   "path": beam["path"] +
                                   [{"frame": frame, "center": predicted, "observed": False}]})
        if not next_beams:
            break
        distinct = {}
        for beam in sorted(next_beams, key=lambda b: b["cost"]):
            key = (round(beam["center"][0]/8), round(beam["center"][1]/8),
                   round((beam["center"][0]-beam["previous"][0])/8) if beam["previous"] else 0,
                   round((beam["center"][1]-beam["previous"][1])/8) if beam["previous"] else 0)
            distinct.setdefault(key, beam)
            if len(distinct) == 100:
                break
        beams = list(distinct.values())
        leader = min(beams, key=lambda beam: beam["cost"])
        competitors = [beam for beam in beams if math.dist(beam["center"],
                                                          leader["center"]) > 25]
        all_candidates[-1]["instantaneous_margin"] = (
            min(beam["cost"] for beam in competitors)-leader["cost"]
            if competitors else None)
        previous = current
    cap.release()
    best = min(beams, key=lambda b: b["cost"]) if beams else {"path": [], "cost": None}
    rows = []
    for item in best["path"]:
        label = labels.get(item["frame"])
        truth = label["box"] if label and label["state"] == "visible" else None
        if truth:
            correct = ((truth[0]+truth[2])/2, (truth[1]+truth[3])/2)
            item["error_px"] = math.dist(correct, item["center"])
        rows.append(item)
    scored = [r for r in rows if "error_px" in r]
    reliability = mark_reliability(rows, anchor, goal)
    return {"clip": clip_id, "anchor": anchor, "goal_center": goal,
            "source_name": source_name,
            "seed_kind": seed_kind or ("in_flight" if goal is None else "contact_frame"),
            "mode": "both_signed" if both_signed else "positive" if positive else "negative" if negative else "absolute",
            "appearance_weight": appearance_weight,
            "appearance_max_hist_distance": appearance_max_hist_distance,
            "appearance_gate_frames": appearance_gate_frames,
            "launch_progress_px_per_frame": launch_progress_px_per_frame,
            "launch_soft_weight": launch_soft_weight,
            "start_box": start_box,
            "frames_attempted": length, "frames_tracked": len(rows),
            "evaluated": len(scored), "within_40px": sum(r["error_px"] <= 40 for r in scored),
            "within_20px": sum(r["error_px"] <= 20 for r in scored),
            "within_10px": sum(r["error_px"] <= 10 for r in scored),
            "observed": sum(r["observed"] for r in rows),
            "status": ("lost" if len(rows) < length else
                       "uncertain" if reliability["status"] == "uncertain" else
                       "candidate_trajectory_unverified"),
            "candidate_counts": all_candidates, "cost": best["cost"],
            "reliability": reliability, "path": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-progress", type=float, default=0)
    parser.add_argument("--launch-soft-weight", type=float, default=0)
    parser.add_argument("--include-negative", action="store_true")
    parser.add_argument("--report-name", default="report.json")
    args = parser.parse_args()
    if (args.launch_progress < 0 or args.launch_soft_weight < 0 or
            Path(args.report_name).name != args.report_name):
        parser.error("launch settings must be nonnegative and report name must be a filename")
    modes = [(False, False), (True, False)]
    if args.include_negative:
        modes.append((False, True))
    results = [track(clip, anchor, goal, positive=positive, negative=negative,
                     launch_progress_px_per_frame=args.launch_progress,
                     launch_soft_weight=args.launch_soft_weight)
               for clip, (anchor, goal) in anchors.items() for positive, negative in modes]
    (output / args.report_name).write_text(json.dumps({
        "status": "offline_pilot_not_production",
        "notes": ["The contact or in-flight seed is a reviewed manual label.",
                  "Goal centres, where used, were visually read from the contact frame.",
                  "A close centre does not prove ball identity or goal entry.",
                  "These clips informed pilot selection; no untouched test set yet.",
                  "The positive/absolute motion cue is not automatically selected."],
        "results": results}, indent=2))
    for result in results:
        print(result["clip"], result["mode"],
              f"<=10px {result['within_10px']}/{result['evaluated']}",
              f"<=40px {result['within_40px']}/{result['evaluated']}",
              "observed", result["observed"], "status", result["status"], flush=True)


if __name__ == "__main__":
    main()
