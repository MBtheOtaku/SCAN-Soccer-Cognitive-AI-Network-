"""Paired development scoring of causal tracker reports against human boxes.

Reference labels are read only after inference. Predictions never count as
observations. Uncertain/hidden frames are excluded, not treated as negatives.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import median

from scripts.ball_dataset import file_sha256, validate_labels
from experiments.ball_tracking.score_stationary_shot import consecutive_miss_runs


def score_report(manifest, labels, report, *, padding_working_px=3.0):
    if (type(padding_working_px) not in (int, float)
            or not math.isfinite(padding_working_px) or padding_working_px < 0):
        raise ValueError("Expected finite nonnegative working-pixel padding")
    validate_labels(manifest, labels)
    if report.get("source_sha256") != manifest["source_sha256"]:
        raise ValueError("Report and labels have different source hashes")
    expected = {f["frame"]: f for f in manifest["frames"]}
    if len(expected) != len(manifest["frames"]) or not expected:
        raise ValueError("Expected unique reference frames")
    reviewed = {r["frame"]: r for r in labels["annotations"]}
    if set(reviewed) != set(expected) or any(r["state"] == "unreviewed" for r in reviewed.values()):
        raise ValueError("Review every reference frame, including uncertain and absent states")
    path = report["path"]
    by_frame = {r["frame"]: r for r in path}
    if len(path) != len(by_frame) or not set(expected) <= set(by_frame):
        raise ValueError("Report must cover every reference frame without duplicates")
    width, height = report["width"], report["height"]
    native = report["frame_transform"]
    for dim in (width, height, native["source_width"], native["source_height"]):
        if type(dim) is not int or dim < 1:
            raise ValueError("Expected positive integer image dimensions")
    sx, sy = native["source_width"] / width, native["source_height"] / height
    rows = []
    for frame, reference in sorted(expected.items()):
        if (reference["width"], reference["height"]) != (native["source_width"], native["source_height"]):
            raise ValueError("Reference and report coordinate systems disagree")
        label, point = reviewed[frame], by_frame[frame]
        if type(point["detected"]) is not bool:
            raise ValueError("Expected boolean detected state")
        center = point["center"] if point["detected"] else None
        if point["detected"] and (not isinstance(center, (tuple, list)) or len(center) != 2
                or not all(type(v) in (int, float) and math.isfinite(v) for v in center)
                or not 0 <= center[0] < width or not 0 <= center[1] < height):
            raise ValueError("Invalid observed center")
        source_point = [center[0] * sx, center[1] * sy] if center is not None else None
        state = label["state"]
        hit = strict = False
        error = None
        if state == "visible" and source_point is not None:
            x1, y1, x2, y2 = label["box"]
            x, y = source_point
            strict = x1 <= x <= x2 and y1 <= y <= y2
            hit = (x1 - padding_working_px * sx <= x <= x2 + padding_working_px * sx
                   and y1 - padding_working_px * sy <= y <= y2 + padding_working_px * sy)
            error = math.hypot((x - (x1 + x2) / 2) / sx, (y - (y1 + y2) / 2) / sy)
        rows.append({"frame": frame, "reference_state": state,
                     "observed_source_center": source_point, "correct_observation": hit,
                     "strict_box_hit": strict, "center_error_working_px": error,
                     "false_acceptance": source_point is not None and
                         (state == "absent" or state == "visible" and not hit)})
    visible = [r for r in rows if r["reference_state"] == "visible"]
    absent = [r for r in rows if r["reference_state"] == "absent"]
    correct = sum(r["correct_observation"] for r in visible)
    wrong = sum(r["false_acceptance"] for r in visible + absent)
    errors = [r["center_error_working_px"] for r in visible if r["center_error_working_px"] is not None]
    runs = consecutive_miss_runs(rows, lambda r: r["reference_state"] == "visible" and not r["correct_observation"])
    return {"visible_frames": len(visible), "absent_frames": len(absent),
            "uncertain_excluded": sum(r["reference_state"] == "uncertain" for r in rows),
            "correct_observations": correct, "strict_box_hits": sum(r["strict_box_hit"] for r in rows),
            "missed_visible_frames": len(visible) - correct, "false_acceptances": wrong,
            "false_acceptances_absent": sum(r["false_acceptance"] for r in absent),
            "visible_observation_recall": correct / len(visible) if visible else None,
            "selected_observation_precision": correct / (correct + wrong) if correct + wrong else None,
            "median_selected_center_error_working_px": median(errors) if errors else None,
            "longest_visible_gap_frames": max((r["length_frames"] for r in runs), default=0),
            "visible_gap_runs": runs, "rows": rows}


def compare(manifest, labels, baseline, candidate):
    if baseline["source_seed"] != candidate["source_seed"]:
        raise ValueError("Paired runs must use the same initial seed")
    if baseline["weights_sha256"] != candidate["weights_sha256"]:
        raise ValueError("Paired runs must use the same detector weights")
    for key in ("start_frame", "width", "height", "frame_transform"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError("Paired runs must use the same start frame and image geometry")
    a, b = (score_report(manifest, labels, r) for r in (baseline, candidate))
    return {"status": "labelled_development_comparison_not_held_out_validation",
            "protocol": {"padding_working_px": 3, "predictions_count_as_observations": False,
                         "uncertain_frames": "excluded", "labels_used_for_inference": False},
            "source_sha256": manifest["source_sha256"], "baseline": a, "candidate": b,
            "improved_frames": [y["frame"] for x, y in zip(a["rows"], b["rows"])
                                if y["correct_observation"] and not x["correct_observation"]],
            "regressed_frames": [x["frame"] for x, y in zip(a["rows"], b["rows"])
                                 if x["correct_observation"] and not y["correct_observation"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "labels", "baseline", "candidate", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier scores; choose a new output path")
    read = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
    result = compare(*(read(getattr(args, key)) for key in ("manifest", "labels", "baseline", "candidate")))
    result["input_sha256"] = {key: file_sha256(getattr(args, key))
                              for key in ("manifest", "labels", "baseline", "candidate")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for key in ("baseline", "candidate"):
        s = result[key]
        print(f'{key}: {s["correct_observations"]}/{s["visible_frames"]} visible-frame hits; '
              f'{s["false_acceptances"]} false acceptances; longest gap {s["longest_visible_gap_frames"]} frames')


if __name__ == "__main__":
    main()
