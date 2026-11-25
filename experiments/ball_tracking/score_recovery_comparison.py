"""Paired development scoring and miss auditing of frozen tracker reports.

Reference labels are read only after inference. Predictions never count as
observations. Uncertain/hidden frames are excluded, not treated as negatives.
"""
import argparse
from collections import Counter
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


def audit_misses(manifest, labels, report, *, auxiliary_search=None):
    """Explain visible misses from recorded checks, never issue model queries.

    An auxiliary search is evaluation evidence only. A matching proposal in
    that report did not necessarily pass the primary policy's acceptance gates.
    Missing boxes in a logged crop do not establish universal detector failure.
    """
    scored = score_report(manifest,labels,report)
    sources = [('primary',report)]
    if auxiliary_search is not None:
        compare(manifest,labels,report,auxiliary_search)
        sources.append(('auxiliary',auxiliary_search))
    reference = {a['frame']:a for a in labels['annotations']}
    sx = report['frame_transform']['source_width']/report['width']
    sy = report['frame_transform']['source_height']/report['height']
    rows = []
    for score in scored['rows']:
        if score['reference_state'] != 'visible' or score['correct_observation']:
            continue
        frame,box = score['frame'],reference[score['frame']]['box']
        def hit(point):
            if (not isinstance(point,(list,tuple)) or len(point) != 2
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in point)):
                raise ValueError('Invalid recorded audit point')
            return box[0]-3*sx <= point[0] <= box[2]+3*sx and box[1]-3*sy <= point[1] <= box[3]+3*sy
        checks = []
        for source_name,source in sources:
            for index,window in enumerate(source.get('learned_recovery',{}).get('windows',[])):
                if window.get('status') != 'completed' or frame < window['seed_frame']:
                    continue
                for check in window.get('checks',[]):
                    if check['frame'] != frame:
                        continue
                    if type(check['model_visible']) is not bool:
                        raise ValueError('Invalid recorded model visibility')
                    matches = [p for p in check['native_proposals'] if hit(p['center_source'])]
                    crop = window.get('crop_xyxy')
                    reference_center = [(box[0]+box[2])/2,(box[1]+box[3])/2]
                    if crop is not None and (len(crop) != 4 or any(type(v) not in (int,float)
                            or not math.isfinite(v) for v in crop) or crop[0]>=crop[2] or crop[1]>=crop[3]):
                        raise ValueError('Invalid recorded native crop')
                    checks.append({'source':source_name,'window_index':index,
                        'status':check['status'],'model_visible':check['model_visible'],
                        'learned_point_matches_reference':hit(check['learned_estimate_source']),
                        'matching_detector_proposals':len(matches),
                        'detector_search_ran':check.get('native_crop_count',0)>0,
                        'reference_center_in_supplied_crop':(crop[0]<=reference_center[0]<crop[2]
                            and crop[1]<=reference_center[1]<crop[3]) if crop is not None else None,
                        'support_observation_frames':check['support_observation_frames']})
        if any(c['matching_detector_proposals'] for c in checks):
            cause = 'matching_detector_box_withheld_by_confirmation'
        elif any(c['learned_point_matches_reference'] and c['detector_search_ran'] for c in checks):
            cause = 'no_matching_detector_box_in_logged_search'
        elif any(c['learned_point_matches_reference'] for c in checks):
            cause = 'matching_learned_point_withheld_before_search'
        elif checks and all(c['reference_center_in_supplied_crop'] is False for c in checks):
            cause = 'reference_outside_supplied_learned_crop'
        elif checks:
            cause = 'learned_query_missed_reference'
        else:
            cause = 'no_recorded_recovery_window'
        rows.append({'frame':frame,'category':cause,'checks':checks})
    return {'status':'evaluation_only_no_inference_queries','padding_working_px':3,
            'auxiliary_search_used':auxiliary_search is not None,
            'missed_visible_frames':len(rows),'categories':dict(Counter(r['category'] for r in rows)),
            'rows':rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "labels", "baseline", "candidate", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument('--audit-misses',action='store_true',help='Classify visible misses from recorded recovery checks')
    parser.add_argument('--auxiliary-search-report',type=Path,
        help='Optional separately frozen search report for evaluation-only miss auditing')
    args = parser.parse_args()
    if args.auxiliary_search_report and not args.audit_misses:
        parser.error('--auxiliary-search-report requires --audit-misses')
    if args.output.exists():
        parser.error("Preserve earlier scores; choose a new output path")
    read = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
    result = compare(*(read(getattr(args, key)) for key in ("manifest", "labels", "baseline", "candidate")))
    result["input_sha256"] = {key: file_sha256(getattr(args, key))
                              for key in ("manifest", "labels", "baseline", "candidate")}
    if args.audit_misses:
        result['miss_audit'] = audit_misses(read(args.manifest),read(args.labels),read(args.candidate),
            auxiliary_search=read(args.auxiliary_search_report) if args.auxiliary_search_report else None)
        if args.auxiliary_search_report:
            result['input_sha256']['auxiliary_search_report'] = file_sha256(args.auxiliary_search_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for key in ("baseline", "candidate"):
        s = result[key]
        print(f'{key}: {s["correct_observations"]}/{s["visible_frames"]} visible-frame hits; '
              f'{s["false_acceptances"]} false acceptances; longest gap {s["longest_visible_gap_frames"]} frames')


if __name__ == "__main__":
    main()
