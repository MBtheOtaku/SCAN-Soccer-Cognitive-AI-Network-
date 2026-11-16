"""Offline human-seeded ball tracking. Evaluation labels are read AFTER inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re

import cv2

from scripts.ball_dataset import validate_labels
from scripts.manual_ball_tracker import ManualBallTracker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/reference-review/manual-ball"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def source_path(clip):
    if not re.fullmatch(r"[0-9a-f]{12}", clip):
        raise ValueError("Use a 12-character analysis ID.")
    paths = list((ROOT / "data/input").glob(clip + ".*"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one original video.")
    return paths[0]


def prepare(args):
    source = source_path(args.clip)
    cap = cv2.VideoCapture(str(source))
    frame, fps = None, float(cap.get(cv2.CAP_PROP_FPS))
    try:
        for _ in range(args.frame+1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Seed frame could not be decoded.")
    finally:
        cap.release()
    out = OUT / args.clip / ("seed-" + str(args.frame))
    out.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(out / "frame.jpg"), frame):
        raise OSError("Seed image could not be written.")
    h, w = frame.shape[:2]
    data = {"schema": 1, "clip_id": args.clip, "source_sha256": sha(source),
            "frame": args.frame, "width": w, "height": h, "box": None}
    template = (ROOT / "scripts/manual_ball_seed.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(template.replace("__SEED__", json.dumps(data)), encoding="utf-8")
    print("Select the ball here: " + str(out / "index.html"))


def validate_seed(seed, source, clip):
    if seed.get("schema") != 1 or seed.get("clip_id") != clip or seed.get("source_sha256") != sha(source):
        raise ValueError("Seed does not match this source video.")
    if type(seed.get("frame")) is not int or seed["frame"] < 0:
        raise ValueError("Seed frame must be a nonnegative integer.")
    if not isinstance(seed.get("box"), list) or len(seed["box"]) != 4:
        raise ValueError("Draw and confirm a ball box before downloading the seed.")


def evaluate(rows, manifest, labels, seed_frame):
    """Post-hoc sparse reviewed-frame check; no labels are returned to inference."""
    reviewed = validate_labels(manifest, labels)
    by_frame = {row["frame"]: row for row in rows if row["frame"] > seed_frame}
    counts = Counter(visible=0, tracked_visible=0, center_inside_label=0,
                     absent=0, accepted_on_absent=0)
    details = []
    for frame, truth in reviewed:
        row = by_frame.get(frame["frame"])
        if row is None:
            continue
        accepted = row["status"] == "tracking" and row["box"] is not None
        detail = {"frame": frame["frame"], "status": row["status"], "truth": truth}
        if truth is None:
            counts["absent"] += 1
            counts["accepted_on_absent"] += int(accepted)
        else:
            counts["visible"] += 1
            counts["tracked_visible"] += int(accepted)
            if accepted:
                cx, cy = ManualBallTracker.center(row["box"])
                inside = truth[0] <= cx <= truth[2] and truth[1] <= cy <= truth[3]
                counts["center_inside_label"] += int(inside)
                detail["center_error_px"] = math.dist((cx, cy), ManualBallTracker.center(truth))
                detail["center_inside_label"] = inside
        details.append(detail)
    return {"counts": dict(counts), "reviewed_frames": details,
            "limitations": "Sparse user labels, seed excluded; centre containment is not ball identity or contact validation."}


def run(args):
    source = source_path(args.clip)
    seed = json.loads(args.seed.read_text(encoding="utf-8"))
    validate_seed(seed, source, args.clip)
    out = OUT / args.clip / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (out / "frames").mkdir(parents=True)
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Invalid source FPS.")
    rows, previews, tracker, index = [], [], None, 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if index < seed["frame"]:
                index += 1
                continue
            if args.max_frames and len(rows) >= args.max_frames:
                break
            if tracker is None:
                h, w = image.shape[:2]
                if (w, h) != (seed["width"], seed["height"]):
                    raise ValueError("Seed dimensions differ from decoded video.")
                tracker = ManualBallTracker(image, seed["box"], index,
                                            min_score=args.min_score, margin=args.margin)
                result = {"status": "seeded", "box": tracker.box.copy(),
                          "reason": "Human selection only; not yet a stable track.",
                          "match_score": None, "alternative_score": None}
            else:
                result = tracker.update(image, index)
            row = {"frame": index, "time_s": index/fps, **result}
            rows.append(row)
            box = row["box"]
            if box:
                colour = (80, 255, 80) if row["status"] == "tracking" else (0, 200, 255)
                cv2.rectangle(image, tuple(round(v) for v in box[:2]), tuple(round(v) for v in box[2:]), colour, 2)
            cv2.putText(image, f'{index}: {row["status"]}', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            name = f"frames/{index:06d}.jpg"
            if not cv2.imwrite(str(out / name), image):
                raise OSError("Preview write failed.")
            previews.append({"image": name, "frame": index, "status": row["status"], "reason": row["reason"]})
            index += 1
    finally:
        cap.release()
    if not rows:
        raise ValueError("No frames at or after seed decoded.")
    # Only now load the other labels; they never influence target selection.
    folder = ROOT / "data/reference-review/ball-labels" / args.clip
    evaluation = None
    if (folder / "labels.json").is_file():
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
        if manifest["source_sha256"] != seed["source_sha256"]:
            raise ValueError("Evaluation labels belong to a different recording.")
        evaluation = evaluate(rows, manifest, labels, seed["frame"])
    summary = {"seed": seed, "method": "fixed_seed_multiscale_template_local_search",
               "min_score": args.min_score, "ambiguity_margin": args.margin,
               "confirm_frames_including_seed": tracker.confirm_frames,
               "search_radius_px": tracker.radius,
               "status_counts": dict(Counter(r["status"] for r in rows)), "evaluation": evaluation,
               "opencv": cv2.__version__, "first_stop_frame": next((r["frame"] for r in rows
                   if r["status"] in {"lost", "ambiguous"}), None),
               "limitations": "No automatic reacquisition. Similar balls can cause undetected identity switches. Scores are not probabilities."}
    write(out / "summary.json", summary)
    write(out / "trace.json", rows)
    write(out / "seed.json", seed)
    page = ('<!doctype html><meta charset="utf-8"><title>SCAN manual tracking review</title>'
            '<style>body{background:#09120c;color:#eef5ee;font:16px system-ui}img{max-height:72vh;max-width:100%}input{width:80%}pre{white-space:pre-wrap}</style>'
            '<h1>Human-seeded tracking experiment</h1><p>Green: accepted appearance match, identity unverified. Yellow: seed/calibration. '
            'No box after loss or ambiguity. Later labels were used only after tracking for evaluation.</p>'
            '<input id="slider" type="range" min="0" value="0" max="' + str(len(rows)-1) + '"><p id="state"></p><img id="image">'
            '<pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre><script>const frames=' + json.dumps(previews) + ';'
            'const slider=document.getElementById("slider");function show(){const f=frames[Number(slider.value)];'
            'document.getElementById("image").src=f.image;document.getElementById("state").textContent=`Frame ${f.frame}: ${f.status}. ${f.reason}`;}'
            'slider.oninput=show;document.onkeydown=e=>{if(e.key==="ArrowRight")slider.value=Number(slider.value)+1;'
            'if(e.key==="ArrowLeft")slider.value=Number(slider.value)-1;show();};show();</script>')
    (out / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"status_counts": summary["status_counts"], "first_stop_frame": summary["first_stop_frame"],
                      "evaluation": evaluation["counts"] if evaluation else None}, indent=2))
    print("Review: " + str(out / "index.html"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("clip")
    prep.add_argument("--frame", type=int, default=0)
    runner = sub.add_parser("run")
    runner.add_argument("clip")
    runner.add_argument("--seed", type=Path, required=True)
    runner.add_argument("--max-frames", type=int, default=0)
    runner.add_argument("--min-score", type=float, default=0.65)
    runner.add_argument("--margin", type=float, default=0.08)
    args = parser.parse_args()
    if args.command == "prepare" and args.frame < 0 or args.command == "run" and args.max_frames < 0:
        parser.error("Frame counts must be nonnegative.")
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
