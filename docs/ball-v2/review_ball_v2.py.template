"""Offline-only preview: does not replace SCAN's tracker or call external APIs."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import time

import cv2
from scripts.ball_detector_v2 import BallDetector
from scripts.ball_temporal_v2 import TemporalBall

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_id")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--mode", choices=("single", "tiled"), default="tiled")
    parser.add_argument("--ball-class-id", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--name", default="experiment")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{12}", args.analysis_id):
        parser.error("Use a 12-character SCAN analysis ID.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", args.name) or args.max_frames < 0:
        parser.error("Use a short plain run name and nonnegative --max-frames.")
    if not args.weights.is_file():
        parser.error("Weights file missing: save the dedicated model before running.")
    originals = list((ROOT / "data" / "input").glob(args.analysis_id + ".*"))
    if len(originals) != 1:
        parser.error("Expected exactly one saved original for that analysis ID.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "data" / "reference-review" / "ball-v2" / args.analysis_id / (args.name + "-" + stamp)
    (out / "frames").mkdir(parents=True, exist_ok=False)
    detector = BallDetector(args.weights, args.mode, args.ball_class_id, args.device)
    tracker = TemporalBall(seed=args.seed)
    cap = cv2.VideoCapture(str(originals[0]))
    if not cap.isOpened():
        raise RuntimeError("Cannot decode original recording.")
    frames, counts = [], Counter()
    started = time.perf_counter()
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Invalid source FPS; investigate video timing.")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if args.seed and not (0 <= args.seed[0] < width and 0 <= args.seed[1] < height):
            raise ValueError("Seed must be inside the original image, not the resized browser preview.")
        with (out / "trace.jsonl").open("w", encoding="utf-8") as trace:
            index = 0
            while args.max_frames == 0 or index < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                candidates = detector.detect(frame)
                result = tracker.update(index, index / fps, candidates)
                trace.write(json.dumps({**result, "candidates": [asdict(c) for c in candidates]}, allow_nan=False) + "\n")
                counts[result["status"]] += 1
                for candidate in candidates:
                    cv2.rectangle(frame, (round(candidate.x1), round(candidate.y1)),
                                  (round(candidate.x2), round(candidate.y2)), (150, 150, 150), 1)
                observed = result["observation"]
                if observed is not None:
                    centre = (round((observed["x1"] + observed["x2"]) / 2),
                              round((observed["y1"] + observed["y2"]) / 2))
                    colour = (0, 255, 0) if result["status"] == "observed" else (0, 165, 255)
                    cv2.circle(frame, centre, 10, colour, 2)
                elif result["prediction"] is not None:
                    point = result["prediction"]
                    x, y = point["x"], point["y"]
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(frame, (round(x), round(y)), 10, (255, 180, 0), 1)
                cv2.putText(frame, f'{index}: {result["status"]} (segment {result["segment"]})',
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                if not cv2.imwrite(str(out / "frames" / f"{index:06d}.jpg"), frame):
                    raise RuntimeError("Preview write failed; check free disk space.")
                frames.append({"frame": index, "time_s": index / fps, "status": result["status"],
                               "reason": result["reason"], "segment": result["segment"]})
                if index % 30 == 0:
                    print(f'Frame {index}: {result["status"]}; candidates {len(candidates)}', flush=True)
                index += 1
    finally:
        cap.release()
    elapsed = time.perf_counter() - started
    if not frames:
        raise RuntimeError("No video frames decoded.")
    metadata = {
        "analysis_id": args.analysis_id, "weights": str(args.weights.resolve()),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "model_classes": detector.names, "ball_class_id": detector.ball_class_id,
        "mode": args.mode, "seed": args.seed, "device": args.device,
        "fps": fps, "timestamp_method": "frame_index_divided_by_fps_not_verified_pts",
        "frame_count": len(frames), "state_counts": dict(counts), "elapsed_s": elapsed,
        "limitations": ["Not connected to SCAN feedback or shot timing.",
                        "Observed means accepted detector evidence, not verified ball identity.",
                        "No camera-motion compensation; tracking may be lost after abrupt movement.",
                        "HSV appearance and image-motion thresholds are provisional.",
                        "Predictions are not observations or contact evidence."],
    }
    (out / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    title = html.escape(f"{args.analysis_id}: {args.name}, {args.mode}")
    page = '''<!doctype html><meta charset="utf-8"><title>SCAN ball v2 review</title>
    <style>body{font:16px system-ui;background:#09110c;color:#eef7f0;margin:24px}
    img{max-width:100%;max-height:78vh;display:block;margin-top:12px}
    input{width:min(90%,720px)}button{padding:10px;margin:4px}</style>'''
    page += f"<h1>{title}</h1>"
    page += '''<p>Grey boxes: raw candidates. Green: accepted detector match.
    Orange: tentative match. Blue: prediction only. Lost: no selected ball.
    These states do not establish correct identity or confirm contact.</p>
    <button id="previous">Previous frame</button><button id="next">Next frame</button>
    <input id="position" aria-label="Frame" type="range" min="0" value="0">
    <p id="caption"></p><img id="preview" alt="Tracking preview"><script>'''
    page += "const frames=" + json.dumps(frames) + ";"
    page += '''const slider=document.getElementById('position');
    slider.max=frames.length-1;
    function show(){const n=Number(slider.value),r=frames[n];
    document.getElementById('preview').src='frames/'+String(r.frame).padStart(6,'0')+'.jpg';
    document.getElementById('caption').textContent='Frame '+r.frame+' | '+r.time_s.toFixed(3)+
    's | '+r.status+' | segment '+r.segment+' | '+r.reason;}
    slider.addEventListener('input',show);
    function step(d){slider.value=Math.max(0,Math.min(frames.length-1,Number(slider.value)+d));show();}
    document.getElementById('previous').onclick=()=>step(-1);
    document.getElementById('next').onclick=()=>step(1);
    document.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){e.preventDefault();step(1);}
    if(e.key==='ArrowLeft'){e.preventDefault();step(-1);}});show();</script>'''
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"Review: {out / 'index.html'}")
    print(f"Frames: {len(frames)}; time: {elapsed:.1f}s; states: {dict(counts)}")


if __name__ == "__main__":
    main()
