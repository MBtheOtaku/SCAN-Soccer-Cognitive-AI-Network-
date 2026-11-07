# Next step: independent contact proposals

Your reviewed reference intervals are saved in `tests/fixtures/shot_contact_references.json`:

| Clip | Approximate contact frame | First movement frame | Previous estimate |
| --- | ---: | ---: | ---: |
| 13a1022c5009 | 79 | 80 | 60 |
| e9935750581e | 117 | 118 | 97 |
| 96b382d82d92 | 136 | 137 | 38 |

These are user-reviewed approximate labels, not instrumented contact measurements. Keep them out of candidate-generation logic.

The first saved ball trajectory is inconsistent with the visible ball at frame 79: the stored centre is around (639, 941), while the visible ball is near the athlete's feet around (270, 765) in the 720×1280 original. Candidate generation must not trust that track merely because its detection rate is high.

This first offline pass finds **possible contacts**, not confirmed contacts or shots. It uses independent full-frame YOLO observations and existing pose coordinates. No old strike time is supplied, no predicted/optical points are used, and no reference label is used to select candidates. Missing proposals expose detection or pose limitations rather than triggering invented contact times.

## 1. Create `scripts/propose_contacts.py`

Paste this code into the new file. Existing backend files stay unchanged. All thresholds are provisional inspection settings, not validated probabilities or biomechanics criteria.

```python
import argparse
import csv
import html
import json
import math
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.backend.ball_tracking import _run_yolo, _detections_from_result


ROOT = Path(__file__).resolve().parents[1]


def proximity(pose, ball):
    """Return nearest ankle and image-distance / hip-to-ankle scale."""
    if float(pose["pose_confidence"]) < 0.50:
        return None
    hip = float(pose["hip_mid_x"]), float(pose["hip_mid_y"])
    candidates = []
    for side in ("left", "right"):
        ankle = float(pose[f"{side}_ankle_x"]), float(pose[f"{side}_ankle_y"])
        scale = math.dist(hip, ankle)
        distance = math.dist(ankle, (ball.center_x, ball.center_y))
        if math.isfinite(scale) and math.isfinite(distance) and scale >= 15:
            candidates.append((distance / scale, side))
    return min(candidates) if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_id")
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    args = parser.parse_args()
    if len(args.analysis_id) != 12 or any(c not in "0123456789abcdef" for c in args.analysis_id):
        parser.error("Use a 12-character SCAN analysis ID.")
    if not args.weights.is_file():
        parser.error("Weights not found. Pass --weights with your existing YOLO weights path.")

    folder = ROOT / "data" / "output" / args.analysis_id
    with (folder / "scan_pose_metrics_v2.csv").open(newline="") as handle:
        poses = {int(row["frame_idx"]): row for row in csv.DictReader(handle)}
    originals = list((ROOT / "data" / "input").glob(args.analysis_id + ".*"))
    if len(originals) != 1:
        raise RuntimeError("Expected exactly one original recording.")
    out = ROOT / "data" / "reference-review" / "contact-candidates" / args.analysis_id
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    cap = cv2.VideoCapture(str(originals[0]))
    if not cap.isOpened():
        raise RuntimeError("Cannot open original recording.")
    observations, proposals = [], []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Invalid FPS; investigate source timing first.")
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            # Raw pixels, independent detections: no strike-conditioned tracker.
            result = _run_yolo(model, frame, confidence=0.10, image_size=1280)
            balls = _detections_from_result(
                result, frame_idx, frame_idx / fps, width, height,
            )
            pose = poses.get(frame_idx)
            frame_proposals = []
            for ball in balls:
                observations.append({
                    "frame": frame_idx, "time_s": frame_idx / fps,
                    "x": ball.center_x, "y": ball.center_y,
                    "confidence": ball.confidence,
                    "bbox_width": ball.bbox_width, "bbox_height": ball.bbox_height,
                })
                near = proximity(pose, ball) if pose else None
                if near is None or near[0] > 0.65:
                    continue
                ratio, side = near
                frame_proposals.append({
                    "frame": frame_idx, "time_s": frame_idx / fps,
                    "nearest_ankle": side, "distance_over_leg_scale": ratio,
                    "ball_confidence": ball.confidence,
                    "ball_x": ball.center_x, "ball_y": ball.center_y,
                    "status": "unverified_proximity_candidate_not_confirmed_contact",
                })
            if frame_proposals:
                # Save every eligible frame; do not pick a shot or throw away
                # other candidate balls just because one has a higher score.
                preview = frame.copy()
                for proposal in frame_proposals:
                    centre = round(proposal["ball_x"]), round(proposal["ball_y"])
                    cv2.circle(preview, centre, 12, (0, 255, 255), 2)
                if not cv2.imwrite(str(out / f"frame-{frame_idx}.jpg"), preview):
                    raise RuntimeError("Cannot save candidate preview; check storage.")
                proposals.extend(frame_proposals)
            if frame_idx % 30 == 0:
                print(f"Processed frame {frame_idx}; proximity proposals: {len(proposals)}", flush=True)
            frame_idx += 1
    finally:
        cap.release()

    payload = {
        "analysis_id": args.analysis_id,
        "method": "independent_raw_yolo_plus_ankle_proximity",
        "fps": fps, "processed_frames": frame_idx,
        "limitations": [
            "Not a shot classifier or confirmed contact detector.",
            "Ankle landmarks are not the foot surface; missing proximity is not proof of no contact.",
            "Camera motion and perspective affect image distances.",
            "Raw YOLO detections may be false positives; confidence is not calibrated accuracy.",
            "Pose identity and source presentation timestamps are not verified by this script.",
        ],
        "observations": observations, "proposals": proposals,
    }
    (out / "candidates.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    groups = {}
    for proposal in proposals:
        groups.setdefault(proposal["frame"], []).append(proposal)
    cards = []
    for number, group in groups.items():
        details = "; ".join(
            f'{p["nearest_ankle"]} ankle distance/scale {p["distance_over_leg_scale"]:.2f}, '
            f'ball score {p["ball_confidence"]:.2f}' for p in group
        )
        cards.append(f'<figure><img src="frame-{number}.jpg"><figcaption>Frame {number}; '
                     f'{number / fps:.3f}s<br>{html.escape(details)}</figcaption></figure>')
    page = '''<!doctype html><meta charset="utf-8"><title>Contact proposals</title>
    <style>body{background:#09110c;color:#eef7f0;font:16px system-ui;padding:24px}
    main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
    figure{margin:0}img{width:100%}</style><h1>Possible foot–ball contacts</h1>
    <p>Yellow circles mark raw detector proposals near an ankle. These are not
    confirmed contacts or shots. Inspect the actual ball and neighbouring frames.</p>'''
    page += '<main>' + ''.join(cards) + '</main>' if cards else '<p>No proximity candidates. Inspect raw observations and pose evidence.</p>'
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"Review: {out / 'index.html'}")


if __name__ == "__main__":
    main()
```

## 2. Start with one clip

Run in SCAN's root, using the same Python environment as the backend. Module invocation (`-m`) is intentional so imports resolve from the repository root.

```powershell
python -m scripts.propose_contacts 13a1022c5009
Start-Process "data/reference-review/contact-candidates/13a1022c5009/index.html"
```

This reruns YOLO over the clip, so allow it to finish. It uses your existing local weights; it does not need API keys, Expo, or a backend restart. If the default weights file is elsewhere, supply its actual path with `--weights`.

Check whether proposals include frames 79–80 and whether the yellow circle is on the actual ball. Also inspect earlier proposals to see whether they are run-up false positives. If frames 79–80 are absent, inspect `candidates.json`: are there independent ball observations at those frames, and are the pose ankle coordinates plausible? Improve the missing evidence before choosing thresholds to force a match.

Then run the other two clips:

```powershell
python -m scripts.propose_contacts e9935750581e
python -m scripts.propose_contacts 96b382d82d92
```

Open their corresponding review pages. Compare against intervals 117–118 and 136–137. The third clip should produce multiple potential touches if its ball and pose observations support them; multiple candidates are useful information, not a reason to silently select the final one.

## 3. Next decision

Only after independent observations recover plausible contact candidates should we add temporal association and measured post-contact separation to rank possible shot events. A ball lifting off is useful evidence where visible, but must not be required for a ground-level shot. Missing detections are not proof of ball movement or no shot.

Keep candidate generation separate from choosing the shooting action. Evaluate contact-interval error, missed contacts and false proposals. These three clips are development examples, not a generalisation test; reserve new clips for subsequent evaluation.

Do not replace the production strike frame with this script's proposals yet. It deliberately has no automatic winning shot, no coaching output and no LLM judgement stage.
