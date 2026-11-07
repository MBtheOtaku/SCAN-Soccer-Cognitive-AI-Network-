# Tonight: audit shot timing before changing the detector

The current estimate is the largest hip-relative ankle-speed peak anywhere in the clip. It is also passed to the ball tracker, which calls `motion.begin_shot()` at that frame. Existing ball trajectories are therefore influenced by the old estimate; they are not independent confirmation of contact.

Both pose and ball timestamps currently use `frame_idx / fps`. This inspection does not establish that source-video presentation timestamps align with that approximation, especially for variable-frame-rate phone recordings.

First compare the exact original clips with their saved estimates. The three reference contact times below are approximate user observations, not verified frame labels.

## 1. Create `scripts/review_shot_timing.py`

Create the root `scripts` folder if it does not exist. Paste this into the new file. It uses the OpenCV dependency already used by the backend. It does not run Jev, call external services, or modify recordings or saved analyses.

```python
import html
import json
import math
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "reference-review" / "shot-timing"
CASES = [
    ("13a1022c5009", 3.00),
    ("e9935750581e", 4.23),
    ("96b382d82d92", 5.00),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sections = []
    for analysis_id, reference_s in CASES:
        summary_path = ROOT / "data" / "output" / analysis_id / "scan_shot_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        candidates = list((ROOT / "data" / "input").glob(analysis_id + ".*"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one original recording for {analysis_id}.")
        cap = cv2.VideoCapture(str(candidates[0]))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open original for {analysis_id}.")
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or total <= 0:
                raise RuntimeError(f"Invalid video timing for {analysis_id}.")
            estimate = summary.get("probable_strike_time_s")
            centres = [("Your approximate contact time", reference_s)]
            if estimate is not None:
                centres.insert(0, ("SCAN estimate", float(estimate)))
            groups = []
            for group, (label, centre) in enumerate(centres):
                cards = []
                for offset in (-0.20, -0.10, 0.0, 0.10, 0.20):
                    requested = round((centre + offset) * fps)
                    if not 0 <= requested < total:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, requested)
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Cannot decode frame {requested} in {analysis_id}.")
                    name = f"{analysis_id}-{group}-{requested}.jpg"
                    if not cv2.imwrite(str(OUT / name), frame):
                        raise RuntimeError("Could not save review image; check storage.")
                    cards.append(
                        f'<figure><img src="{name}" alt="Original video frame">'
                        f'<figcaption>Requested frame {requested}; '
                        f'frame/FPS time {requested / fps:.3f}s</figcaption></figure>'
                    )
                groups.append(f'<h3>{html.escape(label)}: {centre:.3f}s</h3>'
                              + '<div class="frames">' + ''.join(cards) + '</div>')
            sections.append(
                f'<section><h2>{analysis_id}</h2><p>Reported FPS: {fps:.6f}; '
                f'frames: {total}. Approximate reference: {reference_s:.2f}s.</p>'
                + ''.join(groups) + '</section>'
            )
        finally:
            cap.release()
    page = '''<!doctype html><meta charset="utf-8"><title>SCAN shot timing review</title>
    <style>body{font:16px system-ui;background:#09110c;color:#eef7f0;padding:24px}
    section{margin-bottom:48px}.frames{display:flex;gap:12px;overflow:auto}
    figure{margin:0;flex:0 0 350px}img{width:100%}figcaption{padding:8px 0}</style>
    <h1>Original-video shot timing review</h1>
    <p>Reference times are approximate. These requested frame positions use FPS,
    not verified presentation timestamps. Compare with the original video player.</p>'''
    target = OUT / "index.html"
    target.write_text(page + ''.join(sections), encoding="utf-8")
    print(f"Review created: {target}")


if __name__ == "__main__":
    main()
```

## 2. Generate and open the review

Run from SCAN's root in your existing backend Python environment:

```powershell
python scripts/review_shot_timing.py
Start-Process "data/reference-review/shot-timing/index.html"
```

The review output is under the existing ignored `data/reference-review/` directory. Backend and Expo restarts are unnecessary for this offline inspection.

For each clip, record:

- Does the group around your reference show foot–ball contact?
- What action appears around SCAN's estimate: a dribbling touch, running step, shot, or tracking error?
- Does the frame at `reference_s * fps` agree with the time shown in the original video player?
- Is the ball visible near contact and in the following frames?

If the expected contact appears at a different frame/FPS time, investigate video presentation timestamps before changing event selection. If times align but SCAN selected an earlier action, improve candidate selection.

## 3. Next implementation, after checking these frames

Build an offline event-candidate evaluator first, keeping the current backend behaviour while evaluating it. It should use independently obtained real ball observations, foot proximity normalised to player/ball scale, and post-candidate ball separation. Exclude predicted points as contact confirmation. Dribbling touches must remain separate candidates; do not assume the last touch is automatically a shot.

Return ambiguous/unknown when candidates cannot be distinguished. Test occlusion, multiple touches, no shot, moving cameras and missing post-contact ball observations. Only integrate a better estimator into summary generation, ball-tracking mode, goal analysis and strike windows once its errors against verified contact frames are understood.

Tonight's first checkpoint is three verified reference events and a diagnosis of the timing error, rather than a fixed one-second offset or more forceful coaching language.
