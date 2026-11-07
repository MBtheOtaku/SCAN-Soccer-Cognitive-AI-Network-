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
            coarse_offsets = (-0.20, -0.10, 0.0, 0.10, 0.20)
            centres = [("Your approximate contact time", reference_s, coarse_offsets)]
            if estimate is not None:
                centres.insert(0, ("SCAN estimate", float(estimate), coarse_offsets))
            centres.append((
                "Frame-by-frame contact search (approximate reference minus 1.2s to plus 0.1s)",
                reference_s,
                tuple(step / fps for step in range(-round(1.2 * fps), round(0.1 * fps) + 1)),
            ))
            groups = []
            for group, (label, centre, offsets) in enumerate(centres):
                cards = []
                for offset in offsets:
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
                layout = "frames fine" if len(offsets) > 5 else "frames"
                groups.append(f'<h3>{html.escape(label)}: {centre:.3f}s</h3>'
                              + f'<div class="{layout}">' + ''.join(cards) + '</div>')
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
    .frames.fine{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));overflow:visible}
    .frames.fine figure{min-width:0}
    figure{margin:0;flex:0 0 350px}img{width:100%}figcaption{padding:8px 0}</style>
    <h1>Original-video shot timing review</h1>
    <p>Reference times are approximate. These requested frame positions use FPS,
    not verified presentation timestamps. Compare with the original video player.</p>
    <p>In each frame-by-frame search, identify the last frame before contact and
    the first frame showing the ball moving away. If contact is obscured, report
    a frame interval rather than guessing an exact contact frame.</p>'''
    target = OUT / "index.html"
    target.write_text(page + ''.join(sections), encoding="utf-8")
    print(f"Review created: {target}")


if __name__ == "__main__":
    main()
