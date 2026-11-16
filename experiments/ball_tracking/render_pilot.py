"""Render the saved hand-seeded motion pilot for visual inspection."""

import argparse
import html
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/reference-review/ball-labels"
OUTPUT = ROOT / "data/reference-review/beam-motion-track"


def render(report_name="report.json", review_name=None, early_frames=6):
    if Path(report_name).name != report_name:
        raise ValueError("Report name must be a filename")
    destination = OUTPUT / review_name if review_name else OUTPUT
    destination.mkdir(exist_ok=True)
    report = json.loads((OUTPUT / report_name).read_text(encoding="utf-8"))
    if report.get("status") != "offline_pilot_not_production":
        raise ValueError("Expected an offline pilot report.")
    cards = []
    for result in report["results"]:
        clip, mode = result["clip"], result["mode"]
        folder = DATA / clip
        labels = ({item["frame"]: item for item in json.loads((folder / "labels.json").read_text(
            encoding="utf-8"))["annotations"]} if (folder / "labels.json").exists() else {})
        source_name = result["source_name"]
        capture = cv2.VideoCapture(str(ROOT / "data/input" / source_name))
        if not capture.isOpened():
            raise ValueError(f"Cannot open {source_name}")
        chosen = [row for index, row in enumerate(result["path"])
                  if index < early_frames or index % 4 == 0 or
                  index == len(result["path"])-1]
        figures = []
        for row in chosen:
            frame = row["frame"]
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                continue
            label = labels.get(frame)
            if label and label["state"] == "visible":
                box = label["box"]
                cv2.rectangle(image, tuple(round(v) for v in box[:2]),
                              tuple(round(v) for v in box[2:]), (60, 255, 80), 3)
            cx, cy = (round(v) for v in row["center"])
            cv2.circle(image, (cx, cy), 11, (0, 220, 255), 3)
            if row.get("reliability") == "uncertain":
                cv2.drawMarker(image, (cx, cy), (0, 80, 255),
                               cv2.MARKER_TILTED_CROSS, 25, 2)
            if result["goal_center"] is not None:
                goal = tuple(result["goal_center"])
                cv2.drawMarker(image, goal, (255, 80, 210),
                               cv2.MARKER_CROSS, 20, 2)
            width = 320
            image = cv2.resize(image, (width, round(image.shape[0]*width/image.shape[1])))
            name = f"{clip}-{mode}-{frame:06d}.jpg"
            if not cv2.imwrite(str(destination / name), image,
                               [cv2.IMWRITE_JPEG_QUALITY, 85]):
                raise OSError(name)
            error = (f"; centre error {row['error_px']:.1f}px" if "error_px" in row
                     else "; no reviewed box")
            state = "motion candidate" if row["observed"] else "coasted gap"
            if row.get("reliability") == "uncertain":
                state += "; UNCERTAIN after trajectory jump"
            figures.append(f'<figure><img loading="lazy" src="{name}"><figcaption>'
                           f'Frame {frame}; {state}{html.escape(error)}</figcaption></figure>')
        capture.release()
        reviewed = (f'Reviewed centre within 10px: {result["within_10px"]}/{result["evaluated"]}; '
                    f'within 40px: {result["within_40px"]}/{result["evaluated"]}. '
                    if result["evaluated"] else 'No labels embedded in raw report; see scored review. ')
        cards.append(f'<section><h2>{clip} · {mode}</h2><p>'
                     + reviewed +
                     f'Observed motion frames: {result["observed"]}/{result["frames_tracked"]}.</p>'
                     '<div class="frames">' + ''.join(figures) + '</div></section>')
    independent = not any(result["evaluated"] for result in report["results"])
    description = ('These additional recordings were not used to select the pilot settings. '
                   'Sparse later ball positions are scored in the separate new-clips-scored '
                   'review; this raw report has no embedded accuracy score.' if independent else
                   'The clips informed method selection, so results are not independent test performance.')
    legend = ('Yellow: candidate trajectory. Pink: hand-read goal centre. ' if independent else
              'Yellow: candidate trajectory. Green: reviewed ball box. Pink: hand-read goal centre. ')
    page = ('<!doctype html><meta charset="utf-8"><title>SCAN hand-seeded motion pilot</title>'
            '<style>body{background:#08120d;color:#e9f3eb;font:16px system-ui;margin:24px}'
            'section{border-top:1px solid #42664b;padding:16px 0}.frames{display:flex;'
            'overflow-x:auto;gap:12px}figure{flex:none;width:320px;margin:0}img{width:100%}'
            'figcaption{font-size:13px;color:#b9cbbd}p{max-width:900px}</style>'
            '<h1>Hand-seeded motion pilot</h1><p>' + legend +
            'This is an offline test seeded with a manually chosen ball box; '
            'it does not establish automatic contact, ball identity, physical speed, or goal entry. '
            + description + '</p>'
            + ''.join(cards))
    (destination / "index.html").write_text(page, encoding="utf-8")
    print(destination / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-name", default="report.json")
    parser.add_argument("--review-name")
    parser.add_argument("--early-frames", type=int, default=6)
    args = parser.parse_args()
    if args.review_name and (Path(args.review_name).name != args.review_name or
                             args.review_name in (".", "..")):
        parser.error("review name must be a directory name")
    if args.early_frames < 0:
        parser.error("early frames must be nonnegative")
    render(args.report_name, args.review_name, args.early_frames)
