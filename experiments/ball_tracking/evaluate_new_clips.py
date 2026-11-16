"""Score frozen motion paths against sparse, manually reviewed ball centres."""

import hashlib
import html
import json
import math
import statistics
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORT = ROOT / "data/reference-review/beam-motion-track/report-new-clips.json"
LABELS = HERE / "reviewed_new_clips.json"
OUTPUT = ROOT / "data/reference-review/beam-motion-track/new-clips-scored"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(report, labels):
    """Use only labelled frames; do not treat tracker continuation as a hit."""
    results = {}
    for result in report["results"]:
        clip, mode = result["clip"], result["mode"]
        truth = {entry["frame"]: entry["center"]
                 for entry in labels["clips"][clip]["annotations"]}
        path = {entry["frame"]: entry for entry in result["path"]}
        rows = []
        for frame, center in truth.items():
            estimate = path.get(frame)
            rows.append({"frame": frame, "truth": center,
                         "estimate": estimate["center"] if estimate else None,
                         "observed": estimate["observed"] if estimate else False,
                         "reliability": estimate.get("reliability") if estimate else None,
                         "error_px": math.dist(center, estimate["center"])
                         if estimate else None})
        errors = [row["error_px"] for row in rows if row["error_px"] is not None]
        results.setdefault(clip, {})[mode] = {
            "labelled": len(rows), "tracked": len(errors),
            "median_error_px": statistics.median(errors) if errors else None,
            "max_error_px": max(errors) if errors else None,
            "within_10px": sum(error <= 10 for error in errors),
            "within_20px": sum(error <= 20 for error in errors),
            "within_40px": sum(error <= 40 for error in errors),
            "flagged_uncertain": sum(row["reliability"] == "uncertain"
                                     for row in rows),
            "rows": rows,
        }
    return results


def render(results, labels, destination=OUTPUT):
    destination.mkdir(parents=True, exist_ok=True)
    cards = []
    for clip, modes in results.items():
        source = ROOT / "data/input" / labels["clips"][clip]["source_name"]
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Cannot open {source}")
        figures = []
        for positive_row, absolute_row in zip(modes["positive"]["rows"],
                                              modes["absolute"]["rows"], strict=True):
            frame = positive_row["frame"]
            if frame != absolute_row["frame"]:
                raise ValueError("Mismatched review frames")
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode {clip} frame {frame}")
            cv2.drawMarker(image, tuple(positive_row["truth"]), (70, 255, 70),
                           cv2.MARKER_CROSS, 26, 2)
            for row, colour in ((positive_row, (0, 215, 255)),
                                (absolute_row, (255, 130, 70))):
                if row["estimate"] is not None:
                    cv2.circle(image, tuple(round(v) for v in row["estimate"]),
                               15, colour, 3)
            width = 480
            image = cv2.resize(image, (width, round(image.shape[0] * width /
                                                   image.shape[1])))
            name = f"{clip}-{frame:06d}.jpg"
            if not cv2.imwrite(str(destination / name), image,
                               [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise OSError(name)
            positive_error = positive_row["error_px"]
            absolute_error = absolute_row["error_px"]
            positive_text = f'{positive_error:.1f}px' if positive_error is not None else 'lost'
            absolute_text = f'{absolute_error:.1f}px' if absolute_error is not None else 'lost'
            figures.append(f'<figure><img loading="lazy" src="{name}"><figcaption>'
                           f'Frame {frame}; positive {positive_text}; '
                           f'absolute {absolute_text}; '
                           f'positive {positive_row["reliability"]}; '
                           f'absolute {absolute_row["reliability"]}</figcaption></figure>')
        capture.release()
        positive_median = modes["positive"]["median_error_px"]
        absolute_median = modes["absolute"]["median_error_px"]
        positive_median_text = f'{positive_median:.1f}px' if positive_median is not None else 'not available'
        absolute_median_text = f'{absolute_median:.1f}px' if absolute_median is not None else 'not available'
        summary = (f'Positive: median {positive_median_text}, '
                   f'tracked {modes["positive"]["tracked"]}/{modes["positive"]["labelled"]}, '
                   f'within 20px {modes["positive"]["within_20px"]}/'
                   f'{modes["positive"]["labelled"]}. '
                   f'Absolute: median {absolute_median_text}, '
                   f'tracked {modes["absolute"]["tracked"]}/{modes["absolute"]["labelled"]}, '
                   f'within 20px {modes["absolute"]["within_20px"]}/'
                   f'{modes["absolute"]["labelled"]}. '
                   f'Flagged uncertain marks: positive '
                   f'{modes["positive"]["flagged_uncertain"]}, absolute '
                   f'{modes["absolute"]["flagged_uncertain"]}.')
        cards.append(f'<section><h2>{html.escape(clip)}</h2><p>{summary}</p>'
                     '<div class="frames">' + "".join(figures) + '</div></section>')
    page = ('<!doctype html><meta charset="utf-8"><title>SCAN sparse ball path check</title>'
            '<style>body{background:#08120d;color:#e9f3eb;font:16px system-ui;margin:24px}'
            'section{border-top:1px solid #42664b;padding:16px 0}.frames{display:flex;'
            'overflow-x:auto;gap:12px}figure{flex:none;width:480px;margin:0}img{width:100%}'
            'figcaption{font-size:13px;color:#b9cbbd}p{max-width:950px}</style>'
            '<h1>Sparse ball trajectory check</h1><p>Green cross: manually marked ball centre. '
            'Yellow circle: positive change. Blue circle: absolute change. '
            'Original frames were visually marked without tracker overlays. '
            'These approximate centres have human marking error. The pilot was hand-seeded; '
            'A suspicious late jump stays uncertain through the rest of its path. '
            'This is a small local diagnostic, not a deployment accuracy estimate or '
            'proof of shot outcome.</p>' + "".join(cards))
    (destination / "index.html").write_text(page, encoding="utf-8")


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    if report["status"] != "offline_pilot_not_production" or labels["schema"] != 1:
        raise ValueError("Unexpected report or label schema")
    for clip, item in labels["clips"].items():
        source = ROOT / "data/input" / item["source_name"]
        if file_sha256(source) != item["source_sha256"]:
            raise ValueError(f"Source video changed: {clip}")
    results = score(report, labels)
    render(results, labels)
    (OUTPUT / "scores.json").write_text(json.dumps({
        "status": "sparse_manual_review_not_production_accuracy",
        "method": labels["method"], "results": results,
    }, indent=2), encoding="utf-8")
    for clip, modes in results.items():
        for mode, summary in modes.items():
            print(clip, mode, f'median {summary["median_error_px"]:.1f}px',
                  f'<=20px {summary["within_20px"]}/{summary["labelled"]}',
                  f'max {summary["max_error_px"]:.1f}px')
    print(OUTPUT / "index.html")


if __name__ == "__main__":
    main()
