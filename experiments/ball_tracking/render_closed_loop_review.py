"""Build a local interactive review of saved closed-loop ball tracker runs.

Rendering never changes the tracker or runs inference. Manual labels are
overlays for comparison only; they are not selected tracker positions.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

from scripts.ball_dataset import validate_labels

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS = ROOT / "data/reference-review/model-comparison-20261116"
DEFAULT_OUTPUT = ROOT / "data/reference-review/closed-loop-tracker"
TITLES = {
    "36bf848f409c": "Daylight · through the goal area",
    "e9935750581e": "Daylight · second shot",
    "e3949580d0f2": "Night · shot one",
    "735e17f02bd5": "Night · shot two",
    "13a1022c5009": "Night · earlier recording",
}


def load_clip(report_path: Path, destination: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "offline_pilot_not_production":
        raise ValueError(f"Expected an offline tracker report: {report_path}")
    clip = report["clip"]
    metadata = ROOT / "experiments/ball_tracking/reviewed_labels" / clip
    images = ROOT / "data/reference-review/ball-labels" / clip
    if not metadata.is_dir():
        metadata = images
    manifest = json.loads((metadata / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((metadata / "labels.json").read_text(encoding="utf-8"))
    validate_labels(manifest, labels)
    if report["source_sha256"] != manifest["source_sha256"]:
        raise ValueError(f"Report does not match reviewed source: {clip}")
    frames = {row["frame"]: row for row in manifest["frames"]}
    annotations = {row["frame"]: row for row in labels["annotations"]}
    proposals = {row["frame"]: row for row in report["rows"]}
    path = sorted(report["path"], key=lambda row: row["frame"])
    expected = list(range(report["start_frame"], report["end_frame"] + 1))
    if [row["frame"] for row in path] != expected:
        raise ValueError(f"Expected a contiguous reviewed path: {clip}")
    records = []
    for row in path:
        frame = row["frame"]
        info, label = frames[frame], annotations[frame]
        source = images / info["image"]
        if not source.is_file():
            raise ValueError(f"Missing reviewed source image: {source}")
        image_name = f"{clip}-{frame:06d}.jpg"
        shutil.copyfile(source, destination / "frames" / image_name)
        selected = row["center"] if row["detected"] else None
        box = label["box"] if label["state"] == "visible" else None
        truth = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) if box else None
        records.append({"frame": frame, "time_s": info["time_s"],
                        "width": info["width"], "height": info["height"],
                        "image": f"frames/{image_name}", "selected": selected,
                        "predicted": row["predicted"], "score": row["confidence"],
                        "label": box, "candidates": proposals[frame].get("candidates", []),
                        "error_px": math.dist(selected, truth) if selected and truth else None})
    return {"id": clip, "title": TITLES.get(clip, clip), "split": report["split"],
            "fps": manifest["fps"], "start": report["start_frame"],
            "end": report["end_frame"], "seed": report["seed"],
            "report_name": report_path.name, "frames": records}


def preview_sheet(clips: list[dict], destination: Path) -> None:
    """A static overview for checking generated image geometry."""
    import cv2
    import numpy as np

    panel_w, panel_h = 440, 330
    sheet = np.full((len(clips) * panel_h, panel_w * 3, 3), (13, 20, 16), dtype=np.uint8)
    for row, clip in enumerate(clips):
        for col, index in enumerate((0, len(clip["frames"]) // 2, len(clip["frames"]) - 1)):
            frame = clip["frames"][index]
            image = cv2.imread(str(destination / frame["image"]))
            if image is None:
                raise ValueError(f"Unreadable copied frame: {frame['image']}")
            scale = min(panel_w / image.shape[1], (panel_h - 44) / image.shape[0])
            if frame["label"]:
                box = frame["label"]
                cv2.rectangle(image, tuple(map(round, box[:2])), tuple(map(round, box[2:])),
                              (255, 225, 30), 2)
            if frame["selected"]:
                cv2.circle(image, tuple(map(round, frame["selected"])), 12, (35, 222, 255), 3)
            image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
            x, y = col * panel_w + (panel_w - image.shape[1]) // 2, row * panel_h + 42
            sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
            error = f"{frame['error_px']:.1f}px" if frame["error_px"] is not None else "missing"
            caption = f"{clip['id']}  frame {frame['frame']}  error {error}"
            cv2.putText(sheet, caption, (col * panel_w + 10, row * panel_h + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (232, 242, 235), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination / "overview.jpg"), sheet):
        raise OSError("Could not write overview image")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    reports = [args.reports_dir / name for name in (
        "day-goal-closed-loop.json", "day1-closed-loop.json", "night1-closed-loop.json",
        "night2-closed-loop.json", "night0-closed-loop.json")]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frames").mkdir(exist_ok=True)
    clips = [load_clip(report, args.output) for report in reports]
    data = json.dumps({"clips": clips}, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("closed_loop_review.html").read_text(encoding="utf-8")
    if template.count("__REVIEW_DATA__") != 1:
        raise ValueError("Review template must contain one data placeholder")
    (args.output / "index.html").write_text(template.replace("__REVIEW_DATA__", data), encoding="utf-8")
    preview_sheet(clips, args.output)
    print(f"Review: {(args.output / 'index.html').resolve()}")
    print(f"Clips: {len(clips)}; reviewed frames: {sum(len(clip['frames']) for clip in clips)}")


if __name__ == "__main__":
    main()
