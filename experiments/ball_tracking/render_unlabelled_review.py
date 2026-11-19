"""Render unlabelled, manually seeded offline ball tracks for local visual review.

This only displays saved detector output. It does not run inference, score a
track, or establish that the selected point is the ball.
"""

import argparse
import json
import math
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("closed_loop_review.html")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_CLIP_ID = re.compile(r"[A-Za-z0-9_-]+\Z")


def _finite_number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be a finite number")
    return result


def _point(value: object, description: str, *, optional: bool = False) -> list[float] | None:
    if value is None and optional:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{description} must be a two-coordinate point")
    return [_finite_number(value[0], description), _finite_number(value[1], description)]


def _frame_map(items: object, description: str, expected: list[int]) -> dict[int, dict]:
    if not isinstance(items, list) or len(items) != len(expected):
        raise ValueError(f"{description} must contain one row for every requested frame")
    result = {}
    for item in items:
        if not isinstance(item, dict) or isinstance(item.get("frame"), bool) or not isinstance(item.get("frame"), int):
            raise ValueError(f"{description} has an invalid frame number")
        frame = item["frame"]
        if frame in result:
            raise ValueError(f"{description} repeats frame {frame}")
        result[frame] = item
    if sorted(result) != expected:
        raise ValueError(f"{description} does not align with the requested frame range")
    return result


def _frames_dir(report: dict) -> Path:
    location = report.get("frames_directory")
    if not isinstance(location, str) or not location.strip():
        raise ValueError("Missing frames_directory")
    directory = Path(location)
    if not directory.is_absolute():
        directory = ROOT / directory
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"Missing frames directory: {directory}")
    return directory


def _source_image(directory: Path, image_name: object) -> Path:
    if not isinstance(image_name, str) or not image_name.strip():
        raise ValueError("Missing frame image name")
    relative = Path(image_name)
    if relative.is_absolute():
        raise ValueError("Frame image must be relative to frames_directory")
    image = (directory / relative).resolve()
    if not image.is_relative_to(directory) or not image.is_file():
        raise ValueError(f"Missing or unsafe frame image: {image_name}")
    return image


def load_clip(report_path: Path, output: Path, reports_dir: Path) -> dict:
    """Validate one saved run and copy its original frames into the review."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "offline_pilot_not_production":
        raise ValueError(f"Expected an offline pilot report: {report_path}")
    if report.get("split") != "unseen_session_unlabelled":
        raise ValueError(f"Expected an unlabelled unseen-session report: {report_path}")
    clip = report.get("clip")
    if not isinstance(clip, str) or not _CLIP_ID.fullmatch(clip):
        raise ValueError("Invalid clip identifier")
    source_name = report.get("source_name")
    if not isinstance(source_name, str) or not source_name.strip():
        raise ValueError(f"Missing source_name: {clip}")
    for key in ("source_sha256", "weights_sha256"):
        if not isinstance(report.get(key), str) or not _SHA256.fullmatch(report[key]):
            raise ValueError(f"Missing or invalid {key}: {clip}")

    fps = _finite_number(report.get("fps"), f"{clip} fps")
    if fps <= 0:
        raise ValueError(f"Invalid FPS: {clip}")
    width, height = report.get("width"), report.get("height")
    if (isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int)
            or not isinstance(height, int) or width <= 0 or height <= 0):
        raise ValueError(f"Invalid source image dimensions: {clip}")
    start, end = report.get("start_frame"), report.get("end_frame")
    if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
            or not isinstance(end, int) or start < 0 or end < start):
        raise ValueError(f"Invalid frame range: {clip}")
    if end - start + 1 > 10000:
        raise ValueError(f"Unreasonably long review: {clip}")
    expected = list(range(start, end + 1))
    seed = _point(report.get("seed"), f"{clip} seed")
    rows = _frame_map(report.get("rows"), f"{clip} proposals", expected)
    path = _frame_map(report.get("path"), f"{clip} path", expected)
    files = _frame_map(report.get("frame_files"), f"{clip} frame files", expected)
    directory = _frames_dir(report)
    records = []
    destination = output / "frames" / clip
    destination.mkdir(parents=True, exist_ok=True)

    for frame in expected:
        info, proposal, selection = files[frame], rows[frame], path[frame]
        if info.get("width") != width or info.get("height") != height:
            raise ValueError(f"Frame {frame} dimensions differ from source: {clip}")
        time_s = _finite_number(info.get("time_s"), f"{clip} frame {frame} time_s")
        if time_s < 0:
            raise ValueError(f"Negative frame time: {clip} {frame}")
        crop_center = _point(proposal.get("crop_center"), f"{clip} frame {frame} crop_center")
        predicted = _point(selection.get("predicted"), f"{clip} frame {frame} predicted")
        if not all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(crop_center, predicted, strict=True)):
            raise ValueError(f"Proposal and path predictions disagree: {clip} {frame}")
        candidates = proposal.get("candidates")
        if not isinstance(candidates, list) or proposal.get("candidate_count") != len(candidates):
            raise ValueError(f"Invalid candidate count: {clip} {frame}")
        checked_candidates = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"Invalid candidate: {clip} {frame}")
            confidence = _finite_number(candidate.get("confidence"), f"{clip} frame {frame} confidence")
            if not 0 <= confidence <= 1:
                raise ValueError(f"Invalid candidate confidence: {clip} {frame}")
            checked_candidates.append({
                "center": _point(candidate.get("center"), f"{clip} frame {frame} candidate"),
                "confidence": confidence,
            })
        selected = _point(selection.get("center"), f"{clip} frame {frame} selected", optional=True)
        detected = selection.get("detected")
        if not isinstance(detected, bool) or (selected is None) != (not detected):
            raise ValueError(f"Selected detection flag and centre disagree: {clip} {frame}")
        score = None if selected is None else _finite_number(
            selection.get("confidence"), f"{clip} frame {frame} selected confidence")
        if score is not None and not 0 <= score <= 1:
            raise ValueError(f"Invalid selected confidence: {clip} {frame}")
        source = _source_image(directory, info.get("image"))
        copied_name = f"{frame:06d}{source.suffix.lower()}"
        shutil.copyfile(source, destination / copied_name)
        records.append({
            "frame": frame, "time_s": time_s, "width": width, "height": height,
            "image": f"frames/{clip}/{copied_name}", "selected": selected,
            "predicted": predicted, "score": score, "label": None,
            "candidates": checked_candidates, "error_px": None,
        })

    report_name = report_path.resolve().relative_to(reports_dir.resolve()).as_posix()
    return {
        "id": clip,
        "title": Path(source_name).name,
        "split": "unseen session · unlabelled",
        "fps": fps, "start": start, "end": end, "seed": seed,
        "report_name": report_name, "frames": records,
    }


def render_review(reports_dir: Path, output: Path) -> tuple[Path, int, int]:
    reports_dir = reports_dir.resolve()
    reports = sorted(reports_dir.rglob("report.json"))
    if not reports:
        raise ValueError(f"No report.json files under {reports_dir}")
    output.mkdir(parents=True, exist_ok=True)
    clips = [load_clip(report, output, reports_dir) for report in reports]
    if len({clip["id"] for clip in clips}) != len(clips):
        raise ValueError("Reports contain repeated clip identifiers")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__REVIEW_DATA__") != 1:
        raise ValueError("Review template must contain exactly one data placeholder")
    substitutions = {
        "Ball identity review": "Unlabelled ball-path review",
        "Step through the original frames and compare the selected detector position with the manually marked ball. This review helps diagnose tracking; it does not certify a shot result.":
            "Frozen detector on an unseen recording session, started from a manually chosen ball position. Inspect the selected path in the original frames; there are no manual labels or accuracy scores for this review.",
        "Offline development review • manual start": "Offline development review • unseen session • manual start",
        '<div class="metric"><span class="metric-label">Distance to manual mark</span>':
            '<div class="metric" style="display:none"><span class="metric-label">Distance to manual mark</span>',
        '<span class="key"><span class="swatch manual"':
            '<span class="key" style="display:none"><span class="swatch manual"',
        '<label><input id="show-label"': '<label style="display:none"><input id="show-label"',
        "A centre within 20 px of a manual mark is a proximity check only.":
            "No manual labels are available in this review; selections require independent visual checking.",
        "with selected AI candidate and optional manual annotations": "with selected AI candidate",
        "160 × 160 source pixels around the AI selection; cyan shows the manual box when enabled.":
            "160 × 160 source pixels around the AI selection; this is an unverified detector position.",
    }
    for old, new in substitutions.items():
        if old not in template:
            raise ValueError(f"Review template changed; missing text: {old[:50]}")
        template = template.replace(old, new)
    embedded = json.dumps({"clips": clips}, ensure_ascii=False).replace("<", "\\u003c")
    target = output / "index.html"
    target.write_text(template.replace("__REVIEW_DATA__", embedded), encoding="utf-8")
    return target, len(clips), sum(len(clip["frames"]) for clip in clips)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target, clips, frames = render_review(args.reports_dir, args.output)
    print(f"Review: {target.resolve()}")
    print(f"Clips: {clips}; unlabelled frames: {frames}")


if __name__ == "__main__":
    main()
