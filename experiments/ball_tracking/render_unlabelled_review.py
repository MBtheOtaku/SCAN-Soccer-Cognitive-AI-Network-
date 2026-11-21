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
_SPLIT_LABELS = {
    "unseen_session_unlabelled": "unseen session · unlabelled",
    "inspected_session_unlabelled": "inspected session · development · unlabelled",
}


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


def _frame_transform(report: dict, width: int, height: int) -> dict | None:
    """Check optional source-to-processing provenance without relabelling old runs."""
    if "frame_transform" not in report:
        return None
    transform = report["frame_transform"]
    if not isinstance(transform, dict):
        raise ValueError("frame_transform must be an object")
    for key in ("source_width", "source_height", "width", "height"):
        value = transform.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Invalid frame_transform {key}")
    if (transform["width"], transform["height"]) != (width, height):
        raise ValueError("frame_transform processing dimensions differ from report")
    source_width, source_height = transform["source_width"], transform["source_height"]
    if width > source_width or height > source_height:
        raise ValueError("frame_transform must not upscale the source")
    for key, expected in (("scale_x", width / source_width), ("scale_y", height / source_height)):
        scale = _finite_number(transform.get(key), f"frame_transform {key}")
        if not math.isclose(scale, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"frame_transform {key} disagrees with dimensions")
    downsampled = (width, height) != (source_width, source_height)
    expected_status = "downsampled" if downsampled else "native"
    expected_interpolation = "INTER_AREA" if downsampled else "none"
    if transform.get("status") != expected_status:
        raise ValueError("frame_transform status disagrees with dimensions")
    if transform.get("interpolation") != expected_interpolation:
        raise ValueError("Invalid frame_transform interpolation")
    return {key: transform[key] for key in (
        "source_width", "source_height", "width", "height", "scale_x", "scale_y",
        "status", "interpolation")}


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
    """Validate one saved run and copy its saved frames into the review."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "offline_pilot_not_production":
        raise ValueError(f"Expected an offline pilot report: {report_path}")
    split = report.get("split")
    if split not in _SPLIT_LABELS:
        raise ValueError(f"Expected an unlabelled offline-session report: {report_path}")
    inspected = split == "inspected_session_unlabelled"
    if inspected and report.get("association_method") not in ("temporal", "recovery", "motion-recovery"):
        raise ValueError(f"Expected temporal association for inspected-session report: {report_path}")
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
    transform = _frame_transform(report, width, height)
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
        record = {
            "frame": frame, "time_s": time_s, "width": width, "height": height,
            "image": f"frames/{clip}/{copied_name}", "selected": selected,
            "predicted": predicted, "score": score, "label": None,
            "candidates": checked_candidates, "error_px": None,
        }
        if inspected:
            reason, missing_run = selection.get("reason"), selection.get("missing_run")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"Missing temporal association reason: {clip} {frame}")
            if isinstance(missing_run, bool) or not isinstance(missing_run, int) or missing_run < 0:
                raise ValueError(f"Invalid temporal missing run: {clip} {frame}")
            record.update(association_reason=reason, missing_run=missing_run)
        records.append(record)

    report_name = report_path.resolve().relative_to(reports_dir.resolve()).as_posix()
    title = Path(source_name).name
    if transform is not None and transform["status"] == "downsampled":
        title += (f" · source {transform['source_width']}×{transform['source_height']}; "
                  f"preview {width}×{height}")
    result = {
        "id": clip,
        "title": title,
        "split": _SPLIT_LABELS[split],
        "split_kind": split,
        "fps": fps, "start": start, "end": end, "seed": seed,
        "report_name": report_name, "frames": records,
    }
    if transform is not None:
        result["frame_transform"] = transform
    return result


def render_review(reports_dir: Path, output: Path) -> tuple[Path, int, int]:
    reports_dir = reports_dir.resolve()
    reports = sorted(reports_dir.rglob("report.json"))
    if not reports:
        raise ValueError(f"No report.json files under {reports_dir}")
    output.mkdir(parents=True, exist_ok=True)
    clips = [load_clip(report, output, reports_dir) for report in reports]
    if len({clip["id"] for clip in clips}) != len(clips):
        raise ValueError("Reports contain repeated clip identifiers")
    has_inspected = any(clip["split_kind"] == "inspected_session_unlabelled" for clip in clips)
    has_unseen = any(clip["split_kind"] == "unseen_session_unlabelled" for clip in clips)
    has_downsampled = any(clip.get("frame_transform", {}).get("status") == "downsampled" for clip in clips)
    if has_inspected and has_unseen:
        subtitle = ("Offline tracks from unseen and previously inspected recording sessions. "
                    "Inspected clips are development examples using temporal association and a manual start. "
                    "Selections need visual checking; these unlabelled paths are not accuracy measurements.")
        badge = "Offline development review • mixed sessions • manual start"
    elif has_inspected:
        subtitle = ("Previously inspected development clips using temporal association and a manual start. "
                    "Inspect selected positions in the original frames; there are no manual labels or accuracy scores here.")
        badge = "Offline development review • inspected session • temporal association • manual start"
    else:
        subtitle = ("Frozen detector on an unseen recording session, started from a manually chosen ball position. "
                    "Inspect the selected path in the original frames; there are no manual labels or accuracy scores for this review.")
        badge = "Offline development review • unseen session • manual start"
    if has_downsampled:
        subtitle = subtitle.replace("original frames", "saved processing frames")
        subtitle += (" Downsampled clips show processing previews, not native-resolution original frames; "
                     "source and preview dimensions appear in their titles.")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__REVIEW_DATA__") != 1:
        raise ValueError("Review template must contain exactly one data placeholder")
    substitutions = {
        "Ball identity review": "Unlabelled ball-path review",
        "Step through the original frames and compare the selected detector position with the manually marked ball. This review helps diagnose tracking; it does not certify a shot result.":
            subtitle,
        "Offline development review • manual start": badge,
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
        'el("frame-meta").textContent = `Frame ${frame.frame} · ${time} (frame/FPS estimate) · ${fps}`;':
            'el("frame-meta").textContent = `Frame ${frame.frame} · ${time} (frame/FPS estimate) · ${fps}`'
            ' + (typeof frame.association_reason === "string" && Number.isInteger(frame.missing_run)'
            ' ? ` · Association: ${frame.association_reason.replaceAll("_", " ")} · missing run: ${frame.missing_run}`'
            ' : "");',
    }
    for old, new in substitutions.items():
        if old not in template:
            raise ValueError(f"Review template changed; missing text: {old[:50]}")
        template = template.replace(old, new)
    if has_downsampled:
        for old, new in {
            "Current original frame": "Current processing frame",
            "Original image pixels": "Processing image pixels",
            "160 × 160 source pixels": "160 × 160 processing pixels",
            "`Original frame ${frame.frame}": "`Processing frame ${frame.frame}",
        }.items():
            if old not in template:
                raise ValueError(f"Review template changed; missing text: {old}")
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
