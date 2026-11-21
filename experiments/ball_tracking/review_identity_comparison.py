"""Compare two saved offline ball tracks against provisional visual references.

This post-inference tool never passes annotations to a detector or tracker.
Centre-distance checks are diagnostic proximity counts, not validated accuracy,
ball identity confidence, contact timing, or shot outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
from pathlib import Path

from experiments.ball_tracking.render_unlabelled_review import (
    ROOT,
    TEMPLATE,
    _finite_number,
    _frame_map,
    _frames_dir,
    _point,
    _source_image,
)


_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")
_CLIP = re.compile(r"[A-Za-z0-9_-]+\Z")
_PROVENANCE = "assistant_visual_review_provisional_needs_human_confirmation"
_PURPOSE = "offline_evaluation_reference_only_not_tracker_input"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: object, description: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"Invalid {description}")
    return value.lower()


def _source_file(name: object) -> Path:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Missing source_name")
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Missing source video: {path}")
    return path


def _report_paths(location: Path) -> list[Path]:
    if location.is_file():
        paths = [location] if location.name == "report.json" else []
    elif location.is_dir():
        paths = sorted(location.rglob("report.json"))
    else:
        paths = []
    if not paths:
        raise ValueError(f"No saved report.json under {location}")
    return paths


def _load_reports(location: Path) -> dict[str, tuple[Path, dict]]:
    reports: dict[str, tuple[Path, dict]] = {}
    for path in _report_paths(location):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("status") != "offline_pilot_not_production":
            raise ValueError(f"Expected offline pilot report: {path}")
        if data.get("split") not in {"unseen_session_unlabelled", "inspected_session_unlabelled"}:
            raise ValueError(f"Expected unlabelled offline-session report: {path}")
        clip = data.get("clip")
        if not isinstance(clip, str) or not _CLIP.fullmatch(clip) or clip in reports:
            raise ValueError(f"Invalid or repeated clip identifier in {path}")
        _require_sha(data.get("source_sha256"), f"source_sha256 in {path}")
        _require_sha(data.get("weights_sha256"), f"weights_sha256 in {path}")
        reports[clip] = (path, data)
    return reports


def _prepare_report(path: Path, report: dict, annotation: dict) -> dict:
    clip = report["clip"]
    if _require_sha(report["source_sha256"], f"source SHA for {clip}") != annotation["source_sha256"]:
        raise ValueError(f"Source SHA mismatch between report and reference: {clip}")
    source = _source_file(report.get("source_name"))
    if _sha256(source) != annotation["source_sha256"]:
        raise ValueError(f"Source video bytes do not match reference SHA: {clip}")
    if source.name != annotation.get("source_name"):
        raise ValueError(f"Source video name differs from reference: {clip}")
    width, height = report.get("width"), report.get("height")
    if (isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int)
            or not isinstance(height, int) or width <= 0 or height <= 0
            or (width, height) != (annotation["source_width"], annotation["source_height"])):
        raise ValueError(f"Source dimensions differ from reference: {clip}")
    fps = _finite_number(report.get("fps"), f"{clip} fps")
    if fps <= 0:
        raise ValueError(f"Invalid FPS: {clip}")
    start, end = report.get("start_frame"), report.get("end_frame")
    if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
            or not isinstance(end, int) or start < 0 or end < start or end - start > 10000):
        raise ValueError(f"Invalid frame range: {clip}")
    expected = list(range(start, end + 1))
    settings = report.get("settings")
    if not isinstance(settings, dict) or not {"crop_size", "slice_offset", "confidence", "class_id", "inference_size", "device"} <= settings.keys():
        raise ValueError(f"Missing detector settings: {clip}")
    implementation = report.get("implementation_sha256")
    if not isinstance(implementation, dict):
        raise ValueError(f"Missing implementation hashes: {clip}")
    detector_sha = _require_sha(implementation.get("detector"), f"detector implementation SHA: {clip}")
    _require_sha(implementation.get("association"), f"association implementation SHA: {clip}")
    if report.get("seed_kind") != "one_manual_starting_position_no_reseeding":
        raise ValueError(f"Expected one manual seed without reseeding: {clip}")
    return {
        "path": path,
        "report": report,
        "source": source,
        "fps": fps,
        "width": width,
        "height": height,
        "start": start,
        "end": end,
        "settings": settings,
        "detector_sha256": detector_sha,
        "seed": _point(report.get("seed"), f"{clip} seed"),
        "directory": _frames_dir(report),
        "files": _frame_map(report.get("frame_files"), f"{clip} frame files", expected),
        "rows": _frame_map(report.get("rows"), f"{clip} proposals", expected),
        "path_rows": _frame_map(report.get("path"), f"{clip} path", expected),
    }


def _load_annotation(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("provenance") != _PROVENANCE or data.get("purpose") != _PURPOSE:
        raise ValueError(f"Expected provisional evaluation-only reference: {path}")
    clip = data.get("clip")
    if not isinstance(clip, str) or not _CLIP.fullmatch(clip):
        raise ValueError(f"Invalid reference clip identifier: {path}")
    data["source_sha256"] = _require_sha(data.get("source_sha256"), f"reference source SHA: {clip}")
    for key in ("source_width", "source_height"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Invalid {key}: {clip}")
    frame_range = data.get("frame_range")
    if (not isinstance(frame_range, list) or len(frame_range) != 2
            or any(isinstance(f, bool) or not isinstance(f, int) for f in frame_range)
            or frame_range[0] < 0 or frame_range[1] < frame_range[0]):
        raise ValueError(f"Invalid reference frame range: {clip}")
    records = data.get("frames")
    if not isinstance(records, list) or not records:
        raise ValueError(f"No reviewed frames: {clip}")
    indexed = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid reference row: {clip}")
        frame = row.get("frame")
        if (isinstance(frame, bool) or not isinstance(frame, int) or frame in indexed
                or not frame_range[0] <= frame <= frame_range[1]):
            raise ValueError(f"Invalid or repeated reference frame: {clip}")
        _require_sha(row.get("frame_sha256"), f"reference frame SHA: {clip} {frame}")
        if row.get("reviewed_original_frame") is not True:
            raise ValueError(f"Unreviewed original frame: {clip} {frame}")
        status = row.get("status")
        if status == "visible":
            point = _point(row.get("center"), f"reference centre: {clip} {frame}")
            uncertainty = _finite_number(row.get("uncertainty_px"), f"reference uncertainty: {clip} {frame}")
            if (uncertainty <= 0 or not 0 <= point[0] < data["source_width"]
                    or not 0 <= point[1] < data["source_height"]):
                raise ValueError(f"Invalid visible reference position: {clip} {frame}")
        elif status in {"uncertain", "occluded", "absent"}:
            if row.get("center") is not None or row.get("uncertainty_px") is not None:
                raise ValueError(f"Non-visible reference must not have a centre: {clip} {frame}")
        else:
            raise ValueError(f"Invalid reference status: {clip} {frame}")
        indexed[frame] = row
    data["indexed"] = indexed
    return data


def _selection(run: dict, frame: int) -> tuple[list[float] | None, float | None, list[float], list[dict], float]:
    clip = run["report"]["clip"]
    info = run["files"][frame]
    if info.get("width") != run["width"] or info.get("height") != run["height"]:
        raise ValueError(f"Frame dimensions differ from source: {clip} {frame}")
    time_s = _finite_number(info.get("time_s"), f"{clip} frame time")
    if time_s < 0:
        raise ValueError(f"Negative frame time: {clip} {frame}")
    row, path = run["rows"][frame], run["path_rows"][frame]
    predicted = _point(path.get("predicted"), f"{clip} predicted {frame}")
    crop_center = _point(row.get("crop_center"), f"{clip} crop centre {frame}")
    if not all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(predicted, crop_center, strict=True)):
        raise ValueError(f"Proposal/path prediction mismatch: {clip} {frame}")
    selected = _point(path.get("center"), f"{clip} selected {frame}", optional=True)
    if not isinstance(path.get("detected"), bool) or path["detected"] != (selected is not None):
        raise ValueError(f"Selected flag mismatch: {clip} {frame}")
    score = None if selected is None else _finite_number(path.get("confidence"), f"{clip} score {frame}")
    if score is not None and not 0 <= score <= 1:
        raise ValueError(f"Invalid detector score: {clip} {frame}")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or row.get("candidate_count") != len(candidates):
        raise ValueError(f"Invalid candidate count: {clip} {frame}")
    checked = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"Invalid candidate: {clip} {frame}")
        confidence = _finite_number(candidate.get("confidence"), f"candidate score: {clip} {frame}")
        if not 0 <= confidence <= 1:
            raise ValueError(f"Invalid candidate score: {clip} {frame}")
        checked.append({"center": _point(candidate.get("center"), f"candidate: {clip} {frame}"),
                        "confidence": confidence})
    return selected, score, predicted, checked, time_s


def _metrics(rows: list[dict], run_name: str) -> dict:
    visible = [row for row in rows if row["reference_status"] == "visible"]
    matched_10 = matched_20 = wrong = missing = longest = current = recoveries = 0
    previous_frame = None
    for row in visible:
        frame = row["frame"]
        if previous_frame is not None and frame != previous_frame + 1:
            current = 0  # A gap or non-visible reference cannot imply a continuous miss.
        previous_frame = frame
        item = row[run_name]
        distance = item["error_px"]
        if distance is None:
            missing += 1
            current += 1
        elif distance > 20:
            wrong += 1
            current += 1
        else:
            matched_20 += 1
            if distance <= 10:
                matched_10 += 1
            if current:
                recoveries += 1
            current = 0
        longest = max(longest, current)
    if matched_20 + wrong + missing != len(visible):
        raise AssertionError("Proximity buckets do not cover visible references")
    return {
        "visible_reference_frames": len(visible),
        "selected_within_10_px": matched_10,
        "selected_within_20_px": matched_20,
        "wrong_selected_over_20_px": wrong,
        "missing_selection": missing,
        "longest_contiguous_nonmatch_frames": longest,
        "recovered_after_nonmatch_events": recoveries,
    }


def _table(clips: list[dict]) -> str:
    body = []
    for clip in clips:
        name = html.escape(clip["clip"])
        cells = []
        for name_key in ("baseline", "revised"):
            metric = clip[name_key]
            cells.append(
                f"<td>{metric['selected_within_20_px']}/{metric['visible_reference_frames']} within 20 px"
                f"<br><small>{metric['selected_within_10_px']} within 10 px · "
                f"{metric['wrong_selected_over_20_px']} wrong selected · "
                f"{metric['missing_selection']} missing · "
                f"longest nonmatch {metric['longest_contiguous_nonmatch_frames']} frames · "
                f"recoveries {metric['recovered_after_nonmatch_events']}</small></td>"
            )
        body.append(f"<tr><th scope='row'>{name}</th>{''.join(cells)}</tr>")
    return (
        '<section class="comparison-summary" aria-label="Provisional comparison counts">'
        '<h2>Provisional visual-reference proximity</h2>'
        '<p>Inspected development data. Assistant-reviewed centres need human confirmation. '
        'Counts apply only to matching visible frames in the saved reports. '
        'They are neither validated accuracy nor shot outcomes. '
        'The cyan region marks each centre’s stated pixel uncertainty.</p>'
        '<div class="table-scroll"><table><thead><tr><th scope="col">Clip</th>'
        '<th scope="col">Baseline · yellow</th><th scope="col">Revised · green</th>'
        '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div></section>'
    )


def _render_html(clips: list[dict], summary: list[dict], output: Path) -> Path:
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__REVIEW_DATA__") != 1:
        raise ValueError("Review template must have one data placeholder")
    substitutions = {
        "Ball identity review": "Provisional identity comparison",
        "Step through the original frames and compare the selected detector position with the manually marked ball. This review helps diagnose tracking; it does not certify a shot result.":
            "Compare saved offline tracks against provisional assistant-reviewed ball centres. These references need human confirmation and do not establish a goal or shot result.",
        "Offline development review • manual start": "Inspected development comparison • provisional references • manual start",
        "Distance to manual mark": "Distance to provisional centre",
        "Manual ball box": "Provisional reference region",
        "Manual box": "Provisional reference",
        "AI selected ball candidate": "Selected candidate · yellow baseline / green revised",
        "A centre within 20 px of a manual mark is a proximity check only.":
            "A centre within 20 px of a provisional visual reference is only a proximity check. The cyan region shows annotation uncertainty, not the physical ball boundary.",
        'ctx.strokeStyle = "#ffd75c";':
            'ctx.strokeStyle = clip.split === "revised" ? "#a6ef85" : "#ffd75c";',
        'const focus = point(frame.selected) || point(frame.predicted) || point(clip.seed) || [width / 2, height / 2];':
            'const reference = Array.isArray(frame.label) && frame.label.length === 4 '
            '? [(frame.label[0] + frame.label[2]) / 2, (frame.label[1] + frame.label[3]) / 2] : null;\n'
            '      const focus = reference || point(frame.selected) || point(frame.predicted) || point(clip.seed) || [width / 2, height / 2];',
        'el("zoom-note").textContent = point(frame.selected)\n'
        '        ? "160 × 160 source pixels around the AI selection; cyan shows the manual box when enabled."\n'
        '        : "No detection selected. Closeup follows the predicted position; a prediction is not a ball observation.";':
            'el("zoom-note").textContent = reference\n'
            '        ? "160 × 160 source pixels around the provisional centre; cyan marks review uncertainty."\n'
            '        : "No visible reference. Closeup follows the selected or predicted point; this is not a ball observation.";',
        'setMetric("error-value", "No scored position", "neutral");':
            'setMetric("error-value", frame.reference_status === "visible" '
            '? "Missing selection on visible reference" : "Not scored: no visible reference", '
            'frame.reference_status === "visible" ? "bad" : "neutral");',
    }
    for before, after in substitutions.items():
        if before not in template:
            raise ValueError(f"Review template changed: missing {before[:50]}")
        template = template.replace(before, after)
    template = template.replace(
        '<nav id="clip-nav" class="clip-nav" aria-label="Review clips"></nav>',
        _table(summary) + '<nav id="clip-nav" class="clip-nav" aria-label="Review clips"></nav>',
    )
    template = template.replace(
        "  </style>",
        """    .comparison-summary { margin: 20px 0 4px; padding: 16px 20px; border: 1px solid #6f6742;
      border-radius: 14px; background: #202318; }
    .comparison-summary h2 { margin: 0 0 5px; font-size: 1.1rem; }
    .comparison-summary p { margin: 0 0 12px; color: var(--muted); }
    .table-scroll { overflow-x: auto; }
    .comparison-summary table { width: 100%; border-collapse: collapse; min-width: 650px; }
    .comparison-summary th, .comparison-summary td { padding: 9px 12px; text-align: left;
      border-top: 1px solid var(--border); vertical-align: top; }
    .comparison-summary th { color: var(--green); }
    .comparison-summary small { color: var(--muted); }
    body.revised .swatch.selected { border-color: #a6ef85; }
  </style>""",
    )
    template = template.replace(
        "      const clip = currentClip();\n      const frames = currentFrames();",
        '      const clip = currentClip();\n'
        '      document.body.classList.toggle("revised", clip?.split === "revised");\n'
        '      const frames = currentFrames();',
    )
    data = json.dumps({"clips": clips}, ensure_ascii=False).replace("<", "\\u003c")
    target = output / "index.html"
    target.write_text(template.replace("__REVIEW_DATA__", data), encoding="utf-8")
    return target


def compare_runs(baseline: Path, revised: Path, annotations: Path, output: Path) -> tuple[Path, dict]:
    """Score matching visible frames after inference and render local review tabs."""
    baseline_reports = _load_reports(baseline)
    revised_reports = _load_reports(revised)
    if not annotations.is_dir():
        raise ValueError(f"Missing annotations directory: {annotations}")
    refs = {}
    for path in sorted(annotations.glob("*.json")):
        data = _load_annotation(path)
        if data["clip"] in refs:
            raise ValueError(f"Repeated reference clip: {data['clip']}")
        refs[data["clip"]] = data
    clips_to_compare = sorted(revised_reports.keys() & baseline_reports.keys() & refs.keys())
    if not clips_to_compare:
        raise ValueError("No clip shared by baseline, revised and provisional references")
    output.mkdir(parents=True, exist_ok=True)
    display_clips = []
    summaries = []
    for clip in clips_to_compare:
        annotation = refs[clip]
        base = _prepare_report(*baseline_reports[clip], annotation)
        new = _prepare_report(*revised_reports[clip], annotation)
        if (base["report"]["weights_sha256"].lower()
                != new["report"]["weights_sha256"].lower()):
            raise ValueError(f"Compared runs use different detector weights: {clip}")
        if base["source"] != new["source"]:
            raise ValueError(f"Compared runs use different source file paths: {clip}")
        if base["seed"] != new["seed"]:
            raise ValueError(f"Compared runs use different manual seed positions: {clip}")
        if (base["start"], base["end"]) != (new["start"], new["end"]):
            raise ValueError(f"Compared runs use different frame coverage: {clip}")
        if base["settings"] != new["settings"]:
            raise ValueError(f"Compared runs use different detector settings: {clip}")
        if base["detector_sha256"] != new["detector_sha256"]:
            raise ValueError(f"Compared runs use different detector implementations: {clip}")
        if (base["width"], base["height"]) != (new["width"], new["height"]) or not math.isclose(base["fps"], new["fps"], abs_tol=1e-6):
            raise ValueError(f"Compared runs use different frame geometry or FPS: {clip}")
        shared = sorted(annotation["indexed"].keys() & base["files"].keys() & new["files"].keys())
        if not shared:
            raise ValueError(f"Reference frames are outside compared run windows: {clip}")
        frames = list(range(shared[0], shared[-1] + 1))
        if any(frame not in base["files"] or frame not in new["files"] for frame in frames):
            raise ValueError(f"Compared runs do not cover a consecutive reference window: {clip}")
        destination = output / "frames" / clip
        destination.mkdir(parents=True, exist_ok=True)
        records = []
        ui = {"baseline": [], "revised": []}
        for frame in frames:
            reference = annotation["indexed"].get(frame)
            base_image = _source_image(base["directory"], base["files"][frame].get("image"))
            new_image = _source_image(new["directory"], new["files"][frame].get("image"))
            image_sha = _sha256(base_image)
            if _sha256(new_image) != image_sha:
                raise ValueError(f"Compared reports have different decoded original JPEGs: {clip} {frame}")
            if reference and image_sha != _require_sha(reference["frame_sha256"], f"frame SHA {clip} {frame}"):
                raise ValueError(f"Reviewed original JPEG SHA mismatch: {clip} {frame}")
            image_name = f"{frame:06d}{base_image.suffix.lower()}"
            shutil.copyfile(base_image, destination / image_name)
            visible = reference is not None and reference["status"] == "visible"
            centre = reference["center"] if visible else None
            uncertainty = float(reference["uncertainty_px"]) if visible else None
            record = {"frame": frame, "reference_status": reference["status"] if reference else "unreviewed",
                      "reference_center": centre, "uncertainty_px": uncertainty}
            for run_name, run in (("baseline", base), ("revised", new)):
                selected, score, predicted, candidates, time_s = _selection(run, frame)
                distance = math.dist(selected, centre) if selected is not None and visible else None
                record[run_name] = {"selected": selected, "error_px": distance,
                                    "detector_score": score}
                ui[run_name].append({
                    "frame": frame, "time_s": time_s, "width": base["width"], "height": base["height"],
                    "image": f"frames/{clip}/{image_name}", "selected": selected,
                    "predicted": predicted, "score": score,
                    "label": ([centre[0] - uncertainty, centre[1] - uncertainty,
                               centre[0] + uncertainty, centre[1] + uncertainty] if visible else None),
                    "candidates": candidates, "error_px": distance,
                    "reference_status": record["reference_status"],
                    "uncertainty_px": uncertainty,
                })
            records.append(record)
        if not any(row["reference_status"] == "visible" for row in records):
            raise ValueError(f"No visible references in compared window: {clip}")
        counts = {name: _metrics(records, name) for name in ("baseline", "revised")}
        summary = {"clip": clip, "source_sha256": annotation["source_sha256"],
                   "window": [frames[0], frames[-1]], "reference_provenance": _PROVENANCE,
                   "annotation_uncertainty_px": sorted({row["uncertainty_px"] for row in records
                                                        if row["uncertainty_px"] is not None}),
                   "baseline": counts["baseline"], "revised": counts["revised"], "frames": records}
        summaries.append(summary)
        for name, run in (("baseline", base), ("revised", new)):
            display_clips.append({"id": f"{clip}-{name}", "title": f"{clip} · {name.title()}",
                                  "split": name, "fps": run["fps"], "start": frames[0],
                                  "end": frames[-1], "seed": run["seed"],
                                  "report_name": str(run["path"]), "frames": ui[name]})
    result = {"status": "offline_provisional_comparison_not_production",
              "split": "inspected_development_with_provisional_references",
              "reference_provenance": _PROVENANCE,
              "limitations": "Assistant visual references need human confirmation. Centre proximity is not validated identity accuracy, contact timing, or shot outcome.",
              "clips": summaries}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return _render_html(display_clips, summaries, output), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--revised", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    page, summary = compare_runs(args.baseline, args.revised, args.annotations, args.output)
    print(f"Provisional comparison: {page.resolve()}")
    for clip in summary["clips"]:
        a, b = clip["baseline"], clip["revised"]
        print(f"{clip['clip']}: baseline {a['selected_within_20_px']}/{a['visible_reference_frames']} "
              f"vs revised {b['selected_within_20_px']}/{b['visible_reference_frames']} "
              "within 20 px of provisional visible references")


if __name__ == "__main__":
    main()
