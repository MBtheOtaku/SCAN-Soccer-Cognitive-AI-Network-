"""Bounded native-detail detector diagnostic on matched scene regions.

This does not run or score a native-resolution tracker. Query centres come from
a previously saved causal 720-coordinate run; no ball labels are read. Each
selected source frame is compared with its own area-downsampled version.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import math
import re
import time
from pathlib import Path

from experiments.ball_tracking.closed_loop_detector_track import detect_slices
from scripts.ball_dataset import file_sha256


SETTINGS = {"crop_size": 384, "slice_offset": 128, "confidence": 0.05,
            "class_id": 0, "inference_size": 640, "device": "cpu"}
MAX_STRIDE = 256
MAX_BATCH = 16
MAX_FRAMES_PER_CLIP = 24


def _dimension(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 384:
        raise ValueError("Expected integer image dimensions of at least 384 pixels")
    return value


def frozen_origins(width: int, height: int, center: list[float]) -> list[tuple[int, int]]:
    """Use precisely the unchanged detector's clamping and rounding rules."""
    _dimension(width)
    _dimension(height)
    if len(center) != 2 or not all(math.isfinite(v) for v in center):
        raise ValueError("Query centre must be finite XY")
    return list(dict.fromkeys(
        (max(0, min(width - 384, round(center[0] + dx - 192))),
         max(0, min(height - 384, round(center[1] + dy - 192))))
        for dx, dy in itertools.product((-128, 0, 128), repeat=2)))


def paired_regions(width: int, height: int, native_width: int, native_height: int,
                   center: list[float]) -> dict:
    """Project the exact union of frozen crops, rather than assume a full 640-pixel ROI."""
    _dimension(native_width)
    _dimension(native_height)
    if min(width, height) != 720 or native_width < width or native_height < height:
        raise ValueError("Expected 720-short-edge processing without source enlargement")
    sx, sy = native_width / width, native_height / height
    if abs(sx - sy) / max(sx, sy) > 0.005:
        raise ValueError("Source and processing aspect ratios do not match")
    origins = frozen_origins(width, height, center)
    left, top = min(x for x, _ in origins), min(y for _, y in origins)
    right, bottom = max(x for x, _ in origins) + 384, max(y for _, y in origins) + 384
    native = [max(0, math.floor(left * sx)), max(0, math.floor(top * sy)),
              min(native_width, math.ceil(right * sx)), min(native_height, math.ceil(bottom * sy))]
    return {"processing_xyxy": [left, top, right, bottom], "native_xyxy": native,
            "source_per_processing_xy": [sx, sy], "frozen_origins": origins,
            "dedup_native_px": 15 * min(sx, sy)}


def axis_origins(low: int, high: int, *, size: int = 384,
                 max_stride: int = MAX_STRIDE) -> list[int]:
    if size != 384 or not 1 <= max_stride <= 256 or high - low < size:
        raise ValueError("Native ROI must fit 384-pixel crops with a stride of at most 256 pixels")
    result = list(range(low, high - size + 1, max_stride))
    if result[-1] != high - size:
        result.append(high - size)
    return result


def native_origins(region: list[int]) -> list[tuple[int, int]]:
    left, top, right, bottom = region
    return list(itertools.product(axis_origins(left, right), axis_origins(top, bottom)))


def deduplicate(candidates: list[dict], radius: float) -> list[dict]:
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Deduplication radius must be positive and finite")
    result = []
    for item in sorted(candidates, key=lambda p: -p["confidence"]):
        point, confidence = item["center"], item["confidence"]
        if (len(point) != 2 or not all(math.isfinite(v) for v in point)
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("Invalid detector proposal")
        if all(math.dist(point, prior["center"]) > radius for prior in result):
            result.append(item)
    return result


def detect_native(model, image, region: dict) -> tuple[list[dict], int]:
    """Bound batch memory; every model input is a real 384-pixel native crop."""
    origins = native_origins(region["native_xyxy"])
    candidates = []
    for start in range(0, len(origins), MAX_BATCH):
        batch = origins[start:start + MAX_BATCH]
        crops = [image[y:y + 384, x:x + 384] for x, y in batch]
        if any(crop.shape[:2] != (384, 384) for crop in crops):
            raise ValueError("Native crop exceeds source image")
        results = model.predict(crops, imgsz=640, conf=0.05, classes=[0],
                                device="cpu", verbose=False)
        for (left, top), result in zip(batch, results, strict=True):
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            scores = result.boxes.conf.cpu().numpy().tolist()
            for box, confidence in zip(boxes, scores, strict=True):
                candidates.append({"center": [left + (box[0] + box[2]) / 2,
                                                top + (box[1] + box[3]) / 2],
                                   "confidence": confidence})
    return deduplicate(candidates, region["dedup_native_px"]), len(origins)


def proposal_spaces(candidates: list[dict], scales: list[float], *, native: bool) -> list[dict]:
    sx, sy = scales
    return [{"native_center": (p["center"] if native else [p["center"][0] * sx, p["center"][1] * sy]),
             "processing_center": ([p["center"][0] / sx, p["center"][1] / sy] if native else p["center"]),
             "confidence": p["confidence"]} for p in candidates]


def validate_frames(value: object, available: set[int]) -> list[int]:
    if (not isinstance(value, list) or not 1 <= len(value) <= MAX_FRAMES_PER_CLIP
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in value)
            or len(set(value)) != len(value) or not set(value) <= available):
        raise ValueError("Choose 1–24 unique probe frames already present in the normalized run")
    return sorted(value)


def validate_source_geometry(report: dict, width: int, height: int) -> str:
    """Legacy reports can omit a transform; supplied provenance must agree."""
    transform = report.get("frame_transform")
    if transform is None:
        return "legacy_report_transform_unavailable"
    if (not isinstance(transform, dict)
            or (transform.get("source_width"), transform.get("source_height")) != (width, height)
            or (transform.get("width"), transform.get("height")) != (report["width"], report["height"])):
        raise ValueError("Decoded source geometry differs from normalized report transform")
    for key, expected in [("scale_x", report["width"] / width),
                          ("scale_y", report["height"] / height)]:
        value = transform.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-10)):
            raise ValueError("Normalized report transform scale does not match decoded geometry")
    return "matched_normalized_report_transform"


def prepare_case(case: dict, runs: Path, weights_sha: str) -> dict:
    clip = case["clip"]
    if not isinstance(clip, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", clip):
        raise ValueError("Unsafe clip identifier")
    source = Path(case["source"]).resolve()
    source_sha = file_sha256(source)
    report_path = runs / clip / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    detector_sha = file_sha256(Path(__file__).with_name("closed_loop_detector_track.py"))
    if (report.get("status") != "offline_pilot_not_production" or report.get("clip") != clip
            or report.get("source_sha256") != source_sha
            or report.get("weights_sha256") != weights_sha or report.get("settings") != SETTINGS
            or not isinstance(report.get("implementation_sha256"), dict)
            or report["implementation_sha256"].get("detector") != detector_sha):
        raise ValueError("Normalized run must use this exact source and frozen detector")
    rows = {row["frame"]: row for row in report["rows"]}
    if len(rows) != len(report["rows"]):
        raise ValueError("Normalized run repeats a frame")
    frames = validate_frames(case.get("native_probe_frames"), set(rows))
    return {"clip": clip, "source": source, "source_sha256": source_sha,
            "source_colour": case.get("ball_colour", case.get("ball_color", case.get("colour", case.get("colour_metadata_not_tracker_input")))),
            "report_path": report_path, "report": report, "rows": rows, "frames": frames}


def run_case(case: dict, model, output: Path) -> dict:
    import cv2

    source, report = case["source"], case["report"]
    width, height = report["width"], report["height"]
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not decode {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0 or abs(fps - report["fps"]) > 0.01:
        capture.release()
        raise ValueError("Source FPS differs from normalized report")
    orientation = {"opencv_orientation_auto": (capture.get(cv2.CAP_PROP_ORIENTATION_AUTO)
                   if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO") else None),
                   "opencv_orientation_meta": (capture.get(cv2.CAP_PROP_ORIENTATION_META)
                   if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else None)}
    directory = output / case["clip"]
    directory.mkdir()
    records, number = [], 0
    wanted = set(case["frames"])
    try:
        while number <= max(wanted):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Source ended before probe frame {number}")
            if number not in wanted:
                number += 1
                continue
            native_height, native_width = image.shape[:2]
            geometry_status = validate_source_geometry(report, native_width, native_height)
            center = case["rows"][number]["crop_center"]
            region = paired_regions(width, height, native_width, native_height, center)
            reduced = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            started = time.perf_counter()
            base, base_count = detect_slices(model, reduced, center, 384, 128, 0.05)
            base_time = time.perf_counter() - started
            started = time.perf_counter()
            native, native_count = detect_native(model, image, region)
            native_time = time.perf_counter() - started
            a, b, c, d = region["processing_xyxy"]
            x, y, r, bottom = region["native_xyxy"]
            baseline_file = f"{number:06d}-720.png"
            native_file = f"{number:06d}-native.png"
            for filename, crop in [(baseline_file, reduced[b:d, a:c]),
                                   (native_file, image[y:bottom, x:r])]:
                if not cv2.imwrite(str(directory / filename), crop):
                    raise OSError(f"Could not save {filename}")
            records.append({"frame": number, "frame_over_fps_s": number / fps,
                            "decoder_pos_msec": capture.get(cv2.CAP_PROP_POS_MSEC),
                            "native_width": native_width, "native_height": native_height,
                            "geometry_provenance_status": geometry_status,
                            "processing_width": width, "processing_height": height,
                            "source_decoded_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                            "reduced_pixels_sha256": hashlib.sha256(reduced.tobytes()).hexdigest(),
                            "query_center_processing": center, "region": region,
                            "baseline": {"image": f"{case['clip']}/{baseline_file}",
                                         "image_sha256": file_sha256(directory / baseline_file),
                                         "crop_count": base_count, "elapsed_s": base_time,
                                         "proposals": proposal_spaces(base, region["source_per_processing_xy"], native=False)},
                            "native": {"image": f"{case['clip']}/{native_file}",
                                       "image_sha256": file_sha256(directory / native_file),
                                       "crop_count": native_count, "elapsed_s": native_time,
                                       "proposals": proposal_spaces(native, region["source_per_processing_xy"], native=True)}})
            print(f"{case['clip']} frame {number}: 720-pixel branch {len(base)} proposals; native branch {len(native)} proposals", flush=True)
            number += 1
    finally:
        capture.release()
    return {"clip": case["clip"], "ball_colour": case["source_colour"],
            "source_name": str(source), "source_sha256": case["source_sha256"],
            "normalized_report": str(case["report_path"].resolve()),
            "normalized_report_sha256": file_sha256(case["report_path"]),
            "fps": fps, "orientation": orientation, "probe_frames": case["frames"], "frames": records}


def render_review(report: dict, destination: Path) -> None:
    cards = []
    for clip in report["clips"]:
        for row in clip["frames"]:
            panels = []
            for method, label, colour, key in [("baseline", "Same source downsampled to a 720-pixel short edge", "#ffe35a", "processing"),
                                               ("native", "Native source detail", "#80ff9a", "native")]:
                left, top, right, bottom = row["region"][f"{key}_xyxy"]
                w, h = right - left, bottom - top
                circles = "".join(
                    f'<circle cx="{p[f"{key}_center"][0]-left:.3f}" cy="{p[f"{key}_center"][1]-top:.3f}" r="{max(3,w/100):.2f}"><title>Proposal {p["confidence"]:.4f}</title></circle>'
                    for p in row[method]["proposals"])
                panels.append(f'<div><h3>{label}</h3><p>{len(row[method]["proposals"])} proposals · {row[method]["crop_count"]} crops</p>'
                              f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label} proposals"><image href="{html.escape(row[method]["image"], quote=True)}" width="{w}" height="{h}"/>'
                              f'<g fill="none" stroke="{colour}" stroke-width="{max(1,w/350):.2f}">{circles}</g></svg></div>')
            cards.append(f'<section><h2>{html.escape(clip["clip"])} · frame {row["frame"]} · ~{row["frame_over_fps_s"]:.3f}s · {html.escape(str(clip["ball_colour"] or "colour unspecified"))}</h2><div class="pair">{"".join(panels)}</div></section>')
    destination.write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Matched native-detail proposal diagnostic</title><style>body{background:#08120c;color:#e7f4e9;font:16px system-ui;margin:24px}p{max-width:1000px;line-height:1.5}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}.pair>div{min-width:0}svg{width:100%;background:#111;max-height:85vh}section{border-top:1px solid #36503b;margin-top:28px}h3{margin-bottom:0}@media(max-width:700px){.pair{grid-template-columns:1fr}}svg:hover{max-height:none}</style>'
        '<h1>Matched native-detail proposal diagnostic</h1><p><strong>Circles are raw detector proposals, not verified balls or tracker selections.</strong> '
        'Both panels use the same decoded source frame, weights and confidence. Yellow uses an image downsampled to a 720-pixel short edge; green uses 384-pixel native tiles over the matched scene region. '
        'Query centres come from the saved 720-pixel tracker, so this tests detector proposals within shared search regions, rather than a native tracker running independently. More circles do not mean better accuracy. No labels, heatmaps, identity verification or outcome decisions are involved. '
        'The native branch uses more inference crops. Regions may differ by less than one native pixel through integer rounding. Hover over a circle for confidence; enlarge the browser to inspect detail. '
        'Times shown are frame/FPS estimates, not verified contact or presentation timestamps.</p>'+"".join(cards), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--baseline-runs", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists():
        raise FileExistsError("Preserve previous results; choose a new probe directory")
    manifest = json.loads(args.cases.read_text(encoding="utf-8"))
    weights_sha = file_sha256(args.weights)
    cases = [prepare_case(case, args.runs, weights_sha) for case in manifest["clips"]]
    if not cases or len({case["clip"] for case in cases}) != len(cases):
        raise ValueError("Expected nonempty unique probe cases")
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    model = YOLO(str(args.weights))
    if model.task != "detect" or model.names != {0: "ball"}:
        raise ValueError("Expected the frozen one-class ball detector")
    args.output.mkdir(parents=True)
    report = {"status": "offline_native_detail_proposal_diagnostic_not_tracking",
              "settings": SETTINGS, "native_crop_size": 384, "native_max_stride": MAX_STRIDE,
              "native_max_batch": MAX_BATCH, "native_dedup": "15_times_min_source_per_processing_scale",
              "weights": str(args.weights.resolve()), "weights_sha256": weights_sha,
              "manifest_sha256": file_sha256(args.cases), "cpu_threads": args.threads,
              "implementation_sha256": file_sha256(Path(__file__)),
              "detector_sha256": file_sha256(Path(__file__).with_name("closed_loop_detector_track.py")),
              "baseline_runs_context_only_not_inference_input": str(args.baseline_runs) if args.baseline_runs else None,
              "limitation": "Predetermined development frames; search regions come from the saved 720-pixel tracker and are shared between branches. Proposal counts are not accuracy or verified identity. Native inference uses additional crops. No annotation is read. This cannot establish colour effects, shot outcomes or end-to-end native tracker performance.",
              "clips": []}
    (args.output / "frozen-config.json").write_text(json.dumps({**report, "cases": manifest["clips"]}, indent=2)+"\n", encoding="utf-8")
    for case in cases:
        result = run_case(case, model, args.output)
        report["clips"].append(result)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    render_review(report, args.output / "index.html")
    print(f"Review: {(args.output / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
