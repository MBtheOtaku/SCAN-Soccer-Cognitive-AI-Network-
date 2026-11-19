"""Offline, proposal-only contrast/sharpening check on frozen review frames.

This reuses each frozen run's crop centre but never updates a track, reads labels,
or changes the baseline detector. Saved JPEGs are used for both sides of the
comparison, so results are diagnostic and not directly comparable to detections
made on the original decoded video buffer.
"""

import argparse
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from experiments.ball_tracking.closed_loop_detector_track import detect_slices
from scripts.ball_dataset import file_sha256

ROOT = Path(__file__).resolve().parents[2]
FROZEN_SETTINGS = {"crop_size": 384, "slice_offset": 128, "confidence": 0.05,
                   "class_id": 0, "inference_size": 640, "device": "cpu"}
SAMPLE_FRAMES = {
    "rep-1795033104216-okonr73x": (66, 67, 68, 69, 70),
    "rep-1795033176858-b9rlx186": (78, 79, 80, 81),
    "rep-1795033883557-f5y3mv5e": (100, 101, 102, 103, 104, 105, 113, 114, 116, 120),
}
CLAHE_SETTINGS = {"color_space": "LAB", "channel": "L", "clip_limit": 1.5,
                  "tile_grid": [8, 8]}
UNSHARP_SETTINGS = {"amount": 0.35, "gaussian_sigma": 1.0,
                    "limitation": "Conservative edge enhancement, not motion deblurring"}


def mild_unsharp(image: np.ndarray) -> np.ndarray:
    """Enhance existing edges without claiming to restore lost motion detail."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an 8-bit BGR frame")
    blurred = cv2.GaussianBlur(image, (0, 0), UNSHARP_SETTINGS["gaussian_sigma"])
    amount = UNSHARP_SETTINGS["amount"]
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def positive_motion_overlay(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Diagnostic brightening map after estimated global camera translation.

    This is not a learned ball heatmap, a target probability or a detector input.
    Keep the original intact; unreliable alignment produces no overlay.
    """
    if previous.shape != current.shape or current.dtype != np.uint8 or previous.dtype != np.uint8:
        raise ValueError("Expected equal-sized 8-bit BGR frames")
    if current.ndim != 3 or current.shape[2] != 3:
        raise ValueError("Expected equal-sized 8-bit BGR frames")
    height, width = current.shape[:2]
    before = cv2.GaussianBlur(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    after = cv2.GaussianBlur(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    shift, response = cv2.phaseCorrelate(before.copy(), after.copy(),
                                        cv2.createHanningWindow((width, height), cv2.CV_32F))
    info = {"method": "translation_aligned_positive_luminance_change",
            "shift_px": list(shift), "alignment_response_not_probability": response,
            "status": "usable_for_visual_diagnosis_only"}
    if (not all(math.isfinite(v) for v in (*shift, response)) or response < 0.1
            or math.hypot(*shift) > min(width, height) * 0.1):
        info["status"] = "unavailable_unreliable_global_alignment"
        # JSON must not contain NaN even for a textureless frame.
        info["shift_px"] = [v if math.isfinite(v) else None for v in shift]
        info["alignment_response_not_probability"] = response if math.isfinite(response) else None
        return None, info
    transform = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    aligned = cv2.warpAffine(before, transform, (width, height))
    valid = cv2.warpAffine(np.ones((height, width), np.uint8), transform, (width, height))
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8)) > 0
    change = np.maximum(after - aligned, 0)
    change[(change < 6) | ~valid] = 0
    intensity = np.clip(change * 4, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    overlay = current.copy()
    active = intensity > 0
    blended = cv2.addWeighted(current, 0.45, heat, 0.55, 0)
    overlay[active] = blended[active]
    return overlay, info


def luminance_clahe(image: np.ndarray) -> np.ndarray:
    """Apply mild local contrast to luminance, preserving input size and type."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an 8-bit BGR frame")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, chroma_a, chroma_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=CLAHE_SETTINGS["clip_limit"],
                                tileGridSize=tuple(CLAHE_SETTINGS["tile_grid"])).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, chroma_a, chroma_b)),
                        cv2.COLOR_LAB2BGR)


def probe_frame(model, image: np.ndarray, center: tuple[float, float]) -> tuple[dict, dict[str, np.ndarray]]:
    """Run exactly the same detector/crop geometry on all image variants."""
    enhanced = luminance_clahe(image)
    sharpened = mild_unsharp(image)
    settings = FROZEN_SETTINGS
    original, original_crops = detect_slices(
        model, image, center, settings["crop_size"], settings["slice_offset"],
        settings["confidence"])
    contrast, contrast_crops = detect_slices(
        model, enhanced, center, settings["crop_size"], settings["slice_offset"],
        settings["confidence"])
    unsharp, unsharp_crops = detect_slices(
        model, sharpened, center, settings["crop_size"], settings["slice_offset"],
        settings["confidence"])
    if len({original_crops, contrast_crops, unsharp_crops}) != 1:
        raise AssertionError("The variants used different crop geometry")
    return {"crop_center": list(center), "crop_count": original_crops,
            "original": original, "clahe": contrast, "unsharp": unsharp}, {"clahe": enhanced, "unsharp": sharpened}


def _validated_run(report_path: Path, weights_hash: str, detector_hash: str,
                   frames: tuple[int, ...]) -> tuple[dict, dict[int, dict], dict[int, dict]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["weights_sha256"] != weights_hash:
        raise ValueError(f"Weights do not match frozen run: {report_path}")
    if report["implementation_sha256"]["detector"] != detector_hash:
        raise ValueError(f"Detector implementation differs from frozen run: {report_path}")
    if report["settings"] != FROZEN_SETTINGS:
        raise ValueError(f"Detector settings differ from frozen run: {report_path}")
    rows = {int(row["frame"]): row for row in report["rows"]}
    files = {int(item["frame"]): item for item in report["frame_files"]}
    if len(rows) != len(report["rows"]) or len(files) != len(report["frame_files"]):
        raise ValueError(f"Duplicate frame in frozen run: {report_path}")
    if any(frame not in rows or frame not in files for frame in frames):
        raise ValueError(f"Requested frame missing from frozen run: {report_path}")
    return report, rows, files


def _proposal_svg(proposals: list[dict], color: str) -> str:
    circles = []
    for proposal in proposals:
        x, y = proposal["center"]
        confidence = proposal["confidence"]
        circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="12" '
                       f'fill="none" stroke="{color}" stroke-width="3">'
                       f'<title>proposal confidence {confidence:.3f}</title></circle>')
    return "".join(circles)


def _panel(image_path: Path, output: Path, row: dict, variant: str,
           width: int, height: int) -> str:
    relative = quote(Path(os.path.relpath(image_path, output)).as_posix(), safe="/.")
    color = {"original": "#f5dd31", "clahe": "#39e0d0", "unsharp": "#e692ff"}[variant]
    center_x, center_y = row["crop_center"]
    proposals = row[variant]
    circles = _proposal_svg(proposals, color)
    return (f'<section class="panel"><strong>{html.escape(variant)} · '
            f'{len(proposals)} proposals</strong><div class="image">'
            f'<img src="{relative}" alt="{html.escape(variant)} frame">'
            f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
            f'<path d="M {center_x - 12:.1f} {center_y:.1f} h 24 '
            f'M {center_x:.1f} {center_y - 12:.1f} v 24" '
            f'stroke="white" stroke-width="2"/>{circles}</svg></div></section>')


def _write_review(rows: list[dict], output: Path) -> None:
    cards = []
    for row in rows:
        clip, frame = row["clip"], row["frame"]
        original_path = Path(row["original_image"])
        width, height = row["width"], row["height"]
        cards.append(f'<article><h2>{html.escape(clip)} · frame {frame}</h2>'
                     '<div class="pair">' + ''.join(_panel(Path(row[f'{variant}_image']), output, row, variant, width, height)
                                                   for variant in ('original', 'clahe', 'unsharp'))
                     + (f'<section class="panel"><strong>Positive motion · diagnostic only</strong>'
                        f'<div class="image"><img src="{quote(Path(os.path.relpath(row["motion_image"], output)).as_posix(), safe="/.")}" '
                        f'alt="Brightening after estimated camera alignment; not ball identity"></div></section>'
                        if row.get("motion_image") else '<section class="panel"><p>Motion map unavailable: unreliable camera alignment.</p></section>')
                     + '</div></article>')
    page = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
            "<title>SCAN offline contrast and sharpening check</title><style>"
            "body{font:16px system-ui;background:#08110d;color:#e9f4ec;margin:2rem}"
            "article{border-top:1px solid #335046;padding:1.5rem 0}"
            ".pair{display:flex;gap:1rem;flex-wrap:wrap}.panel{width:min(43vw,360px)}"
            ".image{position:relative}.image img{width:100%;display:block}"
            ".image svg{position:absolute;inset:0;width:100%;height:100%}"
            "small{color:#b6c9bd}</style><h1>Original vs contrast vs mild sharpening</h1>"
            "<p>Same saved JPEG, frozen crop centre, detector weights and threshold."
            " White cross = crop centre; yellow/cyan/purple circles = raw detector proposals."
            " This is a proposal-only diagnostic, not a corrected track or accuracy test.</p>"
            "<p>Unsharp is conservative edge enhancement, not recovery of detail lost to motion blur."
            " The motion panel highlights positive luminance changes between neighbouring frames"
            " after estimated global camera translation. Players, shadows and exposure changes can"
            " also light up. It is not a learned ball heatmap and never feeds the detector or coaching.</p>"
            + "".join(cards) + "</html>")
    (output / "index.html").write_text(page, encoding="utf-8")


def run(runs_dir: Path, weights: Path, output: Path, model,
        samples: dict[str, tuple[int, ...]] = SAMPLE_FRAMES) -> dict:
    if output.exists():
        raise FileExistsError(f"Choose a new output folder to preserve the previous run: {output}")
    weights_hash = file_sha256(weights)
    detector_hash = file_sha256(ROOT / "experiments/ball_tracking/closed_loop_detector_track.py")
    checked = {}
    for clip, frames in samples.items():
        if not frames or len(frames) != len(set(frames)) or any(frame < 0 for frame in frames):
            raise ValueError(f"Invalid frame list: {clip}")
        checked[clip] = _validated_run(runs_dir / clip / "report.json",
                                       weights_hash, detector_hash, frames)
    output.mkdir(parents=True)
    for variant in ('clahe', 'unsharp'):
        (output / variant).mkdir()
    (output / "motion").mkdir()
    results = []
    for clip, frames in samples.items():
        report, source_rows, files = checked[clip]
        for frame in frames:
            original_path = Path(report["frames_directory"]) / files[frame]["image"]
            image = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable frozen review frame: {original_path}")
            if image.shape[:2] != (report["height"], report["width"]):
                raise ValueError(f"Unexpected frame size: {original_path}")
            center = tuple(float(value) for value in source_rows[frame]["crop_center"])
            if len(center) != 2 or not all(map(math.isfinite, center)):
                raise ValueError(f"Invalid frozen crop centre: {clip} {frame}")
            comparison, variants = probe_frame(model, image, center)
            previous_info = files.get(frame - 1)
            motion_image, motion_info, motion_path = None, {"status": "unavailable_no_previous_frame"}, None
            if previous_info:
                previous = cv2.imread(str(Path(report["frames_directory"]) / previous_info["image"]), cv2.IMREAD_COLOR)
                if previous is not None:
                    motion_image, motion_info = positive_motion_overlay(previous, image)
            if motion_image is not None:
                motion_path = output / "motion" / f"{clip}-{frame:06d}.jpg"
                if not cv2.imwrite(str(motion_path), motion_image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Could not save diagnostic motion map: {motion_path}")
            variant_paths = {}
            for variant, enhanced in variants.items():
                variant_path = output / variant / f"{clip}-{frame:06d}.jpg"
                if not cv2.imwrite(str(variant_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Could not save enhanced review frame: {variant_path}")
                variant_paths[f'{variant}_image'] = str(variant_path.resolve())
            results.append({"clip": clip, "frame": frame, "width": report["width"],
                            "height": report["height"],
                            "original_image": str(original_path.resolve()),
                            **variant_paths,
                            "motion_image": str(motion_path.resolve()) if motion_path else None,
                            "motion_diagnostic": motion_info, **comparison})
            print(f"{clip} frame {frame}: original {len(comparison['original'])}, "
                  f"CLAHE {len(comparison['clahe'])}, unsharp {len(comparison['unsharp'])} proposals", flush=True)
    result = {"status": "offline_proposal_only_not_production",
              "source": "frozen new-session runs, same saved JPEG for both variants",
              "weights": str(weights.resolve()), "weights_sha256": weights_hash,
              "detector_sha256": detector_hash, "settings": FROZEN_SETTINGS,
              "clahe": CLAHE_SETTINGS, "unsharp": UNSHARP_SETTINGS,
              "probe_sha256": file_sha256(Path(__file__)), "sample_count": len(results),
              "limitation": "Frozen baseline crop centres are reused without association or reseeding. Saved JPEGs differ from the source video buffers used by the baseline run. These unlabelled proposal counts are not detection accuracy, verified ball identity, or shot outcome.",
              "rows": results}
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                        encoding="utf-8")
    _write_review(results, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    model = YOLO(str(args.weights))
    if model.task != "detect" or model.names != {0: "ball"}:
        raise ValueError("Expected the frozen one-class ball detector")
    result = run(args.runs, args.weights, args.output, model)
    print(f"Saved {result['sample_count']} proposal-only comparisons to "
          f"{(args.output / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
