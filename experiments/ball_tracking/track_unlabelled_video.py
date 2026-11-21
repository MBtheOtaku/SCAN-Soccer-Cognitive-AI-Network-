"""Run the frozen offline detector/tracker on ordinary videos without labels.

One manually chosen starting ball centre is the only positional input.
Every decoded frame from that start to the end is processed with the same
detector. Legacy association is the default; temporal association is an
optional inspected-session development experiment, with opt-in local recovery.
No labels, contact
estimates, outcome claims, or training are involved.
"""

import argparse
import json
import math
import time
from pathlib import Path

from experiments.ball_tracking.associate_detector_candidates import associate_next
from experiments.ball_tracking.closed_loop_detector_track import detect_slices
from scripts.ball_dataset import file_sha256

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = {"crop_size": 384, "slice_offset": 128, "confidence": 0.05,
            "class_id": 0, "inference_size": 640, "device": "cpu"}
ASSOCIATION_FILES = {
    "legacy": "associate_detector_candidates.py",
    "temporal": "temporal_identity.py",
    "recovery": "recovery_identity.py",
    "motion-recovery": "motion_recovery_identity.py",
}


def association_hashes(method: str) -> dict:
    hashes = {"association": file_sha256(Path(__file__).with_name(ASSOCIATION_FILES[method]))}
    if method in ("recovery", "motion-recovery"):
        hashes["base_association"] = file_sha256(Path(__file__).with_name("temporal_identity.py"))
    if method == "motion-recovery":
        hashes["recovery_association"] = file_sha256(Path(__file__).with_name("recovery_identity.py"))
    return hashes


def validate_case(case: dict) -> dict:
    source = Path(case["source"]).resolve()
    seed = tuple(float(value) for value in case["seed"])
    start = int(case.get("start_frame", 0))
    if not source.is_file() or len(seed) != 2 or not all(map(math.isfinite, seed)) or start < 0:
        raise ValueError("Expected an existing video, finite XY seed and non-negative start frame")
    if isinstance(case.get("start_frame", 0), float) and case["start_frame"] != start:
        raise ValueError("Start frame must be an integer")
    return {"source": source, "seed": seed, "start_frame": start,
            "clip": str(case.get("clip", source.stem))}


def prepare_frame(image, processing_short_edge: int | None = None):
    """Optional downsampling before all perception; never manufacture detail."""
    if (processing_short_edge is not None
            and (isinstance(processing_short_edge, bool)
                 or not isinstance(processing_short_edge, int) or processing_short_edge <= 0)):
        raise ValueError("Processing short edge must be a positive integer")
    source_height, source_width = image.shape[:2]
    if processing_short_edge is not None and processing_short_edge > min(source_width, source_height):
        raise ValueError("Cannot upscale a source to the requested processing short edge")
    ratio = (processing_short_edge / min(source_width, source_height)
             if processing_short_edge is not None else 1.0)
    width, height = round(source_width * ratio), round(source_height * ratio)
    transformed = (width, height) != (source_width, source_height)
    if transformed:
        import cv2
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image, {"source_width": source_width, "source_height": source_height,
                   "width": width, "height": height,
                   "scale_x": width / source_width, "scale_y": height / source_height,
                   "status": "downsampled" if transformed else "native",
                   "interpolation": "INTER_AREA" if transformed else "none"}


def run_video(case: dict, model, weights: Path, destination: Path, threads: int,
              progress_every: int = 15, association: str = "legacy",
              processing_short_edge: int | None = None,
              native_detail_fallback: bool = False) -> dict:
    import cv2

    if association not in ASSOCIATION_FILES:
        raise ValueError("Choose legacy, temporal, recovery or motion-recovery association")
    if native_detail_fallback:
        if processing_short_edge != 720 or association != "motion-recovery":
            raise ValueError("Native fallback requires motion-recovery and 720-short-edge processing")
        from experiments.ball_tracking.native_detail_fallback import augment_native_detail, SETTINGS as NATIVE_SETTINGS
        from experiments.ball_tracking.native_evidence_verifier import NativeEvidenceVerifier
    if association != "legacy":
        from experiments.ball_tracking.temporal_identity import TemporalIdentity, estimate_camera_translation
        if association == "recovery":
            from experiments.ball_tracking.recovery_identity import RecoveryIdentity
        elif association == "motion-recovery":
            from experiments.ball_tracking.motion_recovery_identity import MotionRecoveryIdentity
    case = validate_case(case)
    source, seed, start = case["source"], case["seed"], case["start_frame"]
    source_seed = seed
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "report.json"
    if report_path.exists():
        raise FileExistsError(f"Preserve previous results; choose a new output: {report_path}")
    frames_dir = destination / "frames"
    frames_dir.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Invalid reported FPS: {source}")
    # Decode sequentially; frame seeking can be inaccurate for phone videos.
    rows, path, frame_files = [], [], []
    accepted_frames = []
    last, velocity = seed, (0.0, 0.0)
    tracker, previous, native_verifier = None, None, None
    split = "unseen_session_unlabelled" if association == "legacy" else "inspected_session_unlabelled"
    number, width, height = 0, None, None
    frame_transform = None
    started = time.perf_counter()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if number < start:
                number += 1
                continue
            source_height, source_width = image.shape[:2]
            if frame_transform is not None and (source_width, source_height) != (
                    frame_transform["source_width"], frame_transform["source_height"]):
                raise ValueError("Decoded source image dimensions changed within video")
            source_image = image
            image, current_transform = prepare_frame(image, processing_short_edge)
            if width is None:
                frame_transform = current_transform
                height, width = image.shape[:2]
                if not (0 <= source_seed[0] < source_width and 0 <= source_seed[1] < source_height):
                    raise ValueError(f"Seed {source_seed} is outside decoded source image {source_width}x{source_height}")
                seed = tuple(source_seed[i] * frame_transform[("scale_x", "scale_y")[i]]
                             for i in range(2))
                if native_detail_fallback:
                    native_verifier = NativeEvidenceVerifier(source_image, image, source_seed)
                last, velocity = seed, (0.0, 0.0)
                if not (0 <= seed[0] < width and 0 <= seed[1] < height):
                    raise ValueError(f"Seed {seed} is outside decoded image {width}x{height}")
                if association != "legacy":
                    tracker_class = (MotionRecoveryIdentity if association == "motion-recovery" else
                                     RecoveryIdentity if association == "recovery" else TemporalIdentity)
                    tracker = tracker_class(seed, scale=min(width, height) / 720.0)
            elif image.shape[:2] != (height, width):
                raise ValueError("Decoded image dimensions changed within video")
            shift, camera_motion = (0.0, 0.0), {"status": "first_processed_frame"}
            if tracker is not None and previous is not None:
                shift, camera_motion = estimate_camera_translation(previous, image)
            predicted = tracker.predict(shift) if tracker is not None else (last[0] + velocity[0], last[1] + velocity[1])
            proposals, count = detect_slices(model, image, predicted,
                                             SETTINGS["crop_size"], SETTINGS["slice_offset"],
                                             SETTINGS["confidence"])
            native_evidence = None
            native_verification = None
            if native_detail_fallback:
                proposals, native_evidence = augment_native_detail(
                    model, source_image, frame_transform, predicted, proposals)
                proposals, native_verification = native_verifier.filter(
                    number, source_image, image, proposals, accepted_frames=accepted_frames)
            row = {"clip": case["clip"], "frame": number,
                   "split": split, "crop_center": predicted,
                   "crop_count": count, "candidate_count": len(proposals), "candidates": proposals}
            if native_evidence is not None:
                row["native_detail"] = native_evidence
                row["native_evidence_verification"] = native_verification
            if tracker is None:
                selected, last, velocity = associate_next(row, last, velocity)
            else:
                row["camera_motion"] = camera_motion
                selected = tracker.update(row, shift)
                previous = image.copy()
            if selected["detected"]:
                accepted_frames = (accepted_frames + [number])[-2:]
            filename = f"{number:06d}.jpg"
            if not cv2.imwrite(str(frames_dir / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Could not write review frame {number}")
            rows.append(row)
            path.append(selected)
            frame_files.append({"frame": number, "image": filename, "time_s": number / fps,
                                "width": width, "height": height})
            number += 1
            if len(path) == 1 or len(path) % progress_every == 0:
                print(f"{case['clip']}: frame {number - 1}/{reported_frames - 1}; "
                      f"{'candidate selected' if selected['detected'] else 'no selected detection'}",
                      flush=True)
    finally:
        capture.release()
    if not path:
        raise ValueError(f"No readable frames at or after seed frame {start}: {source}")
    report = {"status": "offline_pilot_not_production", "clip": case["clip"],
              "split": split, "source_name": str(source),
              "source_sha256": file_sha256(source), "fps": fps,
              "width": width, "height": height,
              "start_frame": start, "end_frame": path[-1]["frame"],
              "seed": seed, "seed_kind": "one_manual_starting_position_no_reseeding",
              "source_seed": source_seed, "frame_transform": frame_transform,
              "processing_short_edge": processing_short_edge,
              "native_detail_fallback": native_detail_fallback,
              "native_detail_settings": NATIVE_SETTINGS if native_detail_fallback else None,
              "native_verification_settings": native_verifier.configuration() if native_verifier is not None else None,
              "native_detail_implementation_sha256": ({
                  "fallback": file_sha256(Path(__file__).with_name("native_detail_fallback.py")),
                  "native_detector": file_sha256(Path(__file__).with_name("native_box_detector.py")),
                  "native_geometry_helper": file_sha256(Path(__file__).with_name("probe_native_detail.py")),
                  "verification": file_sha256(Path(__file__).with_name("native_evidence_verifier.py")),
              } if native_detail_fallback else None),
              "reported_video_frames": reported_frames, "decoded_video_frames": number,
              "reached_reported_end": number >= reported_frames,
              "weights": str(weights.resolve()), "weights_sha256": file_sha256(weights),
              "settings": SETTINGS, "cpu_threads": threads,
              "association_method": association,
              "association_settings": tracker.configuration() if tracker is not None else None,
              "implementation_sha256": {
                  "detector": file_sha256(ROOT / "experiments/ball_tracking/closed_loop_detector_track.py"),
                  **association_hashes(association)},
              "elapsed_s": time.perf_counter() - started,
              "frames_directory": str(frames_dir.resolve()),
              "processed_frames": len(path),
              "selected_detection_frames": sum(item["detected"] for item in path),
              "frames_without_selected_detection": sum(not item["detected"] for item in path),
              "limitation": ("Inspected development, manually seeded unlabelled run. "
                             if association != "legacy" else
                             "Unlabelled, manually seeded unseen-session run. ")
                            + "Selected detection counts are NOT accuracy or verified target identity. Missing predictions are not observations. Decode completion is compared with container frame count; shot outcome and contact timing are unverified.",
              "rows": rows, "path": path, "frame_files": frame_files}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {report_path}: {len(path)} processed; "
          f"{report['selected_detection_frames']} selected (not accuracy)", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--video", type=Path)
    sources.add_argument("--cases", type=Path, help="JSON with clips: [{source, seed, start_frame, clip}]")
    parser.add_argument("--seed", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--processing-short-edge", type=int,
                        help="Optional downsample before inference; seeds remain decoded source XY. Never upscales.")
    parser.add_argument("--association", choices=tuple(ASSOCIATION_FILES), default="legacy",
                        help="Temporal, recovery and motion-recovery are inspected-session offline identity experiments")
    parser.add_argument("--native-detail-fallback", action="store_true",
                        help="Offline native crops with background/temporal checks; requires motion-recovery and processing short edge 720")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.processing_short_edge is not None and args.processing_short_edge < 1:
        parser.error("--processing-short-edge must be positive")
    if args.native_detail_fallback and (args.processing_short_edge != 720 or args.association != "motion-recovery"):
        parser.error("--native-detail-fallback requires --processing-short-edge 720 --association motion-recovery")
    if args.video and args.seed is None:
        parser.error("--video requires --seed")
    cases = (json.loads(args.cases.read_text(encoding="utf-8"))["clips"] if args.cases else
             [{"source": str(args.video), "seed": args.seed, "start_frame": args.start_frame}])
    # Validate all sources and freeze provenance before the first inference.
    validated = [validate_case(case) for case in cases]
    if len({case["clip"] for case in validated}) != len(validated):
        parser.error("Clip identifiers must be unique")
    for case in validated:
        if Path(case["clip"]).name != case["clip"] or case["clip"] in (".", ".."):
            parser.error("Clip identifiers must be safe directory names")
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    model = YOLO(str(args.weights))
    if model.task != "detect" or model.names != {0: "ball"}:
        raise ValueError("Expected the frozen one-class ball detector")
    args.output.mkdir(parents=True, exist_ok=True)
    freeze_path = args.output / "frozen-config.json"
    if freeze_path.exists():
        raise FileExistsError("Output already contains frozen results; choose a new directory")
    freeze = {"weights": str(args.weights.resolve()), "weights_sha256": file_sha256(args.weights),
              "settings": SETTINGS, "cpu_threads": args.threads,
              "processing_short_edge": args.processing_short_edge,
              "native_detail_fallback": args.native_detail_fallback,
              "input_seed_coordinates": "decoded_source_pixels_before_optional_downsample",
              "association_method": args.association,
              "cases": [{"source": str(case["source"]), "source_sha256": file_sha256(case["source"]),
                         "clip": case["clip"], "seed": case["seed"], "start_frame": case["start_frame"]}
                        for case in validated],
              "runner_sha256": file_sha256(Path(__file__)),
              "detector_sha256": file_sha256(ROOT / "experiments/ball_tracking/closed_loop_detector_track.py"),
              **{f"{key}_sha256": value for key, value in association_hashes(args.association).items()}}
    if args.native_detail_fallback:
        from experiments.ball_tracking.native_detail_fallback import SETTINGS as NATIVE_SETTINGS
        from experiments.ball_tracking.native_evidence_verifier import SETTINGS as VERIFICATION_SETTINGS
        freeze["native_detail_settings"] = NATIVE_SETTINGS
        freeze["native_verification_settings"] = VERIFICATION_SETTINGS
        freeze["native_detail_implementation_sha256"] = {
            "fallback": file_sha256(Path(__file__).with_name("native_detail_fallback.py")),
            "native_detector": file_sha256(Path(__file__).with_name("native_box_detector.py")),
            "native_geometry_helper": file_sha256(Path(__file__).with_name("probe_native_detail.py")),
            "verification": file_sha256(Path(__file__).with_name("native_evidence_verifier.py"))}
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        checked = validate_case(case)
        run_video(case, model, args.weights, args.output / checked["clip"], args.threads,
                  association=args.association, processing_short_edge=args.processing_short_edge,
                  native_detail_fallback=args.native_detail_fallback)


if __name__ == "__main__":
    main()
