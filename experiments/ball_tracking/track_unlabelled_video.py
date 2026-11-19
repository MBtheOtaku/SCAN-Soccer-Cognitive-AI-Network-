"""Run the frozen offline detector/tracker on ordinary videos without labels.

One manually chosen starting ball centre is the only positional input.
Every decoded frame from that start to the end is processed with the same
detector and association functions used by the prior development experiment.
No labels, contact estimates, outcome claims, or training are involved.
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


def run_video(case: dict, model, weights: Path, destination: Path, threads: int,
              progress_every: int = 15) -> dict:
    import cv2

    case = validate_case(case)
    source, seed, start = case["source"], case["seed"], case["start_frame"]
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
    last, velocity = seed, (0.0, 0.0)
    number, width, height = 0, None, None
    started = time.perf_counter()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if number < start:
                number += 1
                continue
            if width is None:
                height, width = image.shape[:2]
                if not (0 <= seed[0] < width and 0 <= seed[1] < height):
                    raise ValueError(f"Seed {seed} is outside decoded image {width}x{height}")
            elif image.shape[:2] != (height, width):
                raise ValueError("Decoded image dimensions changed within video")
            predicted = (last[0] + velocity[0], last[1] + velocity[1])
            proposals, count = detect_slices(model, image, predicted,
                                             SETTINGS["crop_size"], SETTINGS["slice_offset"],
                                             SETTINGS["confidence"])
            row = {"clip": case["clip"], "frame": number,
                   "split": "unseen_session_unlabelled", "crop_center": predicted,
                   "crop_count": count, "candidate_count": len(proposals), "candidates": proposals}
            selected, last, velocity = associate_next(row, last, velocity)
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
              "split": "unseen_session_unlabelled", "source_name": str(source),
              "source_sha256": file_sha256(source), "fps": fps,
              "width": width, "height": height,
              "start_frame": start, "end_frame": path[-1]["frame"],
              "seed": seed, "seed_kind": "one_manual_starting_position_no_reseeding",
              "reported_video_frames": reported_frames, "decoded_video_frames": number,
              "reached_reported_end": number >= reported_frames,
              "weights": str(weights.resolve()), "weights_sha256": file_sha256(weights),
              "settings": SETTINGS, "cpu_threads": threads,
              "implementation_sha256": {
                  "detector": file_sha256(ROOT / "experiments/ball_tracking/closed_loop_detector_track.py"),
                  "association": file_sha256(ROOT / "experiments/ball_tracking/associate_detector_candidates.py")},
              "elapsed_s": time.perf_counter() - started,
              "frames_directory": str(frames_dir.resolve()),
              "processed_frames": len(path),
              "selected_detection_frames": sum(item["detected"] for item in path),
              "frames_without_selected_detection": sum(not item["detected"] for item in path),
              "limitation": "Unlabelled, manually seeded unseen-session run. Selected detection counts are NOT accuracy or verified target identity. Missing predictions are not observations. Decode completion is compared with container frame count; shot outcome and contact timing are unverified.",
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
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
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
              "cases": [{"source": str(case["source"]), "source_sha256": file_sha256(case["source"]),
                         "clip": case["clip"], "seed": case["seed"], "start_frame": case["start_frame"]}
                        for case in validated],
              "runner_sha256": file_sha256(Path(__file__)),
              "detector_sha256": file_sha256(ROOT / "experiments/ball_tracking/closed_loop_detector_track.py"),
              "association_sha256": file_sha256(ROOT / "experiments/ball_tracking/associate_detector_candidates.py")}
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        checked = validate_case(case)
        run_video(case, model, args.weights, args.output / checked["clip"], args.threads)


if __name__ == "__main__":
    main()
