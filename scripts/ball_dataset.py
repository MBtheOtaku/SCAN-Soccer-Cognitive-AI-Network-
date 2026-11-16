"""Offline ball annotation preparation and reviewed-label YOLO export."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "reference-review" / "ball-labels"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_prepare_source(clip_id, source):
    """Give named camera files a stable content ID without copying the video."""
    if source is not None:
        if clip_id is not None:
            raise ValueError("Use either an analysis ID or --source, not both.")
        path = source.resolve()
        if not path.is_file() or not path.is_relative_to((ROOT / "data" / "input").resolve()):
            raise ValueError("--source must be a recording in data/input.")
        sha = file_sha256(path)
        return sha[:12], path, sha
    if clip_id is None or not re.fullmatch(r"[0-9a-f]{12}", clip_id):
        raise ValueError("Use a 12-character SCAN analysis ID or --source.")
    sources = list((ROOT / "data" / "input").glob(clip_id + ".*"))
    if len(sources) != 1:
        raise ValueError("Expected exactly one saved original recording.")
    return clip_id, sources[0], file_sha256(sources[0])


def sample_indices(count, stride, start=None, end=None):
    if count <= 0 or stride <= 0:
        raise ValueError("Frame count and stride must be positive.")
    selected = set(range(0, count, stride))
    if (start is None) != (end is None):
        raise ValueError("Supply both dense window endpoints.")
    if start is not None:
        if not 0 <= start <= end < count:
            raise ValueError("Dense window must be inside the recording.")
        selected.update(range(start, end + 1))
    return sorted(selected)


def validate_labels(manifest, labels):
    if labels.get("schema") != 1 or labels.get("clip_id") != manifest["clip_id"]:
        raise ValueError("Label schema or clip ID does not match.")
    if labels.get("source_sha256") != manifest["source_sha256"]:
        raise ValueError("Labels belong to a different source recording.")
    expected = {f["frame"]: f for f in manifest["frames"]}
    accepted, seen = [], set()
    rows = labels.get("annotations", [])
    if not isinstance(rows, list):
        raise ValueError("Annotations must be a list.")
    for row in rows:
        index = row.get("frame")
        if type(index) is not int or index not in expected or index in seen:
            raise ValueError("Unknown or duplicate frame in labels.")
        seen.add(index)
        state = row.get("state")
        if state not in {"visible", "absent", "uncertain", "unreviewed"}:
            raise ValueError("Unknown annotation state.")
        box = row.get("box")
        if state != "visible":
            if box is not None:
                raise ValueError("Only visible frames may have a box.")
            if state == "absent":
                accepted.append((expected[index], None))
            continue
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Visible ball requires [x1,y1,x2,y2].")
        if not all(type(v) in (int, float) and math.isfinite(v) for v in box):
            raise ValueError("Box coordinates must be finite numbers.")
        x1, y1, x2, y2 = box
        frame = expected[index]
        if not (0 <= x1 < x2 <= frame["width"] and 0 <= y1 < y2 <= frame["height"]):
            raise ValueError("Box must have positive area inside the image.")
        accepted.append((frame, box))
    return accepted


def validate_splits(config):
    groups, clips = {}, set()
    for item in config["clips"]:
        clip = item["clip_id"]
        if not re.fullmatch(r"[0-9a-f]{12}", clip) or clip in clips:
            raise ValueError("Invalid or duplicate clip ID.")
        clips.add(clip)
        split, group = item["split"], item["group"]
        if split not in {"train", "val", "test"} or not isinstance(group, str) or not group.strip():
            raise ValueError("Each clip needs train/val/test and a recording group.")
        if group in groups and groups[group] != split:
            raise ValueError("Recording group appears in more than one split: " + group)
        groups[group] = split
    if not clips:
        raise ValueError("No clips configured.")


def prepare(args):
    import cv2

    clip_id, source, source_sha256 = resolve_prepare_source(args.clip_id, args.source)
    out = DATA / clip_id
    if out.exists():
        raise ValueError("Annotation folder already exists; open it to resume, or use a new clip.")
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open recording.")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid recording FPS.")
        selected = sample_indices(count, args.stride, args.dense_start, args.dense_end)
        (out / "frames").mkdir(parents=True)
        frames, index = [], 0
        # Sequential decoding avoids seek/keyframe ambiguity for the dense window.
        wanted = set(selected)
        while index <= selected[-1]:
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Decode failed at frame {index}; incomplete annotation set.")
            if index in wanted:
                name = f"frames/{index:06d}.jpg"
                if not cv2.imwrite(str(out / name), image, [cv2.IMWRITE_JPEG_QUALITY, 97]):
                    raise OSError("Image write failed.")
                h, w = image.shape[:2]
                frames.append({"frame": index, "time_s": index / fps,
                               "width": w, "height": h, "image": name})
            index += 1
    finally:
        cap.release()
    manifest = {"schema": 1, "clip_id": clip_id,
                "source_sha256": source_sha256,
                "source_name": source.name, "fps": fps, "frames": frames}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    template = (ROOT / "scripts" / "ball_label_ui.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(template.replace("__MANIFEST__", json.dumps(manifest)), encoding="utf-8")
    print(f"Prepared {len(frames)} frames. Open: {out / 'index.html'}")


def export_dataset(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_splits(config)
    out = args.output.resolve()
    if out.exists():
        raise ValueError("Output already exists; choose a new directory to preserve earlier exports.")
    plan, counts, identities = [], {}, {}
    for item in config["clips"]:
        folder = DATA / item["clip_id"]
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
        if manifest["clip_id"] != item["clip_id"]:
            raise ValueError("Manifest clip ID mismatch.")
        digest = manifest["source_sha256"]
        if digest in identities and identities[digest] != item["split"]:
            raise ValueError("Identical source recording appears across splits.")
        identities[digest] = item["split"]
        reviewed = validate_labels(manifest, labels)
        if not reviewed:
            raise ValueError("No exportable reviewed frames for " + item["clip_id"])
        positive = sum(box is not None for _, box in reviewed)
        counts[item["clip_id"]] = {"split": item["split"], "group": item["group"],
                                   "visible": positive, "absent": len(reviewed) - positive,
                                   "excluded": len(manifest["frames"]) - len(reviewed)}
        for frame, box in reviewed:
            source = (folder / frame["image"]).resolve()
            if not source.is_relative_to((folder / "frames").resolve()) or not source.is_file():
                raise ValueError("Missing or invalid frame image.")
            plan.append((item, frame, box, source))
    # Validate every input before writing output.
    for item, frame, box, source in plan:
        split = item["split"]
        images, labels = out / "images" / split, out / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        stem = f'{item["clip_id"]}_{frame["frame"]:06d}'
        shutil.copy2(source, images / (stem + ".jpg"))
        line = ""
        if box is not None:
            x1, y1, x2, y2 = box
            w, h = frame["width"], frame["height"]
            line = f"0 {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}\n"
        (labels / (stem + ".txt")).write_text(line, encoding="utf-8")
    splits = {item["split"] for item in config["clips"]}
    yaml = "path: " + json.dumps(out.as_posix()) + "\n"
    yaml += "".join(f"{s}: images/{s}\n" for s in ("train", "val", "test") if s in splits)
    yaml += "names:\n  0: ball\n"
    (out / "dataset.yaml").write_text(yaml, encoding="utf-8")
    (out / "export_report.json").write_text(json.dumps({"clips": counts}, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print("Exported: " + str(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("clip_id", nargs="?", help="Existing 12-character SCAN analysis ID.")
    prep.add_argument("--source", type=Path, help="Named original recording in data/input.")
    prep.add_argument("--stride", type=int, default=10)
    prep.add_argument("--dense-start", type=int)
    prep.add_argument("--dense-end", type=int)
    export = commands.add_parser("export")
    export.add_argument("--config", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        (prepare if args.command == "prepare" else export_dataset)(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
