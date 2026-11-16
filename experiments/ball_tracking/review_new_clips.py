"""Create contact sheets to choose review seeds in recordings without labels."""

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data/reference-review/new-ball-clips"


def contact_sheet(clip_id, start_frame=None, step=1, save_frames=False):
    source = ROOT / "data/input" / f"{clip_id}.mov"
    if not source.is_file():
        source = source.with_suffix(".mp4")
    if not source.is_file():
        raise FileNotFoundError(source)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {source}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = (np.linspace(0, max(0, count-1), 16, dtype=int)
              if start_frame is None else
              np.arange(start_frame, min(count, start_frame+16*step), step))
    DEST.mkdir(parents=True, exist_ok=True)
    tiles = []
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = cap.read()
        if not ok:
            continue
        if save_frames:
            if not cv2.imwrite(str(DEST / f"{clip_id}-frame-{frame:04d}.jpg"), image):
                raise OSError(f"Cannot save frame {frame}")
        height = round(image.shape[0] * 240 / image.shape[1])
        image = cv2.resize(image, (240, height))
        tile = np.full((height+32, 240, 3), 18, dtype=np.uint8)
        tile[:height] = image
        cv2.putText(tile, f"frame {frame}, {frame/fps:.2f}s", (6, height+23),
                    cv2.FONT_HERSHEY_SIMPLEX, .52, (245, 245, 245), 1)
        tiles.append(tile)
    cap.release()
    if not tiles:
        raise ValueError(f"No frames from {source}")
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 18))
    sheet = np.vstack([np.hstack(tiles[index:index+4])
                       for index in range(0, len(tiles), 4)])
    suffix = "contact-sheet" if start_frame is None else f"frames-{start_frame}-{step}"
    path = DEST / f"{clip_id}-{suffix}.jpg"
    if not cv2.imwrite(str(path), sheet):
        raise OSError(path)
    print(f"{clip_id}: {count} frames at {fps:.2f} fps; {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_ids", nargs="+")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--save-frames", action="store_true")
    args = parser.parse_args()
    if args.step < 1 or (args.start_frame is not None and args.start_frame < 0):
        parser.error("Frame selection must be nonnegative and step positive")
    for clip_id in args.clip_ids:
        if not clip_id.isalnum():
            parser.error("clip IDs must be alphanumeric")
        contact_sheet(clip_id, args.start_frame, args.step, args.save_frames)


if __name__ == "__main__":
    main()
