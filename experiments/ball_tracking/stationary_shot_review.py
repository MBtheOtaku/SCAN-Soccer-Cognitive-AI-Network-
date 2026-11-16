"""Review a hand-seeded stationary shot with the offline motion-only pilot.

This does not identify contact, verify ball identity, or feed app coaching.
"""

import argparse
import html
import json
import uuid
from pathlib import Path

import cv2

from .pilot_motion_track import output, track


def validate_seed(video: Path, contact_frame: int, box: list[float],
                  goal: list[float], length: int) -> None:
    if not video.is_file():
        raise ValueError(f"Video does not exist: {video}")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if frames < 2:
        raise ValueError("Video must contain at least two readable frames.")
    if contact_frame < 0 or contact_frame >= frames - 1:
        raise ValueError(f"Contact frame must be between 0 and {frames - 2}.")
    if length < 1 or length > 90:
        raise ValueError("Review length must be between 1 and 90 frames.")
    if not (0 <= box[0] < box[2] <= width and
            0 <= box[1] < box[3] <= height):
        raise ValueError(f"Ball box must fit inside the {width}x{height} video.")
    if not (0 <= goal[0] < width and 0 <= goal[1] < height):
        raise ValueError(f"Goal centre must fit inside the {width}x{height} video.")


def render(video: Path, result: dict, destination: Path) -> None:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, result["anchor"])
        ok, seed_image = capture.read()
        if not ok:
            raise ValueError(f"Could not decode seed frame of {video}")
        cv2.rectangle(seed_image,
                      tuple(map(round, result["start_box"][:2])),
                      tuple(map(round, result["start_box"][2:])),
                      (60, 255, 80), 3)
        cv2.drawMarker(seed_image, tuple(map(round, result["goal_center"])),
                       (255, 80, 210), cv2.MARKER_CROSS, 20, 2)
        seed_image = cv2.resize(seed_image, (360, round(
            seed_image.shape[0] * 360 / seed_image.shape[1])))
        if not cv2.imwrite(str(destination / "seed.jpg"), seed_image):
            raise OSError("Could not write seed.jpg")
        rows = result["path"]
        selected = [row for index, row in enumerate(rows)
                    if index < 8 or index % 4 == 0 or index == len(rows) - 1]
        figures = []
        for row in selected:
            frame = row["frame"]
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Could not decode frame {frame} of {video}")
            centre = tuple(round(value) for value in row["center"])
            cv2.circle(image, centre, 12, (0, 220, 255), 3)
            if row["reliability"] == "uncertain":
                cv2.drawMarker(image, centre, (0, 80, 255),
                               cv2.MARKER_TILTED_CROSS, 25, 2)
            cv2.drawMarker(image, tuple(map(round, result["goal_center"])),
                           (255, 80, 210), cv2.MARKER_CROSS, 20, 2)
            image = cv2.resize(image, (360, round(image.shape[0] * 360 / image.shape[1])))
            filename = f"frame-{frame:06d}.jpg"
            if not cv2.imwrite(str(destination / filename), image):
                raise OSError(f"Could not write {filename}")
            state = "observed proposal" if row["observed"] else "predicted gap"
            figures.append(f'<figure><img loading="lazy" src="{filename}">'
                           f'<figcaption>Frame {frame}: {state}; '
                           f'{row["reliability"]}</figcaption></figure>')
    finally:
        capture.release()
    title = html.escape(video.name)
    reason = html.escape(result["reliability"].get("reason") or "none")
    page = ('<!doctype html><meta charset="utf-8"><title>Stationary shot review</title>'
            '<style>body{background:#08120d;color:#e9f3eb;font:16px system-ui;margin:24px}'
            'figure{width:360px;flex:none;margin:0}img{width:100%}.frames{display:flex;'
            'overflow-x:auto;gap:12px}figcaption{font-size:13px}</style>'
            f'<h1>Stationary shot: {title}</h1><p>Yellow circle: motion-only candidate. '
            'Red cross: uncertain path. Pink cross: manually chosen goal centre. '
            'This review starts at a manually chosen contact frame and ball box. '
            'It does not verify contact, ball identity, shot outcome, or physical speed.</p>'
            f'<p>Tracker status: {html.escape(result["status"])}. '
            f'Frames tracked: {result["frames_tracked"]}/{result["frames_attempted"]}. '
            f'Uncertainty reason: {reason}.</p>'
            '<figure><img src="seed.jpg"><figcaption>Manual seed frame: green ball box; '
            'pink goal centre.</figcaption></figure>'
            '<div class="frames">' + ''.join(figures) + '</div>')
    (destination / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--contact-frame", type=int, required=True)
    parser.add_argument("--ball-box", nargs=4, type=float, required=True,
                        metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--goal-center", nargs=2, type=float, required=True,
                        metavar=("X", "Y"))
    parser.add_argument("--length", type=int, default=30)
    args = parser.parse_args()
    video = args.video.resolve()
    try:
        validate_seed(video, args.contact_frame, args.ball_box,
                      args.goal_center, args.length)
    except ValueError as exc:
        parser.error(str(exc))
    destination = output / "stationary-shot" / f"{video.stem}-{uuid.uuid4().hex[:8]}"
    destination.mkdir(parents=True)
    result = track(video.stem, args.contact_frame, tuple(args.goal_center),
                   length=args.length, seed_box=args.ball_box,
                   source_name=str(video), seed_kind="manual_contact_frame",
                   both_signed=True, launch_progress_px_per_frame=12,
                   launch_soft_weight=.1)
    (destination / "trajectory.json").write_text(json.dumps({
        "status": "offline_pilot_not_production", "result": result,
    }, indent=2), encoding="utf-8")
    render(video, result, destination)
    print(destination / "index.html")
    print(f'Path: {result["status"]}; observed {result["observed"]}/'
          f'{result["frames_tracked"]} frames. Review the images before drawing conclusions.')


if __name__ == "__main__":
    main()
