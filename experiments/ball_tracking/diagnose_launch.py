"""Inspect motion proposals during the manually seeded launch interval."""

import json
import math

import cv2

from .pilot_motion_track import align_lk, anchors, candidates, label_root, root


def main():
    for clip_id, (anchor, goal) in anchors.items():
        if clip_id == "ee73dffd61b5":
            continue
        folder = label_root / clip_id
        manifest = json.loads((folder / "manifest.json").read_text())
        labels = {a["frame"]: a for a in json.loads(
            (folder / "labels.json").read_text())["annotations"]}
        seed_box = labels[anchor]["box"]
        seed = ((seed_box[0] + seed_box[2]) / 2,
                (seed_box[1] + seed_box[3]) / 2)
        capture = cv2.VideoCapture(str(root / "data/input" / manifest["source_name"]))
        capture.set(cv2.CAP_PROP_POS_FRAMES, anchor)
        ok, previous = capture.read()
        if not ok:
            raise ValueError(clip_id)
        print(clip_id)
        for frame in range(anchor + 1, anchor + 11):
            ok, current = capture.read()
            if not ok:
                break
            aligned = align_lk(previous, current)
            label = labels.get(frame)
            if label and label["state"] == "visible":
                box = label["box"]
                truth = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                parts = []
                for positive in (False, True):
                    proposals = candidates(aligned, current, positive)
                    nearby = sorted(((math.dist(x["center"], truth), x)
                                     for x in proposals), key=lambda pair: pair[0])
                    best = nearby[0] if nearby else None
                    parts.append(f'{"positive" if positive else "absolute"}: '
                                 f'{len(proposals)} proposals, '
                                 f'nearest {best[0]:.1f}px area {best[1]["area"]}'
                                 if best else f'{"positive" if positive else "absolute"}: none')
                progress = math.dist(goal, seed) - math.dist(goal, truth)
                print(frame - anchor, f"goal progress {progress:.1f}px", " | ".join(parts))
            previous = current
        capture.release()


if __name__ == "__main__":
    main()
