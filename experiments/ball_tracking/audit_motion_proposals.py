"""Measure whether raw motion proposals contain the reviewed ball at all.

This is an offline diagnostic, not a tracker or an accuracy estimate on new
recordings. It uses labels only for scoring, never to generate proposals.
"""

import argparse
import json
import math
from pathlib import Path

import cv2

from scripts.ball_dataset import file_sha256, validate_labels

from .pilot_motion_track import align_lk, anchors, candidates, label_root, output, root


def nearest_distance(proposals: list[dict], truth: tuple[float, float]) -> float | None:
    return min((math.dist(proposal["center"], truth) for proposal in proposals),
               default=None)


def audit_clip(clip_id: str) -> dict:
    folder = label_root / clip_id
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    validate_labels(manifest, labels)
    source = root / "data" / "input" / manifest["source_name"]
    if file_sha256(source) != manifest["source_sha256"]:
        raise ValueError(f"Source hash changed for {clip_id}")
    visible = {row["frame"]: row for row in labels["annotations"]
               if row["state"] == "visible" and row["frame"] > 0}
    if not visible:
        raise ValueError(f"No reviewed visible ball frames for {clip_id}")
    capture = cv2.VideoCapture(str(source))
    rows = []
    try:
        start, end = min(visible), max(visible)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
        ok, previous = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode frame {start - 1} of {source}")
        for frame in range(start, end + 1):
            ok, current = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {frame} of {source}")
            label = visible.get(frame)
            if label is not None:
                aligned = align_lk(previous, current)
                positive = candidates(aligned, current, positive=True)
                negative = candidates(aligned, current, negative=True)
                x1, y1, x2, y2 = label["box"]
                truth = ((x1 + x2) / 2, (y1 + y2) / 2)
                pos_distance = nearest_distance(positive, truth)
                neg_distance = nearest_distance(negative, truth)
                distances = [d for d in (pos_distance, neg_distance) if d is not None]
                rows.append({
                    "frame": frame,
                    "truth_center": truth,
                    "ball_box_size": [x2 - x1, y2 - y1],
                    "positive_candidates": len(positive),
                    "negative_candidates": len(negative),
                    "positive_nearest_px": pos_distance,
                    "negative_nearest_px": neg_distance,
                    "either_nearest_px": min(distances) if distances else None,
                })
            previous = current
    finally:
        capture.release()
    return {
        "clip_id": clip_id,
        "source_name": manifest["source_name"],
        "source_sha256": manifest["source_sha256"],
        "reviewed_visible_frames": len(rows),
        "positive_within_20px": sum(row["positive_nearest_px"] is not None and
                                    row["positive_nearest_px"] <= 20 for row in rows),
        "negative_within_20px": sum(row["negative_nearest_px"] is not None and
                                    row["negative_nearest_px"] <= 20 for row in rows),
        "either_within_20px": sum(row["either_nearest_px"] is not None and
                                  row["either_nearest_px"] <= 20 for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_ids", nargs="+", help="Prepared ball-label clip IDs")
    parser.add_argument("--output", type=Path,
                        default=output / "reviewed-proposal-recall.json")
    args = parser.parse_args()
    report = {"status": "offline_development_diagnostic_not_production",
              "note": "Per-frame nearest raw proposal to separately reviewed visible ball; "
                      "labels are used for scoring only. These recordings informed development.",
              "clips": [audit_clip(clip_id) for clip_id in args.clip_ids]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for clip in report["clips"]:
        print(clip["clip_id"],
              f'either cue {clip["either_within_20px"]}/{clip["reviewed_visible_frames"]}',
              f'positive {clip["positive_within_20px"]}',
              f'negative {clip["negative_within_20px"]}', flush=True)
        misses = [row["frame"] for row in clip["rows"]
                  if row["either_nearest_px"] is None or row["either_nearest_px"] > 20]
        print("  no nearby proposal:", misses, flush=True)
        if clip["clip_id"] in anchors:
            anchor = anchors[clip["clip_id"]][0]
            post = [row for row in clip["rows"]
                    if anchor < row["frame"] <= anchor + 50]
            found = sum(row["either_nearest_px"] is not None and
                        row["either_nearest_px"] <= 20 for row in post)
            post_misses = [row["frame"] for row in post
                           if row["either_nearest_px"] is None or
                           row["either_nearest_px"] > 20]
            print(f"  after manual seed frame {anchor}: {found}/{len(post)}; "
                  f"no nearby proposal {post_misses}", flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
