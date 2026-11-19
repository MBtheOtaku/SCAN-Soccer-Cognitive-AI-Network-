"""Offline identity diagnostic for a saved fixed-frame detector search.

Labels score selections; they are never used to choose a candidate.
"""

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def score(report: dict) -> dict:
    labels_by_clip = {}
    rows = []
    for item in report["rows"]:
        clip, frame = item["clip"], item["frame"]
        if clip not in labels_by_clip:
            tracked = ROOT / "experiments/ball_tracking/reviewed_labels" / clip / "labels.json"
            local = ROOT / "data/reference-review/ball-labels" / clip / "labels.json"
            path = tracked if tracked.is_file() else local
            labels_by_clip[clip] = {
                label["frame"]: label for label in json.loads(path.read_text(encoding="utf-8"))["annotations"]}
        label = labels_by_clip[clip][frame]
        if label["state"] != "visible":
            raise ValueError(f"Expected visible ball: {clip} frame {frame}")
        box = label["box"]
        truth = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        proposals = item.get("candidates", [])
        if len(proposals) != item["candidate_count"]:
            raise ValueError(f"Candidate count mismatch: {clip} frame {frame}")
        center = item.get("crop_center")
        by_confidence = max(proposals, key=lambda p: p["confidence"], default=None)
        by_path = (min(proposals, key=lambda p: math.dist(p["center"], center))
                   if center is not None and proposals else None)
        is_hit = lambda proposal: bool(proposal and math.dist(proposal["center"], truth) <= 20)
        true_conf = [p["confidence"] for p in proposals if is_hit(p)]
        other_conf = [p["confidence"] for p in proposals if not is_hit(p)]
        rows.append({"clip": clip, "frame": frame, "split": item["split"],
                     "ball_candidate": bool(true_conf),
                     "top_confidence_hit": is_hit(by_confidence),
                     "nearest_path_hit": is_hit(by_path),
                     "best_ball_confidence": max(true_conf, default=None),
                     "best_other_confidence": max(other_conf, default=None),
                     "other_proposals": len(other_conf)})
    summary = {}
    for split in ("train", "val"):
        selected = [row for row in rows if row["split"] == split]
        summary[split] = {"frames": len(selected),
                          "ball_candidate": sum(row["ball_candidate"] for row in selected),
                          "top_confidence_hit": sum(row["top_confidence_hit"] for row in selected),
                          "nearest_path_hit": sum(row["nearest_path_hit"] for row in selected),
                          "other_proposals": sum(row["other_proposals"] for row in selected)}
    return {"summary": summary, "rows": rows,
            "limitation": "Manual labels only score offline candidate choices. The path centre is from an unverified manually seeded tracker, and neither selection proves identity."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(json.loads(args.report.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
