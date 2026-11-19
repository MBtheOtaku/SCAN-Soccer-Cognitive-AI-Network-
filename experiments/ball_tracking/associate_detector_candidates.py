"""Offline causal association pilot over saved detector proposals.

The initial ball centre is supplied manually. Review labels are used only
after association to score the resulting candidate path.
"""

import argparse
import json
import math
from pathlib import Path

from experiments.ball_tracking.score_detector_candidates import score


def associate_next(row: dict, last: tuple[float, float],
                   velocity: tuple[float, float]) -> tuple[dict, tuple[float, float], tuple[float, float]]:
    """Choose from one frame without using its reviewed ball label."""
    predicted = (last[0] + velocity[0], last[1] + velocity[1])
    speed = math.hypot(*velocity)
    # A struck ball can travel many pixels in its first frame. Keep a
    # generous motion gate, then weigh detection strength and trajectory
    # agreement together instead of snapping to the seed's nearest dot.
    gate = max(95.0, 2.0 * speed + 40.0)
    eligible = [item for item in row.get("candidates", [])
                if math.dist(item["center"], predicted) <= gate]

    def cost(item: dict) -> float:
        return -math.log(max(item["confidence"], 1e-6)) + math.dist(item["center"], predicted) / 60.0

    chosen = min(eligible, key=cost) if eligible else None
    if chosen is not None and cost(chosen) > 3.0:
        chosen = None
    if chosen is not None:
        center = tuple(chosen["center"])
        measured = (center[0] - last[0], center[1] - last[1])
        velocity = (0.65 * measured[0] + 0.35 * velocity[0],
                    0.65 * measured[1] + 0.35 * velocity[1])
        last = center
    else:
        # A prediction is not a detection. It can guide the next search but
        # must remain marked as missing and cannot verify ball identity.
        last = predicted
    result = {"clip": row["clip"], "frame": row["frame"],
              "split": row["split"], "center": chosen["center"] if chosen else None,
              "confidence": chosen["confidence"] if chosen else None,
              "predicted": predicted, "gate_px": gate,
              "candidate_count": len(row.get("candidates", [])),
              "eligible_count": len(eligible), "detected": chosen is not None}
    return result, last, velocity


def associate(rows: list[dict], seed: tuple[float, float]) -> list[dict]:
    if not rows or len({row["clip"] for row in rows}) != 1:
        raise ValueError("Expected one clip with at least one frame")
    ordered = sorted(rows, key=lambda row: row["frame"])
    if any(right["frame"] != left["frame"] + 1 for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Causal association requires contiguous frames")
    last = tuple(seed)
    velocity = (0.0, 0.0)
    output = []
    for row in ordered:
        result, last, velocity = associate_next(row, last, velocity)
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--seed", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    path = associate(report["rows"], tuple(args.seed))
    # Reuse the independent offline scorer after the association path is fixed.
    scored_rows = []
    for source, result in zip(sorted(report["rows"], key=lambda row: row["frame"]), path, strict=True):
        candidate = ({"center": result["center"], "confidence": result["confidence"]}
                     if result["detected"] else None)
        scored_rows.append({**source, "candidates": [candidate] if candidate else [],
                            "candidate_count": int(candidate is not None)})
    assessment = score({"rows": scored_rows})["summary"]
    output = {"source_report": str(args.report), "seed": args.seed,
              "status": "offline_pilot_not_production",
              "method": "causal_constant_velocity_distance_and_detection_score",
              "limitation": "Manually seeded development check. A selected detection near a prediction is not verified ball identity; do not use for coaching or outcomes.",
              "assessment": assessment, "path": path}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
