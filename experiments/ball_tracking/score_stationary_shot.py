"""Score an offline stationary-shot path against separately reviewed ball boxes."""

import argparse
import json
import math
from pathlib import Path

from scripts.ball_dataset import file_sha256, validate_labels


def score_path(result: dict, annotations: list[dict]) -> dict:
    start = result["anchor"]
    end = start + result["frames_attempted"]
    selected = [row for row in annotations if start < row["frame"] <= end]
    visible = [row for row in selected if row["state"] == "visible"]
    path = {row["frame"]: row for row in result["path"]}
    rows = []
    for label in visible:
        frame = label["frame"]
        estimate = path.get(frame)
        box = label["box"]
        truth = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        error = math.dist(truth, estimate["center"]) if estimate else None
        rows.append({"frame": frame, "truth": truth,
                     "estimate": estimate["center"] if estimate else None,
                     "error_px": error,
                     "observed": estimate["observed"] if estimate else False,
                     "reliability": estimate["reliability"] if estimate else "lost"})
    errors = [row["error_px"] for row in rows if row["error_px"] is not None]
    supported = [row for row in rows if row["observed"] and
                 row["reliability"] == "provisional"]
    return {"visible_reviewed": len(visible),
            "uncertain_reviewed": sum(row["state"] == "uncertain" for row in selected),
            "absent_reviewed": sum(row["state"] == "absent" for row in selected),
            "unreviewed": sum(row["state"] == "unreviewed" for row in selected),
            "located": len(errors),
            "within_10px": sum(error <= 10 for error in errors),
            "within_20px": sum(error <= 20 for error in errors),
            "within_40px": sum(error <= 40 for error in errors),
            "provisional_observed_within_20px": sum(
                row["error_px"] <= 20 for row in supported),
            "provisional_observed_total": len(supported),
            "rows": rows}


def validate_review_window(result: dict, manifest: dict, labels: dict) -> None:
    start = result["anchor"] + 1
    end = result["anchor"] + result["frames_attempted"]
    sampled = {row["frame"] for row in manifest["frames"]}
    reviewed = {row["frame"] for row in labels["annotations"]
                if row["state"] != "unreviewed"}
    required = set(range(start, end + 1))
    if not required <= sampled:
        raise ValueError(f"Annotation page must include every frame {start}–{end}.")
    if not required <= reviewed:
        raise ValueError(f"Review every frame {start}–{end}, including hidden/absent states.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="trajectory.json from stationary_shot_review")
    parser.add_argument("labels", type=Path, help="labels.json from ball_dataset")
    args = parser.parse_args()
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    if trajectory.get("status") != "offline_pilot_not_production":
        parser.error("Expected an offline pilot trajectory.")
    result = trajectory["result"]
    manifest_path = args.labels.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    validate_labels(manifest, labels)
    source = Path(result["source_name"]).resolve()
    if (source.name != manifest["source_name"] or
            file_sha256(source) != manifest["source_sha256"]):
        parser.error("Trajectory and labels do not belong to the same video.")
    try:
        validate_review_window(result, manifest, labels)
    except ValueError as exc:
        parser.error(str(exc))
    scores = score_path(result, labels["annotations"])
    report = {"status": "offline_scored_not_validated_identity",
              "source_sha256": manifest["source_sha256"],
              "trajectory": str(args.trajectory.resolve()),
              "labels": str(args.labels.resolve()),
              "score": scores,
              "note": "Raw pixel proximity on reviewed frames; predicted or uncertain positions are not supported observations."}
    destination = args.trajectory.with_name("score.json")
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f'Visible marks within 20 px: {scores["within_20px"]}/{scores["visible_reviewed"]}')
    print(f'Provisional observed positions within 20 px: '
          f'{scores["provisional_observed_within_20px"]}/'
          f'{scores["visible_reviewed"]} visible marks')
    print(destination)


if __name__ == "__main__":
    main()
