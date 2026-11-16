"""Reproducible, sparse review of distinct phone recordings.

The seed and reference centres below were read from original video frames
before running the frozen offline tracker. This is not production validation.
"""

import json

from .evaluate_new_clips import file_sha256, render, score
from .pilot_motion_track import output, root, track


CASES = {
    "36bf848f409c": {
        "kind": "shot", "anchor": 49, "seed_box": [875, 747, 910, 781],
        "goal_center": (420, 700),
        "centers": {51: [860, 746], 52: [812, 731], 53: [769, 710],
                    55: [698, 684], 56: [665, 672], 58: [613, 655],
                    59: [588, 649], 61: [548, 640], 62: [531, 641],
                    65: [487, 639]},
    },
    "7de88d94f23c": {
        "kind": "dribble_stress_case", "anchor": 41,
        "seed_box": [375, 771, 406, 804], "goal_center": (365, 600),
        "centers": {44: [404, 775], 47: [417, 770], 53: [445, 754],
                    56: [449, 746], 59: [453, 742]},
    },
}


def main():
    labels = {"schema": 1, "method": "Visual ball centres on original video frames before tracker run.",
              "clips": {clip: {"source_name": f"{clip}.mov",
                               "source_sha256": file_sha256(root / "data/input" / f"{clip}.mov"),
                               "kind": case["kind"],
                               "annotations": [{"frame": frame, "center": center}
                                               for frame, center in case["centers"].items()]}
                        for clip, case in CASES.items()}}
    results = []
    for clip, case in CASES.items():
        for positive in (True, False):
            result = track(clip, case["anchor"], case["goal_center"], length=30,
                           positive=positive, launch_progress_px_per_frame=12,
                           launch_soft_weight=.1, seed_box=case["seed_box"],
                           seed_kind="pre_shot_ball" if case["kind"] == "shot" else "dribble_ball")
            results.append(result)
            print(clip, result["mode"], result["status"],
                  result["frames_tracked"], flush=True)
    report = {"status": "offline_pilot_not_production", "results": results,
              "notes": ["Two additional byte-distinct phone recordings: one shot and one dribble stress case.",
                        "Frozen settings; hand-picked seed and approximate goal centre."]}
    scores = score(report, labels)
    dest = output / "holdout-additional-clips"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (dest / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (dest / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    render(scores, labels, dest)
    for clip, modes in scores.items():
        for mode, item in modes.items():
            median = item["median_error_px"]
            print(clip, mode, f'tracked {item["tracked"]}/{item["labelled"]}',
                  f'median {median:.1f}px' if median is not None else 'median unavailable',
                  f'within20 {item["within_20px"]}/{item["labelled"]}',
                  f'uncertain {item["flagged_uncertain"]}', flush=True)
            for row in item["rows"]:
                error = row["error_px"]
                print(row["frame"], f'{error:.1f}px' if error is not None else "lost",
                      row["reliability"])
    print(dest)


if __name__ == "__main__":
    main()
