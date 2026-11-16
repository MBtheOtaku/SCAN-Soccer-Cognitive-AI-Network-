"""Compare existing motion cue/launch settings on the reviewed holdout frames."""

import json

from .evaluate_new_clips import score
from .holdout_shot_check import CASES
from .pilot_motion_track import output, track


VARIANTS = (
    ("positive_no_launch", True, False, 0, 0),
    ("negative_soft_launch", False, True, 12, .1),
    ("negative_no_launch", False, True, 0, 0),
)


def main():
    labels = json.loads((output / "holdout-additional-clips" / "labels.json").read_text())
    results = []
    for clip, case in CASES.items():
        for name, positive, negative, progress, weight in VARIANTS:
            result = track(clip, case["anchor"], case["goal_center"], length=30,
                           positive=positive, negative=negative,
                           launch_progress_px_per_frame=progress,
                           launch_soft_weight=weight, seed_box=case["seed_box"],
                           seed_kind=case["kind"])
            result["mode"] = name
            results.append(result)
            print(clip, name, result["status"], flush=True)
    report = {"status": "offline_pilot_not_production", "results": results}
    scores = score(report, labels)
    dest = output / "holdout-additional-clips"
    (dest / "cue-variants.json").write_text(json.dumps({"report": report,
                                                       "scores": scores}, indent=2))
    for clip, modes in scores.items():
        for mode, item in modes.items():
            print(clip, mode, f'tracked {item["tracked"]}/{item["labelled"]}',
                  f'within20 {item["within_20px"]}/{item["labelled"]}',
                  f'flagged {item["flagged_uncertain"]}')


if __name__ == "__main__":
    main()
