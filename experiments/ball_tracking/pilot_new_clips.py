"""Frozen hand-seeded check on additional phone recordings.

These seeds and approximate goal centres were selected before running the
tracker. Sparse post-seed reference centres are scored separately by
evaluate_new_clips.py; this raw report is not an accuracy score.
"""

import json

from .pilot_motion_track import output, track


# Zero-based frame, visually chosen ball box, approximate goal centre, seed type.
CASES = {
    "e3949580d0f2": (105, [303, 769, 330, 799], (550, 650), "pre_shot_ball"),
    "735e17f02bd5": (83, [399, 791, 426, 820], (157, 649), "pre_shot_ball"),
    "675e44d940ff": (47, [663, 482, 692, 513], (280, 468), "first_flight_frame"),
}


def main():
    results = []
    for clip, (frame, box, goal, kind) in CASES.items():
        for positive in (True, False):
            result = track(clip, frame, goal, length=30, positive=positive,
                           launch_progress_px_per_frame=12,
                           launch_soft_weight=.1, seed_box=box, seed_kind=kind)
            results.append(result)
            print(clip, result["mode"], result["status"],
                  result["frames_tracked"], "frames,",
                  result["observed"], "motion observations", flush=True)
    report = {
        "status": "offline_pilot_not_production",
        "notes": ["Additional phone recordings, not used to select pilot parameters.",
                  "One manually selected pre-shot or first-flight ball box per clip.",
                  "Goal centres were visually estimated before tracking.",
                  "Sparse later ball centres are scored separately; zero evaluated frames here is not a score.",
                  "Candidate trajectories and reliability warnings require independent validation."],
        "results": results,
    }
    path = output / "report-new-clips.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
