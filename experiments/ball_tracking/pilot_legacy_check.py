"""Run the identity candidate on six already reviewed development recordings."""

import json

from .pilot_motion_track import anchors, output, track


def main():
    results = []
    for clip, (anchor, goal) in anchors.items():
        result = track(clip, anchor, goal, length=30, both_signed=True,
                       appearance_weight=1, appearance_max_hist_distance=.85,
                       appearance_gate_frames=6,
                       launch_progress_px_per_frame=12, launch_soft_weight=.1)
        results.append(result)
        print(clip, result["status"],
              f'within 20 px: {result["within_20px"]}/{result["evaluated"]}',
              flush=True)
    destination = output / "identity-pilot"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "legacy-reviewed.json").write_text(json.dumps({
        "status": "development_check_not_independent_validation",
        "notes": ["Six earlier clips were used during motion-pilot development.",
                  "Only reviewed visible boxes in the first 30 post-seed frames are scored.",
                  "All paths remain provisional or uncertain, never confirmed identity."],
        "results": results,
    }, indent=2), encoding="utf-8")
    print(destination / "legacy-reviewed.json")


if __name__ == "__main__":
    main()
