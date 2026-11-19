"""Keep offline detector evaluations aligned with recording-session splits."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/ball_tracking"


def test_fixed_frame_selections_follow_recording_group_split():
    split_file = EXPERIMENT / "splits-dense-pilot.json"
    clips = json.loads(split_file.read_text(encoding="utf-8"))["clips"]
    split_by_clip = {item["clip_id"]: item["split"] for item in clips}
    assert len(split_by_clip) == len(clips)

    split_by_group = {}
    for item in clips:
        group = item["group"]
        assert group not in split_by_group or split_by_group[group] == item["split"]
        split_by_group[group] = item["split"]

    selections = [EXPERIMENT / "oracle-dense-frames.json",
                  *EXPERIMENT.glob("predicted-*-frames.json")]
    assert len(selections) >= 2
    for selection in selections:
        for case in json.loads(selection.read_text(encoding="utf-8"))["clips"]:
            assert case["split"] == split_by_clip[case["clip_id"]], selection.name
