"""Compare the appearance-aware offline candidate with frozen earlier paths.

These settings were chosen after examining all five clips. The report is a
development regression check, not an independent accuracy estimate.
"""

import html
import json
from pathlib import Path

import cv2

from .evaluate_new_clips import file_sha256, score
from .holdout_shot_check import CASES as HOLDOUT
from .pilot_motion_track import output, root, track
from .pilot_new_clips import CASES as EARLIER


def cases_and_labels():
    earlier = json.loads((Path(__file__).parent / "reviewed_new_clips.json").read_text())
    cases = {}
    for clip, case in HOLDOUT.items():
        cases[clip] = (case["anchor"], case["seed_box"], case["goal_center"],
                       "dribble_ball" if case["kind"] == "dribble_stress_case" else "pre_shot_ball")
        earlier["clips"][clip] = {
            "source_name": f"{clip}.mov",
            "source_sha256": file_sha256(root / "data/input" / f"{clip}.mov"),
            "annotations": [{"frame": frame, "center": center}
                            for frame, center in case["centers"].items()],
        }
    cases.update(EARLIER)
    return cases, earlier


def render(scores, labels, destination):
    figures = []
    for clip, modes in scores.items():
        baseline = modes["baseline_positive"]
        candidate = modes["identity_candidate"]
        motion_only = modes["both_motion_only"]
        source = root / "data/input" / labels["clips"][clip]["source_name"]
        cap = cv2.VideoCapture(str(source))
        cards = []
        for before, after, motion in zip(baseline["rows"], candidate["rows"],
                                         motion_only["rows"], strict=True):
            frame = before["frame"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Could not decode {source}, frame {frame}")
            cv2.drawMarker(image, tuple(before["truth"]), (50, 240, 70),
                           cv2.MARKER_CROSS, 25, 2)
            if before["estimate"] is not None:
                cv2.circle(image, tuple(map(round, before["estimate"])), 14,
                           (230, 130, 50), 3)
            if after["estimate"] is not None:
                cv2.circle(image, tuple(map(round, after["estimate"])), 10,
                           (0, 220, 255), 3)
            if motion["estimate"] is not None:
                cv2.drawMarker(image, tuple(map(round, motion["estimate"])),
                               (255, 80, 220), cv2.MARKER_TILTED_CROSS, 20, 2)
            image = cv2.resize(image, (360, round(image.shape[0] * 360 / image.shape[1])))
            name = f"{clip}-{frame:06d}.jpg"
            if not cv2.imwrite(str(destination / name), image):
                raise OSError(name)
            old_error = before["error_px"]
            new_error = after["error_px"]
            old_text = f'{old_error:.1f}px' if old_error is not None else 'lost'
            new_text = f'{new_error:.1f}px' if new_error is not None else 'lost'
            motion_error = motion["error_px"]
            motion_text = f'{motion_error:.1f}px' if motion_error is not None else 'lost'
            cards.append(f'<figure><img loading="lazy" src="{name}"><figcaption>'
                         f'Frame {frame}: baseline {old_text}, appearance {new_text}, '
                         f'motion-only {motion_text}; '
                         f'{after["reliability"]}</figcaption></figure>')
        cap.release()
        figures.append(f'<section><h2>{html.escape(clip)}</h2><p>Baseline '
                       f'{baseline["within_20px"]}/{baseline["labelled"]}; '
                       f'appearance {candidate["within_20px"]}/{candidate["labelled"]}; '
                       f'motion-only {motion_only["within_20px"]}/{motion_only["labelled"]} '
                       'within 20 px of sparse marks.</p><div class="frames">' +
                       ''.join(cards) + '</div></section>')
    page = ('<!doctype html><meta charset="utf-8"><title>Ball identity pilot</title>'
            '<style>body{background:#08120d;color:#e9f3eb;font:16px system-ui;margin:24px}'
            '.frames{display:flex;overflow-x:auto;gap:12px}figure{flex:none;width:360px;margin:0}'
            'img{width:100%}figcaption{font-size:13px}section{border-top:1px solid #42664b}'
            '</style><h1>Hand-seeded ball identity pilot</h1><p>Green cross: manual ball centre. '
            'Blue circle: frozen positive-change baseline. Yellow circle: appearance-aware '
            'candidate. Magenta cross: pooled motion without appearance. '
            'Errors are raw pixel distances. An uncertain candidate is not accepted '
            'ball evidence. These clips were used to choose the settings; this is not an '
            'independent accuracy estimate.</p>' + ''.join(figures))
    (destination / "index.html").write_text(page, encoding="utf-8")


def main():
    cases, labels = cases_and_labels()
    for clip, item in labels["clips"].items():
        source = root / "data/input" / item["source_name"]
        if file_sha256(source) != item["source_sha256"]:
            raise ValueError(f"Source video changed: {clip}")
    old_reports = [output / "holdout-additional-clips/report.json",
                   output / "report-new-clips.json"]
    baseline = {result["clip"]: result for path in old_reports
                for result in json.loads(path.read_text())["results"]
                if result["mode"] == "positive"}
    results = []
    for clip, (anchor, box, goal, kind) in cases.items():
        old = dict(baseline[clip])
        old["mode"] = "baseline_positive"
        results.append(old)
        result = track(clip, anchor, goal, length=30, seed_box=box, seed_kind=kind,
                       launch_progress_px_per_frame=12, launch_soft_weight=.1,
                       both_signed=True, appearance_weight=1,
                       appearance_max_hist_distance=.85,
                       appearance_gate_frames=30 if kind == "dribble_ball" else 6)
        result["mode"] = "identity_candidate"
        results.append(result)
        motion_only = track(clip, anchor, goal, length=30, seed_box=box,
                            seed_kind=kind, launch_progress_px_per_frame=12,
                            launch_soft_weight=.1, both_signed=True)
        motion_only["mode"] = "both_motion_only"
        results.append(motion_only)
        print(clip, kind, result["status"], result["frames_tracked"], flush=True)
    report = {"status": "offline_pilot_not_production", "results": results,
              "notes": ["The settings were chosen after examining these five clips.",
                        "Seed boxes and goal centres are manually supplied.",
                        "The candidate is not accepted evidence while uncertain."]}
    scores = score(report, labels)
    destination = output / "identity-pilot"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2))
    (destination / "scores.json").write_text(json.dumps(scores, indent=2))
    render(scores, labels, destination)
    for clip, modes in scores.items():
        print(clip, 'baseline', modes['baseline_positive']['within_20px'],
              'candidate', modes['identity_candidate']['within_20px'],
              'motion_only', modes['both_motion_only']['within_20px'],
              'of', modes['identity_candidate']['labelled'], flush=True)
    print(destination / "index.html")


if __name__ == "__main__":
    main()
