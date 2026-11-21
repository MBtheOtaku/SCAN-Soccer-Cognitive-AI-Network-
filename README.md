# SCAN Pose MVP v2 — first real shooting clip

Live Session v0.1: see [phone setup and testing](docs/live-session.md) for camera
recording, configurable API URLs, trusted HTTPS and server start commands.

Ball tracking development: see the [active workflow and current limitations](docs/ball-tracking.md).
Superseded tracking experiments are isolated in the [research archive](experiments/ball_tracking/archive/README.md).

Training intentions: see [power, curl and placement measurements](docs/training-intention.md)
for the per-clip flow, formulas, evidence requirements and published reference context.

Saved-rep experiments: see [single-tap ball tracking and replay](docs/experimental-tracking.md).
New Expo recordings also support [microphone audio](src/mobile/README.md#recording-sound).

This version is tuned for the first SCAN test clip: a fixed 1280x720 side-angle shooting video.
It crops the pose model to the player region, draws the skeleton back onto the original frame,
computes simple pose features, estimates the probable striking leg/moment using ankle motion
relative to the hips, and writes a JSON summary that can later feed `RepFeatures`.

## Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install mediapipe opencv-python numpy
```

Current MediaPipe releases support Windows x86-64. If your Python environment gives package
compatibility trouble, use Python 3.12 for the least-friction setup.

Download Google's Pose Landmarker model and place it beside the script as:

`pose_landmarker_full.task`

## Run this specific clip

```bash
python scan_pose_mvp_v2.py IMG_5702.mov --roi 540,280,1180,650
```

Outputs:
- `scan_annotated_v2.mp4` — skeleton + metrics overlay
- `scan_pose_metrics_v2.csv` — per-frame pose measurements
- `scan_shot_summary.json` — estimated strike event + one prototype cue

## Why the ROI?

The player is relatively small in the full 1280x720 frame. The fixed ROI covers the full run-up,
strike, and early follow-through while giving the pose estimator many more useful pixels.
Without a manual ROI, the backend now samples the local person detector at
`data/models/yolo11n.pt` and builds a padded movement envelope before running MediaPipe.
This also supports decoded portrait phone footage without resizing the saved video.
The detector weights must already be present; analysis never downloads them. Missing weights
or failed detector inference retain the pose-only/full-frame fallback. Multiple detected people
leave automatic player identity uncertain and withhold pose measurements.

The summary's `pose_tracking` section records the crop source, native coordinates and counts
of accepted, rejected and missing pose frames. The landmark-confidence check (0.35) and
provisional whole-clip coverage gate (60%) are unchanged. A player outside the camera frame
cannot be recovered by cropping, and required landmarks outside the crop are rejected even
when their model confidence is high. Keep both feet and the whole body visible through the approach
and follow-through. Higher accepted-frame coverage is not validated biomechanics or confirmed
ball-contact timing.

## Important interpretation

The strike detector is an engineering heuristic: it picks the ankle with the highest motion
relative to the hip midpoint. The feedback is also provisional. This is enough to validate the
pipeline, but it should not be described as validated soccer biomechanics.

Written reports withhold a technique grade when perception is unusable, Jev is unavailable,
or the primary-issue judgement is missing or favours insufficient evidence. They lead with
pose coverage and, when usable, the descriptive strike-window measurements. These reports
use deterministic wording instead of language synthesis so uncertainty cannot be rewritten
as an execution grade. Raw Jev scores remain in the analysis diagnostics; controller and
speech policy are unchanged. Outcome claims still require their own ball/goal evidence.
