# SCAN Pose MVP v2 — first real shooting clip

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
The ROI is only for this first fixed-camera test. Later we should replace it with player tracking.

## Important interpretation

The strike detector is an engineering heuristic: it picks the ankle with the highest motion
relative to the hip midpoint. The feedback is also provisional. This is enough to validate the
pipeline, but it should not be described as validated soccer biomechanics.
