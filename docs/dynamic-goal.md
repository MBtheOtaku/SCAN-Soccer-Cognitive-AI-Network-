# Camera-motion-aware goal calibration

Web uploads and Live Session now use [automatic proposals with per-setup
confirmation](goal-setup.md). The shared-file instructions below describe the
standalone CLI workflow.

Goal corners still need to be calibrated for the clip. They now follow observed
camera motion instead of remaining fixed throughout the video. The existing
annotated video and goal-entry analysis use the same per-frame calibration.

Restart the backend and re-analyze the original video to generate a new annotated
output. Previously generated videos will retain their old overlays. No frontend
restart is needed for this change.

The tracker observes visual features around the calibrated goal and estimates
translation, rotation and uniform scale with a robust affine fit. Each frame is
matched to the calibration reference frame rather than accumulating transforms
between frames. Forward/backward flow checks, reprojection inliers and distributed
feature support reject uncertain matches. Moving players are treated as outliers
when sufficient stationary goal/background features remain.

Lost frames show **GOAL TRACKING UNCERTAIN** instead of a frozen grid. They cannot
support a goal-entry judgement, and a ball segment cannot bridge a lost goal track.
No observed crossing is not treated as a miss. Tracking can resume when the
reference features are visible again.

The analysis JSON includes `goal_motion` with accepted/lost frame counts and the
reference frame. Existing calibration JSON defaults to reference frame 0. The
interactive calibrator saves `reference_frame` when `--frame` selects another frame:

```powershell
python -m src.backend.goal_tracking calibrate "C:\Users\Matthew Bui\Downloads\IMG_6440.mov" --output data/goal_calibration.json --frame 0
```

Click top-left, top-right, bottom-right and bottom-left, then Enter. The current
global calibration file must be updated for a new camera setup/clip; this change
does not automatically locate a previously uncalibrated goal. Do not use one clip's
corners as ground truth for an unrelated view.

Large viewpoint changes, heavy occlusion, scene cuts, weak texture and large zoom
can make tracking unavailable. This model supports modest camera motion, not a full
3D reconstruction. An incorrect initial calibration remains incorrect. Pose/player
calibration and opponent-pressure motion compensation are separate concerns.

Validation includes static-camera and translated synthetic frames, lost-track and
recovery behaviour, camera-induced false crossings, real crossings with moving
bounds, unknown outcomes across lost frames, legacy calibration loading and visual
checks on `IMG_6440.mov`.
