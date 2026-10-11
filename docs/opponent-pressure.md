# Opponent pressure and goal beacon prototype

This milestone adds opt-in **post-rep training assessment**, not continuous safety
monitoring. Uploaded videos and 4.5-second Live Session reps still use `/analyze`.
Pose, ball tracking, goal tracking, SituatedState, Jev and coaching synthesis are
unchanged. Optional pressure analysis runs afterward; failures leave that report intact.

## Test the supplied clip

1. Restart the backend and frontend using the [Live Session setup](live-session.md).
2. Upload `IMG_6440.mov` from Downloads.
3. In opponent-pressure setup choose **Athlete on camera-right**: the white/purple
   shirt player starts on the right; the dark shirt player starts on the left.
4. Analyze. Review **Opponent pressure · post-rep review** separately from the
   existing technique report. Camera-side describes the image, not body-relative left.
5. For Live Session, position exactly two players separately at the start and select
   the training athlete's side before START REP. The selection persists, so recheck
   it before each rep. Pressure is off by default and adds detector work when enabled.

`athlete_start_side=left|right` is an optional multipart field on `/analyze`.
The response adds optional `pressure_report`: evidence, prototype event timings,
image-space closing rate, confidence note, controller policy and deterministic text.
The local `yolo26n.pt` is required; a missing model is not downloaded automatically.

## What the detector supports

Sample person detections at approximately 5 fps, bounded to the first 20 seconds.
Assign the initial two-player roles from the user's side selection. Associate by
foot-point continuity; stop at ambiguity, occlusion, low-confidence detection, a
large jump, long gap or a third player. No predicted points become observations.

Compare projected foot-point separation normalized by mean bounding-box height over
0.6–1.2 second windows. A candidate needs at least three intervals, mostly consistent
closure, a gap reduction of 0.5 projected body heights, a rate of 0.75 body heights/s,
and an ending gap under 1.8 body heights. These are **unvalidated prototype thresholds**,
not measurements of metres, collision probability or safe distance. Camera movement
and perspective can affect the result; use a stationary camera. Events before tracking
is interrupted may be reported, but nothing is inferred after the interruption.

The controller permits one short supported pressure cue after the rep and protects
mid-action silence. The frontend selects that cue instead of speaking multiple cues
over each other. Without a supported pressure cue, the existing coaching policy and
silence behaviour remain. Detailed pressure text is written independently of any LLM.
Approaching people are treated as training pressure, not automatically as a hazard.
No instruction to cut right/left is generated without athlete heading and open-space
evidence. Lack of a detection does not establish safety.

## Bluetooth speaker goal beacon

Pair the phone to the speaker through iOS. In Live Session, enable the camera, turn
VOICE ON and tap **START GOAL BEACON** while READY. A quiet mono pulse repeats every
two seconds. Place the speaker near the goal target for a physical auditory landmark;
if it stays on the sideline, the tone comes from the sideline. There is no synthetic
spatial direction, automatic goal localization or distance guidance.

Beacon audio requires the button gesture on Safari. It pauses during recording,
analysis and feedback, avoids speaking overlap, follows VOICE OFF, and stops on close.
The goal is a manually configured training target. Pass-receiver targets and live
approach callouts are future extensions and are not represented as implemented here.

Test Bluetooth routing on the actual phone. Verify VOICE OFF, close/reopen, START REP
while a tone plays, analysis errors, pressure-off uploads and repeated reps. The beacon
must be silent during action. Neither this detector nor the speaker setup is a collision
avoidance system. Actual live warnings would require a separate low-latency observation
path and validated player-relative geometry; waiting for `/analyze` cannot provide them.
