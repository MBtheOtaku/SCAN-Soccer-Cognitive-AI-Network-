# Automatic goal proposals and confirmed setup

SCAN can now propose a goal outline from a captured camera frame or the first
frame of an uploaded video. Proposals are never used for analysis until the user
confirms the four corners. The existing camera-motion tracker then maintains them.

## Try it

Restart backend and frontend using the existing HTTPS commands in
[Live Session setup](live-session.md). Refresh SCAN on the phone.

**Uploaded clip:** choose a video, tap FIND GOAL, review the captured frame, then
CONFIRM GOAL. If several candidates appear, choose the intended one. Drag numbered
handles to adjust; MARK CORNERS clears them for four taps in order: top-left,
top-right, bottom-right, bottom-left. Then analyse as usual. Changing the uploaded
file clears its goal setup. If analysis fails, confirm setup again before retrying.

For precise adjustment, tap a numbered handle or a corner-name button. The magnified
view centres its crosshair on that exact corner. Use arrow buttons to move it by
1 or 5 reference-image pixels per tap; a focused precision panel also accepts arrow
keys. Movement stops at the image edges. Any edit clears confirmation. RESET TO
PROPOSED BOUNDS restores the currently selected automatic candidate and requires
confirmation again; this button is available when an automatic proposal exists.
Uploaded clips are decoded on the backend for this preview, avoiding browser MOV
black-frame capture. FIND GOAL uploads a temporary preview copy before the normal
analysis upload. It tries a few early frames to skip blank introductions, then deletes
the temporary clip. A preview with no detected goal still supports manual marking.

**Live Session:** enable the camera, point it at the goal, tap FIND GOAL and confirm
or adjust the outline. START REP reuses that setup across reps. A new camera access,
closing the session or substantially unreliable goal tracking clears it and requires
a new capture/confirmation. Reconfirm whenever changing camera position or target.

SKIP GOAL leaves shot outcome unknown while keeping technique analysis available.
Detector/API failure also leaves the reference photo available for manual marking.
The saved global `data/goal_calibration.json` is no longer silently applied to web
uploads or live clips. The standalone CLI retains its legacy calibration support.

## Implementation and uncertainty

The initial detector is an OpenCV geometry prototype for reasonably front-facing,
light/white goal frames. It searches for a crossbar and post support, rejects some
pitch-line configurations, and ranks candidate polygons. It is not a trained goal
recognition model. Rank scores are not probabilities. Dark frames, strong camera
tilt, grass-backed nets, obstruction, tiny/distant goals, unusual perspective and
multiple goals can cause abstention or incorrect candidates. Check every proposal;
manual correction remains essential in these cases.

Confirmed corners and their captured reference image are sent with each analysis.
They belong only to that request; the server does not update a shared calibration
file or reuse another user's/session's setup. Live Session retains its own confirmed
setup in browser memory until cleared. A page refresh clears that memory.

The goal-motion tracker compares even frame zero against the confirmed reference
photo, so the recording is not assumed to match a potentially older preview. It
supports modest motion; invalid aspect ratio or lost visual support leaves outcome
uncertain. Below 70% accepted goal-tracking frames, the live UI clears confirmation
and asks for recapture. This is a conservative prototype quality threshold, not a
calibrated probability or a guarantee of accuracy.

## API

- `POST /goals/preview`: multipart video field `video`. Returns a visible JPEG
  `image_base64`, its `frame_index`, and provisional candidates requiring confirmation.
- `POST /goals/propose`: multipart JPEG/image field `frame`, maximum 5 MB and
  12 million decoded pixels. Returns candidates and `requires_confirmation: true`.
- `POST /analyze`: optional multipart `goal_setup` JSON of
  `{"confirmed":true,"corners":[[x,y],[x,y],[x,y],[x,y]]}` and matching image field
  `goal_reference`. Coordinates are normalized 0–1 in TL/TR/BR/BL order. A confirmed
  setup requires its photo. `goal_setup=null` or absence means no goal calibration.
- The pipeline still uses the existing pose, ball tracking, SituatedState, Jev,
  controller and coaching synthesis. Its JSON records `goal_setup` and `goal_motion`.

Validation covers malformed/uncertain geometry, blank scenes, a synthetic goal,
separate request setups, reference-image verification at clip frame zero, endpoint
fallbacks and the existing regression tests. Recheck tap/drag behaviour and camera
capture on the actual iPhone; desktop checks do not establish mobile usability.

Repeatable frontend contract checks: from `src/frontend`, run
`node --test tests/goal-setup.test.cjs`. Backend checks: from the repository root,
run `python -m pytest tests` with `PYTHONPATH=src/backend` for existing controller tests.
