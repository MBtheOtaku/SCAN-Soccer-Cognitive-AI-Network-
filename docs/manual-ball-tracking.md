# Manual ball initialisation: offline control experiment

This implements the first test of selecting the actual ball before a rep.
It is an offline research tool, not yet a live camera calibration or recording gate.
Production perception, uploads and coaching are unchanged. A tap defines an initial
target; it does not calibrate camera geometry or prove that later tracking is correct.

## Select your target

In an idle SCAN-root terminal:

```powershell
Set-Location "C:\Users\Matthew Bui\OneDrive\SCAN"
python -m scripts.review_manual_ball prepare 13a1022c5009 --frame 0
Start-Process "data/reference-review/manual-ball/13a1022c5009/seed-0/index.html"
```

Tap the actual ball. Adjust the square size before tapping again, or drag a tight
rectangle around it. Include the ball rather than a large patch of surrounding turf.
Click **Confirm ball & download seed**. This downloads `13a1022c5009-seed.json`.
Place a copy beside that seed page and name it `seed.json`:

```powershell
Copy-Item "$env:USERPROFILE\Downloads\13a1022c5009-seed.json" "data/reference-review/manual-ball/13a1022c5009/seed-0/seed.json"
python -m scripts.review_manual_ball run 13a1022c5009 --seed data/reference-review/manual-ball/13a1022c5009/seed-0/seed.json
```

The original video hash, frame index, dimensions and box are saved in the seed.
The runner validates the source before tracking. Preview coordinates are mapped
back to the full-resolution image. Reopen the seed page to make another selection;
`prepare` refuses to overwrite an existing seed directory.

## Review the result

Open the printed review path. The slider and arrow keys step through every decoded
frame from the seed onwards. Green boxes are accepted appearance matches; yellow
is human seed or provisional calibration. No position is emitted after loss or
ambiguity. This avoids presenting stale predictions as measurements.

The tracker searches locally around the recent position/velocity using the fixed
seed appearance at three scales. It needs a short confirmation sequence. Weak or
competing matches stop the track permanently for that run. There is no detector,
whole-frame reacquisition, ball-contact estimate or external API. Match scores
are correlation heuristics, not probabilities or validated identity confidence.
The fixed template helps make drift visible but will struggle with rotation,
motion blur, occlusion, camera motion and rapid kicks. Scale change is bounded.
These thresholds are experimental, not tuned accuracy guarantees.

`trace.json` contains each frame's status and box. `summary.json` records the seed,
thresholds, first stop frame, status counts and optional evaluation. The other
manual ball labels are loaded only after tracking finishes and used only for
evaluation. The seed frame is excluded from the score. In particular, a high
`tracking` count is not an accuracy result: compare it with `center_inside_label`
and inspect the images. Sparse frame labels cannot establish complete trajectories.

## Additional controls

Repeat on the daylight clips using a clear seed before contact. An alternative
seed closer to contact helps distinguish long-term drift from failure during the
kick, but report that it is a different, easier initialisation condition. For example:

```powershell
python -m scripts.review_manual_ball prepare e9935750581e --frame 100
Start-Process "data/reference-review/manual-ball/e9935750581e/seed-100/index.html"
```

Select and download a fresh seed, then use the matching clip and seed path with
`run`. Do not reset the tracker automatically from later annotation labels.

## A second ball entering the scene

Manual selection gives the track a target origin, not permanent identity.
This prototype can stop when two separated local template matches compete, and
ignores objects outside its local search region. It cannot recognise every second
ball or guarantee identity through a crossing or occlusion. If an identical ball
replaces the original at a plausible location, an undetected switch is possible.
Two extremely close matches may also merge under the peak-deduplication heuristic.
Unit tests exercise detectable ambiguity; they do not demonstrate robustness on
real multi-ball footage.

Collect controlled clips with: an unrelated ball entering far away; two balls
approaching/crossing; original ball occluded while the other remains; original
leaving the image; and a fast shot. Annotate target identity and occlusion separately
for these cases rather than reusing single-ball absence labels for another ball.
Measure wrong-target switches, missed/lost intervals and recovery behaviour.

## Eventual mobile flow

If this control works on the footage, the next implementation can capture a seed
frame and normalised target box in the mobile app, show a provisional lock, then
record. Loss/ambiguity should mark affected analysis unavailable and request a
fresh selection for a new segment. Recording may still be retained. The controller
would separately decide whether evidence supports feedback.

Live pre-recording tracking on the phone is a separate engineering step; the
current Expo camera preview does not run this Python tracker. No claim of a live
lock or start permission should be shown until that integration is implemented.
The mobile integration must also map taps through preview cropping/rotation into
video coordinates and align the seed to an actual recorded frame. A tap on a
preview cannot simply be assumed to identify the same pixels in the recording.

## Initial recorded controls (10 November 2026)

Each control used one existing human ball box as its seed and fixed default
thresholds. No other manual positions corrected the tracker. Frame numbers are
zero-based; these development clips do not constitute a new independent benchmark.

| Clip | Seed frame | First stop | Accepted tracked centres inside later visible labels | Later reviewed visible frames |
|---|---:|---|---:|---:|
| `13a1022c5009` | 0 | lost at 72, before reviewed contact 79–80 | 9 | 36 |
| `e9935750581e` | 10 | ambiguous at 118, at first reviewed ball movement | 20 | 48 |
| `96b382d82d92` | 20 | ambiguous at 22 | 0 | 49 |

The inside-label counts apply only where tracking was accepted. They must be read
with the coverage denominator: these are not 100% successful full-rep trajectories.
The stationary/run-up control looks promising in two clips; ball movement and the
dribbling example remain unresolved. An `ambiguous` status indicates competing
image matches; it does not prove a second ball was present. No real multi-ball
crossing experiment has been run yet.

Files: `scripts/manual_ball_tracker.py`, `scripts/review_manual_ball.py`,
`scripts/manual_ball_seed.html`. Tests:
`python -m pytest tests/test_manual_ball.py -q`.

The image matcher uses OpenCV's normalised template matching:
[OpenCV documentation](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html).
