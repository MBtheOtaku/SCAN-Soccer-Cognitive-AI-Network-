# Experimental ball tracking in saved reps

Saved Expo reps now support an optional, manually initialised tracking replay.
The backend runs the guarded motion-recovery pilot with the existing detector
weights. The result is a visual experiment, kept separate from ordinary analysis,
SituatedState, Jev judgements, coaching and shot outcomes.

## Using the phone

Restart the laptop backend and the stable Expo preview to load these changes.
Keep the usual Tailscale connection active on the phone and laptop.

1. Open a saved stationary-shot rep.
2. Choose **Load initial frame** under **Experimental ball tracking**. This
   uploads the original clip once and prepares a visible frame near its start.
3. Tap the centre of the intended ball in the preview. The cyan ring is your
   manual starting position. Adjust it before starting if needed.
4. Choose **Run experimental tracking**. Progress comes from processed frames.
   Processing can take several minutes on the laptop.
5. Play the original video above the controls to inspect the overlay. Hide the
   overlay to use ordinary replay, including fullscreen playback.

Yellow rings show selected detector candidates with unverified identity. Purple
dashed rings show predictions without a selected observation. Lost frames have
no marker. The app never draws an observed point from a neighbouring frame or
fills a gap with an observation. Replay alignment uses frame index divided by
reported FPS and is approximate, particularly for variable-frame-rate media.

The original video is the playback source, so embedded sound is preserved.
There is no re-encoded tracking video. New recordings support microphone sound;
see [recording sound](../src/mobile/README.md#recording-sound). Older silent clips
cannot recover audio.

**Stop waiting**, navigation and backgrounding stop phone polling and pause
replay. A job already running on the laptop continues. **Check status** reconnects
to the same job while this replay screen remains open, without another upload.
If the screen is unmounted, reopen the rep and prepare a new job. Only one
experimental tracking job runs at a time, so wait for an existing job to finish
before starting another. A failed or interrupted job needs a new upload.

## Backend setup and job boundaries

The default model is the ignored local file
`data/models/scan-experimental-ball.pt`, copied from the reviewed guarded pilot.
Its SHA256 is
`8e1467e9fd3e2ce213ec262ab8aa10deb3422a67e99c008f4f50e0b7e296cdd0`.
`SCAN_EXPERIMENTAL_BALL_WEIGHTS` can select another local checkpoint. The model
must be the one-class ball detector; a missing model produces an error rather
than a download. Weights are not committed with the source code.

Uploads, previews, job state, raw tracking reports and compact results are stored
under ignored `data/reference-review/experimental-tracking/jobs/`. Original uploads
are copied byte-for-byte. Preview extraction sequentially decodes the exact frame
used to initialise the tracker; coordinates are normalized against that oriented
source image. No later labels or reseeding guide the run.

For sources with a short edge of at least 720 pixels, inference uses the current
720-pixel processing scale, motion-recovery association and guarded native-detail
supplement. Smaller sources stay at their native size, with no upscaling and no
native-detail supplement. The detector and association settings are unchanged.
The app path omits saved JPEGs for every frame; it returns normalized positions
for the original replay instead.

Uploads are limited to 512 MiB, 30 seconds, 1800 frames and supported 4K dimensions.
The worker enforces decoded-frame/image bounds and a cooperative 20-minute runtime
limit. Cancellation and timeout checks run between frames and inference calls;
they cannot forcibly interrupt a native inference call that has stalled. An OS
file lock prevents concurrent experimental inference, and job state is written
atomically. Abandoned active jobs are marked interrupted when checked after a
backend restart. The ordinary `/analyze` endpoint does not wait for this job.

## API

| Request | Purpose |
| --- | --- |
| `POST /experimental-tracking/preview` | Multipart `video` upload; returns preview metadata and a job ID |
| `GET /experimental-tracking/{id}/preview` | Initial frame JPEG |
| `POST /experimental-tracking/{id}/start` | JSON `{"seed":{"x":0.4,"y":0.6}}`, with coordinates in `[0,1)` |
| `GET /experimental-tracking/{id}` | Status and frame progress |
| `GET /experimental-tracking/{id}/result` | Completed compact trajectory and provenance |
| `POST /experimental-tracking/{id}/cancel` | Cooperative server cancellation |

Job IDs and artifact paths are validated. Starting the same job again with the
same seed is idempotent; changing a started seed requires a new job. The public
contract has no goal, contact, physical speed, curl or technique assessment.
Every candidate retains `identity_verified: false`.

## Offline prediction-drift experiment

`experiments.ball_tracking.track_unlabelled_video` additionally accepts
`--association drift-recovery`. The phone continues to use `motion-recovery`.
This is an inspected-footage development experiment, not a validated upgrade.

During an aligned, short detection gap, the opt-in native search moves its
existing crop budget to the last accepted observation. Its admission radius is
96 working pixels, bounded independently of the drifting velocity prediction.
Native candidates still pass the existing registration, background and temporal
verification checks before association. The alternative association requires
three consecutive unambiguous observations within that anchor, bounded steps,
at least six pixels of net motion and a bounded constant-velocity residual.
Existing off-target stationary history remains a veto. Recovery selects only
the current detection, resets velocity from those observations and relatches
the local guard. Missing frames remain missing; recovery after the existing
lost deadline remains disabled.

These pixel thresholds are provisional settings at the 720-short-edge working
scale, not physical measurements or calibrated identity probabilities. A moving
distractor can still fool them. Compare exact source hashes and initial seeds
with preserved baseline runs, inspect recovery and subsequent identity, and
label difficult frames before reporting accuracy or enabling this in the app.

`--merge-nested-native` is a further offline option with native-detail fallback.
It removes a lower-score native box when it is strongly nested inside a second
box with nearly the same centre and comparable scale. It does not merge by
centre distance alone, merge primary/native evidence, relax background checks,
or verify identity. This addresses duplicate tile proposals resetting temporal
support. Its flag and per-frame removal counts are recorded in each report.
The app does not enable it. Development comparisons must report both gained
and regressed observations, including changes outside the labelled window.

`--trajectory-native-support` is another explicit offline option requiring
native-detail fallback. It retains a native-evidence hypothesis across at most
one empty detector frame. Three actual observations within five frames are
required, with mutual unique assignment, a bounded image-speed check and a
short-horizon constant-velocity residual check in the registered reference
image. A predicted location supplies a matching prior only: empty frames stay
empty, detector coordinates are never replaced with a fitted curve, and this
option does not change association or bypass the background-patch veto. A long
gap, failed registration or unresolved competing candidates cannot contribute
fabricated support. The default verifier and phone configuration remain off.
Reports record the flag, trajectory settings, implementation hash, supporting
frame numbers and whether a detector gap was bridged.

This is a limited test of motion-predicted recovery, not a physical flight
model. Image-plane velocity depends on perspective and camera registration;
it cannot establish depth, spin, a net landing position or a goal-plane crossing.
Retain the measurement/outcome separation even if a plotted path looks smooth.

### Pretrained-model screening

The November 23/24 development review also screens public Hugging Face models
locally, without uploading recordings or changing production weights:

- [acatorcini/yolov9-soccer-ball](https://huggingface.co/acatorcini/yolov9-soccer-ball),
  an amateur-football detector with a fixed 1280-pixel ONNX input;
- [Dura-S/reco-yolo26s-football](https://huggingface.co/Dura-S/reco-yolo26s-football),
  a detector trained for amateur panoramic-camera footage with a fixed
  1920-pixel ONNX input;
- [Meta CoTracker3](https://huggingface.co/facebook/cotracker3), a general learned
  point tracker, evaluated separately as an offline recovery-window estimate.
  Its CC BY-NC 4.0 research licence is distinct from the detector licences.

Downloaded weights and pinned official source stay in ignored
`data/models/research/`, with exact revisions, hashes and licences recorded.
The ONNX/einops research dependencies are installed in that ignored directory,
not added to the app requirements. Screening artifacts live under
`data/reference-review/predictive-recovery-20261124/`.

External-detector screening uses a 768-source-pixel crop around the frozen
tracker's causal query, sampled by frame number before labels are read. It is
not compute- or geometry-matched to the current multiscale detector. A separate
whole-frame check guards against drawing conclusions solely from that crop.
Report candidate recall independently of accepted tracking recall. An unmatched
candidate is a distractor for this target, not automatically a detector false
positive, because the reference boxes do not label every other ball.

CoTracker's probe is explicitly noncausal: it can use later video frames in a
short window, starts from a frozen past tracker selection rather than a human
reference, and produces model visibility/point estimates, not independent
detections. Exclude its supplied seed frame from scoring. Do not merge those
estimates into perception evidence or coaching merely because they are smooth.

The multi-mode reconstruction approach in
[Vorobev et al.](https://arxiv.org/html/2506.07981v1) supports combining motion
priors with detections, but assumes camera calibration and player positions
and reports proprietary broadcast-footage results. It is conceptual guidance,
not an available validated goal-outcome implementation for phone recordings.

The completed development comparison decoded 1,200 frames in eight causal
runs (the rejected replacement and revised additive versions). The additive
option selected correct observations on 79/105 visible reference frames versus
70/105 for the nested-merge baseline, with no regressed or wrong selected
positions in those labelled windows. The difficult QGXD6716 window improved
from 3/26 to 11/26, while remaining substantially incomplete. There are no
absent-target references, and these inspected development windows do not
establish a general false-positive rate. 187 relevant automated checks passed.

Neither external ONNX detector found the reference target on the 34 sampled
visible frames under the tested local-crop setup; the current multiscale
detector supplied a near-target candidate on 27/34. Neither external model
found the target in a separate six-frame whole-image check. These results
reject swapping detectors under those tested settings, not every possible
configuration or deployment domain.

CoTracker3's unfiltered point estimates fell inside the reference boxes on all
79 visible frames across the three clearer short windows, but on only 7/26
visible frames in the occluded window. Its own visibility flag retained 36/79
of the clearer-window estimates and none of the occluded-window estimates.
These are learned estimates using later video frames and automatic seeds from
frozen past selections; they are not 79 additional verified detections or an
end-to-end automatic recovery benchmark. Detector-corroborated learned recovery
is worth testing next. Do not globally override learned visibility, promote the
model into coaching, or infer a successful goal from this comparison.

### Detector-confirmed learned recovery

`experiments/ball_tracking/detector_confirmed_recovery.py` is a separate offline
postprocessor. It does not change the phone tracker, analysis defaults, Jev,
coaching policy or shot outcomes. Its inputs are an existing frozen baseline
report, the exact original video and the same one-class detector weights.

After at least three consecutive selected detector observations, a missing
frame can trigger a short CoTracker3 window. The query is the last detector
selection strictly before the gap, with two real context observations. No
reference labels or manually chosen contact times select these windows. The
recording still has the baseline's initial ball seed; automatic recovery does
not remove that initial setup requirement.

The model requests original-resolution pixels around its proposed location.
Recovery requires exactly one nearby current detector box, successful
background registration, the existing static-background/seed-area vetoes, and
consistent registered detector observations. Accepted coordinates come from
that current box, never from CoTracker or interpolation. Existing selected
baseline positions are preserved byte for byte; learned estimates and checks
are recorded separately. Every added observation remains provisional, with
`identity_verified: false`. These checks reduce ambiguity but do not prove
that a competing ball is the intended ball.

The default requires the official CoTracker visibility flag. An explicit
`--search-uncertain-points` ablation permits uncertain estimates to request
pixels while retaining their false visibility flags. Acceptance then requires
four consecutive actual detector observations, in addition to the other
checks. An invisible estimate alone cannot add a position, and no model
visibility probability is recalibrated or presented as certainty.

Windows are bounded to 28 frames by default, at most four per clip, with a
12-frame maximum gap from a real observation and a cooperative 240-second
budget. Torch inference cannot be interrupted halfway through a call, so this
is not a strict wall-clock timeout. Fixed native crops are at most 768×576;
the detector searches a 384×384 region inside them. Source geometry, hashes,
frame alignment, checkpoint and recorded inference-source hashes are checked.
Missing local model files or failed/malformed model windows preserve the
usable baseline and record the optional recovery failure.

The adapter in `cotracker_recovery.py` uses the previously downloaded and
pinned `facebook/cotracker3` scaled-offline research checkpoint and official
code under ignored `data/models/research/cotracker3/`. It makes no network
request or installation. The CC BY-NC 4.0 research model and isolated research
dependencies are not added to the app's runtime requirements.

Example from the repository root, using the already prepared local assets:

```powershell
python -m experiments.ball_tracking.detector_confirmed_recovery `
  --baseline data/reference-review/predictive-recovery-20261124/trajectory-union/VULK7057/report.json `
  --weights data/models/scan-experimental-ball.pt `
  --cotracker-root data/models/research/cotracker3 `
  --research-runtime data/models/research/runtime `
  --output data/reference-review/detector-confirmed-recovery/manual-VULK7057/report.json
```

Choose a new output path for each run; existing reports are not overwritten.
The model uses later frames within each window, so this is post-recording
recovery, not a causal live tracker or a feedback-latency improvement.

The completed automatic-gap study evaluated four 150-frame recordings with
14 model windows. It added six current detector observations across two
recordings while preserving every existing baseline selection. On the same
105 visible development-reference frames, correct selected positions increased
from 79 to 80 (strict box hits increased from 78 to 79), with zero wrong
selected positions or regressions in those labelled windows. Per clip:

| Clip | Baseline | Detector-confirmed learned recovery |
| --- | --- | --- |
| VULK7057 | 20/24 | 21/24 |
| LPRK9986 | 21/24 | 21/24 |
| DGQH9591 | 27/31 | 27/31 |
| QGXD6716 | 11/26 | 11/26 |

Five additions were outside the labelled windows. The six native-pixel crops
look visually plausible, but those five are not counted as measured accuracy.
There are no absent-target labels; this is not a general false-positive or
held-out accuracy estimate. The existing difficult occlusion window remains
incomplete. The original full runs added about 80–155 seconds per clip on this
CPU, excluding shared model startup and the original baseline tracker.

The uncertain-search ablation reused identical frozen learned estimates and
added no extra observations. Retain the official-visibility default. Its replay
time excludes cached neural-network inference and must not be presented as
deployment latency. Missing detector confirmations and insufficient supported
sequences remain, even with a plausible learned point estimate.

Reports, exact implementation snapshots, model/source hashes, window traces,
both policy comparisons and the interactive visual review are in ignored
`data/reference-review/detector-confirmed-recovery-20261124/`. 169 relevant
automated checks passed, including competing-ball ambiguity, model failure,
mid-window failure without partial publication, geometry/frame alignment and
preservation of the existing path. This is a tested offline implementation
with a modest development gain; continuous recovery is not solved.

### Human reference comparison

Recovery labels use the existing `scripts/ball_label_ui.html` template, now with
Fit, 2× and 4× views. Zoom changes display size only; boxes remain in original
decoded source pixels. Label the intended ball on every prepared frame without
tracker overlays. Include clearly localizable blur in a tight box. Mark hidden
or ambiguous positions uncertain; use absent only when the target is visibly
out of the scene. Never infer a reference position from a prediction.

`python -m experiments.ball_tracking.score_recovery_comparison` accepts
`--manifest`, `--labels`, `--baseline`, `--candidate` and `--output` JSON paths.
It rejects mismatched source hashes, image geometry, seeds, weights and
incomplete reviews. It converts selected observations to source coordinates,
reports strict box hits and hits within the box plus three working pixels,
and reports wrong selections and consecutive visible-frame gaps. Predictions
never count as observations. Uncertain frames are excluded and break gap runs.
These development windows were chosen after inspecting failures and cannot
establish accuracy on an unseen recording session. Reference labels must not
be supplied to tracking inference or used to select its starting seed.

## Ordinary analysis latency

The ordinary `/analyze` request retains native-resolution perception and uses
an annotated replay with a longest edge of at most 1280 pixels. This replay is
a derived visual artifact; it does not replace or modify the source recording,
pose measurements, goal coordinates or ball detections. Trail coordinates are
scaled on copies of detections for the replay. Browser-compatible H.264 encoding
runs once after the optional trail overlay, with a bounded fast encoder. CLI
analysis preserves native artifact dimensions unless a caller supplies a bound.

Player calibration sequentially decodes the same requested sample indices
instead of repeatedly seeking through compressed phone footage. Confidence,
sampling count and player-selection checks are unchanged. Output JSON and API
responses include additive `processing` metadata with stage timings, total
analysis seconds and native/render dimensions. These server timings exclude
upload/body parsing and client polling. External-service timeouts and
deterministic fallbacks remain in place. A local benchmark with services
disabled is not a phone-to-report latency measurement.

## Verification

Automated checks cover upload/seed validation, exact preview coordinates,
resource limits, interrupted/cancelled/failed jobs, separate model loading,
progress and normalized results. Existing offline tests check that default
review output is preserved. Mobile checks cover letterboxing, frame lookup,
observed/predicted/lost separation, multipart upload and status reconnection.

A real-model run on an already inspected clip is an integration smoke test,
not new accuracy evidence. The remaining phone check is to select a ball,
run a job, seek and replay with the overlay, navigate away to confirm pause,
then record, replay and export a short new clip with sound.
