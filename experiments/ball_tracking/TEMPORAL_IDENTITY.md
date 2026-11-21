# Temporal association development check — 2026-11-19

This is an **offline candidate**, separate from SCAN perception, SituatedState,
judgements, policy and coaching. The committed legacy association stays the
default. This experiment does not establish goal crossing, contact, physical
speed, curve or dip.

## What changed

`temporal_identity.py` retains the adapted one-class YOLO detector and its nine
overlapping, prediction-centred slices. It changes which proposal updates the
next search position:

- Prefer a plausible nearby observation over a distant, higher detector score.
- Keep at most two tentative moving paths outside the narrow prediction gate.
  A single distant proposal cannot redirect the main path. Promotion needs two
  consecutive observations with some camera-compensated motion.
- Estimate consecutive-frame background translation. Failed alignment supplies
  no correction; it does not prove that the camera was stationary. This is a
  translation approximation, not full camera calibration or perspective recovery.
- Missing evidence stays missing. Predictions remain separate from observed
  candidates and do not become ball observations. After twelve missing frames,
  the pilot requires a new manual seed instead of adopting a newly appearing ball.

All selected candidates remain **provisional**. Consistent motion and a detector
score do not verify identity, and a distractor inside the narrow prediction gate
can still be selected. In particular, this is not a solved multiple-ball tracker.
There is one manually chosen starting position per clip and no later re-seeding.
The reviewer references are not loaded by the tracker.

## Matched experiment

The baseline is preserved under ignored
`data/reference-review/new-session-20261119/runs/`. The revised full-video run is
under `data/reference-review/identity-v2/runs-confirmed/`. Both use the same
eleven distinct videos, source bytes, starting frames/positions, detector weights
and detector implementation/settings. Predictions and therefore the searched
crop positions may differ, so inference is rerun closed-loop on source video.

The model is the three-epoch adapted YOLO11n checkpoint at
`data/reference-review/ball-finetune/20261117T004329962346Z/train/weights/best.pt`.
Its SHA-256 is
`8e1467e9fd3e2ce213ec262ab8aa10deb3422a67e99c008f4f50e0b7e296cdd0`.
Detector settings remain 384-pixel crops, offsets of 128 pixels, confidence 0.05,
class 0, 640-pixel model input and CPU inference. Settings and implementation
hashes are frozen before running; reports preserve original review JPEGs.

These recordings were inspected while developing this rule. They are now
**development evidence**, not a fresh evaluation session. All are the original
720 × 1280 exports; this run cannot establish a benefit from 1080p or 4K capture.

An initial three-clip trial under `identity-v2/runs/` missed the fast launch in
`k9tpxsbg`. Its tentative-path allowance was too narrow during acceleration.
That trial is preserved. The confirmed version expands the tentative-path
padding from 8 to 16 pixels at this resolution while retaining two-frame
confirmation. A regression check covers that launch pattern. The corrected
version is frozen for the full-session comparison; it is not tuned per clip.

## Provisional visual references

Three short failure windows have 62 reviewed original frames, with 59 approximate
visible ball centres and three uncertain frames excluded from proximity counts.
These are **assistant visual references requiring human confirmation**, not
independently verified ground truth. The references record source/video and
frame hashes, visibility and pixel uncertainty. They are used only after
inference. The comparison rejects differences in source, seed, frame coverage,
detector weights/settings/implementation or reviewed frame bytes.

`review_identity_comparison.py` reports selected centres within 10/20 pixels,
selected centres over 20 pixels away, missing selections, contiguous nonmatches
and recovery events. These are local diagnostic counts, not calibrated identity
probabilities or whole-session accuracy. A 20-pixel tolerance is large relative
to some distant balls. Unlabelled frames elsewhere have no accuracy score.

## Completed comparison

Both versions processed all 1,531 frames after the same starting points in eleven
videos. The two delayed starts exclude sixty earlier frames. The baseline selected
1,272 candidates; the revised version selected 1,063. These are coverage counts,
not correct-identity counts.

On the three preselected failure windows, selected centres within 20 pixels of
the provisional visual reference changed as follows:

| Clip suffix | Visible references | Baseline nearby | Revised nearby | Baseline wrong / missing | Revised wrong / missing |
| --- | ---: | ---: | ---: | ---: | ---: |
| `okonr73x` | 19 | 7 | 13 | 1 / 11 | 0 / 6 |
| `b9rlx186` | 17 | 5 | 16 | 12 / 0 | 0 / 1 |
| `5mcmluz8` | 23 | 7 | 10 | 13 / 3 | 0 / 13 |
| Total | 59 | 19 | 39 | 26 / 14 | 0 / 20 |

Here "wrong" means selected more than 20 pixels from the provisional reference;
it is not an independently verified identity judgement. These small, deliberately
chosen failure windows do not represent the complete session. Nearby counts in
this comparison are also within 10 pixels. References still need human review.

Original-frame spot checks support the gains in the first two windows: the
revised path stays on the visibly flying ball where the baseline switches to
clothing or a distant candidate. In the third window it rejects the false turf
branch, but loses thirteen visible positions instead. This is a real tradeoff,
not a complete fix.

The broader check also finds a **goal-area regression** in `vaokupxq`: the
revised version follows launch/early flight but fails to reacquire the ball
bouncing/settling in the net after approximately frame 87. The baseline recovers
around frame 93 and tracks much of the remaining net/resting sequence. In
`k9tpxsbg`, both versions find the real ball near the left post; the revised
version adds gaps during the rebound and after it comes to rest. Late missing
states in `b9rlx186` also omit a faintly visible ball near the goal base.

**Decision:** keep temporal association optional and offline. Do not replace the
legacy default or feed this candidate into coaching/outcomes. The next identity
change needs to distinguish a justified bounce/low-motion recovery from a static
distractor, and to preserve continuity in previously working goal-area cases.
Keep these reports frozen when evaluating that change.

The completed local viewers are `identity-v2/review/index.html` (all eleven
unlabelled paths) and `identity-v2/comparison/index.html` (three provisional
reference windows). Per-clip coverage, hashes and limitations are in
`identity-v2/summary.json` and `identity-v2/RESULTS.md`. Original run metadata
contained inherited "unseen-session" wording in its limitation text; this was
corrected to inspected development without changing inference. Original report
bytes and a field/hash correction log are preserved under `identity-v2/provenance/`.

## Reproduce and review

Run from the repository root, using a new run directory to preserve earlier
reports. The local source videos, ignored references and checkpoint must exist.

```powershell
python -m experiments.ball_tracking.track_unlabelled_video --cases data/reference-review/new-session-20261119/seeds.json --weights data/reference-review/ball-finetune/20261117T004329962346Z/train/weights/best.pt --output data/reference-review/identity-v2/runs-next --threads 8 --association temporal
python -m experiments.ball_tracking.render_unlabelled_review --reports-dir data/reference-review/identity-v2/runs-next --output data/reference-review/identity-v2/review-next
python -m experiments.ball_tracking.review_identity_comparison --baseline data/reference-review/new-session-20261119/runs --revised data/reference-review/identity-v2/runs-next --annotations data/reference-review/identity-v2/annotations --output data/reference-review/identity-v2/comparison-next
```

The full-session viewer shows selected candidates in yellow, optional proposals
and predictions, and the reason for an association or missing result. The
targeted comparison uses separate baseline/revised tabs: yellow is baseline,
green is revised and cyan is the provisional reference region. Cyan denotes
review uncertainty around a centre, not a segmentation mask or physical ball box.

## Remaining limitations

The main tradeoff is losing valid observations after abrupt deceleration,
rebounds, long gaps or a ball coming to rest. The consecutive-motion rule can
reject the real stationary ball after a gap. A camera translation estimate can
also be inaccurate during rotation or strong perspective change. These failure
modes need explicit handling and a genuinely new recording session before this
candidate can replace a production perception component.

The next recording check should keep the camera steady and frame the entire
flight and goal. Verify the dimensions of exported 1080p recordings, keep the
tracker frozen, and evaluate the new session separately from these development
clips. Preserve originals for detection and use annotations only for evaluation.

## Verification

The expanded association, recovery, unlabelled-runner, review and comparison suite
passes 67 checks. It covers weak ball/strong distractor competition, missing observations,
launch acceleration, long loss, camera translation, preservation of earlier
reports and comparison provenance. Both generated viewers also pass JavaScript
syntax checks and image-link/dimension checks. Physical phone capture still needs
the export check described in the mobile README.

```powershell
python -m pytest tests/test_recovery_identity.py tests/test_temporal_identity.py tests/test_unlabelled_ball_video.py tests/test_associate_detector_candidates.py tests/test_render_unlabelled_review.py tests/test_identity_comparison.py -q --basetemp data/reference-review/pytest-identity-check
```

## Optional local rebound/rest recovery

`recovery_identity.py` subclasses the unchanged temporal rule. It adds a separate
local hypothesis only when that rule abstains. It reads current detector
proposals and past observations, never reference labels or future frames.

This hypothesis needs at least five accepted observations, a last estimated
ball velocity no greater than 12 pixels/frame, and available translation
alignment. It considers only candidates within 32 pixels of the camera-aligned
last observed ball position. Three observations must agree within a seven-frame
window, with at most three absent frames between observations and no more than
eight pixels/frame of relative movement. Small direction reversals and little
or no motion are permitted. Pixel limits scale with the decoded shorter image
dimension relative to 720. These are provisional settings for these roughly
30-fps clips, not validated physical limits.

Two nearby candidates leave recovery ambiguous; detector score alone cannot
select between them. The original twelve-frame loss horizon remains in force.
Recovery selects the **current** detector observation only; it does not fill
the earlier gap or label an unobserved position as the ball. The new report
reason is `slow_recovery_provisional`; `identity_verified` remains false.

After a recovery, association stays bounded to 32 pixels from the last observed
position (adjusted for camera translation) until a new manual seed. The anchor
updates with real detections, so a locally moving ball can still be followed.
This prevents a later moving-branch promotion from escaping to distant white
background marks, but can lose a real ball that moves farther during a gap.
Raw detector proposals remain in the reports; only the association view is gated.

The first recovery trial under `identity-recovery/runs/` correctly recovered
several balls, but `j2aq2uji` frame 139 joined two different white background
proposals into a false moving branch after recovery. That trial was stopped and
preserved. The guarded version is evaluated separately in
`identity-recovery/runs-guarded/`; a regression test covers the observed escape.

An important remaining negative is a single persistent **nearby static
distractor**. A synthetic fixed proposal 20 pixels from a last stationary ball
position can satisfy this rule after three observations. With these inputs,
nearby repetition cannot prove that the object is the same ball. This needs
hard-negative evaluation and potentially additional appearance evidence; do not
present the hypothesis as solved identity verification or enable coaching from it.

The new mode is opt-in as `--association recovery`; legacy remains the default,
and `--association temporal` preserves the prior behavior. Both the new module
and inherited temporal module hashes are recorded. All earlier reports stay
preserved. The local run/review root is
`data/reference-review/identity-recovery/`.

```powershell
python -m experiments.ball_tracking.track_unlabelled_video --cases data/reference-review/new-session-20261119/seeds.json --weights data/reference-review/ball-finetune/20261117T004329962346Z/train/weights/best.pt --output data/reference-review/identity-recovery/runs-next --threads 8 --association recovery
python -m experiments.ball_tracking.render_unlabelled_review --reports-dir data/reference-review/identity-recovery/runs-next --output data/reference-review/identity-recovery/review-next
python -m experiments.ball_tracking.review_identity_comparison --baseline data/reference-review/identity-v2/runs-confirmed --revised data/reference-review/identity-recovery/runs-next --annotations data/reference-review/identity-recovery/annotations --output data/reference-review/identity-recovery/comparison-next
```

The recovery references add original-frame windows around `vaokupxq` rebound
and `k9tpxsbg` rest. Their author had already inspected candidate paths; this
prior exposure is disclosed. All centres are provisional, require human
confirmation and remain evaluation-only. These inspected clips are not an
unseen test. Ball colour has not changed in this experiment. A later colour
comparison should use the same resolution, camera placement and lighting for
both balls to separate colour from capture changes.

### Guarded recovery results

The guarded version completed all eleven clips and all 1,531 requested frames.
It selected 1,185 candidates versus 1,063 in temporal-v2 and 1,272 in the original
legacy run. These remain coverage counts, not correct-identity counts.

The five provisional windows contain 107 visible references in 112 reviewed
frames; five uncertain frames are excluded. All three versions use the same
references and decoded source images:

| Window | Visible references | Original legacy nearby | Temporal-v2 nearby | Guarded recovery nearby |
| --- | ---: | ---: | ---: | ---: |
| `k9tpxsbg` resting ball | 19 | 14 | 2 | 15 |
| `okonr73x` failure window | 19 | 7 | 13 | 13 |
| `b9rlx186` failure window | 17 | 5 | 16 | 16 |
| `vaokupxq` rebound | 29 | 17 | 2 | 15 |
| `5mcmluz8` failure window | 23 | 7 | 10 | 10 |
| Total | 107 | 50 | 43 | 69 |

"Nearby" is a selected centre within 20 pixels of an approximate visual
reference. In these windows the nearby centres are also within 10 pixels.
Selected centres beyond 20 pixels number 26, 0 and 0 respectively; missing
selections number 31, 64 and 38. This is an inspected development comparison,
not validated accuracy or a claim of zero identity errors over the full session.

Original-frame audits support the added goalpost/net/rest observations. The
first trial's wrong `j2aq2uji` frame 139 is rejected by the guard; the corrected
closed-loop run subsequently recovers the real ball at frame 141. Original
frames and earlier trial reports are preserved with the audit evidence.

There is a **real fast-flight regression** in `ao346of1`: recovery at frame 78
correctly finds the ball before the later shot, but the permanent local guard
then blocks a genuine fast-flight return and loses the original-v2 observation
at frame 110. It also removes an apparent old false selection at frame 111;
therefore its selected count changing from 51 to 50 is not an accuracy measure.
The guard needs a justified way to release when the ball accelerates again.
In `5mcmluz8`, the feature adds only one correct-looking observation at frame
111; sparse detections still prevent durable recovery.

**Decision:** keep this an opt-in offline mode. It improves the reviewed
stationary/goal-area cases but cannot replace the default or support shot
outcomes/coaching yet. Preserve the frozen results before working on motion
transitions. A fresh session can then test generalisation and the colour/capture
comparison without changing the tracker during evaluation.

Local outputs are `identity-recovery/review/index.html` (all eleven clips),
`identity-recovery/comparison/index.html` (versus temporal-v2), and
`identity-recovery/comparison-original/index.html` (versus original legacy).
Detailed counts, provenance and limitations are in `summary.json` and
`RESULTS.md` under that directory. Source/QC audits are in `audit/`.

## Optional release after renewed movement

`motion_recovery_identity.py` adds the opt-in `--association motion-recovery`
mode. The temporal and guarded-recovery implementations remain unchanged;
legacy remains the default. The new mode only changes behaviour after a slow
recovery has activated the local guard. It never supplies coaching evidence or
shot outcomes.

Release is earned by **accepted local observations**, rather than by distant
tentative proposals. It requires three consecutive camera-adjusted steps of at
least 12 pixels/frame with pairwise direction cosine at least 0.8. A gap,
unavailable camera alignment, slow/reversing movement or a competing nearby
proposal clears the partial support. At these settings four accepted positions
are needed to supply the three steps. Thresholds scale with image dimensions;
they are provisional image-space settings for these approximately 30-fps clips,
not physical ball speed or frame-rate independent limits.

Once armed, candidates must remain in a prediction-centred corridor equal to
the parent's speed-dependent core radius plus its 16-pixel branch padding.
The parent still requires temporal support after gaps. Observed slowing or
direction change, unresolved ambiguity, camera-alignment failure or hard loss
restores the local guard. The twelve-frame loss horizon is unchanged; missing
frames are never backfilled and a lost target still requires a new seed.

A second safeguard retains up to eight prior frames of camera-aligned detector
proposals at least 48 pixels away from the observed target (or last observation
when missing). A current corridor candidate repeating within four pixels in at
least three prior frames is vetoed. This is **stationary off-target evidence**,
not a confirmed background class. It can reject a real ball that stops at an
overlapping position. It does not solve appearance identity or multiple balls.
Raw proposals, veto centres, release support and reasons remain in each report.
No reference annotations or future frames are inputs to either safeguard.

### Frozen regression run

All eleven original clips completed closed-loop detection again: 1,531 requested
frames, unchanged videos, weights, detector, seeds and settings. Three-module
association hashes and a source snapshot are preserved. All original JPEG
hashes match the previous guarded run. The outputs are under ignored
`data/reference-review/identity-motion-recovery/`.

Only `ao346of1` activates release, at frame 95, after the accepted launch chain.
It adds actual-looking ball observations at frames 110, 115, 116 and 117 while
rejecting recurring bag/base and goal-wheel proposals. The known false
`j2aq2uji` frame 139 remains missing. The other ten paths are unchanged within
one pixel and have the same selected-frame counts. This is deliberately narrow
regression evidence, not a claim of broad recovery performance.

Selected-frame coverage changes from 1,185 to 1,189; it is not accuracy. The five
previous provisional windows retain 69/107 nearby visible reference positions.
A new, non-blind original-frame review at `ao346of1` frames 110–121 has nine
visible centres and three excluded occluded/uncertain frames. Guarded recovery
selects 0/9 nearby centres there; motion recovery selects 4/9. Five visible
positions still go missing. The earlier temporal-v2 run has 1/9 nearby and an
apparent bag/base false selection at occluded frame 111. That excluded frame
is not folded into the visible-reference proximity score.

The six windows together have 116 visible centres: guarded 69 nearby / 47
missing versus motion recovery 73 nearby / 43 missing, with no selected centre
over 20 pixels away in these windows. All nearby matches here are also within
10 pixels. References remain approximate assistant visual marks needing human
confirmation, exposed to previous candidates, and evaluation-only. These are
chosen development windows, not session-wide accuracy or an independent test.

The final focused suite passes 88 checks, including acceleration, static
proposals, ambiguous candidates, camera pan/failure, scale, slow relatching,
hard loss and invalid-frame state preservation. Keep this version frozen for
the next recording/colour comparison. Remaining distant-ball detections,
occlusion gaps, curved paths and multiple-ball identity need further evidence.

```powershell
python -m experiments.ball_tracking.track_unlabelled_video --cases data/reference-review/identity-motion-recovery/cases.json --weights data/reference-review/ball-finetune/20261117T004329962346Z/train/weights/best.pt --output data/reference-review/identity-motion-recovery/runs-next --threads 8 --association motion-recovery
python -m experiments.ball_tracking.render_unlabelled_review --reports-dir data/reference-review/identity-motion-recovery/runs-next --output data/reference-review/identity-motion-recovery/review-next
python -m experiments.ball_tracking.review_identity_comparison --baseline data/reference-review/identity-recovery/runs-guarded --revised data/reference-review/identity-motion-recovery/runs-next --annotations data/reference-review/identity-motion-recovery/annotations --output data/reference-review/identity-motion-recovery/comparison-next
python -m pytest tests/test_motion_recovery_identity.py tests/test_recovery_identity.py tests/test_temporal_identity.py tests/test_unlabelled_ball_video.py tests/test_associate_detector_candidates.py tests/test_render_unlabelled_review.py tests/test_identity_comparison.py -q --basetemp data/reference-review/pytest-motion-recovery-next
```

The completed full viewer is `identity-motion-recovery/review/index.html`.
`comparison/index.html` compares the permanent guard with motion release;
`comparison-v2/index.html` adds context against the earlier temporal version.
Yellow is the baseline selection, green the revised selection, and cyan the
provisional reference. In the full unlabelled viewer, yellow is the new tracker
selection. Predictions and missing positions remain separate from detections.
