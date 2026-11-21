# Hand-seeded post-kick ball-motion pilot

The latest guarded detector and motion-recovery runner also has an opt-in
[saved-rep replay adapter](../../docs/experimental-tracking.md). That adapter
uses one initial ball tap and keeps every result separate from coaching and
outcomes. The experiments and historical results below remain research tools.

This is an **offline experiment**, separate from SCAN's production perception
pipeline. It asks whether motion can recover a ball trajectory after a human
supplies one reviewed contact-frame box and an approximate goal centre. One
additional case starts with an in-flight box and no goal centre. The pilot
does not find either seed or goal centre automatically.

`pilot_motion_track.py` aligns consecutive frames using background feature
motion, extracts small moving regions, and uses a beam search to prefer a
smooth path initially moving toward the goal. It compares absolute frame
change with positive brightness change. Up to two frames may be coasted when
the ball is obscured at impact. It never calls the output confirmed ball
identity or a confirmed goal.

Run from the repository root:

```powershell
python -m experiments.ball_tracking.probe_motion_candidates
python -m experiments.ball_tracking.pilot_motion_track
python -m experiments.ball_tracking.render_pilot
python -m experiments.ball_tracking.diagnose_launch
python -m experiments.ball_tracking.pilot_motion_track --launch-progress 12 --report-name report-launch12.json
python -m experiments.ball_tracking.render_pilot --report-name report-launch12.json --review-name launch12 --early-frames 12
python -m experiments.ball_tracking.pilot_motion_track --launch-progress 12 --launch-soft-weight 0.1 --report-name report-launch-soft01.json
python -m experiments.ball_tracking.render_pilot --report-name report-launch-soft01.json --review-name launch-soft01 --early-frames 12
python -m experiments.ball_tracking.pilot_motion_track --launch-progress 12 --launch-soft-weight 0.1 --include-negative --report-name report-three-cues.json
```

`probe_detector_recall.py` is a separate fixed-sample confidence sweep for the
dedicated YOLO detector. On 27 held-out crops, the one-epoch checkpoint at
640-pixel inference and 0.25 confidence found 1 of 18 labelled balls with 2
false proposals. Lowering confidence to 0.01 found 5 of 18 but produced 178
false proposals. This motivated the motion-candidate pilot; simply relaxing
the detector threshold is not a viable tracker.

The reports and visual review go to `data/reference-review/motion-candidates/`
and `data/reference-review/beam-motion-track/`. The scripts use six existing
reviewed clips and hand-read goal centres recorded in the pilot script. The
image review marks the estimated trajectory in yellow, reviewed ball boxes in
green, and goal centre in pink.

## Current diagnostic result

These are **reviewed post-kick frames in the first 50 frames after the manual
seed**. The pilot chose settings after inspecting these clips, so the numbers
are development checks, not independent performance estimates.

| Clip | Motion cue | Centre within 10 px | Centre within 40 px |
| --- | --- | ---: | ---: |
| `13a1022c5009` | Brightening | 13/18 | 18/18 |
| `560706a458b1` | Absolute change | 19/22 | 21/22 |
| `6292a1b2b194` | Absolute change | 18/22 | 20/22 |
| `e9935750581e` | Brightening | 18/19 | 19/19 |
| `96b382d82d92` | Absolute change | 14/17 | 16/17 |
| `ee73dffd61b5` (in-flight seed, no goal cue) | Absolute change | 6/6 | 6/6 |

The held-out daylight clips were used while choosing this approach; a fresh
recording from another session is needed before claiming generalisation.
Frames without reviewed ball boxes are not counted, and the candidate may
drift after the last reviewed frame. A nearby centre does not prove the
candidate is the ball. The current fixed pixel thresholds also assume roughly
30-fps, 720p/1280p footage and need calibration across camera setups.
The in-flight case has only six reviewed frames in this interval, so its
6/6 result says little about tracking through the later flight or rebound.

## Early-flight launch experiment

In `13a1022c5009`, the brightening cue had a proposal near the marked ball
from the first reviewed frame after contact, yet the beam initially selected
a slower player/foot region. The optional `--launch-progress 12` run requires
the candidate to move at least 12 image pixels toward the hand-read goal per
frame during the first six frames. It changed this clip's brightening path from
13/18 to 18/18 reviewed centres within 10 px; in the first ten post-seed
frames the count changed from 4 to 9. This did not use later ball labels to
choose the path, but the rate and rule were selected while inspecting these
same clips, so this is a development result only.

The hard rule can also **destroy a usable path**: with the absolute-change cue,
three clips terminated early. A soft shortfall penalty (weight 0.1) gives the
same 18/18 early-flight improvement in `13a1022c5009` while leaving the other
five clips' reviewed counts and tracked lengths unchanged in this small pilot.
The soft version remains optional and is not a validated production default.
The positive/absolute cue is still selected manually for these reviews, and a
reliable automatic choice and explicit lost/ambiguous state are needed before
integration. The detailed soft-launch review is at
`data/reference-review/beam-motion-track/launch-soft01/index.html`.

A third **darkening** cue was also tried because a brightening region in
`560706a458b1` appeared one frame behind the ball. Darkening did not improve
that clip over absolute change (15/22 versus 19/22 within 10 px) and failed
badly on two others. It remains an opt-in diagnostic, not a recommended cue.

The next implementation step is a reliable automatic handoff from the
pre-kick seed to motion tracking, plus validation of the provisional
ambiguity/loss warning. Only after independent recordings confirm the path
should this feed shot timing, goal
crossing, or coaching. Pixel displacement is not physical ball speed; dip and
curve need calibrated geometry and reliable trajectory observations.

## Additional Expo recordings (unlabelled visual check)

`pilot_new_clips.py` applies the same soft-launch settings to three other
recordings, with one manually chosen starting ball box and goal centre per
clip. It runs both positive and absolute change for 30 frames after the seed.
These recordings were not used to select the pilot settings. Review the
overlays at `data/reference-review/beam-motion-track/new-clips/index.html`.

| Clip | Visual check |
| --- | --- |
| `e3949580d0f2` (night) | Both cues follow the visible flight into the goal area through about frame 130. The positive path drifts off the ball by frame 135 while the ball remains visible in the net. |
| `735e17f02bd5` (night) | Positive follows the ball across the inspected flight frames, including frames 94, 101 and 112. Absolute differs near launch but converges on the same ball later. |
| `675e44d940ff` (day) | Both cues visually follow the ball from the first flight frame through frame 77, including its path across the goal. |

Eight evenly spaced post-seed frames in each clip were subsequently marked on
the original full-resolution video without tracker overlays. The resulting
small check is at `data/reference-review/beam-motion-track/new-clips-scored/`
and its approximate ball-centre labels are in `reviewed_new_clips.json`.

| Clip | Positive within 20 px | Absolute within 20 px | Largest positive error |
| --- | ---: | ---: | ---: |
| `e3949580d0f2` | 7/8 | 6/8 | 69.8 px at frame 135, where it follows net motion |
| `735e17f02bd5` | 8/8 | 5/8 | 13.5 px |
| `675e44d940ff` | 8/8 | 8/8 | 5.3 px |

Positive change reached 23/24 of these sparse marks within 20 px; absolute
change reached 19/24. In `735e17f02bd5`, absolute change missed three early
flight marks by about 91–186 px before reacquiring the ball. These figures describe
only 24 selected frames in three hand-seeded clips, with approximate human
centres. They are not full-sequence tracking accuracy, an identity-swap rate,
or an independent deployment benchmark. The setup still requires a hand-picked
seed and goal centre and is not connected to the app's analysis pipeline.

## Provisional path reliability warning

The offline pilot now marks a path **uncertain** from a late near-goal motion
jump following several slow frames. The boundary is sticky because the pilot
cannot independently re-identify the ball after a suspicious handoff. It keeps
the candidate path for visual debugging but does not treat those later points
as reliable ball evidence. `reliability.py` owns this provisional check.

On the three sparsely reviewed new recordings, both cues are flagged from
frame 132 in `e3949580d0f2`. The one labelled frame after that boundary is
indeed wrong (positive error 69.8 px; absolute error 102.7 px). Neither of the
other two clips is flagged. Several unlabelled late transitions in older
development clips are also flagged; they may include real rebounds, so these
are warnings rather than confirmed identity swaps. The early-flight absolute
errors in `735e17f02bd5` remain undetected by this rule. The rule was chosen
while inspecting these same clips. It is offline only and must not gate app
feedback yet.

## Additional distinct-recording check

`holdout_shot_check.py` freezes the same tracker settings and runs two more
byte-distinct phone recordings. The starting boxes, approximate goal centres,
and sparse ball centres were read from original frames before this tracker run.
The frame-by-frame overlay and machine-readable scores are at
`data/reference-review/beam-motion-track/holdout-additional-clips/`.
Many other filenames in `data/input/` are exact copies of earlier recordings;
they were not counted as new test clips.

| Clip | Situation | Positive within 20 px | Absolute within 20 px | Warning on wrong marks |
| --- | --- | ---: | ---: | ---: |
| `36bf848f409c` | Daytime shot | 5/10 | 0/10 (only 1/10 tracked) | 0/5 |
| `7de88d94f23c` | Nighttime dribble stress case | 0/5 | 0/5 (0/5 tracked) | 0/5 |

In the daytime shot, the positive path misses the first five reviewed flight
positions by 36–57 px, then converges to within 14 px on the last five. For
those early marks, comparing each marked ball with the *next* tracker frame
reduces median error from 37.4 to 7.4 px, suggesting the brightening proposal
tracks the just-vacated ball location rather than the current ball. This is a
diagnostic, not a reason to shift every trajectory by one frame. In the
nighttime dribble, it follows the player rather than the visible ball, with
121–170 px error. The warning stays `provisional` in both cases, even though
those marked positions are wrong. The dribble is a stress case, not a second
shot benchmark. This small check is evidence that the current warning misses
early and player-handoff failures. More automatic starting-box work should
wait until ball identity and loss/ambiguity are made reliable. Neither path
should be used for shot outcome or coaching yet.

## Current-frame identity experiment

`pilot_identity_check.py` compares the frozen positive-change baseline with a
new **offline-only** candidate. The candidate pools brightening and darkening
motion regions, adds a seed-patch appearance cost, and rejects obviously
different patches during the first six post-seed frames of a shot. The
nighttime dribble stress case applies that gate through its 30-frame review
window. This distinction is a manually specified pilot profile, not an
automatically inferred ball state. Appearance is deliberately a soft cue
after early shot flight: in one valid night clip, the true ball's colour and
texture change enough that a fixed seed-template gate would lose it.

Run from the repository root:

```powershell
python -m experiments.ball_tracking.pilot_identity_check
python -m experiments.ball_tracking.pilot_legacy_check
python -m pytest tests/test_ball_motion_appearance.py tests/test_ball_motion_reliability.py tests/test_ball_motion_evaluation.py -q
```

Review the generated overlay and `scores.json` at
`data/reference-review/beam-motion-track/identity-pilot/`. Green is the
manually marked ball, blue the frozen positive-change baseline, and yellow
the appearance candidate. The magenta cross is pooled positive/negative
motion without appearance. Raw distance to **sparse** reviewed centres is:

| Clip | Baseline within 20 px | Appearance within 20 px | Pooled motion within 20 px |
| --- | ---: | ---: | ---: |
| `36bf848f409c` (day shot) | 5/10 | 10/10 | 10/10 |
| `7de88d94f23c` (night dribble) | 0/5 | 5/5 | 0/5 |
| `e3949580d0f2` (night shot) | 7/8 | 7/8 | 7/8 |
| `735e17f02bd5` (night shot) | 8/8 | 7/8 | 8/8 |
| `675e44d940ff` (day flight) | 8/8 | 8/8 | 8/8 |

The current-frame check improves the two recent failure cases but still
misses one early mark in `735e17f02bd5` and follows net motion at the last
mark in `e3949580d0f2`. All five candidate paths are currently marked
`uncertain` because a predicted position or weak appearance occurred; that
warning remains sticky even if later positions happen to be near manual
marks. The settings were chosen while examining these same five recordings,
so the table is a regression check, not an independent accuracy claim. The
production app still uses its existing perception pipeline. Before an
automatic seed or shot-outcome integration, this needs newly recorded clips,
dense frame labels, and a more reliable identity/loss decision.

The separate `legacy-reviewed.json` checks six earlier development clips in
their first 30 post-seed frames. The candidate is within 20 px on 6/17, 20/20,
14/20, 16/17, 16/16 and 3/3 visible reviewed frames respectively, in the
order listed in `pilot_motion_track.py`. The poor 6/17 result on
`13a1022c5009` shows that this appearance profile does not transfer reliably
to a seed placed at the contact frame. These clips informed the earlier
motion pilot and are not fresh holdouts.

## Contact-seed regression audit

`contact_regression_audit.py` checks raw motion proposals against the marked
ball in `13a1022c5009`; it does not use the marks to guide tracking. A nearby
brightening proposal exists on every reviewed visible frame from 80 through
100 (1–9 px from the marked centre). The appearance-aware path instead
diverges after contact. Removing only the six-frame appearance gate does not
fix it: both appearance variants reach 6/17 within 20 px, whereas pooled
motion without appearance reaches 17/17. Positive motion plus appearance
performs worse at 1/17. The seed-patch appearance cost, rather than missing
motion candidates, is the main failure in this case.

The six earlier development clips yield 92/93 reviewed positions within
20 px for pooled signed motion without appearance, versus 75/93 for the
appearance candidate. On the five newer clips, pooled motion matches the
appearance candidate on three shots, improves `735e17f02bd5` by one mark,
but fails all five reviewed marks in the night dribble where appearance helps.
This supports keeping motion-only and appearance-aware paths as separate
offline hypotheses, with a capture-condition or track-state decision still
to be validated. It does **not** justify selecting whichever path is closest
to labels per clip at runtime. The labelled clips were used to explore both
settings, and neither path establishes identity, contact, goal entry or speed.

## Stationary-shot review on an exported rep

Use `stationary_shot_review.py` to run the motion-only candidate on any local
video without a labels file. First inspect the original video and identify
the contact frame, a tight ball box in that frame, and an approximate goal
centre, all in **original video pixel coordinates**. For the already reviewed
`13a1022c5009` example:

```powershell
python -m experiments.ball_tracking.stationary_shot_review data/input/13a1022c5009.mov --contact-frame 78 --ball-box 250 749 287 783 --goal-center 260 610
```

The command prints a new `data/reference-review/beam-motion-track/stationary-shot/`
review page and saves `trajectory.json` beside it. Yellow marks the candidate;
a red cross marks uncertainty. The output is a visual debugging aid, not a
verified shot result. This path requires a manually marked contact frame and
ball, a manually chosen goal centre, and a stationary camera. It is **not**
connected to mobile uploads, Jev, goal outcomes, or coaching. The next step
is to score newly exported stationary shots through contact and flight before
designing an app seed workflow or replacing the production tracker.

For a **new, distinct stationary-shot recording**, keep the tracker output
closed until the original video has been reviewed and the ball boxes saved.
Place a copy of the exported clip in `data/input/`, find the contact frame in
the original video, then prepare a dense annotation window of at least 30
frames after contact:

```powershell
python -m scripts.ball_dataset prepare --source data/input/NEW_CLIP.mov --stride 10 --dense-start CONTACT_FRAME --dense-end END_FRAME
```

Use the generated `index.html` to mark the target ball on **every** frame from
contact + 1 through contact + 30; mark occluded frames uncertain rather than
guessing. Download and save its `labels.json` as explained in
`docs/ball-labelling-guide.md`. Only then run `stationary_shot_review.py`
with the contact-frame ball box and goal centre read from the original image.
Finally score the saved path:

```powershell
python -m experiments.ball_tracking.score_stationary_shot PATH_TO_TRAJECTORY_JSON PATH_TO_LABELS_JSON
```

The scorer checks the source hash and requires a complete dense review window.
Its 20-pixel proximity count includes lost frames in the denominator and
reports provisional observed positions separately. It does not establish
ball identity or a confirmed goal. Do not adjust tracker settings using this
clip and then call the same clip an independent test.

An end-to-end check on the already reviewed `13a1022c5009` frames 79–95
returned 16/16 visible positions within 20 px, but 0/16 as provisional
observed positions. Frame 79 is marked uncertain in the manual labels; the
tracker coasts it, so the current sticky reliability warning marks the later
path uncertain. This illustrates why raw position proximity alone cannot
authorize a shot outcome or coaching claim.

An extended, **development-only** check on the existing `IMG_5703.mov`
stationary shot used the reviewed frame-50 ball box and tracked frames 51–100.
The motion-only candidate was within 10 px on all 20 visible labelled
positions through frame 80, including frame 80 in flight. The path then
diverged near the goal: its errors were 101.6 px at frame 90 and 92.8 px at
frame 100. Its reliability check became uncertain at frame 83 with reason
`late_near_goal_motion_jump`, before the two large labelled errors. Of the
50-frame window, 22 positions have visible labels and frame 63 is unreviewed;
the rest of the late flight is sparsely labelled. These figures are therefore
an exploratory sparse-label comparison, **not** a passing result from the
dense-window scorer or independent validation. The generated review is under
`data/reference-review/beam-motion-track/stationary-shot/IMG_5703-1ed7f8ca/`.
Next, investigate late-flight identity near the goal while preserving the
uncertainty gate; do not infer goal entry or trajectory quality from this path.

## Six-shot extended development check (2026-11-16)

The same motion-only settings were run for 50 frames on six other shot clips
already used during development. The first three were seeded at a reviewed
contact frame; the last three used existing pre-shot seeds. These are sparse
reviewed-position checks, **not** new holdouts or dense flight validation.
The first three source files match their annotation SHA-256 hashes.

| Clip | Seed | Reviewed centres within 20 px | Provisional observed hits | First uncertainty |
| --- | --- | ---: | ---: | --- |
| `13a1022c5009` | contact | 18/18 | 0 | frame 79, missing proposal |
| `IMG_5702` | contact | 21/22 | 21 | frame 90, missing proposal |
| `e9935750581e` | contact | 18/19 | 0 | frame 118, missing proposal |
| `36bf848f409c` | pre-shot | 10/10 | 10 | frame 85, late near-goal jump |
| `e3949580d0f2` | pre-shot | 8/8 | 7 | frame 132, missing proposal |
| `735e17f02bd5` | pre-shot | 8/8 | 8 | none within 50 frames |

The two labelled misses were 98.2 px at `IMG_5702` frame 90 and 22.5 px at
`e9935750581e` frame 118; both occur at an uncertainty boundary. The 13a
and e993 paths later return close to sparse labels, but their warnings stay
sticky because the pilot cannot verify re-identification. The 36bf warning
falls after its last reviewed position, so its late flight is unscored.
Even the unflagged 735 path has no reviewed positions after frame 113;
visual proximity to the goal cannot validate its identity or outcome.
The [local review](../../data/reference-review/beam-motion-track/multi-shot-check-20261116/index.html)
and `summary.json` show frame-level results. Across these six clips, 83/85
reviewed centres were within 20 px, but only 46 were provisional observed
hits. The sparse, development-selected labels and manual seeds make that
83/85 figure unsuitable as an accuracy estimate for new shots.
The reviewed ball box in `IMG_5702` shrinks from roughly 26×28 px at contact
to 10×16 px at the frame-90 miss, so small apparent size may contribute.
In `IMG_5703`, the reviewed ball box is still about 15×26 px at its frame-90
miss. Size alone therefore does not explain both goal-area failures; nearby
goal structure and motion remain hypotheses to test with denser labels.

## Moving-ball stress check (2026-11-16)

Three other distinct, previously reviewed moving-ball recordings were run for
50 frames with the same hand-seeded pooled-motion settings. This is a stress
check outside the stationary-shot target, not independent validation. The
`96b382d82d92` and `ee73dffd61b5` files match their annotation SHA-256
hashes. A comparison also ran the existing appearance-aware settings (with
the previously specified 30-frame gate for the dribble case).

| Clip | Motion-only within 20 px | Appearance within 20 px | Reliability concern |
| --- | ---: | ---: | --- |
| `96b382d82d92` | 15/17 | 15/17 | Both paths diverge after early flight; motion flags frame 155, appearance frame 157. |
| `ee73dffd61b5` | 0/6 | 6/6 | Motion follows the player with no warning; appearance is unflagged on sparse marks. |
| `7de88d94f23c` | 0/5 | 5/5 | Motion follows a wrong target with no warning; appearance flags uncertainty from frame 42. |

On `96b382d82d92`, motion's two late errors are 255 and 314 px despite
15 early labelled positions being within about 5 px. On `ee73dffd61b5`
and `7de88d94f23c`, a 50-frame wrong path is marked provisional throughout.
Those false-negative reliability decisions are more serious than a missed
candidate: a wrong trajectory could otherwise look acceptable to later
layers. The appearance comparison shows a possible cue for moving-ball
identity, but the 7de path remains uncertain and no labels verify the full
late flight. Do not route any of these paths into outcome or coaching.
The generated local review is under
`data/reference-review/beam-motion-track/moving-shot-stress-20261116/`.

## High-resolution crop feasibility gate (2026-11-16)

Before adding predicted-path crop search or changing the sticky uncertainty
rule, we checked whether a public soccer-ball detector could detect the ball
when given an *oracle* crop centred on its manual label. This is an optimistic
upper-bound diagnostic, not a deployable tracker: inference must not use the
manual label. The public `martinjolif/yolo-football-ball-detection` checkpoint
(SHA-256 `fb37942448e7de08745e8aab148d0794f680a738ddd55e5f17abe9ab2d6313fb`)
was run at 960-pixel input size and 0.05 confidence on nine visible labelled
frames from four SCAN clips. A candidate counted as a hit when its centre was
within 20 original-image pixels of the labelled ball centre.

| Input | Ball hits | Interpretation |
| --- | ---: | --- |
| Full frame | 1/9 | Missed the selected late-flight and goal-area examples. |
| 512-pixel oracle crop | 0/9 | Four other detections; no labelled-ball hit. |
| 256-pixel oracle crop | 0/9 | Fourteen other detections; no labelled-ball hit. |

The local per-frame reports are `data/reference-review/model-comparison-20261116/visionpitch-results.json`
and `oracle-crop-results.json`. These deliberately difficult, development-selected
frames do not estimate overall accuracy. The full-frame hit disappearing under
the tighter crops also shows that zooming can remove useful context. This
checkpoint is a **no-go** for the proposed crop-search pipeline: more search
windows would increase candidate volume without demonstrating ball recall.
Keep the offline tracker and its uncertainty behavior unchanged. A future
candidate detector must first recover the manually labelled ball in these
oracle crops while limiting hard-negative detections, then pass a separate
clip/session-held-out test before temporal re-identification is evaluated.

## SCAN-specific small-detector pilot (2026-11-16)

We fine-tuned the same one-class YOLO11n checkpoint offline on the six-clip
`export-v3` labels. Four clips from two recording groups formed the training
split; two daylight clips from one recording group formed validation. The
validation group has already informed earlier experiments, so it is a
development split, **not** an untouched test set. All checkpoints and data
snapshots stay under ignored `data/reference-review/`; no app or production
tracking path uses them.

The first run used 512-pixel crops (223 positive and 144 empty train crops;
160 positive and 101 empty validation crops). After three CPU epochs, its
validation mAP50 was 0.0345. At 0.05 confidence on the fixed oracle-crop
check, it found 2/7 training balls and 0/2 daylight-validation balls in
512-pixel crops. Tightening those same oracle crops to 256 pixels found 5/7
training balls and still 0/2 daylight balls, with 45 other proposals across
the nine frames. This suggests resolution can help on seen scenes, but does
not establish usable detection.

A second run built 256-pixel crops (199 positive and 144 empty train crops;
137 positive and 101 empty validation crops). The baseline detector produced
false alarms on 22 empty *training* crops; we reviewed their montage and added
those 22 crops as hard-negative oversamples. After three CPU epochs, validation
mAP50 was 0.00031 and the fixed 256-pixel oracle check found 0/9 balls at
0.05 confidence. The baseline checkpoint also found 0/9 under the same
640-pixel inference settings. This arm is a **no-go**. The 512- and 256-crop
mAP figures use different datasets and should not be compared as if they were
the same benchmark.

Reproducible scripts are `evaluate_oracle_crops.py` and
`mine_hard_negative_crops.py` in this directory. Reports and checkpoints are
under `data/reference-review/ball-finetune/20261116T082024656866Z/`,
`data/reference-review/ball-finetune/20261116T084216933323Z/`, and
`data/reference-review/model-comparison-20261116/`. The hard-negative review
montage and provenance are in `data/reference-review/ball-crops/20261116-256-hard-v3/`.
The next detector experiment needs more labelled recording groups, especially
daylight and late-flight small balls. It must pass an untouched session-held-out
crop test before predicted-path search and temporal identity checks are worth
connecting to the app.

## Dense stationary-shot identity check (2026-11-16)

The input folder contains 12 byte-distinct recordings. Nine have reviewed ball
labels for the current detector pilot; the other three are an indoor ball
close-up or moving/multi-player shots, so they cannot serve as an untouched
stationary-shot test. Three additional stationary-shot recordings now have
contiguous reviewed flight windows. Their label metadata is in `reviewed_labels/`; the
source videos and frame images remain local under ignored `data/`.

| Clip | Scene | Visible positions within 20 px | Provisional observed hits | First critical failure |
| --- | --- | ---: | ---: | --- |
| `36bf848f409c` | Daylight, through goal area | 33/49 | 33/49 | Frame 85: trajectory switches to a goal feature. |
| `735e17f02bd5` | Night | 41/41 | 41/41 | No labelled failure in this window; one uncertain manual frame is excluded. |
| `e3949580d0f2` | Night | 38/39 | 25/39 | Missing proposal at frame 132 makes later identity uncertain. |
| `e9935750581e` | Daylight, shorter window | 15/16 | 0/16 | First post-contact frame lacks a proposal; warning stays sticky. |

These are **manually contact-seeded development cases**, not a deployable
start-up or independent accuracy estimate. The scorer now reports consecutive
misses, unsupported runs, and wrong positions accepted without a warning.
The daylight failure matters more than aggregate proximity: nearby goal motion
can make an incorrect path look smooth. An experimental abrupt-jump filter
improved raw proximity on `36bf848f409c` from 33/49 to 37/49, but marked 12
wrong positions provisional. It was removed rather than weakening the safety
gate. Reviewed trajectories and scores are under
`data/reference-review/beam-motion-track/stationary-shot/`.

The earlier dedicated ball checkpoint found **0/18** balls in fixed 384- or
256-pixel oracle crops from these dense labels. A frozen public COCO YOLO11n
`sports ball` model found **5/18** in 384-pixel oracle crops. Without access to
manual ball locations, 384-pixel crops centred on the motion path found only
**2/18**. A nine-slice search found 12/21 positions on `36bf848f409c`
frames 80–100, but only 4/16 on the other daylight shot and 7/12 on sampled
night frames, while proposing many distractors. Slicing improved candidate
recall in one scene but did not establish ball identity across frames. The
fixed selections and trajectory references are in `oracle-dense-frames.json`
and `predicted-*-frames.json`. The latter require regenerating the ignored
local trajectory reports before use on another machine. Per-frame output is
under `data/reference-review/model-comparison-20261116/`.

For the next small-model adaptation, `splits-dense-pilot.json` keeps recording
groups together: eight clips from four groups train, and the already-inspected
`36bf848f409c` clip is the sole development validation clip. A first one-epoch
CPU smoke run used Ultralytics `fraction=0.25`; that option takes the first
alphabetically sorted 235 of 941 training crops and omitted two groups. Its
near-zero validation score is **invalid as a full-dataset comparison**. The
script no longer offers that misleading shortcut. The full-data run used all
941 training crops and 132 validation crops. Its three-epoch development
validation mAP50 was about 0.756. This crop metric is not a measure of
end-to-end tracking or a separate-session test. No detector or trajectory
from this section feeds app coaching or outcome claims.

## Adapted small-ball detector and identity pilot (2026-11-16)

We compared the frozen COCO YOLO11n `sports ball` checkpoint (class 32) with
a three-epoch adaptation of that checkpoint to SCAN's reviewed 384-pixel crops
(one `ball` class). The adapted weights and per-frame reports remain in ignored
`data/reference-review/ball-finetune/20261117T004329962346Z/` and
`data/reference-review/model-comparison-20261116/`. The fixed 18-frame
oracle-crop check found the ball in **18/18** adapted frames versus **5/18**
with the frozen checkpoint. The oracle crops know where the ball is and test
the detector alone.

With nine overlapping crops centred on the earlier motion track, the adapted
checkpoint proposed the labelled ball in **49/49** frames of the daylight
`36bf848f409c` goal-area window, versus **34/49** with the frozen checkpoint.
It also proposed the ball in **16/16** frames of the shorter daylight
`e9935750581e` window, versus **4/16** frozen, and **39/39** frames of the
night `e3949580d0f2` window. It produced extra distractors: 45 non-ball
proposals in the first daylight window and 29 in the second. Candidate recall
must not be mistaken for choosing the correct identity.

The offline causal association in `associate_detector_candidates.py` combines
detection strength with predicted motion and marks a rejected detection as
missing. On these *saved* proposals it selected the labelled ball in 49/49,
16/16, and 39/39 frames respectively. A first distance-dominated version
selected a false ball beside the seed and then lost the trajectory; the
revised rule is an experimental response to that failure, not a validated
calibration. Crucially, the saved proposals were searched around an **older
motion trajectory**. They do not show whether the detector can search around
its own choices without drifting.

`closed_loop_detector_track.py` tests that stronger question: after a manual
contact seed it centres the next search on its own predicted position, never
on a manual label or saved trajectory. Its first development run on
`36bf848f409c` found and selected the labelled ball in **49/49** reviewed
frames through the goal area; 60 other proposals were present. The same
closed-loop check selected the labelled ball in **16/16** frames of the other
daylight shot and **39/39**, **41/41**, and **16/16** visible frames of three
night windows, for **161/161** reviewed visible frames total. These windows
include four training clips and the already-inspected development validation
clip. Manual starting points were supplied. The count is a within-20-pixel
comparison against labels; it is **not** a verified identity probability or
unbiased accuracy estimate.

An exhaustive 384-pixel tiled search on six frames labelled *target ball
absent* produced proposals on two frames. One night frame had a weak 0.07
proposal; the daylight frame contained several other soccer balls and five
ball proposals. Detecting a real *different* ball would still be an identity
failure for SCAN. This absence check does not simulate a continuous track
through disappearance. A distinct recording session, multiple-ball cases,
and a supported/uncertain decision rule are needed before allowing this path
into SituatedState, outcome evaluation, or coaching. Reports are under
`data/reference-review/model-comparison-20261116/`.

The linked [YOLO/interpolation walkthrough](https://www.youtube.com/watch?v=t18k3mGVNgQ&t=1584s)
illustrates a useful distinction: keep detector observations separate from
positions filled across missing frames. Short-gap interpolation may improve a
review display after identity is established, but it cannot verify that a
newly found point after a gap is the same ball. In SCAN, any interpolated
position must stay marked inferred and must not become shot-outcome evidence.

### Interactive review of saved closed-loop runs

Run `python -m experiments.ball_tracking.render_closed_loop_review` to build
`data/reference-review/closed-loop-tracker/index.html` from the five saved
closed-loop reports. This renders existing detections without rerunning the
model. Yellow marks the selected tracker position; cyan marks the manual
reference box. The local viewer supports playback, frame stepping, a ball
close-up, and optional candidate/prediction overlays. It shows only the
reviewed windows, with manually supplied starting points. Its displayed
detector score is not a probability that ball identity is correct.

### Unlabelled recording-session check (2026-11-19)

`track_unlabelled_video.py` runs the frozen adapted ball detector and the same
offline association rule on ordinary video files, without per-frame ball
labels. It decodes sequentially and processes every frame from one manually
selected starting ball position through the decoded end of each video. It does
not re-seed after tracking starts. The current local case list is
`data/reference-review/new-session-20261119/seeds.json`: nine seeds are on
frame zero and two deliberately start later, at frames 45 and 15. Those two
starts omit earlier frames from tracking, even though the videos are decoded
sequentially. Seed coordinates were chosen by visual inspection before these
new runs; they are not automatic contact detections.

For a repeatable run, use a **new** output directory each time, because the
runner refuses to overwrite a prior frozen configuration or report:

```powershell
python -m experiments.ball_tracking.track_unlabelled_video --cases data/reference-review/new-session-20261119/seeds.json --weights data/reference-review/ball-finetune/20261117T004329962346Z/train/weights/best.pt --output data/reference-review/new-session-20261119/runs-next --threads 8
python -m experiments.ball_tracking.render_unlabelled_review --reports-dir data/reference-review/new-session-20261119/runs-next --output data/reference-review/new-session-20261119/review-next
```

The runner records the frozen model hash, fixed settings, and each source-video
hash in `frozen-config.json`, with a `report.json` and original-frame JPEGs for each
completed clip. The renderer builds a local interactive `index.html` from
completed reports, displaying selected detections, a close-up, and optional
candidate/prediction overlays. Yellow indicates a selected candidate, not
verified ball identity; predicted positions are not observations. There are
no manual frame labels or accuracy scores in this check. A selected-detection
count alone cannot establish correct ball identity, goal crossing, speed,
curve, or shot outcome. The source exports, reports, and rendered review stay
under ignored `data/` and do not feed the app, SituatedState, or coaching.

The first new-session batch completed all 11 distinct videos from 13 exports
(two exact duplicates). It processed 1,531 frames after the chosen starts;
60 earlier frames were excluded by the two delayed starts. All runs reached
the container-reported end with the same frozen weights, detector settings,
and association implementation. No per-frame labels or later re-seeds guided
the tracks. The local review is
`data/reference-review/new-session-20261119/review/index.html`, with provenance
and counts in `summary.json` beside it in the session directory.

Visual inspection shows several plausible launches and flights, but confirmed
identity failures remain: `okonr73x` switches to a distractor at frame 67,
`b9rlx186` selects the player's clothing at frame 79, and `5mcmluz8` switches
to a turf marking at frame 98. In some failures the real ball is still among
the detector's proposals. Goal-area/post selections in other clips also need
independent identity checks. The 1,272 selected-candidate frames are therefore
**not** an accuracy score or evidence of reliable goal-area tracking. Preserve
this first pass as the baseline; the next development target is candidate
association and identity verification through gaps, followed by a separate
evaluation. These inspected clips are now development evidence, not a fresh
test set for later tuned versions.

### Capture quality and image-processing probe (2026-11-19)

The mobile recording screen now requests 1080p by default, with 720p and 4K
options, higher requested bitrates, and Auto/Off stabilization controls. See
`src/mobile/README.md` from the repository root for capture limitations and the
physical-phone save/replay/export check. This does not change the frozen
720p tracking baseline or establish better tracking on higher resolutions.

`probe_contrast_detector.py` compares the frozen ball detector on original
review JPEGs, mild luminance-only CLAHE, and conservative unsharp masking.
Every variant uses the same saved prediction-centred crops and thresholds.
It never updates a track or uses a manual ball label to guide the search.
Because the inputs are saved JPEGs, its original branch can differ slightly
from the baseline's source-video buffers. This is a proposal-level diagnostic,
not an end-to-end tracking or accuracy benchmark.

```powershell
python -m experiments.ball_tracking.probe_contrast_detector --runs data/reference-review/new-session-20261119/runs --weights data/reference-review/ball-finetune/20261117T004329962346Z/train/weights/best.pt --output data/reference-review/new-session-20261119/contrast-sharpen-next --threads 8
```

The completed `contrast-sharpen-pilot/index.html` in that session directory
shows 19 difficult frames from three clips. The results are mixed: CLAHE
raises the ball proposal's score at `okonr73x` frame 67 but lowers it at
`b9rlx186` frame 79. Unsharp masking strengthens the ball proposals at
`f5y3mv5e` frames 113–114, but neither processing method recovers a proposal
near the visually apparent ball at frames 102 or 120. Detector scores are not
identity probabilities, and fewer or more proposals do not prove improvement.
Neither filter is enabled in mobile recording or production perception.

The review also includes a positive-luminance-change overlay after estimated
global translation alignment. It rejects unreliable alignment and does not
alter either detector input. This is **not a learned ball heatmap**: moving
players, shadows, exposure variation and imperfect alignment can also appear.
It is diagnostic evidence only, with no ball identity, coaching or outcome
claim. Unsharp masking enhances existing edges; it does not reconstruct
detail lost to motion blur. Originals and all previous reviews are preserved.

### Temporal association development check (2026-11-19)

The optional `--association temporal` runner keeps a narrow main prediction and
requires consecutive moving observations before promoting a more distant path.
The separate optional `--association recovery` mode adds a bounded slow/rebound
hypothesis after repeated nearby detector observations, preserving missing frames.
The opt-in `--association motion-recovery` mode can release that local guard
after three consecutive accepted movement steps, with a bounded prediction
corridor and a causal veto for recurring off-target stationary proposals.
It is an offline development option, not verified ball identity.
It preserves missing detections and estimates background translation rather than
allowing one stronger clothing/turf proposal to redirect the track. The legacy
association remains the default and all work stays offline. Read
[TEMPORAL_IDENTITY.md](TEMPORAL_IDENTITY.md) for the matched experiment, provisional
reference protocol, limitations and repeatable commands. The recordings inspected
for this change are development clips; they are no longer a fresh test session.

### Original-file and native-detail evaluation

The unlabelled runner accepts optional `--processing-short-edge 720` for a
matched processing-scale comparison of higher-resolution originals with earlier
720p exports. It only downsamples, using area interpolation, before camera-motion
estimation and inference. Input seeds are in the oriented decoded **source**
coordinates; reports preserve both source and processing seeds, dimensions and
exact XY scales. Without the option, existing native processing is unchanged.
The review labels downsampled frames as processing previews rather than original
resolution. Source files and earlier runs are never overwritten.

`probe_native_detail.py` is a separate detector diagnostic. Predetermined frames
use the saved 720-coordinate track's query region. The original frame and its
own downsampled version use the same model and confidence. Native 384-pixel crops
cover the exact projected scene region with more tiles; no association is run.
Source, detector, weights, resize and ROI provenance are checked and recorded.
Circles in its viewer are raw proposals, not verified balls or tracker paths.
More proposals do not establish higher accuracy, colour benefit or shot outcome.

```powershell
python -m experiments.ball_tracking.track_unlabelled_video --cases data/reference-review/new-session-20261120-originals/cases.json --weights data/reference-review/ball-finetune/20261117T004329962346Z/train/weights/best.pt --output data/reference-review/new-session-20261120-originals/runs-next --association motion-recovery --processing-short-edge 720 --threads 8
python -m experiments.ball_tracking.probe_native_detail --cases data/reference-review/new-session-20261120-originals/cases.json --runs data/reference-review/new-session-20261120-originals/runs-next --weights data/reference-review/ball-finetune/20261117T004329962346Z/train/weights/best.pt --output data/reference-review/new-session-20261120-originals/native-detail-next --threads 8
```

The mail originals in this session include 1080p white/neon-green recordings and
4K yellow recordings. Colour and capture resolution are confounded. Matched
original-versus-Discord files support a transfer-fidelity comparison, not a pure
1080p-versus-4K camera benchmark. The existing strict identity-reference comparator
continues to reject different source bytes; these cross-encode checks are kept
separate. No production perception, feedback or outcome contract changes here.

### Opt-in native gap evidence

`--native-detail-fallback` adds a small native-resolution search only when the
720-coordinate detector has no proposal within 32 pixels of its causal query.
Existing reduced-image proposals are retained. Additional proposals must remain
within that local gate; this is not a full-frame detector reset or a new seed.
The option requires `--association motion-recovery --processing-short-edge 720`.

The first native-only fusion ablation incorrectly followed a fixed pale object
beside the goal in the white-ball clip. Its larger selected-frame count is not
an improvement in identity. That run and its exact implementation are retained
under the original-file session's ignored evaluation directory.

The current opt-in pilot therefore checks extra proposals against the first
source image. SIFT features and RANSAC register the initial background to the
current image; unsupported registration withholds extra evidence. Similar
registered patches are treated as ambiguous pre-existing background. Extra
observations in the initial seed region are withheld, leaving that phase to the
primary detector. This prevents a supplemental boot detection at contact from
bootstrapping a new target path; it does not claim to classify all occlusions.
Outside that region, extra
proposals can use the last two consecutive accepted detector observations for
continuity; otherwise each native hypothesis needs three consistent observations.
Several plausible objects remain separate candidates for the unchanged
association layer. Missing frames stay missing;
no earlier gaps are filled, identity is never marked verified, and filename
colour/outcome metadata and evaluation references are not read by inference.

This image registration is **not** goal calibration, 3D reconstruction, depth,
contact detection or proof of identity. Moving nets, occlusion, parallax, similar
objects and very small balls can still defeat these checks. A single-patch seed
colour veto was rejected because a printed ball panel can supply the wrong hue;
no colour gate is enabled. The legacy/default runner and production pipeline
remain unchanged. This mode is an inspected-session development experiment.

Native tile duplicates now merge by box IoU rather than distance between centres.
The earlier 15-processing-pixel radius removed a correct low-score yellow-ball
box beside a different higher-score goal-area object. Non-overlapping boxes now
survive even when close together; this preserves evidence for association rather
than asserting that either object is the target. The initial matched native-detail
diagnostic preserves its original radius merge as a frozen control. The new
native-only merger does not change that diagnostic or the baseline detector.
The local native search also explicitly includes a tile centred on the causal
query. A regular grid alone can straddle that point; the query-centred tile is
deduplicated when the grid already contains it. This adds source context rather
than changing the target-admission gate or creating image detail.

The registration follows the [OpenCV feature matching and homography workflow](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html).

```powershell
python -m experiments.ball_tracking.track_unlabelled_video --cases data/reference-review/new-session-20261120-originals/cases.json --weights data/reference-review/ball-finetune/20261117T004329962346Z/train/weights/best.pt --output data/reference-review/new-session-20261120-originals/native-verified-next --association motion-recovery --processing-short-edge 720 --native-detail-fallback --threads 8
```
