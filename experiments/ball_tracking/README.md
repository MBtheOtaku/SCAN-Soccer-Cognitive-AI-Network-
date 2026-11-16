# Hand-seeded post-kick ball-motion pilot

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
