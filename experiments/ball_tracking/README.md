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
the new candidate. Raw distance to **sparse** reviewed centres is:

| Clip | Baseline within 20 px | Candidate within 20 px |
| --- | ---: | ---: |
| `36bf848f409c` (day shot) | 5/10 | 10/10 |
| `7de88d94f23c` (night dribble) | 0/5 | 5/5 |
| `e3949580d0f2` (night shot) | 7/8 | 7/8 |
| `735e17f02bd5` (night shot) | 8/8 | 7/8 |
| `675e44d940ff` (day flight) | 8/8 | 8/8 |

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
