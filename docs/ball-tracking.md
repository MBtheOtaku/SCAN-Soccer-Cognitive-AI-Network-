# Ball tracking: active workflow

The guarded motion-recovery tracker now has an opt-in app replay path. See
[experimental tracking in saved reps](experimental-tracking.md) for single-tap
initialisation, job progress, replay overlays and limitations. Its output remains
separate from normal analysis, SituatedState, Jev judgements and coaching.
The dated experiment results below describe earlier development evidence.

This is the entry point for ball-tracking development. Updated 15 November 2026.
The mobile app and backend are operational prototypes; continuous ball localisation,
identity through contact/flight, and shot outcome are not validated.

The latest hand-seeded motion and appearance experiment is documented in the
[offline identity pilot](../experiments/ball_tracking/README.md). Its
appearance cue improves a nighttime dribble but misroutes two contact-seeded
shots; a motion-only comparison recovers those reviewed shot positions.
All paths still need independent identity validation. The pilot has not
been connected to app analysis or coaching.

## Current evidence

- Independent generic/dedicated detector proposals frequently select boots, turf
  or lights and miss the real ball near contact.
- [Sliding-window inference](sliced-ball-inference.md) is implemented offline.
  On two sampled daylight contact frames it still missed the ball while raising
  false detections from 2→21 and 3→38; do not use it for coaching yet.
- The three-epoch development fine-tune at 960 pixels produced 1 matching box,
  98 missed labelled balls and 222 false detections on 100 validation images at
  confidence 0.25 and matching IoU 0.5. The original checkpoint produced 0 matches,
  99 misses and 180 false detections. This did not establish useful improvement.
- Manually seeded fixed-template tracking stayed on the reviewed ball during parts
  of two run-ups but failed before/around contact; the dribbling example became
  ambiguous early. Manual selection is useful target initialisation, not a solved
  tracker or camera calibration.
- A correctly initialised tracker can still switch to an identical second ball.
  Real multi-ball crossings and occlusions have not been evaluated.

## One development pipeline

Reviewed recordings and labels → detector sanity check → human-seeded temporal
tracking → independent contact/flight evaluation → production integration only
after measured improvements.

The training-image sanity check compares the original and adapted detector on
the saved night-session training images at confidence 0.01, 0.10 and 0.25. It
answers whether adaptation located examples it saw during training; it cannot
establish performance on new footage. Run it with `python -m scripts.finetune_ball
train-sanity data/reference-review/ball-finetune/20261108T000537258416Z` and
review `train-sanity/index.html` and `report.json` in that run directory.

On the 42 saved training frames (37 labelled balls), the adapted checkpoint
matched **0/37 at confidence 0.25**, **1/37 at 0.10** and **2/37 at 0.01**
(with eight false detections at 0.01). The baseline matched 0/37 at every
cutoff. The two low-confidence adapted matches were frames 84 and 86, after the
reviewed contact near frame 79. This checkpoint has not learned reliable
localisation even on its training frames, so temporal association alone cannot
rescue it. The next experiment should address ball appearance/scale and detector
training on these labelled images; then measure on a new held-out session before
connecting a tracker to coaching.

Further changes should isolate one hypothesis at a time. Preserve full-resolution
detail, separately score contact/flight and run-up frames, count wrong-target
switches, and keep prediction/occlusion gaps separate from supported observations.
Do not tune repeatedly on development clips and describe them as an untouched test.
Do not use later manual labels to correct tracking during evaluation.

## Active tools

| Purpose | Entry point | Guide |
|---|---|---|
| Review contact timing in original footage | `scripts/review_shot_timing.py` | [Timing audit](shot-timing-audit.md) |
| Prepare, label and export ball images | `python -m scripts.ball_dataset` | [Labelling](ball-labelling-guide.md) |
| Controlled detector adaptation and validation comparison | `python -m scripts.finetune_ball` | [Fine-tuning experiment](ball-finetuning-experiment.md) |
| Offline full-image sliced detection and labelled review | `python -m scripts.review_sliced_ball` | [Sliced inference](sliced-ball-inference.md) |
| Build auditable training crops matching tile inference | `python -m scripts.ball_crop_dataset` | [Crop pilot](cropped-ball-pilot.md) |
| Select a target ball and track forward offline | `python -m scripts.review_manual_ball` | [Manual initialisation](manual-ball-tracking.md) |

The manual seed UI and annotation UI remain beside their Python runners in
`scripts/`. Active tracking logic is in `scripts/manual_ball_tracker.py`.
These tools do not replace `src/backend/ball_tracking.py`, change the app's recording
flow, or supply coaching judgements.

Active implementation checks:

```powershell
python -m pytest tests/test_ball_dataset.py tests/test_finetune_ball.py tests/test_manual_ball.py -q
```

## Data and historical work

- Original recordings: `data/input/`.
- Existing backend analysis: `data/output/`.
- Your reviewed contact references: `tests/fixtures/shot_contact_references.json`.
- Annotation files, seeds, exports and review pages: `data/reference-review/`.
- Original/downloaded models: `data/models/`; experiment checkpoints remain under
  their existing `data/reference-review/ball-finetune/` run directories.

Those locations were preserved during cleanup. Generated reviews/models remain
ignored rather than being added to Git. Existing browser links remain valid.

Old detector-v2 association, WASB probes, contact proposals and the initial research
survey are in the [archive](../experiments/ball_tracking/archive/README.md). Their
checks are opt-in and do not clutter normal test discovery. Revisit an archived
method only with a specific hypothesis and an evaluation criterion.
