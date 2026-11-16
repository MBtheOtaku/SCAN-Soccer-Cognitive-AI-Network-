# Tile-scale ball detector pilot

This offline experiment trains the saved one-class detector on 512×512 crops,
matching the windows used by full-image sliced inference. It keeps the reviewed
night recording in training and both daylight recordings from the same session
in development validation. These clips have already guided design choices; a
new recording session is needed for an independent final evaluation.

The crop exporter uses complete manual ball boxes, chooses up to two tile
positions containing each ball, and adds an empty tile with strong image edges
from each frame where possible. It also mines model-selected distractors from
the night-only training report. When the real ball falls inside a mined crop,
the ball stays labelled; otherwise the crop is labelled empty. Partly cut-off
ball crops are excluded. The output records the source frame, crop coordinates,
label, image/label hashes, category and candidate provenance for auditing.

Build a new, immutable crop export:

```powershell
Set-Location "C:\Users\Matthew Bui\OneDrive\SCAN"
python -m scripts.ball_crop_dataset
```

The current export is
`data/reference-review/ball-crops/20261111T000708449784Z`. It contains 68
ordinary ball crops, seven ball-plus-distractor crops, 47 empty crops and one
mined empty crop for training. Development validation contains 160 ball crops
and 101 empty crops. Review `negative-review.html` in the export folder for
an unlabelled second ball before trusting mined examples. These counts are
from just three source recordings, not 384 independent scenes.

The one-epoch CPU pilot uses the same 640-pixel model input for both cropped
training and subsequent sliced inference:

```powershell
python -m scripts.finetune_ball train --dataset data/reference-review/ball-crops/20261111T000708449784Z --epochs 1 --imgsz 640 --batch 2 --device cpu --skip-compare
```

It snapshots the dataset and original checkpoint in a new timestamped run.
`--skip-compare` avoids a costly two-model pass over all 261 validation crops;
training still performs its own development validation and saves `best.pt`.
One epoch checks feasibility and first learning direction; it does not establish
convergence or deployment quality. The original full-frame labels remain
available for a matched-scale sliced review using `--source-dataset`:

```powershell
python -m scripts.review_sliced_ball RUN_FOLDER --source-dataset data/reference-review/ball-labels/export-v1 --split val --checkpoint adapted --imgsz 640 --tile-size 512 --overlap 128 --frame e9935750581e_000118 --frame 96b382d82d92_000136
```

For crop-level review, omit `--source-dataset` and pass exported crop names
with `--frame`, using `--tile-size 512 --imgsz 640`. Compare positives and
empty crops at the same fixed confidence threshold. We should promote a model
only after it improves ball recall and false detections on an untouched session
and maintains identity through contact and flight.

The completed one-epoch run is
`data/reference-review/ball-finetune/20261111T001518781194Z`. At 0.25
confidence, on two **original daylight contact frames** (one from each clip),
both checkpoints missed both labelled balls. The baseline made 13 false
detections and the adapted checkpoint made one. On five selected daylight
crops, the baseline made 15 false detections and the adapted checkpoint made
one; both missed all three labelled ball crops. The adapted checkpoint also
missed all four selected labelled night training crops. These small samples
are a diagnostic, not a dataset-wide accuracy estimate. Ultralytics' own
development validation after the single epoch reported 0.106 recall and
0.00115 mAP50 on the 261 daylight crops. That uses its validation scoring,
not the fixed 0.25 threshold above. **Do not integrate this checkpoint into
the application.** It suppresses false positives but does not yet detect the
ball reliably.

The two original-frame comparison reports are in this run's `sliced-review`
folder under `source-val-baseline-*` and `source-val-adapted-*`. Each contains
`report.json` and an annotated `index.html`. Training and crop-level reviews
are in `snapshot-train-adapted-*`, `snapshot-val-baseline-*`, and
`snapshot-val-adapted-*`.

When a newly recorded session has independently labelled frames, export it
as a **separate test-only dataset**. Review it with
`python -m scripts.review_sliced_ball RUN_FOLDER --source-dataset TEST_FOLDER --split test --checkpoint adapted --imgsz 640 --tile-size 512 --overlap 128`.
The review checks that its session group and image hashes do not overlap the
training/development snapshot. Compare the baseline with the same flags.
Do not tune on that test session. The next training iteration should first
improve crop-level recall with more varied, reviewed positive and negative
frames; only then evaluate temporal association and shot tracking.

No mobile, backend, coaching or segmentation code changes in this experiment.
The detector has boxes; segmentation would need pixel-level labels and a
segmentation model.
