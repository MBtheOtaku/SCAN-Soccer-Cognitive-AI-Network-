# Offline sliding-window ball detection

`scripts/sliced_ball.py` covers each full image with overlapping square tiles,
runs the saved one-class YOLO ball detector on every tile, maps boxes back to
full-image pixels, and applies a second non-maximum suppression across tile
boundaries. `scripts/review_sliced_ball.py` checks those boxes against the
reviewed training or development-validation labels and saves an HTML overlay
and JSON report. This is an offline perception experiment; it does not change
the mobile app, backend tracker, contact estimate or coaching.

The default 512-pixel tile with 128-pixel overlap produces six windows on these
720×1280 frames. Each tile is sent to the model at its saved 960-pixel image
size. This preserves/enlarges detail relative to resizing the full frame, but
it also runs the large CPU model six times per frame. The current checkpoint
has detection boxes only; no segmentation mask is produced.

Run a small labelled review, for example:

```powershell
Set-Location "C:\Users\Matthew Bui\OneDrive\SCAN"
python -m scripts.review_sliced_ball data/reference-review/ball-finetune/20261108T000537258416Z --split train --checkpoint adapted --frame 13a1022c5009_000080 --frame 13a1022c5009_000084
python -m scripts.review_sliced_ball data/reference-review/ball-finetune/20261108T000537258416Z --split val --checkpoint adapted --frame e9935750581e_000118 --frame 96b382d82d92_000136
```

With no `--frame`, the runner chooses six evenly spaced images from the split;
`--max-frames` changes that number. Each run makes a new timestamped folder
under the saved experiment's `sliced-review/`. Green rectangles are reviewed
ball boxes, yellow rectangles are merged proposals, and grey lines are tile
boundaries. Reports contain raw and merged proposals, per-frame TP/FP/FN,
inference time and the saved full-frame score at the same 0.25 confidence when
available. They also verify model and dataset snapshot hashes before running.

## Initial result, 10 November 2026

At 0.25 confidence and IoU matching threshold 0.5, the adapted checkpoint
missed the ball on both sampled night training frames (80 and 84), with zero
proposals. It also missed both sampled daylight contact frames:

| Reviewed frame | Full frame FP | Sliced FP | Sliced ball match | Sliced CPU time |
|---|---:|---:|---:|---:|
| `e9935750581e_000118` | 2 | 21 | 0 | 51.5 s |
| `96b382d82d92_000136` | 3 | 38 | 0 | 21.6 s |

The first frame of each run includes model startup cost. The reported times
are measured on this laptop. These four frames are a diagnostic sample, not a
full dataset benchmark. Both daylight frames had zero overlap between any
merged proposal and the reviewed ball box. Tiling increased distractor
responses and did not recover the ball in this sample. It is not ready to
replace the existing tracker. The next controlled experiment is to build
labelled crops with hard negatives and train/evaluate a detector on those crops,
then check a new recording session before considering temporal association.

Checks: `python -m pytest tests/test_sliced_ball.py tests/test_finetune_ball.py -q`
