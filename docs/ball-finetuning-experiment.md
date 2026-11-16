# First SCAN-specific detector fine-tuning experiment

The experiment adapts the existing `football-ball-detection.pt` checkpoint using
the reviewed dataset. It does not change any backend/mobile tracking or coaching.
Implementation: `scripts/finetune_ball.py`. No packages need reinstalling.

Current development split: 42 night-session training images (37 boxes, 5 absent)
and 100 daylight-session validation images (99 boxes, 1 absent). The two daylight
clips remain together because the user confirmed they share a recording session.
Five uncertain images are excluded. This is a small cross-session development
experiment, not a final test of tracking quality. These clips have already informed
model choices. More sessions and untouched test clips remain necessary.

## Run

Use an idle terminal, not the terminal hosting the backend or Metro:

```powershell
Set-Location "C:\Users\Matthew Bui\OneDrive\SCAN"
python -m scripts.finetune_ball train --epochs 3 --imgsz 960 --batch 2 --device cpu
```

The existing dedicated checkpoint is YOLOv8x (about 68 million parameters), not a
nano detector. The experiment is small in data and epoch count; the architecture
is substantial. CPU training can be slow. Keep the laptop powered and the terminal open. Training
prints the output directory at startup and epoch progress. A seed and deterministic
settings improve repeatability within an environment; they do not guarantee identical
results on different hardware or package versions. No GPU is available to the current
PyTorch installation. This script does not install a GPU stack or upload recordings.

For a plumbing check only:

```powershell
python -m scripts.finetune_ball train --epochs 1 --imgsz 640 --batch 2 --device cpu --skip-compare
```

A one-epoch lower-resolution smoke run verifies that training works. It is not an
accuracy experiment or a replacement for the 960-pixel run. Each command creates
a new timestamped directory under `data/reference-review/ball-finetune/`.
Three epochs are an initial bounded pilot, not evidence that training has converged.
Use the saved run times to plan a longer run or GPU training if needed.

## What the runner does

1. Validates image/label pairs, single-ball boxes, recording-group separation and
   exact duplicate images across train/validation.
2. Copies the dataset and original checkpoint into a new experiment directory.
3. Records data/checkpoint SHA256 hashes, package versions and training options.
4. Fine-tunes with a small learning rate, the first ten model layers frozen and
   mild augmentation. Mosaic and mixup are disabled for this initial experiment.
   Nominal batch size equals actual batch size so even a short run makes regular
   optimiser updates. Warmup is disabled: its usual duration and bias learning
   rate are inappropriate defaults for this tiny, short adaptation experiment.
   These are starting hypotheses, not established optimal settings for SCAN.
5. Saves Ultralytics checkpoints under `train/weights/`. Validation selects `best.pt`.
6. Compares the original checkpoint with `best.pt` on the same validation images,
   at the same image size and fixed 0.25 confidence threshold.

The comparison uses full-frame detection, no temporal tracker, NMS IoU 0.5, and
at most 24 proposals. A prediction matches the manual ball box at IoU >=0.5;
there is at most one true positive per image. Wrong objects count as false
positives and a missed labelled ball also counts as a false negative. These
fixed-threshold counts/precision/recall are **not mAP or calibrated confidence**.
For tiny balls, a few pixels can change IoU substantially, so inspect overlays too.
There is only one negative validation image, limiting false-positive assessment.

## Inspect results

At completion, open the printed `comparison/index.html` path. It shows:

- green: human ball box;
- pink: original detector predictions;
- cyan: fine-tuned detector predictions;
- total and per-frame true positives, false positives and misses.

`comparison/report.json` contains all prediction boxes and scores, and
`experiment.json` records configuration and hashes. Results, data and checkpoints
remain under the existing ignored `data/reference-review/` directory.

For a run created with `--skip-compare`, compare later using its actual folder:

```powershell
python -m scripts.finetune_ball compare "C:\Users\Matthew Bui\OneDrive\SCAN\data\reference-review\ball-finetune\YOUR_RUN_FOLDER"
```

An existing comparison directory is not overwritten. Use the saved report rather
than rerunning against changed labels. Validation is also used to select the best
epoch; its reported performance is development performance, not an independent
final evaluation. Do not change thresholds repeatedly until these clips look good
and call that generalisation. A zero-result experiment is still informative.

To check whether the adapted model can locate balls in the **training** images
it saw, run:

```powershell
python -m scripts.finetune_ball train-sanity "C:\Users\Matthew Bui\OneDrive\SCAN\data\reference-review\ball-finetune\20261108T000537258416Z"
```

This creates `train-sanity/index.html` and `train-sanity/report.json` without
changing the earlier validation report. The same low-confidence proposals are
scored at 0.01, 0.10 and 0.25. Checkpoint and dataset hashes are checked first.
Training results show whether the detector learned these examples, not whether
it will work on another recording, through contact, or during ball flight.

For the 3-epoch run above, the adapted model matched 0/37 labelled training
balls at confidence 0.25, 1/37 at 0.10, and 2/37 at 0.01, with eight false
detections at the lowest cutoff. The baseline matched none. The two adapted
matches were frames 84 and 86, after the reviewed contact. Training-image
failure points to detector/scale/data preparation as the next problem to test;
association cannot recover an object that produces no viable candidate.

Do not copy the checkpoint into production yet. First inspect real-ball misses,
boot/light false positives and contact-neighbourhood frames, then test additional
recording sessions. Detection improvements alone do not validate temporal identity,
ball contact timing, shot outcome or coaching conclusions.

## API references

The installed Ultralytics 8.4.166 API and its official documentation were checked:
[training](https://docs.ultralytics.com/modes/train/) and
[prediction](https://docs.ultralytics.com/modes/predict/).
Tests: `python -m pytest tests/test_finetune_ball.py tests/test_ball_dataset.py -q`.
