# SCAN ball annotation: first adaptation dataset

This is an offline preparation tool. It does not modify the app, production tracking,
Jev judgements, or coaching. No API keys or new packages are required beyond the
OpenCV already used by SCAN. A labelled dataset is the next input to model adaptation,
not evidence that detection has improved yet.

## Named camera recordings

For a file such as `data/input/IMG_5703.mov` that has no SCAN analysis ID, pass
`--source` instead of an ID. The tool derives a stable 12-character ID from the
video hash without copying or renaming the original. The `IMG_5703.mov` batch
was prepared with:

```powershell
python -m scripts.ball_dataset prepare --source data/input/IMG_5703.mov --stride 10 --dense-start 38 --dense-end 70
```

It contains 40 frames and is ready at
`data/reference-review/ball-labels/6292a1b2b194/index.html`. Open that page,
label the shooting ball, and download the JSON as described below. The objects
beside the goal in this clip are equipment, not another ball. If the ball is
hidden or cannot be located reliably, mark that frame `uncertain`. Save the download as
`data/reference-review/ball-labels/6292a1b2b194/labels.json` before exporting.
Treat any other videos from the same recording session as one group when making
the later train/validation split.

The reviewed `IMG_5703.mov` labels were imported on 12 November 2026: 39
visible frames and one unreviewed frame (63, excluded). The saved development
export is `data/reference-review/ball-labels/export-v2`, with this clip in the
training split; its crop-scale export is `data/reference-review/ball-crops/20261112-v2`.
The source inventory at `data/reference-review/ball-labels/source-inventory/index.html`
shows 60 files but only 12 distinct video hashes. Multiple analysis IDs are
copies of the same original recording; do not count or split those copies as
independent examples.

## 1. Prepare the first recording

In an idle PowerShell terminal at the SCAN root:

```powershell
Set-Location "C:\Users\Matthew Bui\OneDrive\SCAN"
python -m scripts.ball_dataset prepare 13a1022c5009 --stride 10 --dense-start 69 --dense-end 95
Start-Process "data/reference-review/ball-labels/13a1022c5009/index.html"
```

This samples the whole clip every ten frames and every frame from 69–95, around
your reviewed contact at 79–80. Original frame dimensions are preserved. These
contact references guide annotation coverage only; they are not model inputs.
Times displayed as frame/FPS are approximate, not verified presentation timestamps.

The tool refuses to overwrite an existing annotation folder. To resume, open its
`index.html` and import your downloaded labels JSON; do not rerun preparation.

## 2. Label the actual ball

- Drag a tight rectangle around the visible ball. Redraw to replace it.
- Include visible motion blur when the ball can still be located reliably.
- Use **Uncertain / hidden** for occlusion, ambiguous blur, or a ball too small to
  locate reliably. Never guess its position using the foot or a model prediction.
- Use **Ball absent** only when no ball is visible anywhere in the frame. A ball
  that may be hidden is uncertain, not an absence label.
- **Clear** returns a frame to unreviewed. Arrow keys navigate.
- Click **Download labels** frequently and verify the file downloaded. No labels
  are automatically saved to the project or browser storage.

The tool supports one target ball per image. If multiple visible balls make the
frame ambiguous, exclude it with **Uncertain / hidden** for this first dataset.
Marking a frame absent while another ball is visible would create a false negative.

Keep the complete original frame: boots, turf markings and floodlights become
background examples when the real ball is correctly boxed. Do not label those
objects as balls. Include explicit absent frames from suitable footage too.

Copy the downloaded `13a1022c5009-labels.json` into its annotation folder and rename
the copy to `labels.json`. Keep your downloaded backup. For example, after checking
the actual filename in Downloads:

```powershell
Copy-Item "$env:USERPROFILE\Downloads\13a1022c5009-labels.json" "data/reference-review/ball-labels/13a1022c5009/labels.json"
```

On reopening the page, import that JSON to resume. A fresh page starts unreviewed.

## 3. Prepare the daylight examples

```powershell
python -m scripts.ball_dataset prepare e9935750581e --stride 10 --dense-start 107 --dense-end 133
python -m scripts.ball_dataset prepare 96b382d82d92 --stride 10 --dense-start 126 --dense-end 152
Start-Process "data/reference-review/ball-labels/e9935750581e/index.html"
Start-Process "data/reference-review/ball-labels/96b382d82d92/index.html"
```

Label and save each clip the same way. Dense contact frames help inspect difficult
ball appearances; uniform context includes stationary balls, run-up and flight.
These three clips are development examples already used for model selection.
They cannot establish generalisation to new recordings on their own.

## 4. Assign recording groups before training

Create `data/reference-review/ball-labels/splits.json`. Example:

```json
{
  "clips": [
    {"clip_id": "13a1022c5009", "group": "night-session-1", "split": "train"},
    {"clip_id": "e9935750581e", "group": "day-session-1", "split": "val"},
    {"clip_id": "96b382d82d92", "group": "day-session-1", "split": "val"}
  ]
}
```

Use the real recording session/setup groups. Clips with the same pitch, camera
setup and session should stay together. Do not split neighbouring frames randomly.
The exporter enforces the group names you provide; it cannot infer shared sessions.
This example is a development split, not a representative final benchmark.
Add recordings from separate sessions, camera distances and lighting, and reserve
new untouched recordings as `test` before evaluating a trained detector. Do not
use test results repeatedly to tune the model or thresholds.

## 5. Export reviewed labels

```powershell
python -m scripts.ball_dataset export --config data/reference-review/ball-labels/splits.json --output data/reference-review/ball-labels/export-v1
```

The exporter creates standard one-class YOLO bounding-box labels and `dataset.yaml`:

- Visible reviewed ball → original-resolution image plus normalised ball box.
- Explicitly absent → image plus empty label file.
- Uncertain, hidden, unreviewed → excluded entirely, not treated as negatives.
- `export_report.json` reports counts by clip and split.

It rejects invalid boxes, mismatched recording hashes, duplicate annotation rows,
recording groups crossing splits, and identical source hashes crossing splits.
It will not overwrite an earlier export; use `export-v2` for a later version.
Annotation sets, images and exports stay inside the existing ignored
`data/reference-review/` directory.

## What follows

First inspect a handful of saved boxes at full resolution, especially the foot,
blurred contact and floodlight examples. Then adapt one detector and compare its
predictions on held-out clips: missed visible balls, false objects, centre error
and behaviour around contact. Only after localisation improves should we associate
observations across time and revisit shot detection. A working exporter and unit
tests do not validate detector accuracy.

Implementation files: `scripts/ball_dataset.py`, `scripts/ball_label_ui.html`.
Verification: `python -m pytest tests/test_ball_dataset.py -q`.
