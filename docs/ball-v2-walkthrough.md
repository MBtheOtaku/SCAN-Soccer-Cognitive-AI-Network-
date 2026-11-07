# Roboflow-based ball experiment: implement and compare

This is the larger addition we agreed you would implement yourself. Complete code is provided as copyable templates in `docs/ball-v2/`; application files have not been replaced.

This experiment uses the dedicated weights linked by [Roboflow's official soccer setup](https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh), Ultralytics inference, overlapping tiles and our own conservative association logic. It does not require Roboflow hosting or an API key. The [official example](https://github.com/roboflow/sports/blob/main/examples/soccer/main.py) supplies the detector/slicing pattern; our association code is an experimental SCAN implementation, not Roboflow's tracker.

Validation performed on the provided templates: 16 tests passed; the detector adapter ran on one original frame in single and tiled modes using the existing generic weights; the runner decoded real video and produced a three-frame trace, summary and HTML preview using controlled fixture detections. These are implementation checks, not a test of the dedicated checkpoint or evidence of improved ball accuracy.

## 1. Review and copy the files

Open these templates to read the code. Copy them into the destinations below. No existing SCAN tracking or feedback file needs editing.

| Template in `docs/ball-v2/` | Destination |
| --- | --- |
| `ball_temporal_v2.py.template` | `scripts/ball_temporal_v2.py` |
| `ball_detector_v2.py.template` | `scripts/ball_detector_v2.py` |
| `review_ball_v2.py.template` | `scripts/review_ball_v2.py` |
| `test_ball_temporal_v2.py.template` | `tests/test_ball_temporal_v2.py` |
| `test_ball_detector_v2.py.template` | `tests/test_ball_detector_v2.py` |

From SCAN's root, these commands perform those copies. Do not repeat them after editing the destination files unless you intend to replace your edits.

```powershell
Set-Location "C:\Users\Matthew Bui\OneDrive\SCAN"
Copy-Item docs/ball-v2/ball_temporal_v2.py.template scripts/ball_temporal_v2.py
Copy-Item docs/ball-v2/ball_detector_v2.py.template scripts/ball_detector_v2.py
Copy-Item docs/ball-v2/review_ball_v2.py.template scripts/review_ball_v2.py
Copy-Item docs/ball-v2/test_ball_temporal_v2.py.template tests/test_ball_temporal_v2.py
Copy-Item docs/ball-v2/test_ball_detector_v2.py.template tests/test_ball_detector_v2.py
```

The current environment already has Ultralytics, OpenCV, NumPy and pytest. No dependency install or Metro restart is needed. The checked environment had Ultralytics 8.4.166; the root requirements file's older pin does not describe that installed environment. Do not reinstall or upgrade packages for this experiment merely to reconcile that pin.

## 2. Save the dedicated soccer-ball weights

Create a local model directory:

```powershell
New-Item -ItemType Directory -Force data/models | Out-Null
```

Add this line to the root `.gitignore` so downloaded model binaries remain local:

```gitignore
data/models/
```

Open the [model download linked by Roboflow's setup script](https://drive.google.com/file/d/1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V/view) in your browser. Download it and save it as:

```text
C:\Users\Matthew Bui\OneDrive\SCAN\data\models\football-ball-detection.pt
```

Verify it exists:

```powershell
Get-Item data/models/football-ball-detection.pt | Select-Object Name,Length
```

Use the binary weights, not an HTML download page renamed `.pt`. This source link was checked against the official setup script; the checkpoint itself has not yet been downloaded or tested here. If its download fails, report that rather than quietly substituting generic weights.

## 3. Run the tests

```powershell
python -m pytest tests/test_ball_temporal_v2.py tests/test_ball_detector_v2.py -q
```

These verify confirmation before acceptance, ambiguous initialization, duplicate-box removal, lost/reacquired segments, prediction separation, appearance/size/motion rejection, model class selection and tile coordinate mapping. They do not establish detector accuracy on real balls.

## 4. First real run: short smoke test

```powershell
python -m scripts.review_ball_v2 13a1022c5009 --weights data/models/football-ball-detection.pt --mode tiled --max-frames 10 --name dedicated-smoke
```

The model's metadata identifies the ball class rather than assuming COCO class 32. If the model's label is not recognised, the error prints its classes; inspect those before explicitly passing `--ball-class-id`. Do not guess the ID.

The command prints the exact output `index.html` path. Open it in your browser. This is an offline experiment; the backend, Expo and phone can remain as they are. CPU sliced inference can be slow because it performs multiple predictions per frame.

## 5. Run the first complete clip

**Stop here if the short test has no real-ball candidates.** On 7 November, the downloaded checkpoint (SHA-256 `678fbad05134f19c5094cb8d273812ec9c6691228180d46832551ecf99ed2912`) produced zero candidates in the first ten frames of the night clip at the experiment's 0.10 threshold. Independent spot checks at threshold 0.01 and input sizes 320, 640 and 1280 missed the actual ball at night frames 0 and 79. At daylight contact frame 117, full-frame size-1280 inference missed the ball but gave a turf object a 0.6423 score. This is evidence against using this checkpoint unchanged on these examples; it is not a general benchmark of all Roboflow models or all SCAN footage. Raw results and overlays are in ignored `data/reference-review/ball-v2-model-diagnostics/`.

Do not lower the threshold or relax association merely to produce a track. The next checkpoint is labelled-frame comparison of another ball model (for example WASB's soccer checkpoint), or adapting a detector with labelled SCAN footage and held-out clips. The association tests below remain useful, but passing them does not establish ball recognition quality. Resume full-clip comparisons only once raw candidates include the actual ball on representative inspected frames.

Compare the dedicated model without and with tiling:

```powershell
python -m scripts.review_ball_v2 13a1022c5009 --weights data/models/football-ball-detection.pt --mode single --name dedicated-single
python -m scripts.review_ball_v2 13a1022c5009 --weights data/models/football-ball-detection.pt --mode tiled --name dedicated-tiled
```

Each run gets a distinct output directory under `data/reference-review/ball-v2/<analysis-id>/`. No video or existing analysis result is overwritten. The page has a frame slider, previous/next buttons and arrow-key navigation.

- **Grey boxes:** all retained raw ball candidates.
- **Orange circle:** tentative detector match; not a confirmed track.
- **Green circle:** accepted detector match; not proof of correct identity.
- **Blue circle:** prediction only; never used as an observation.
- **Lost:** no selected ball location.

Inspect frames **79–80** first, then the approach and following flight. If grey boxes miss the real ball, association has no real observation to accept. If grey boxes contain it but the green selection follows the boot, association needs improvement. If the ball remains lost, keep that result visible rather than forcing a green trail.

## 6. Compare correct initialization

The optional `--seed X Y` is a hint for the ball centre in **frame 0 in original-image pixels**. It is not treated as an observed ball; detector matches still require confirmation. It is an experimental control, not a new phone calibration workflow.

For this first clip, an existing raw candidate near `(178, 785)` appears consistent with the starting ball. Confirm that location in the original first frame before using this example:

```powershell
python -m scripts.review_ball_v2 13a1022c5009 --weights data/models/football-ball-detection.pt --mode tiled --seed 178 785 --name dedicated-seeded
```

If an automatically initialized run fails but a correctly seeded run succeeds, initialization deserves attention. If both fail at contact, investigate ball recall, blur, occlusion and association gates. Do not use contact reference times as input to force the desired result.

## 7. Expand to the other clips

```powershell
python -m scripts.review_ball_v2 e9935750581e --weights data/models/football-ball-detection.pt --mode tiled --name dedicated-tiled
python -m scripts.review_ball_v2 96b382d82d92 --weights data/models/football-ball-detection.pt --mode tiled --name dedicated-tiled
```

Their approximate reviewed contact intervals are **117–118** and **136–137**. The moving-camera third clip is especially challenging: camera-motion compensation is not implemented here. A conservative loss of tracking is an honest result.

## 8. What to compare

Every run includes:

- `trace.jsonl`: frame index, timestamp, candidates, selected observation, separate prediction, segment and reason.
- `summary.json`: weight hash, actual model classes, ball class, seed, mode, state counts and runtime.
- `frames/` and `index.html`: original-frame overlays for inspection.

State counts are **not** accuracy scores. Record whether the true ball is detected, whether the selected identity is correct, false boot/turf selections, gaps and reacquisition time. Verify across new clips before production integration.

The tracker uses provisional image-space gates and a coarse HSV appearance comparison. A boot can still pass those checks; a genuine ball can fail after a sudden kick or lighting change. We are testing whether this improves the baseline, not claiming solved tracking. No shot/contact timing, goal result or coaching decision is derived from this experiment.

## Production integration comes after the comparison

Once evidence supports it, adapt the detector behind the existing `BallTrackResult` contract, preserve observed versus predicted sources, and replace strike-dependent initialization. Re-test goal analysis and event timing together. Keep WASB/TrackNet as replaceable detector alternatives if dedicated single-frame detection still misses the true ball near contact.
