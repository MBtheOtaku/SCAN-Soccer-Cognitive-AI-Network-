# Implement next: WASB three-frame CPU probe

This is a small offline model comparison, not a replacement for SCAN tracking. It inspects three consecutive original frames around contact without calling Jev, changing coaching, or installing the author's old CUDA environment.

The official detector wrapper asserts CUDA availability. Our custom CPU probe imports only the author's HRNet architecture and affine utilities and uses the existing PyTorch, OpenCV, NumPy and PyYAML packages. It follows the author's RGB/ImageNet normalisation, three-frame channel concatenation, scale-0 sigmoid heatmaps and weighted connected components. It does not reproduce the complete paper's tracking or oversampling evaluation. [Original wrapper](https://github.com/nttcom/WASB-SBDT/blob/main/src/detectors/detector.py), [preprocessing](https://github.com/nttcom/WASB-SBDT/blob/main/src/dataloaders/dataset_loader.py), [model zoo](https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md).

## 1. Copy the next two files

Read the complete code in `docs/wasb/probe_wasb.py.template` and `docs/wasb/test_wasb_probe.py.template`, then copy them into place. Run from SCAN's root in a spare PowerShell terminal:

```powershell
Set-Location "C:\Users\Matthew Bui\OneDrive\SCAN"
Copy-Item docs/wasb/probe_wasb.py.template scripts/probe_wasb.py
Copy-Item docs/wasb/test_wasb_probe.py.template tests/test_wasb_probe.py
```

Run these copies once before making changes of your own; repeated copies replace the destination files.

## 2. Obtain the author's source in the ignored model directory

The root `.gitignore` should already include `data/models/` from the previous experiment. Keep that entry.

```powershell
git clone --depth 1 https://github.com/nttcom/WASB-SBDT.git data/models/WASB-SBDT
```

If that directory already contains this clone, skip the clone command. Do not install the repository's Dockerfile or old packages into SCAN. We need its source files and configuration only.

## 3. Download the soccer WASB checkpoint

Download the **Soccer column, WASB (Ours) row** from the [official model zoo](https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md). [Direct author-provided download page](https://drive.google.com/file/d/1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l/view).

Save the downloaded binary as:

```text
C:\Users\Matthew Bui\OneDrive\SCAN\data\models\wasb_soccer_best.pth.tar
```

This is a different model from the Roboflow `.pt` file. Do not rename the Roboflow model to this filename.

Check the files and existing imports:

```powershell
Get-Item data/models/wasb_soccer_best.pth.tar | Select-Object Name,Length
python -c "import torch, cv2, numpy, yaml; print('WASB probe dependencies available')"
```

The checked SCAN environment already had PyTorch 2.10.0+cpu and PyYAML 6.0.3. No package installation is needed there. The probe uses strict state-dictionary loading and restricted checkpoint loading. If either reports an error, paste it before changing model architecture, ignoring missing keys or changing checkpoint loading settings.

## 4. Run the helper tests

```powershell
python -m pytest tests/test_wasb_probe.py -q
```

There are seven tests. They check channel order, configuration access, heatmap validity, output shape and inverse-coordinate mapping. Passing them does not establish recognition accuracy.

## 5. Run just the first three-frame sequence

```powershell
python -m scripts.probe_wasb 13a1022c5009 --start-frame 78
```

This processes frames **78, 79 and 80**. The reviewed contact interval is approximately 79–80. There is no reference label supplied to the model; the start frame just chooses the evaluation excerpt.

Open the exact `index.html` path printed at completion. Each frame shows:

- The original image with yellow heatmap candidates.
- The actual transformed model input, so we can see whether the ball was cropped away.
- The position heatmap and candidate coordinates.

Activation values are not calibrated probabilities. A yellow circle is not confirmed ball identity. No candidate remains an explicit empty result.

The author's affine transform was designed around research footage and can crop parts of portrait phone frames. Inspect the actual model input before concluding that a missing candidate proves detector failure. Do not alter the aspect ratio, invert the mapping by eye or silently move points to a desired location.

## 6. If it loads and runs, compare the other two contact windows

```powershell
python -m scripts.probe_wasb e9935750581e --start-frame 116
python -m scripts.probe_wasb 96b382d82d92 --start-frame 135
```

These inspect **116–118** and **135–137** around reviewed contact intervals 117–118 and 136–137. A single three-frame excerpt does not establish full-clip tracking or generalisation.

Save the screenshots/results, including failures. If raw heatmaps recover the actual ball, temporal association can be evaluated next. If they do not, pause model hopping and prepare visible-ball annotations and boot/turf negatives from SCAN footage for adaptation and held-out evaluation.

Backend and Expo restarts are unnecessary. Results are written under ignored `data/reference-review/wasb/`; original videos and saved analyses are unchanged.
