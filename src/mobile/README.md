# SCAN mobile capture

The active recording screen is `src/app/record.tsx`, loaded by Expo Router.
The older `App.tsx` is not the entry point.

## Recording quality

Open **Camera settings** on the recording screen. Each newly opened screen
starts with **1080p** and **Auto** image stabilization. Available requests are:

| Quality | Requested video bitrate |
| --- | --- |
| 720p | 6 Mbps |
| 1080p | 12 Mbps |
| 4K / 2160p | 35 Mbps |

Higher resolutions and bitrates can preserve more small-ball detail, but their
tracking benefit is not yet measured. Files need more storage and upload time.
The camera/encoder may choose a supported format instead, so the preview badge
describes the **requested**, not verified, resolution. On iOS, capture requests
H.264 (`avc1`) so Expo applies the requested bitrate.

Quality and stabilization can change only between recordings. A change
restarts the camera preview and waits for the replacement camera's ready
event. Late events from the previous quality/stabilization session cannot
enable recording.
If capture fails, the screen offers a lower-quality retry without changing
existing clips. Saving, recovery, sharing and upload still use the original
camera file without resizing or filtering it.

Auto stabilization handles camera movement. Off is available for comparisons
on a fixed tripod. Stabilization does not remove a fast ball's motion blur.
Use good lighting and keep the camera steady. Frame rate and exposure remain
device-controlled: the installed Expo Camera SDK 58 API does not expose
manual native shutter-speed or frame-rate settings, or a general live frame
processor for applying contrast/deblur filters.

## Applying changes and checking a phone

The stable preview disables automatic reload. Stop and restart the existing
Expo terminal with the same proxy/network settings:

```powershell
npm run start:stable -- --localhost
```

Open the existing Expo Go remote link. Record a short 1080p rep, save and
replay it, then export the original file. Check its actual dimensions and
frame rate; repeat with 4K when supported. Keep the ball, camera position and
lighting comparable. Do not infer recording format from the preview badge.

Type checking and lint run on the laptop. Physical camera support, resulting
resolution/bitrate and save/replay behaviour still need the phone check.

## Image-processing experiment

`../../experiments/ball_tracking/probe_contrast_detector.py` compares the
frozen detector on original frames, mild luminance CLAHE and conservative
unsharp masking. A separate positive-motion overlay aligns estimated camera
translation before highlighting brightening. It is a diagnostic, not a learned
ball heatmap or identity probability. Sharpening does not recover detail
lost to motion blur.

These experiments run offline on saved review frames. They do not modify
recorded clips or app coaching. See the ball-tracking experiments README for
the command, limitations and observed results.

SDK documentation: <https://docs.expo.dev/versions/v58.0.0/sdk/camera/>.
