# Experimental ball tracking in saved reps

Saved Expo reps now support an optional, manually initialised tracking replay.
The backend runs the guarded motion-recovery pilot with the existing detector
weights. The result is a visual experiment, kept separate from ordinary analysis,
SituatedState, Jev judgements, coaching and shot outcomes.

## Using the phone

Restart the laptop backend and the stable Expo preview to load these changes.
Keep the usual Tailscale connection active on the phone and laptop.

1. Open a saved stationary-shot rep.
2. Choose **Load initial frame** under **Experimental ball tracking**. This
   uploads the original clip once and prepares a visible frame near its start.
3. Tap the centre of the intended ball in the preview. The cyan ring is your
   manual starting position. Adjust it before starting if needed.
4. Choose **Run experimental tracking**. Progress comes from processed frames.
   Processing can take several minutes on the laptop.
5. Play the original video above the controls to inspect the overlay. Hide the
   overlay to use ordinary replay, including fullscreen playback.

Yellow rings show selected detector candidates with unverified identity. Purple
dashed rings show predictions without a selected observation. Lost frames have
no marker. The app never draws an observed point from a neighbouring frame or
fills a gap with an observation. Replay alignment uses frame index divided by
reported FPS and is approximate, particularly for variable-frame-rate media.

The original video is the playback source, so embedded sound is preserved.
There is no re-encoded tracking video. New recordings support microphone sound;
see [recording sound](../src/mobile/README.md#recording-sound). Older silent clips
cannot recover audio.

**Stop waiting**, navigation and backgrounding stop phone polling and pause
replay. A job already running on the laptop continues. **Check status** reconnects
to the same job while this replay screen remains open, without another upload.
If the screen is unmounted, reopen the rep and prepare a new job. Only one
experimental tracking job runs at a time, so wait for an existing job to finish
before starting another. A failed or interrupted job needs a new upload.

## Backend setup and job boundaries

The default model is the ignored local file
`data/models/scan-experimental-ball.pt`, copied from the reviewed guarded pilot.
Its SHA256 is
`8e1467e9fd3e2ce213ec262ab8aa10deb3422a67e99c008f4f50e0b7e296cdd0`.
`SCAN_EXPERIMENTAL_BALL_WEIGHTS` can select another local checkpoint. The model
must be the one-class ball detector; a missing model produces an error rather
than a download. Weights are not committed with the source code.

Uploads, previews, job state, raw tracking reports and compact results are stored
under ignored `data/reference-review/experimental-tracking/jobs/`. Original uploads
are copied byte-for-byte. Preview extraction sequentially decodes the exact frame
used to initialise the tracker; coordinates are normalized against that oriented
source image. No later labels or reseeding guide the run.

For sources with a short edge of at least 720 pixels, inference uses the current
720-pixel processing scale, motion-recovery association and guarded native-detail
supplement. Smaller sources stay at their native size, with no upscaling and no
native-detail supplement. The detector and association settings are unchanged.
The app path omits saved JPEGs for every frame; it returns normalized positions
for the original replay instead.

Uploads are limited to 512 MiB, 30 seconds, 1800 frames and supported 4K dimensions.
The worker enforces decoded-frame/image bounds and a cooperative 20-minute runtime
limit. Cancellation and timeout checks run between frames and inference calls;
they cannot forcibly interrupt a native inference call that has stalled. An OS
file lock prevents concurrent experimental inference, and job state is written
atomically. Abandoned active jobs are marked interrupted when checked after a
backend restart. The ordinary `/analyze` endpoint does not wait for this job.

## API

| Request | Purpose |
| --- | --- |
| `POST /experimental-tracking/preview` | Multipart `video` upload; returns preview metadata and a job ID |
| `GET /experimental-tracking/{id}/preview` | Initial frame JPEG |
| `POST /experimental-tracking/{id}/start` | JSON `{"seed":{"x":0.4,"y":0.6}}`, with coordinates in `[0,1)` |
| `GET /experimental-tracking/{id}` | Status and frame progress |
| `GET /experimental-tracking/{id}/result` | Completed compact trajectory and provenance |
| `POST /experimental-tracking/{id}/cancel` | Cooperative server cancellation |

Job IDs and artifact paths are validated. Starting the same job again with the
same seed is idempotent; changing a started seed requires a new job. The public
contract has no goal, contact, physical speed, curl or technique assessment.
Every candidate retains `identity_verified: false`.

## Verification

Automated checks cover upload/seed validation, exact preview coordinates,
resource limits, interrupted/cancelled/failed jobs, separate model loading,
progress and normalized results. Existing offline tests check that default
review output is preserved. Mobile checks cover letterboxing, frame lookup,
observed/predicted/lost separation, multipart upload and status reconnection.

A real-model run on an already inspected clip is an integration smoke test,
not new accuracy evidence. The remaining phone check is to select a ball,
run a job, seek and replay with the overlay, navigate away to confirm pause,
then record, replay and export a short new clip with sound.
