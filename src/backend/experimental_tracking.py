"""Opt-in HTTP adapter for the manually seeded, guarded ball-tracking pilot.

No output from this service enters SituatedState, Jev, coaching, or goal
decisions. Importing or creating the router does not import vision libraries,
load a model, start a worker, inspect saved jobs, or download anything.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool


PREFIX = "/experimental-tracking"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_DURATION_S = 30.0
MAX_FRAMES = 1800
MAX_DIMENSION = 4096
MAX_PIXELS = 4096 * 2160
MAX_RUNTIME_S = 20 * 60
CPU_THREADS = 8
_ID = re.compile(r"[0-9a-f]{32}\Z")
_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}
_TERMINAL = {"completed", "failed", "cancelled", "interrupted"}
_LIMITATIONS = [
    "Experimental manually seeded tracking; selected detections are provisional candidates, not verified ball identity.",
    "One tap on the exact preview frame initializes the search. No per-frame labels or reseeding are used.",
    "Predictions are search estimates and are never presented as observed detections.",
    "No goal, shot outcome, speed, curl, biomechanics or coaching judgement is produced.",
    "Frame times use frame index divided by reported FPS, not independently verified presentation timestamps.",
    "Cancellation and the runtime limit are cooperative checks between decoded frames and inference calls.",
]


def _finite(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON fields are not supported.")
        result[key] = value
    return result


def _reject_nonfinite(_):
    raise ValueError("Coordinates must be finite numbers.")


def _seed_body(body: bytes) -> dict:
    if len(body) > 4096:
        raise ValueError("The seed request is too large.")
    try:
        value = json.loads(body, object_pairs_hook=_unique_json, parse_constant=_reject_nonfinite)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("Send a JSON object containing one normalized seed.") from exc
    if (type(value) is not dict or 'seed' not in value
            or set(value)-{'seed','recover_misses'}
            or type(value.get('recover_misses',False)) is not bool):
        raise ValueError("Send one seed and an optional boolean recover_misses; labels and trajectories cannot be uploaded.")
    seed = value["seed"]
    if type(seed) is not dict or set(seed) != {"x", "y"}:
        raise ValueError("The seed must contain x and y only.")
    if not all(_finite(seed[key]) and 0 <= seed[key] < 1 for key in ("x", "y")):
        raise ValueError("Seed coordinates must be finite numbers in [0, 1).")
    return {key: float(seed[key]) for key in ("x", "y")}


class _Lease:
    """An OS-held lock, released automatically if a server process exits."""

    def __init__(self, path: Path):
        self.handle = path.open("a+b")
        try:
            self.handle.seek(0, 2)
            if self.handle.tell() == 0:
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.handle.close()
            raise

    def close(self):
        if self.handle.closed:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()


def _bounded_image(image) -> tuple[int, int]:
    height, width = image.shape[:2]
    if width < 1 or height < 1 or max(width, height) > MAX_DIMENSION or width * height > MAX_PIXELS:
        raise ValueError("The decoded clip exceeds the supported 4K image bounds.")
    return width, height


def _extract_preview(source: Path, output: Path) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError("This video could not be decoded. Export the original MOV or MP4 clip.")
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not _finite(fps) or not 0 < fps <= 240 or not _finite(count) or count < 1:
            raise ValueError("This clip has no supported frame-rate or duration metadata.")
        count = int(count)
        duration = count / fps
        if count > MAX_FRAMES or duration > MAX_DURATION_S:
            raise ValueError("Use a clip of at most 30 seconds and 1800 frames.")
        # Sequential decoding preserves the exact frame index used by run_video.
        # Never seek approximately to a frame and then claim it is an exact seed.
        for index in range(min(count, int(fps * 2) + 1, 121)):
            ok, image = capture.read()
            if not ok:
                break
            width, height = _bounded_image(image)
            if (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 12).mean() < 0.01:
                continue
            ratio = min(1.0, 1280.0 / max(width, height))
            preview = (cv2.resize(image, (round(width * ratio), round(height * ratio)),
                                  interpolation=cv2.INTER_AREA) if ratio < 1 else image)
            if not cv2.imwrite(str(output), preview, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError("Could not save the ball-selection preview.")
            return {"width": width, "height": height, "preview_width": preview.shape[1],
                    "preview_height": preview.shape[0], "frame_index": index,
                    "time_s": index / fps, "fps": float(fps), "duration_s": duration,
                    "reported_frames": count, "total_frames": count - index}
        raise ValueError("No visible frame was found near the start. Choose another clip.")
    finally:
        capture.release()


def _normalized_point(value, width, height):
    if (type(value) not in (list, tuple) or len(value) != 2
            or not all(_finite(component) for component in value)):
        return None
    x, y = value[0] / width, value[1] / height
    if not (0 <= x < 1 and 0 <= y < 1):
        return None
    return {"x": x, "y": y}


def _decode_frame_times(source: Path) -> list[float] | None:
    """Read the source timeline, without running or changing ball inference.

    Phone MOV average-rate metadata can disagree with its presentation times.
    Only accept a complete, zero-based, strictly increasing decoded timeline.
    Unsupported timestamps retain the explicitly approximate FPS fallback.
    """
    import cv2

    capture = cv2.VideoCapture(str(source))
    started = time.monotonic()
    times = []
    try:
        if not capture.isOpened():
            return None
        while True:
            if time.monotonic() - started > 20:
                return None
            # The completed tracker already checked every decoded image. grab
            # advances its presentation timestamp without allocating/converting
            # another full 4K BGR image merely to read the clock.
            if not capture.grab():
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if (len(times) >= MAX_FRAMES or not _finite(timestamp)
                    or not 0 <= timestamp <= MAX_DURATION_S
                    or (not times and abs(timestamp) > 1e-6)
                    or (times and timestamp <= times[-1])):
                return None
            times.append(float(timestamp))
        return times or None
    finally:
        capture.release()


def _with_playback_timing(result: dict, times: list[float] | None) -> dict:
    """Change display times only; preserve every candidate, prediction and loss."""
    value = dict(result)
    points = result.get('points', [])
    valid = (type(times) is list and bool(times) and bool(points)
             and len(times) == points[-1]['frame'] + 1
             and len(times) <= MAX_FRAMES
             and all(_finite(t) and 0 <= t <= MAX_DURATION_S for t in times)
             and abs(times[0]) <= 1e-6
             and all(a < b for a, b in zip(times, times[1:])))
    value['playback_timing_version'] = 1
    value['timing_mode'] = 'presentation_timestamps' if valid else 'reported_fps_approximate'
    if valid:
        # The next decoded timestamp closes each frame's display interval.
        # The final interval uses a conservative, bounded duration estimate;
        # it is a display bound, not a motion estimate.
        last_duration = min(times[-1] - times[-2], 1 / result['fps'], 1.0) if len(times) > 1 else min(1 / result['fps'], 1.0)
        ends = times[1:] + [times[-1] + last_duration]
        value['points'] = [{**point, 'time_s': times[point['frame']],
                            'end_time_s': ends[point['frame']]} for point in points]
        value['limitations'] = [item for item in result.get('limitations', [])
                                if not item.startswith('Frame times use frame index')]
        value['limitations'].append('Replay uses decoded presentation timestamps; native video and JavaScript overlay rendering are not frame-locked. The final frame duration is estimated.')
    return value


def _compact_result(tracking_id: str, state: dict, report: dict) -> dict:
    width, height, fps = report.get("width"), report.get("height"), report.get("fps")
    if (not all(_finite(value) and value > 0 for value in (width, height, fps))
            or abs(fps - state["fps"]) > 1e-6):
        raise ValueError("Tracker output has inconsistent image or timing metadata.")
    transform = report.get("frame_transform", {})
    if (transform.get("source_width") != state["width"]
            or transform.get("source_height") != state["height"]
            or report.get("start_frame") != state["frame_index"]):
        raise ValueError("Tracker output does not match the selected source preview.")
    points = []
    path = report.get("path")
    if type(path) is not list or not path or len(path) > MAX_FRAMES:
        raise ValueError("Tracker did not return a bounded frame sequence.")
    for index, item in enumerate(path):
        frame = item.get("frame")
        if type(frame) is not int or frame != state["frame_index"] + index:
            raise ValueError("Tracker output is not a contiguous sequence from the selected preview.")
        reason = item.get("reason", "unverified_tracker_candidate")
        reason = reason if type(reason) is str else "unspecified_tracker_reason"
        center = (_normalized_point(item.get("center"), width, height)
                  if item.get("detected") is True else None)
        prediction = _normalized_point(item.get("predicted"), width, height)
        if center is not None:
            point_state = "observed_candidate"
        elif reason.startswith("lost") or prediction is None:
            point_state, prediction = "lost", None
        else:
            point_state = "predicted"
        if item.get("detected") is True and center is None:
            reason = "selected_candidate_outside_image_or_nonfinite"
        points.append({"frame": frame, "time_s": frame / fps, "state": point_state,
                       "center": center, "prediction": prediction,
                       "identity_verified": False, "reason": reason,
                       "recovered": center is not None and reason == 'detector_confirmed_learned_recovery_provisional'})
    limitations = list(_LIMITATIONS)
    if state.get('recover_misses'):
        limitations.append('Optional recovery uses later frames from the saved clip to confirm measured candidates; it is not a live causal tracker.')
    if min(state["width"], state["height"]) < 720:
        limitations.append("Source short edge is below 720 pixels; native-detail fallback is disabled and no upscaling is used.")
    return {
        "status": "completed", "tracking_id": tracking_id,
        "width": state["width"], "height": state["height"],
        "processing_width": width, "processing_height": height,
        "fps": fps, "duration_s": state["duration_s"],
        "start_frame": state["frame_index"], "seed": state["seed"],
        "points": points, "processed_frames": len(points),
        "observed_candidate_frames": sum(point["state"] == "observed_candidate" for point in points),
        "identity_verified": False, "association": report.get('association_method','motion-recovery'),
        "native_detail_fallback": report.get("native_detail_fallback") is True,
        "limitations": limitations,
        "processing": state.get('processing'),
        "recovery": state.get('recovery'),
        "provenance": {key: report.get(key) for key in (
            "source_sha256", "weights_sha256", "implementation_sha256",
            "native_detail_implementation_sha256", "settings", "association_settings",
        )},
    }


class ExperimentalTrackingService:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.jobs = self.project_root / "data/reference-review/experimental-tracking/jobs"
        self.owner = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.worker = None
        self.cancel_event = None
        self.closed = False
        self.model = None
        self.model_signature = None
        self.recovery_provider = None

    def _recovery_available(self, state):
        root = self.project_root/'data/models/research/cotracker3'
        return (min(state['width'],state['height']) >= 720
            and max(state['width'],state['height']) <= 3840
            and state['width']*state['height'] <= 3840*2160
            and (root/'scaled_offline.pth').is_file()
            and (root/'provenance.json').is_file()
            and (root/'source/cotracker/predictor.py').is_file())

    def _load_recovery(self):
        # Provenance and every pinned source hash are checked on each job, even
        # when a validated model can be retained between research requests.
        from experiments.ball_tracking.cotracker_recovery import CoTrackerRecovery
        provider = CoTrackerRecovery(self.project_root/'data/models/research/cotracker3',
            runtime_directory=self.project_root/'data/models/research/runtime',threads=CPU_THREADS)
        if self.recovery_provider is not None and provider.configuration()==self.recovery_provider.configuration():
            provider.model = self.recovery_provider.model
        self.recovery_provider = provider
        return provider

    def _directory(self, tracking_id: str) -> Path:
        if type(tracking_id) is not str or _ID.fullmatch(tracking_id) is None:
            raise HTTPException(404, "Tracking job not found.")
        directory = (self.jobs / tracking_id).resolve()
        if directory.parent != self.jobs.resolve() or not directory.is_dir():
            raise HTTPException(404, "Tracking job not found.")
        return directory

    def _weights(self) -> Path:
        supplied = os.environ.get("SCAN_EXPERIMENTAL_BALL_WEIGHTS")
        path = Path(supplied) if supplied else self.project_root / "data/models/scan-experimental-ball.pt"
        return (path if path.is_absolute() else self.project_root / path).resolve()

    def _model_available(self) -> bool:
        path = self._weights()
        return path.suffix.lower() == ".pt" and path.is_file()

    def _read(self, tracking_id: str) -> tuple[Path, dict]:
        directory = self._directory(tracking_id)
        try:
            state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            if state["tracking_id"] != tracking_id:
                raise ValueError("Mismatched saved job identifier")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(500, "Saved tracking job metadata is unavailable.") from exc
        if state["status"] in {"queued", "running"} and state.get("owner") != self.owner:
            # A lock held by another live process is evidence that its job is
            # still running. An abandoned JSON state alone is never evidence.
            try:
                lease = _Lease(directory / ".worker.lock")
            except OSError:
                pass
            else:
                try:
                    state.update(status="interrupted", error="The backend restarted before this tracking job finished. Upload the clip again to retry.")
                    _atomic_json(directory / "state.json", state)
                finally:
                    lease.close()
        return directory, state

    def _public(self, state: dict) -> dict:
        value = {key: state[key] for key in (
            "tracking_id", "status", "width", "height", "preview_width", "preview_height",
            "frame_index", "time_s", "fps", "duration_s", "total_frames", "processed_frames", "error",
        )}
        value.update(preview_url=f"{PREFIX}/{state['tracking_id']}/preview",
                     result_url=f"{PREFIX}/{state['tracking_id']}/result" if state["status"] == "completed" else None,
                     model_available=self._model_available(), experimental=True)
        value.update(recovery_available=self._recovery_available(state),
            recovery_requested=state.get('recover_misses',False),phase=state.get('phase','baseline'),
            recovery_processed_windows=state.get('recovery_processed_windows',0),
            recovery_total_windows=state.get('recovery_total_windows',0))
        if state.get("seed") is not None:
            value["seed"] = state["seed"]
        if not value["model_available"] and value["error"] is None:
            value["error"] = "The local experimental ball model is unavailable. Configure SCAN_EXPERIMENTAL_BALL_WEIGHTS or restore data/models/scan-experimental-ball.pt; no model will be downloaded."
        return value

    def preview(self, upload: UploadFile) -> dict:
        if self.closed:
            raise HTTPException(503, "The tracking service is shutting down.")
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in _EXTENSIONS:
            raise HTTPException(400, "Use an original MOV, MP4 or other supported video file.")
        tracking_id = uuid.uuid4().hex
        directory = self.jobs / tracking_id
        directory.mkdir(parents=True)
        source = directory / f"source{suffix}"
        try:
            size = 0
            with source.open("xb") as output:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "The clip exceeds the 512 MiB upload limit.")
                    output.write(chunk)
            if size == 0:
                raise ValueError("The uploaded video is empty.")
            metadata = _extract_preview(source, directory / "preview.jpg")
            state = {"tracking_id": tracking_id, "status": "ready", "source_suffix": suffix,
                     "created_at": time.time(), "owner": self.owner, "processed_frames": 0,
                     "error": None, "seed": None, **metadata}
            _atomic_json(directory / "state.json", state)
            return self._public(state)
        except HTTPException:
            source.unlink(missing_ok=True)
            raise
        except (ValueError, OSError) as exc:
            source.unlink(missing_ok=True)
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            source.unlink(missing_ok=True)
            raise HTTPException(400, "Could not create a preview from this video.") from exc

    def status(self, tracking_id: str) -> dict:
        with self.lock:
            return self._public(self._read(tracking_id)[1])

    def preview_path(self, tracking_id: str) -> Path:
        with self.lock:
            directory, _ = self._read(tracking_id)
            path = directory / "preview.jpg"
            if not path.is_file():
                raise HTTPException(404, "The preview image is unavailable.")
            return path

    def result(self, tracking_id: str) -> dict:
        with self.lock:
            directory, state = self._read(tracking_id)
            if state["status"] != "completed":
                raise HTTPException(409, f"Tracking is {state['status']}; no completed result is available.")
            try:
                result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise HTTPException(500, "The completed tracking result is unavailable.") from exc
            cache = directory / 'result.playback-v1.json'
            try:
                saved = json.loads(cache.read_text(encoding='utf-8'))
                if (saved.get('tracking_id') == tracking_id
                        and saved.get('playback_timing_version') == 1):
                    return saved
            except (OSError, ValueError, AttributeError):
                pass
        # Decode outside the service lock. This also upgrades already completed
        # phone jobs without another upload or several minutes of inference.
        # Preserve the original result and all raw tracking/recovery reports.
        try:
            times = _decode_frame_times(directory / f"source{state['source_suffix']}")
        except Exception:
            times = None
        value = _with_playback_timing(result, times)
        with self.lock:
            _atomic_json(cache, value)
        return value

    def start(self, tracking_id: str, seed: dict, *, recover_misses=False) -> dict:
        with self.lock:
            directory, state = self._read(tracking_id)
            if type(recover_misses) is not bool:
                raise HTTPException(400,"recover_misses must be boolean.")
            if state.get("seed") is not None:
                if state["seed"] == seed and state.get('recover_misses',False)==recover_misses:
                    return self._public(state)
                raise HTTPException(409, "This job already has a different seed or recovery option. Upload the clip again to change them.")
            if self.closed:
                raise HTTPException(503, "The tracking service is shutting down.")
            if state["status"] != "ready":
                raise HTTPException(409, "This job is not ready to start. Upload the clip again.")
            if not self._model_available():
                raise HTTPException(503, self._public(state)["error"])
            if recover_misses and not self._recovery_available(state):
                raise HTTPException(503,"Research recovery needs the local pinned model and a supported original clip of at least 720p. Basic tracking is still available.")
            try:
                lease = _Lease(self.jobs / ".active.lock")
            except OSError as exc:
                raise HTTPException(429, "Another experimental tracking job is running. Wait for it to finish, then start this clip.") from exc
            state.update(seed=dict(seed), status="queued", owner=self.owner, error=None,
                         recover_misses=recover_misses,phase='baseline')
            job_lease = None
            try:
                job_lease = _Lease(directory / ".worker.lock")
                _atomic_json(directory / "state.json", state)
                event = threading.Event()
                self.cancel_event = event
                self.worker = threading.Thread(target=self._execute, args=(directory, state, event, lease, job_lease),
                                               name="scan-experimental-tracker", daemon=True)
                self.worker.start()
            except Exception:
                if job_lease is not None:
                    job_lease.close()
                lease.close()
                state.update(status="failed", error="The tracking worker could not start.")
                _atomic_json(directory / "state.json", state)
                raise HTTPException(503, state["error"])
            return self._public(state)

    def cancel(self, tracking_id: str) -> dict:
        with self.lock:
            directory, state = self._read(tracking_id)
            if state["status"] in _TERMINAL:
                return self._public(state)
            if state["status"] == "ready":
                state.update(status="cancelled", error="Tracking was cancelled before it started.")
                _atomic_json(directory / "state.json", state)
            else:
                # The marker also reaches workers owned by another server
                # instance; no in-memory owner assumption is required.
                (directory / "cancel.requested").touch(exist_ok=True)
                if state.get("owner") == self.owner and self.cancel_event is not None:
                    self.cancel_event.set()
            return self._public(state)

    def _load_model(self, weights: Path):
        signature = (str(weights), weights.stat().st_size, weights.stat().st_mtime_ns)
        if self.model_signature != signature:
            import torch
            from ultralytics import YOLO
            torch.set_num_threads(CPU_THREADS)
            model = YOLO(str(weights))
            if model.task != "detect" or model.names != {0: "ball"}:
                raise ValueError("The configured model must be the validated one-class ball detector.")
            model.to("cpu")
            self.model, self.model_signature = model, signature
        return self.model

    def _execute(self, directory: Path, state: dict, event: threading.Event, lease: _Lease, job_lease: _Lease):
        started, last_saved = time.monotonic(), 0.0
        def cancelled():
            return event.is_set() or (directory / "cancel.requested").exists()
        def progress(value):
            nonlocal last_saved
            state.update(processed_frames=value["processed_frames"])
            if time.monotonic() - last_saved >= 1:
                with self.lock:
                    _atomic_json(directory / "state.json", state)
                last_saved = time.monotonic()
        try:
            from experiments.ball_tracking.track_unlabelled_video import run_video, TrackingCancelledError
            if cancelled():
                raise TrackingCancelledError("Tracking was cancelled.")
            state.update(status="running")
            with self.lock:
                _atomic_json(directory / "state.json", state)
            weights = self._weights()
            if not weights.is_file() or weights.suffix.lower() != ".pt":
                raise ValueError("The local experimental ball model is unavailable; no model will be downloaded.")
            model = self._load_model(weights)
            model_loaded = time.monotonic()
            remaining = MAX_RUNTIME_S - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Model loading exceeded the tracking runtime limit.")
            native = min(state["width"], state["height"]) >= 720
            report = run_video(
                {"source": str(directory / f"source{state['source_suffix']}"),
                 "clip": state["tracking_id"], "start_frame": state["frame_index"],
                 "seed": [state["seed"]["x"] * state["width"], state["seed"]["y"] * state["height"]]},
                model, weights, directory / "tracking", CPU_THREADS,
                association="drift-recovery" if state.get('recover_misses') else "motion-recovery",
                processing_short_edge=720 if native else None,
                native_detail_fallback=native, save_frames=False,
                merge_nested_native=bool(state.get('recover_misses')),
                trajectory_native_support=bool(state.get('recover_misses')),
                progress_callback=progress, cancel_check=cancelled,
                maximum_frames=min(MAX_FRAMES, int(MAX_DURATION_S * state["fps"])), maximum_runtime_s=remaining,
                maximum_source_pixels=MAX_PIXELS, maximum_source_dimension=MAX_DIMENSION,
            )
            if cancelled():
                raise TrackingCancelledError("Tracking was cancelled.")
            if report.get("reached_reported_end") is not True:
                raise ValueError("Video decoding ended before the reported end; this partial run is not a completed result.")
            baseline_finished = time.monotonic()
            state['recovery'] = {'requested':False,'status':'not_requested','added_observations':0}
            if state.get('recover_misses'):
                state.update(phase='recovery',recovery_processed_windows=0,recovery_total_windows=0)
                with self.lock:
                    _atomic_json(directory/'state.json',state)
                def recovery_progress(value):
                    state.update(recovery_processed_windows=value['processed_windows'],
                                 recovery_total_windows=value['total_windows'])
                    with self.lock:
                        _atomic_json(directory/'state.json',state)
                try:
                    from experiments.ball_tracking.detector_confirmed_recovery import run_recovery, LearnedRecoverySettings
                    provider = self._load_recovery()
                    remaining = MAX_RUNTIME_S-(time.monotonic()-started)
                    if remaining <= 0:
                        raise TimeoutError('No recovery time remains')
                    recovered = run_recovery(report,directory/f"source{state['source_suffix']}",
                        model,weights,provider,directory/'recovery/report.json',
                        baseline_path=directory/'tracking/report.json',
                        settings=LearnedRecoverySettings(confirm_measured_history=True,motion_guided_crop=True,
                            maximum_runtime_s=min(240.0,remaining)),
                        cancel_check=cancelled,progress_callback=recovery_progress)
                    info = recovered['learned_recovery']
                    state['recovery'] = {'requested':True,
                        'status':'partial' if info['errors'] else 'completed',
                        'added_observations':len(info['new_observation_frames']),
                        'warning':'Some recovery searches could not finish; existing observations were preserved.' if info['errors'] else None}
                    report = recovered
                except Exception:
                    if cancelled():
                        raise TrackingCancelledError('Recovery was cancelled')
                    import logging
                    logging.getLogger(__name__).exception('Optional recovery unavailable for job %s',state['tracking_id'])
                    state['recovery'] = {'requested':True,'status':'unavailable','added_observations':0,
                        'warning':'Recovery could not finish. The completed basic tracking result is available.'}
            if cancelled():
                raise TrackingCancelledError('Tracking was cancelled')
            finished = time.monotonic()
            state['processing'] = {'model_load_s':model_loaded-started,
                'baseline_s':baseline_finished-model_loaded,'recovery_s':finished-baseline_finished,
                'total_s':finished-started,'excludes_upload':True}
            state['phase'] = 'completed'
            result = _compact_result(state["tracking_id"], state, report)
            _atomic_json(directory / "result.json", result)
            state.update(status="completed", processed_frames=result["processed_frames"],
                         total_frames=result["processed_frames"], error=None)
        except Exception as exc:
            # Do not expose source paths, tracebacks, credentials, or arbitrary
            # model exception strings over the API.
            if cancelled() or type(exc).__name__ == "TrackingCancelledError":
                state.update(status="cancelled", error="Tracking was cancelled. No completed result was produced.")
            elif isinstance(exc, TimeoutError):
                state.update(status="failed", error="Experimental tracking exceeded the 20-minute runtime limit.")
            else:
                state.update(status="failed", error="Experimental tracking failed validation or processing. Check the backend log and try another clip.")
                import logging
                logging.getLogger(__name__).exception("Experimental tracking failed for job %s", state["tracking_id"])
        finally:
            try:
                with self.lock:
                    _atomic_json(directory / "state.json", state)
            finally:
                try:
                    job_lease.close()
                finally:
                    lease.close()

    def close(self):
        with self.lock:
            self.closed = True
            if self.cancel_event is not None:
                self.cancel_event.set()
            worker = self.worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)


def create_tracking_router(project_root: Path) -> APIRouter:
    service = ExperimentalTrackingService(project_root)
    router = APIRouter(prefix=PREFIX, tags=["Experimental ball tracking"])
    router.tracking_service = service

    @router.post("/preview")
    async def preview(video: UploadFile = File(...)):
        try:
            return await run_in_threadpool(service.preview, video)
        finally:
            await video.close()

    @router.get("/{tracking_id}/preview")
    def preview_image(tracking_id: str):
        return FileResponse(service.preview_path(tracking_id), media_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

    @router.post("/{tracking_id}/start")
    async def start(tracking_id: str, request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(400, "Send the normalized seed as application/json.")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 4096:
                raise HTTPException(413, "The seed request is too large.")
        try:
            seed = _seed_body(bytes(body))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        recover_misses = json.loads(body).get('recover_misses',False)
        return await run_in_threadpool(service.start, tracking_id, seed, recover_misses=recover_misses)

    @router.get("/{tracking_id}")
    def status(tracking_id: str):
        return service.status(tracking_id)

    @router.get("/{tracking_id}/result")
    def result(tracking_id: str):
        return service.result(tracking_id)

    @router.post("/{tracking_id}/cancel")
    def cancel(tracking_id: str):
        return service.cancel(tracking_id)

    router.add_event_handler("shutdown", service.close)
    return router
