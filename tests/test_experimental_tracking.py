"""The optional HTTP tracker uses one exact seed and never supplies coaching."""

import builtins
import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient

from src.backend import experimental_tracking as tracking


def metadata(frame_index=0, width=96, height=64, count=3):
    return {"width": width, "height": height, "preview_width": width,
            "preview_height": height, "frame_index": frame_index,
            "time_s": frame_index / 30, "fps": 30.0, "duration_s": count / 30,
            "reported_frames": count, "total_frames": count - frame_index}


def mock_preview(source, destination):
    destination.write_bytes(b"preview-image")
    return metadata()


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.delenv("SCAN_EXPERIMENTAL_BALL_WEIGHTS", raising=False)
    monkeypatch.setattr(tracking, "_extract_preview", mock_preview)
    weights = tmp_path / "data/models/scan-experimental-ball.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"frozen-local-weights")
    router = tracking.create_tracking_router(tmp_path)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client, router.tracking_service, tmp_path


def upload(client):
    response = client.post("/experimental-tracking/preview", files={
        "video": ("phone.mov", b"the-unchanged-original-clip", "video/quicktime"),
    })
    assert response.status_code == 200, response.text
    return response.json()


def await_terminal(service, identifier):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = service.status(identifier)
        if state["status"] in tracking._TERMINAL:
            if service.worker is not None:
                service.worker.join(timeout=2)
            return state
        time.sleep(0.01)
    raise AssertionError("Test worker did not finish")


def fake_report(case, width=96, height=64, native=False):
    start = case["start_frame"]
    return {"width": width, "height": height, "fps": 30.0, "start_frame": start,
            "frame_transform": {"source_width": width, "source_height": height},
            "reached_reported_end": True, "native_detail_fallback": native,
            "path": [
                {"frame": start, "detected": True, "center": [24, 32],
                 "predicted": [23, 32], "reason": "near_prediction_provisional"},
                {"frame": start + 1, "detected": False, "center": None,
                 "predicted": [25, 32], "reason": "awaiting_temporal_support"},
                {"frame": start + 2, "detected": False, "center": None,
                 "predicted": [26, 32], "reason": "lost_requires_new_seed"},
            ]}


def test_router_factory_is_lazy_and_does_not_scan_or_start_workers(tmp_path, monkeypatch):
    original_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        if name.split(".")[0] in {"cv2", "torch", "ultralytics"}:
            raise AssertionError(f"Eager heavy import: {name}")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    router = tracking.create_tracking_router(tmp_path)
    assert not router.tracking_service.jobs.exists()
    assert router.tracking_service.worker is None
    assert router.tracking_service.model is None
    router.tracking_service.close()


def test_preview_preserves_original_and_exposes_only_scoped_artifact_urls(api):
    client, service, _ = api
    item = upload(client)
    identifier = item["tracking_id"]
    assert tracking._ID.fullmatch(identifier)
    assert item["preview_url"] == f"/experimental-tracking/{identifier}/preview"
    assert item["status"] == "ready" and item["model_available"] is True
    assert item["frame_index"] == 0 and item["processed_frames"] == 0
    assert item["result_url"] is None
    assert (service.jobs / identifier / "source.mov").read_bytes() == b"the-unchanged-original-clip"
    image = client.get(item["preview_url"])
    assert image.content == b"preview-image"
    assert image.headers["content-type"] == "image/jpeg"
    assert "no-store" in image.headers["cache-control"]
    assert client.get(f"/experimental-tracking/{identifier}/result").status_code == 409


@pytest.mark.parametrize("body", [
    b"[]", b"null", b"true", b"{}", b'{"seed":null}',
    b'{"seed":{"x":true,"y":0.5}}', b'{"seed":{"x":1,"y":0.5}}',
    b'{"seed":{"x":-0.1,"y":0.5}}', b'{"seed":{"x":NaN,"y":0.5}}',
    b'{"seed":{"x":Infinity,"y":0.5}}', b'{"seed":{"x":0.5,"y":0.5,"z":0}}',
    b'{"seed":{"x":0.5,"y":0.5},"labels":[]}',
    b'{"seed":{"x":0.5,"x":0.6,"y":0.5}}',
    b'{"seed":{"x":1e999,"y":0.5}}',
])
def test_invalid_seed_is_rejected_before_worker_or_state_change(api, body):
    client, service, _ = api
    item = upload(client)
    response = client.post(f"/experimental-tracking/{item['tracking_id']}/start",
                           content=body, headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert service.worker is None
    assert service.status(item["tracking_id"])["status"] == "ready"


def test_seed_body_size_and_content_type_are_bounded(api):
    client, service, _ = api
    item = upload(client)
    route = f"/experimental-tracking/{item['tracking_id']}/start"
    assert client.post(route, content=b"x" * 4097,
                       headers={"content-type": "application/json"}).status_code == 413
    assert client.post(route, content=b"{}", headers={"content-type": "text/plain"}).status_code == 400
    assert service.worker is None


def test_missing_model_is_helpful_and_never_downloaded(api):
    client, service, root = api
    (root / "data/models/scan-experimental-ball.pt").unlink()
    item = upload(client)
    assert item["model_available"] is False
    assert "no model will be downloaded" in item["error"]
    response = client.post(f"/experimental-tracking/{item['tracking_id']}/start",
                           json={"seed": {"x": 0.25, "y": 0.5}})
    assert response.status_code == 503
    assert service.worker is None
    assert service.status(item["tracking_id"])["status"] == "ready"


def test_worker_uses_guarded_runner_and_separates_candidates_predictions_and_loss(api, monkeypatch):
    from experiments.ball_tracking import track_unlabelled_video as runner
    client, service, _ = api
    calls = []
    monkeypatch.setattr(service, "_load_model", lambda _: object())
    def run(case, model, weights, output, threads, **kwargs):
        calls.append((case, kwargs, threads))
        kwargs["progress_callback"]({"processed_frames": 2, "total_frames": 3, "frame": 1})
        return fake_report(case)
    monkeypatch.setattr(runner, "run_video", run)
    item = upload(client)
    identifier = item["tracking_id"]
    response = client.post(f"/experimental-tracking/{identifier}/start", json={"seed": {"x": 0.25, "y": 0.5}})
    assert response.status_code == 200
    terminal = await_terminal(service, identifier)
    assert terminal["status"] == "completed"
    result = client.get(terminal["result_url"]).json()
    assert (result["width"], result["height"]) == (96, 64)
    assert result["seed"] == {"x": 0.25, "y": 0.5}
    assert [point["state"] for point in result["points"]] == ["observed_candidate", "predicted", "lost"]
    assert result["points"][0]["center"] == {"x": 0.25, "y": 0.5}
    assert result["points"][1]["center"] is None
    assert result["points"][1]["prediction"] is not None
    assert result["points"][2]["prediction"] is None
    assert all(point["identity_verified"] is False for point in result["points"])
    assert not {"outcome", "coaching", "goal", "speed"}.intersection(result)
    case, options, threads = calls[0]
    assert case["seed"] == [24, 32] and case["start_frame"] == item["frame_index"]
    assert threads == 8 and options["association"] == "motion-recovery"
    assert options["save_frames"] is False
    assert options["processing_short_edge"] is None
    assert options["native_detail_fallback"] is False
    assert options["maximum_frames"] == 900
    assert "below 720" in " ".join(result["limitations"])


def test_high_resolution_uses_720_working_frame_and_native_fallback(api, monkeypatch):
    from experiments.ball_tracking import track_unlabelled_video as runner
    client, service, _ = api
    def preview(source, destination):
        destination.write_bytes(b"preview")
        return metadata(width=2160, height=3840)
    monkeypatch.setattr(tracking, "_extract_preview", preview)
    monkeypatch.setattr(service, "_load_model", lambda _: object())
    def run(case, model, weights, output, threads, **kwargs):
        assert kwargs["processing_short_edge"] == 720
        assert kwargs["native_detail_fallback"] is True
        assert case["seed"] == [1080, 1920]
        report = fake_report(case, 720, 1280, True)
        report["frame_transform"] = {"source_width": 2160, "source_height": 3840}
        return report
    monkeypatch.setattr(runner, "run_video", run)
    item = upload(client)
    service.start(item["tracking_id"], {"x": 0.5, "y": 0.5})
    assert await_terminal(service, item["tracking_id"])["status"] == "completed"
    assert service.result(item["tracking_id"])["width"] == 2160


def test_one_active_job_idempotent_start_conflicts_and_cancellation(api, monkeypatch):
    client, service, _ = api
    entered, release = threading.Event(), threading.Event()
    def load(_):
        entered.set()
        assert release.wait(timeout=3)
        return object()
    monkeypatch.setattr(service, "_load_model", load)
    first, second = upload(client), upload(client)
    try:
        service.start(first["tracking_id"], {"x": 0.25, "y": 0.5})
        assert entered.wait(timeout=2)
        assert service.start(first["tracking_id"], {"x": 0.25, "y": 0.5})["status"] == "running"
        with pytest.raises(HTTPException) as changed:
            service.start(first["tracking_id"], {"x": 0.5, "y": 0.5})
        assert changed.value.status_code == 409
        with pytest.raises(HTTPException) as busy:
            service.start(second["tracking_id"], {"x": 0.5, "y": 0.5})
        assert busy.value.status_code == 429
        assert service.status(second["tracking_id"])["status"] == "ready"
        assert client.post(f"/experimental-tracking/{first['tracking_id']}/cancel").status_code == 200
    finally:
        release.set()
    assert await_terminal(service, first["tracking_id"])["status"] == "cancelled"


def test_restart_marks_abandoned_running_job_interrupted_without_loading_model(api):
    client, service, root = api
    item = upload(client)
    path = service.jobs / item["tracking_id"] / "state.json"
    state = json.loads(path.read_text())
    state.update(status="running", owner="old-process")
    tracking._atomic_json(path, state)
    restarted = tracking.ExperimentalTrackingService(root)
    assert restarted.status(item["tracking_id"])["status"] == "interrupted"
    assert restarted.model is None and restarted.worker is None
    restarted.close()


def test_another_live_service_is_not_mistaken_for_a_restarted_job(api, monkeypatch):
    client, service, root = api
    entered, release = threading.Event(), threading.Event()
    def load(_):
        entered.set()
        assert release.wait(timeout=3)
        return object()
    monkeypatch.setattr(service, "_load_model", load)
    item = upload(client)
    other = tracking.ExperimentalTrackingService(root)
    try:
        service.start(item["tracking_id"], {"x": 0.2, "y": 0.2})
        assert entered.wait(timeout=2)
        assert other.status(item["tracking_id"])["status"] == "running"
        other.cancel(item["tracking_id"])
    finally:
        release.set()
        other.close()
    assert await_terminal(service, item["tracking_id"])["status"] == "cancelled"


def test_upload_limits_and_empty_file_are_rejected(api, monkeypatch):
    client, service, _ = api
    monkeypatch.setattr(tracking, "MAX_UPLOAD_BYTES", 4)
    assert client.post("/experimental-tracking/preview", files={"video": ("clip.mov", b"12345")}).status_code == 413
    assert client.post("/experimental-tracking/preview", files={"video": ("clip.mov", b"")}).status_code == 400
    assert client.post("/experimental-tracking/preview", files={"video": ("clip.exe", b"123")}).status_code == 400
    assert not list(service.jobs.glob("*/source.*"))


def test_ids_cannot_resolve_arbitrary_local_files(api):
    client, _, _ = api
    for identifier in ("not-a-job", "0" * 32, "A" * 32):
        assert client.get(f"/experimental-tracking/{identifier}").status_code == 404
        assert client.get(f"/experimental-tracking/{identifier}/preview").status_code == 404


def test_nonfinite_or_out_of_bounds_tracker_points_never_become_observations():
    state = {**metadata(), "seed": {"x": 0.2, "y": 0.2}}
    report = fake_report({"start_frame": 0})
    report["path"][0]["center"] = [float("nan"), 2]
    report["path"][0]["predicted"] = [-1, 2]
    result = tracking._compact_result("a" * 32, state, report)
    assert result["points"][0]["center"] is None
    assert result["points"][0]["state"] == "lost"
    json.dumps(result, allow_nan=False)


class FakeCapture:
    def __init__(self, frames, fps=30, count=None):
        self.frames, self.fps = frames, fps
        self.count = len(frames) if count is None else count
        self.position, self.released = 0, False
    def isOpened(self): return True
    def get(self, key): return self.fps if key == cv2.CAP_PROP_FPS else self.count
    def read(self):
        if self.position >= len(self.frames): return False, None
        image = self.frames[self.position]
        self.position += 1
        return True, image
    def release(self): self.released = True


def test_preview_decodes_sequentially_and_keeps_exact_first_visible_frame(tmp_path, monkeypatch):
    capture = FakeCapture([np.zeros((64, 96, 3), np.uint8), np.full((64, 96, 3), 120, np.uint8)])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    result = tracking._extract_preview(tmp_path / "source.mov", tmp_path / "preview.jpg")
    assert result["frame_index"] == 1
    assert result["time_s"] == 1 / 30
    assert capture.released
    assert cv2.imread(str(tmp_path / "preview.jpg")).mean() == pytest.approx(120, abs=1)


@pytest.mark.parametrize("fps,count", [(0, 3), (float("nan"), 3), (30, 1801), (10, 301)])
def test_preview_rejects_invalid_or_oversized_container_metadata(tmp_path, monkeypatch, fps, count):
    capture = FakeCapture([], fps=fps, count=count)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(ValueError):
        tracking._extract_preview(tmp_path / "source.mov", tmp_path / "preview.jpg")
    assert capture.released


def test_real_runner_path_decodes_clip_without_writing_frame_images(tmp_path, monkeypatch):
    from experiments.ball_tracking import track_unlabelled_video as runner
    from experiments.ball_tracking import temporal_identity
    video = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (96, 64))
    assert writer.isOpened()
    for value in (0, 100, 120):
        writer.write(np.full((64, 96, 3), value, np.uint8))
    writer.release()
    weights = tmp_path / "data/models/scan-experimental-ball.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"test-weights")
    service = tracking.ExperimentalTrackingService(tmp_path)
    monkeypatch.delenv("SCAN_EXPERIMENTAL_BALL_WEIGHTS", raising=False)
    monkeypatch.setattr(service, "_load_model", lambda _: object())
    monkeypatch.setattr(temporal_identity, "estimate_camera_translation",
                        lambda *_: ((0, 0), {"status": "test_alignment"}))
    monkeypatch.setattr(runner, "detect_slices", lambda *_: ([{"center": [24, 32], "confidence": 0.9}], 1))
    try:
        item = service.preview(UploadFile(filename="tiny.avi", file=io.BytesIO(video.read_bytes())))
        assert item["frame_index"] == 1
        service.start(item["tracking_id"], {"x": 0.25, "y": 0.5})
        terminal = await_terminal(service, item["tracking_id"])
        assert terminal["status"] == "completed", terminal
        result = service.result(item["tracking_id"])
        assert [point["frame"] for point in result["points"]] == [1, 2]
        assert all(point["state"] == "observed_candidate" for point in result["points"])
        directory = service.jobs / item["tracking_id"]
        assert not (directory / "tracking/frames").exists()
        report = json.loads((directory / "tracking/report.json").read_text())
        assert report["frames_saved"] is False and report["frame_files"] == []
        assert report["source_seed"] == [24, 32]
    finally:
        service.close()


@pytest.mark.parametrize("limit,expected", [
    ({"maximum_frames": 1}, ValueError),
    ({"maximum_source_pixels": 1}, ValueError),
    ({"maximum_source_dimension": 1}, ValueError),
    ({"maximum_runtime_s": 1e-9}, TimeoutError),
    ({"cancel_check": lambda: True}, runner_error := RuntimeError),
])
def test_runner_resource_checks_apply_to_actual_decoded_frames(tmp_path, monkeypatch, limit, expected):
    from experiments.ball_tracking import track_unlabelled_video as runner
    from experiments.ball_tracking import temporal_identity
    source, weights = tmp_path / "clip.mov", tmp_path / "weights.pt"
    source.write_bytes(b"clip")
    weights.write_bytes(b"weights")
    capture = FakeCapture([np.full((64, 96, 3), 120, np.uint8)] * 3, count=1)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(temporal_identity, "estimate_camera_translation", lambda *_: ((0, 0), {}))
    monkeypatch.setattr(runner, "detect_slices", lambda *_: ([], 1))
    with pytest.raises(expected):
        runner.run_video({"source": str(source), "seed": [24, 32]}, object(), weights,
                         tmp_path / "run", threads=8, association="motion-recovery",
                         save_frames=False, **limit)
    assert capture.released
    assert not (tmp_path / "run/frames").exists()
