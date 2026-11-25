"""The optional HTTP tracker uses one exact seed and never supplies coaching."""

import builtins
from copy import deepcopy
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
    monkeypatch.setattr(tracking, "_decode_frame_times", lambda _: None)
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


@pytest.mark.parametrize('option',[True,False])
def test_recovery_request_requires_local_assets_before_starting(api,option):
    client,service,_=api
    item=upload(client)
    assert item['recovery_available'] is False
    response=client.post(f"/experimental-tracking/{item['tracking_id']}/start",
        json={'seed':{'x':.5,'y':.5},'recover_misses':True if option else 'true'})
    assert response.status_code==(503 if option else 400)
    assert service.worker is None
    assert service.status(item['tracking_id'])['status']=='ready'


@pytest.mark.parametrize('failure',[None,'missing_model','cancelled','partial'])
def test_optional_recovery_is_isolated_from_coaching_and_preserves_baseline_on_failure(api,monkeypatch,failure):
    from experiments.ball_tracking import track_unlabelled_video as runner
    from experiments.ball_tracking import detector_confirmed_recovery as recovery
    client,service,_=api
    def preview(source,destination):
        destination.write_bytes(b'preview')
        return metadata(width=2160,height=3840)
    monkeypatch.setattr(tracking,'_extract_preview',preview)
    monkeypatch.setattr(service,'_recovery_available',lambda state:True)
    monkeypatch.setattr(service,'_load_model',lambda _:object())
    def load_recovery():
        if failure=='missing_model':
            raise FileNotFoundError('private internal model path')
        return object()
    monkeypatch.setattr(service,'_load_recovery',load_recovery)
    calls=[]
    def run(case,model,weights,output,threads,**options):
        assert options['association']=='drift-recovery'
        assert options['merge_nested_native'] is options['trajectory_native_support'] is True
        result=fake_report(case,720,1280,True)
        result['frame_transform']={'source_width':2160,'source_height':3840}
        calls.append(deepcopy(result))
        return result
    monkeypatch.setattr(runner,'run_video',run)
    def recover(baseline,source,model,weights,provider,destination,**options):
        assert baseline==calls[0]
        assert options['settings'].confirm_measured_history and options['settings'].motion_guided_crop
        assert options['settings'].maximum_runtime_s<=240
        options['progress_callback']({'processed_windows':1,'total_windows':2})
        if failure=='cancelled':
            service.cancel_event.set()
            raise runner.TrackingCancelledError('cancelled')
        result=deepcopy(baseline)
        result['path'][1].update(detected=True,center=[25,32],reason='detector_confirmed_learned_recovery_provisional')
        result['learned_recovery']={'new_observation_frames':[1],
            'errors':[{'type':'TimeoutError'}] if failure=='partial' else []}
        return result
    monkeypatch.setattr(recovery,'run_recovery',recover)
    item=upload(client)
    assert item['recovery_available'] is True
    response=client.post(f"/experimental-tracking/{item['tracking_id']}/start",
        json={'seed':{'x':.5,'y':.5},'recover_misses':True})
    assert response.status_code==200
    terminal=await_terminal(service,item['tracking_id'])
    if failure=='cancelled':
        assert terminal['status']=='cancelled'
        assert not (service.jobs/item['tracking_id']/'result.json').exists()
        return
    assert terminal['status']=='completed'
    result=service.result(item['tracking_id'])
    assert result['recovery']['requested'] is True
    assert result['processing']['total_s']>=0
    assert result['processing']['excludes_upload'] is True
    assert result['recovery']['status']==('unavailable' if failure=='missing_model' else 'partial' if failure=='partial' else 'completed')
    assert result['points'][1]['recovered']==(failure!='missing_model')
    assert result['recovery']['added_observations']==(0 if failure=='missing_model' else 1)
    assert not {'outcome','coaching','goal','speed'}.intersection(result)
    assert 'private internal' not in json.dumps(result)
    assert all(p['identity_verified'] is False for p in result['points'])
    with pytest.raises(HTTPException) as changed:
        service.start(item['tracking_id'],{'x':.5,'y':.5},recover_misses=False)
    assert changed.value.status_code==409


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
    def __init__(self, frames, fps=30, count=None, times=None):
        self.frames, self.fps = frames, fps
        self.times = times
        self.count = len(frames) if count is None else count
        self.position, self.released = 0, False
    def isOpened(self): return True
    def get(self, key):
        if key == cv2.CAP_PROP_POS_MSEC:
            return (self.times[self.position - 1] if self.times is not None else (self.position - 1) / self.fps) * 1000
        return self.fps if key == cv2.CAP_PROP_FPS else self.count
    def read(self):
        if self.position >= len(self.frames): return False, None
        image = self.frames[self.position]
        self.position += 1
        return True, image
    def grab(self): return self.read()[0]
    def release(self): self.released = True


def test_playback_uses_decoded_timestamps_when_average_fps_is_wrong(tmp_path, monkeypatch):
    image = np.full((64, 96, 3), 120, np.uint8)
    capture = FakeCapture([image] * 150, fps=22.994379151762903,
                          times=[frame / 30 for frame in range(150)])
    monkeypatch.setattr(cv2, 'VideoCapture', lambda _: capture)
    times = tracking._decode_frame_times(tmp_path / 'source.mov')
    assert times[79] == pytest.approx(2.6333333333333333)
    assert times[149] == pytest.approx(4.966666666666667)
    report = {**fake_report({'start_frame': 0}), 'fps': capture.fps}
    result = tracking._compact_result('a' * 32,
                                     {**metadata(), 'fps': capture.fps, 'seed': {'x': .2, 'y': .4}}, report)
    original = deepcopy(result)
    fixed = tracking._with_playback_timing(result, times[:3])
    assert fixed['timing_mode'] == 'presentation_timestamps'
    assert fixed['points'][2]['time_s'] == pytest.approx(2 / 30)
    assert fixed['points'][2]['time_s'] < result['points'][2]['time_s']
    assert fixed['points'][1]['end_time_s'] == fixed['points'][2]['time_s']
    for before, after in zip(result['points'], fixed['points']):
        assert {k: v for k, v in after.items() if k not in {'time_s', 'end_time_s'}} == {
            k: v for k, v in before.items() if k != 'time_s'}
    assert result == original and capture.released


@pytest.mark.parametrize('times', [[0, 0, 0], [0, .1, .05], [0, float('nan'), .1],
                                  [0, float('inf'), .1], [1, 1.1, 1.2], [0, .1, 31]])
def test_unsupported_presentation_timestamps_use_explicit_fallback(tmp_path, monkeypatch, times):
    capture = FakeCapture([np.full((64, 96, 3), 120, np.uint8)] * 3, times=times)
    monkeypatch.setattr(cv2, 'VideoCapture', lambda _: capture)
    assert tracking._decode_frame_times(tmp_path / 'source.mov') is None
    assert capture.released


def test_incomplete_timeline_cannot_shift_or_fill_tracking_points():
    result = tracking._compact_result('a' * 32, {**metadata(), 'seed': {'x': .2, 'y': .4}},
                                     fake_report({'start_frame': 0}))
    fixed = tracking._with_playback_timing(result, [0, 1 / 30])
    assert fixed['timing_mode'] == 'reported_fps_approximate'
    assert fixed['points'] == result['points']


def test_completed_job_retimes_once_without_inference_or_rewriting_original(api, monkeypatch):
    from experiments.ball_tracking import track_unlabelled_video as runner
    client, service, _ = api
    monkeypatch.setattr(service, '_load_model', lambda _: object())
    monkeypatch.setattr(runner, 'run_video', lambda case, *args, **kwargs: fake_report(case))
    item = upload(client)
    identifier = item['tracking_id']
    service.start(identifier, {'x': .25, 'y': .5})
    assert await_terminal(service, identifier)['status'] == 'completed'
    directory = service.jobs / identifier
    original = (directory / 'result.json').read_bytes()
    calls = []
    def decode(source):
        calls.append(source)
        return [0, .04, .09]
    monkeypatch.setattr(tracking, '_decode_frame_times', decode)
    first = client.get(f'/experimental-tracking/{identifier}/result').json()
    second = client.get(f'/experimental-tracking/{identifier}/result').json()
    assert first == second and len(calls) == 1
    assert first['points'][2]['time_s'] == .09
    assert first['points'][2]['state'] == 'lost'
    assert first['points'][2]['center'] is None
    assert (directory / 'result.json').read_bytes() == original
    assert (directory / 'result.playback-v1.json').is_file()


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
