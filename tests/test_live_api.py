"""Upload-boundary checks; no perception models or external services required."""

import asyncio
from dataclasses import dataclass
import importlib.util
import io
import json
from pathlib import Path
import sys
import types

import pytest
from fastapi import HTTPException, UploadFile


@pytest.fixture
def api(monkeypatch, tmp_path):
    calls = []

    @dataclass
    class Summary:
        prototype_feedback: str = "Repeat the movement."

    def existing_analyze(video, model, annotated, csv, report, roi, **kwargs):
        module.received_goal_options = kwargs
        calls.append((video.suffix, video.read_bytes(), roi))
        report.write_text(json.dumps({"coaching_report": {"spoken_cue": None}}))
        return Summary()

    # Isolate only the expensive import. The actual API upload handler is exercised.
    perception = types.ModuleType("src.backend.pose_mvp.scan_pose_mvp_v2")
    perception.analyze = existing_analyze
    monkeypatch.setitem(sys.modules, perception.__name__, perception)
    monkeypatch.setenv("SCAN_CORS_ORIGINS", " https://192.168.1.50:3000,https://scan.example ")
    source = Path(__file__).resolve().parents[1] / "src/backend/api/main.py"
    spec = importlib.util.spec_from_file_location("live_api_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.INPUT_DIR = tmp_path
    module.OUTPUT_DIR = tmp_path
    module.MODEL_PATH = tmp_path / "model.task"
    module.MODEL_PATH.touch()
    return module, calls


@pytest.mark.parametrize("suffix", [".mp4", ".mov", ".webm"])
def test_clips_reuse_existing_analyser(api, suffix):
    module, calls = api
    upload = UploadFile(filename=f"rep{suffix}", file=io.BytesIO(b"clip bytes"))
    result = asyncio.run(module.analyze_video(upload, roi=None))
    assert calls == [(suffix, b"clip bytes", None)]
    assert result["coaching_report"]["spoken_cue"] is None
    assert result["artifacts"]["annotated_video"].startswith("/outputs/")
    assert upload.file.closed


def test_invalid_extension_does_not_run_analysis(api):
    module, calls = api
    upload = UploadFile(filename="rep.txt", file=io.BytesIO(b"invalid"))
    with pytest.raises(HTTPException) as error:
        asyncio.run(module.analyze_video(upload, roi=None))
    assert error.value.status_code == 400
    assert not calls


def test_cors_uses_explicit_origins(api):
    module, _ = api
    origins = module.app.user_middleware[0].kwargs["allow_origins"]
    assert "http://localhost:3000" in origins
    assert "http://127.0.0.1:3000" in origins
    assert "https://localhost:3000" in origins
    assert "https://127.0.0.1:3000" in origins
    assert "https://192.168.1.50:3000" in origins
    assert "https://scan.example" in origins
    assert "*" not in origins


def test_pressure_failure_preserves_completed_analysis(api, monkeypatch):
    module, calls = api
    from src.backend import opponent_pressure

    def fail(*args):
        raise RuntimeError("optional detector unavailable")

    monkeypatch.setattr(opponent_pressure, "analyze_pressure", fail)
    upload = UploadFile(filename="rep.mp4", file=io.BytesIO(b"clip bytes"))
    result = asyncio.run(module.analyze_video(upload, roi=None, athlete_start_side="right"))
    assert calls
    assert result["coaching_report"]["spoken_cue"] is None
    assert result["pressure_report"]["status"] == "unavailable"
    assert not result["pressure_report"]["policy"]["should_speak"]


def test_invalid_pressure_setup_rejected_before_analysis(api):
    module, calls = api
    upload = UploadFile(filename="rep.mp4", file=io.BytesIO(b"clip bytes"))
    with pytest.raises(HTTPException) as error:
        asyncio.run(module.analyze_video(upload, roi=None, athlete_start_side="middle"))
    assert error.value.status_code == 400
    assert not calls


def test_unconfirmed_analysis_does_not_read_shared_goal_setup(api):
    module, _ = api
    upload = UploadFile(filename="rep.mp4", file=io.BytesIO(b"clip bytes"))
    asyncio.run(module.analyze_video(upload, roi=None))
    assert module.received_goal_options["use_legacy_goal_calibration"] is False
    assert module.received_goal_options["goal_calibration_override"] is None


def test_confirmed_setup_requires_reference_photo(api):
    module, calls = api
    setup = json.dumps({"confirmed": True, "corners": [[.2, .2], [.8, .2], [.8, .6], [.2, .6]]})
    with pytest.raises(HTTPException) as error:
        asyncio.run(module.analyze_video(UploadFile(filename="rep.mp4", file=io.BytesIO(b"clip")), roi=None, goal_setup=setup))
    assert error.value.status_code == 400
    assert not calls


def test_confirmed_setups_are_isolated_between_requests(api):
    import cv2
    import numpy as np
    module, _ = api
    _, encoded = cv2.imencode(".jpg", np.zeros((100, 200, 3), np.uint8))
    first = None
    for left in [.1, .3]:
        setup = json.dumps({"confirmed": True, "corners": [[left, .2], [.8, .2], [.8, .6], [left, .6]]})
        reference = UploadFile(filename="ref.jpg", file=io.BytesIO(encoded.tobytes()))
        upload = UploadFile(filename="rep.mp4", file=io.BytesIO(b"clip"))
        asyncio.run(module.analyze_video(upload, roi=None, goal_setup=setup, goal_reference=reference))
        options = module.received_goal_options
        assert options["goal_calibration_override"].top_left[0] == pytest.approx(left)
        assert options["goal_reference_path"].is_file()
        assert reference.file.closed
        if first is None:
            first = options
        else:
            assert options["goal_reference_path"] != first["goal_reference_path"]
    assert first["goal_calibration_override"].top_left[0] == pytest.approx(.1)


def test_goal_proposal_endpoint_closes_photo_and_requires_confirmation(api):
    import cv2
    import numpy as np
    module, _ = api
    _, encoded = cv2.imencode(".jpg", np.zeros((100, 200, 3), np.uint8))
    reference = UploadFile(filename="ref.jpg", file=io.BytesIO(encoded.tobytes()))
    result = asyncio.run(module.propose_goal(reference))
    assert result["candidates"] == []
    assert result["requires_confirmation"] is True
    assert reference.file.closed
