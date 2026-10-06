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

    def existing_analyze(video, model, annotated, csv, report, roi):
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
    assert "https://192.168.1.50:3000" in origins
    assert "https://scan.example" in origins
    assert "*" not in origins
