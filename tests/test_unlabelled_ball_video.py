"""Fresh video evaluation must not depend on per-frame ball annotations."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.ball_tracking import track_unlabelled_video as runner


@pytest.mark.parametrize("reported_frames", [3, 5])
@pytest.mark.parametrize("association", ["legacy", "temporal", "recovery", "motion-recovery"])
def test_unlabelled_run_decodes_every_frame_preserves_missing_and_reports_early_end(
        tmp_path, monkeypatch, reported_frames, association):
    source = tmp_path / "new-recording.mov"
    source.write_bytes(b"original-video")
    weights = tmp_path / "frozen.pt"
    weights.write_bytes(b"frozen-weights")

    class Capture:
        position = 0

        def isOpened(self):
            return True

        def get(self, key):
            return 30.0 if key == 1 else reported_frames

        def read(self):
            if self.position == 3:
                return False, None
            image = np.full((400, 400, 3), self.position, dtype=np.uint8)
            self.position += 1
            return True, image

        def release(self):
            pass

    def write_frame(path, image, settings):
        Path(path).write_bytes(image[:1, :1].tobytes())
        return True

    fake_cv2 = SimpleNamespace(VideoCapture=lambda _: Capture(), CAP_PROP_FPS=1,
                               CAP_PROP_FRAME_COUNT=2, IMWRITE_JPEG_QUALITY=3,
                               imwrite=write_frame)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    from experiments.ball_tracking import temporal_identity
    monkeypatch.setattr(temporal_identity, "estimate_camera_translation",
                        lambda before, after: ((0.0, 0.0), {"status": "test_alignment"}))
    calls = []

    def detect(model, image, predicted, size, offset, confidence):
        number = int(image[0, 0, 0])
        calls.append(number)
        candidates = ([{"center": [50 + number, 50], "confidence": 0.9}]
                      if number < 2 else [])
        return candidates, 9

    monkeypatch.setattr(runner, "detect_slices", detect)
    result = runner.run_video({"source": str(source), "seed": [50, 50]}, object(),
                              weights, tmp_path / "result", threads=4, association=association)
    assert calls == [0, 1, 2]
    assert result["processed_frames"] == 3
    assert result["selected_detection_frames"] == 2
    assert result["frames_without_selected_detection"] == 1
    assert result["path"][-1]["center"] is None
    assert result["path"][-1]["detected"] is False
    assert result["reached_reported_end"] is (reported_frames == 3)
    assert result["split"] == ("unseen_session_unlabelled" if association == "legacy"
                               else "inspected_session_unlabelled")
    assert result["association_method"] == association
    assert result["trajectory_native_support"] is False
    assert ("Inspected development" in result["limitation"]) is (association != "legacy")
    assert ("unseen-session run" in result["limitation"]) is (association == "legacy")
    assert len(result["frame_files"]) == 3
    assert "accuracy" not in result
    saved = json.loads((tmp_path / "result/report.json").read_text())
    assert saved["source_sha256"] == runner.file_sha256(source)
    assert saved["weights_sha256"] == runner.file_sha256(weights)
    assert saved["implementation_sha256"]["association"] == runner.file_sha256(
        runner.ROOT / "experiments/ball_tracking" / runner.ASSOCIATION_FILES[association])
    if association in ("recovery", "motion-recovery"):
        assert saved["implementation_sha256"]["base_association"] == runner.file_sha256(
            runner.ROOT / "experiments/ball_tracking/temporal_identity.py")
    if association == "motion-recovery":
        assert saved["implementation_sha256"]["recovery_association"] == runner.file_sha256(
            runner.ROOT / "experiments/ball_tracking/recovery_identity.py")


def test_unlabelled_run_never_overwrites_a_previous_report(tmp_path):
    source, weights = tmp_path / "video.mov", tmp_path / "weights.pt"
    source.write_bytes(b"video")
    weights.write_bytes(b"weights")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "report.json").write_text("original result")
    with pytest.raises(FileExistsError, match="Preserve previous results"):
        runner.run_video({"source": str(source), "seed": [10, 10]}, object(),
                         weights, output, threads=4)
    assert (output / "report.json").read_text() == "original result"


@pytest.mark.parametrize('enabled,native', [(True,False), (1,True), ('yes',True)])
def test_trajectory_support_requires_explicit_boolean_and_native_fallback(enabled,native,tmp_path):
    with pytest.raises(ValueError, match='Trajectory native support'):
        runner.run_video({},object(),tmp_path/'weights',tmp_path/'output',threads=4,
                         trajectory_native_support=enabled,native_detail_fallback=native)
