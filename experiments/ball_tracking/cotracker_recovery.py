"""Local pinned CoTracker3 adapter for short, offline recovery windows.

No downloads or checkpoint execution occur implicitly. Learned estimates and
visibility are search evidence, never detector observations or outcomes.
"""

import json
import math
from pathlib import Path
import sys

from scripts.ball_dataset import file_sha256


MODEL_REVISION = "bf55ea50d4390e1820a267f131cd6587240fb2c5"
SOURCE_REVISION = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
CHECKPOINT_SHA256 = "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"


class UnavailableRecovery:
    """Represent an unavailable optional model without breaking the baseline."""

    def __init__(self, error):
        self.error = {"type": type(error).__name__, "message": str(error)[:240]}

    def configuration(self):
        return {"status": "unavailable", "error": self.error}

    def predict(self, *args, **kwargs):
        raise RuntimeError("Optional learned recovery model is unavailable")


class CoTrackerRecovery:
    """One query point, native crop coordinates, official visibility threshold."""

    def __init__(self, model_root, *, runtime_directory=None, threads=6):
        if type(threads) is not int or not 1 <= threads <= 16:
            raise ValueError("Expected 1–16 CPU threads")
        self.root = Path(model_root).resolve()
        self.checkpoint = self.root / "scaled_offline.pth"
        self.source = self.root / "source"
        self.runtime = Path(runtime_directory).resolve() if runtime_directory else None
        self.threads = threads
        self.model = None
        record = json.loads((self.root / "provenance.json").read_text(encoding="utf-8"))
        if (record.get("model") != "facebook/cotracker3"
                or record.get("revision") != MODEL_REVISION
                or record.get("commit") != SOURCE_REVISION
                or record.get("checkpoint_sha256") != CHECKPOINT_SHA256
                or file_sha256(self.checkpoint) != CHECKPOINT_SHA256):
            raise ValueError("CoTracker checkpoint/provenance does not match the pinned research model")
        for name, expected in record["source_sha256"].items():
            path = (self.source / name).resolve()
            if not path.is_relative_to(self.source) or file_sha256(path) != expected:
                raise ValueError("CoTracker source hash mismatch")
        required = {"cotracker/predictor.py", "cotracker/models/build_cotracker.py",
                    "cotracker/models/core/cotracker/cotracker3_offline.py"}
        recorded = {str(Path(name)).replace("\\", "/") for name in record["source_sha256"]}
        if not required <= recorded:
            raise ValueError("CoTracker provenance omits inference source")
        self.provenance = record

    def configuration(self):
        return {"model": "facebook/cotracker3_scaled_offline", "revision": MODEL_REVISION,
                "source_commit": SOURCE_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
                "license": "cc-by-nc-4.0", "cpu_threads": self.threads,
                "query": "past_baseline_detector_observation_not_reference_label",
                "uses_future_video_frames": True, "visibility_threshold": "official_predictor_0.9_unchanged",
                "prediction_is_observation": False}

    def _load(self):
        if self.model is not None:
            return
        if self.runtime is not None:
            sys.path.insert(0, str(self.runtime))
        sys.path.insert(0, str(self.source))
        # A pre-imported different installation must not silently change the model.
        import importlib
        module = importlib.import_module("cotracker.predictor")
        if not Path(module.__file__).resolve().is_relative_to(self.source):
            raise ValueError("Another CoTracker installation is already imported")
        import torch
        torch.set_num_threads(self.threads)
        model = module.CoTrackerPredictor(checkpoint=None, window_len=60)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        model.model.load_state_dict(state.get("model", state))
        model.eval()
        self.model = model

    def predict(self, crops, *, start_frame, seed_frame, seed_local):
        if (type(start_frame) is not int or type(seed_frame) is not int
                or start_frame < 0 or not start_frame <= seed_frame < start_frame + len(crops)
                or not 3 <= len(crops) <= 32
                or len(seed_local) != 2 or not all(math.isfinite(v) for v in seed_local)):
            raise ValueError("Expected a bounded window and a finite in-window query")
        import numpy as np
        import cv2
        shape = crops[0].shape
        if (len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 384
                or shape[0] > 576 or shape[1] > 768
                or any(c.shape != shape or c.dtype != np.uint8 for c in crops)
                or not 0 <= seed_local[0] < shape[1] or not 0 <= seed_local[1] < shape[0]):
            raise ValueError("Expected matching native BGR crops and a query inside the crop")
        self._load()
        import torch
        video = torch.from_numpy(np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]))
        video = video.permute(0, 3, 1, 2)[None].float()
        query = torch.tensor([[[float(seed_frame - start_frame), *map(float, seed_local)]]])
        with torch.inference_mode():
            tracks, visibility = self.model(video, queries=query)
        if tracks.shape != (1, len(crops), 1, 2) or visibility.shape != (1, len(crops), 1):
            raise ValueError("Unexpected CoTracker output shape")
        points, visible = tracks[0, :, 0].cpu().numpy(), visibility[0, :, 0].cpu().numpy()
        if not np.isfinite(points).all() or visible.dtype != np.bool_:
            raise ValueError("Invalid CoTracker point/visibility output")
        return [{"frame": start_frame + i, "center_local": [float(v) for v in point],
                 "visible": bool(flag), "independent_detection": False}
                for i, (point, flag) in enumerate(zip(points, visible, strict=True))]
