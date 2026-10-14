from pathlib import Path
import cv2
import numpy as np
import pytest
from src.backend.goal_preview import extract_goal_preview


class Capture:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0
        self.closed = False

    def isOpened(self): return True
    def get(self, field): return 30 if field == cv2.CAP_PROP_FPS else len(self.frames)
    def set(self, field, index): self.index = int(index)
    def read(self): return True, self.frames[self.index]
    def release(self): self.closed = True


def test_blank_opening_frame_is_skipped(monkeypatch):
    capture = Capture([np.zeros((80, 160, 3), np.uint8), np.full((80, 160, 3), 120, np.uint8)])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    image, index = extract_goal_preview(Path("clip.mov"))
    assert index == 1
    assert image.mean() == 120
    assert capture.closed


def test_all_black_clip_returns_useful_error(monkeypatch):
    capture = Capture([np.zeros((80, 160, 3), np.uint8)] * 3)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(ValueError, match="No visible reference frame"):
        extract_goal_preview(Path("clip.mov"))
    assert capture.closed


def test_visible_first_frame_is_kept(monkeypatch):
    capture = Capture([np.full((80, 160, 3), 160, np.uint8)] * 2)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    image, index = extract_goal_preview(Path("clip.mov"))
    assert index == 0
    assert image.shape == (80, 160, 3)
