from dataclasses import dataclass
from pathlib import Path

import pytest

from src.backend.analysis_artifacts import preview_dimensions, detections_for_preview, encode_browser_video
from src.backend.pose_mvp.player_roi import sample_frames_sequential


@dataclass
class Detection:
    center_x: float = 1800
    center_y: float = 900
    bbox_width: float = 30
    bbox_height: float = 24
    center_x_norm: float = .46875
    source: str = 'predicted'


def test_preview_is_smaller_without_upscaling_and_supports_portrait():
    assert preview_dimensions(3840,2160,1280) == (1280,720)
    assert preview_dimensions(2160,3840,1280) == (720,1280)
    assert preview_dimensions(640,480,1280) == (640,480)
    assert preview_dimensions(3840,2160) == (3840,2160)


def test_overlay_scaling_keeps_measurements_and_source_status_unchanged():
    original=Detection()
    rendered=detections_for_preview([original],(3840,2160),(1280,720))[0]
    assert (rendered.center_x,rendered.center_y,rendered.bbox_width,rendered.bbox_height)==(600,300,10,8)
    assert rendered.source=='predicted' and rendered.center_x_norm==original.center_x_norm
    assert original.center_x==1800 and rendered is not original


def test_encoding_is_one_bounded_browser_compatible_pass(monkeypatch):
    calls=[]
    monkeypatch.setattr('src.backend.analysis_artifacts.subprocess.run',lambda *args,**kwargs:calls.append((args,kwargs)))
    encode_browser_video(Path('input.mp4'),Path('output.mp4'))
    args=calls[0][0][0]
    assert len(calls)==1 and calls[0][1]['check']
    assert args[args.index('-preset')+1]=='veryfast'
    assert args[args.index('-threads')+1]=='4'
    assert args[-1]=='output.mp4' and '+faststart' in args and 'yuv420p' in args


def test_player_samples_are_exact_and_monotonic_without_seeking():
    class Capture:
        position=0
        def read(self):
            p=self.position;self.position+=1
            return p<10,p if p<10 else None
    cap=Capture()
    assert list(sample_frames_sequential(cap,[0,3,8,12]))==[(0,0),(3,3),(8,8),(12,None)]
    assert cap.position==11


@pytest.mark.parametrize('indices',[[2,1],[1,1],[-1],[True]])
def test_invalid_sample_indices_rejected_before_read(indices):
    with pytest.raises(ValueError):
        list(sample_frames_sequential(None,indices))
