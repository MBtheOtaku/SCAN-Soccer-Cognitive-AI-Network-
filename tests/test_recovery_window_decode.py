"""Exact frame/crop reuse, memory limits and cancellation during recovery."""
import cv2
import numpy as np
import pytest

from experiments.ball_tracking import detector_confirmed_recovery as recovery


class Capture:
    def __init__(self, count=7):
        self.position=0
        self.count=count
        self.reads=self.grabs=0
        self.released=False
    def grab(self):
        self.position+=1
        self.grabs+=1
        return self.position<=self.count
    def read(self):
        self.reads+=1
        self.position+=1
        return (True,np.full((12,16,3),self.position-1,np.uint8)) if self.position<=self.count else (False,None)
    def release(self):
        self.released=True


def inputs(monkeypatch, count=7):
    capture=Capture(count)
    monkeypatch.setattr(cv2,'VideoCapture',lambda *args:capture)
    monkeypatch.setattr(recovery,'prepare_frame',lambda image,size:(image,{}))
    windows=[{'start_frame':2,'end_frame':4,'crop_xyxy':[0,0,8,6]},
             {'start_frame':3,'end_frame':5,'crop_xyxy':[4,2,12,8]}]
    transform={'source_width':16,'source_height':12,'width':16,'height':12}
    return capture,windows,transform


def test_overlapping_windows_retrieve_each_frame_once_without_seeking(monkeypatch):
    capture,windows,transform=inputs(monkeypatch)
    records,meta=recovery.decode_selected_windows('clip.mov',windows,transform)
    assert capture.grabs==2 and capture.reads==4 and capture.released
    assert meta['retrieved_frames']==4 and meta['skipped_frames']==2
    assert meta['estimated_bytes']==meta['retained_bytes']==3168
    first=records[(3,(0,0,8,6))]
    second=records[(3,(4,2,12,8))]
    assert first['working'] is second['working']
    for (frame,crop),record in records.items():
        assert (record['working']==frame).all() and (record['crop']==frame).all()
        assert not record['working'].flags.writeable and not record['crop'].flags.writeable


def test_memory_overflow_falls_back_before_decoding_or_allocating(monkeypatch):
    capture,windows,transform=inputs(monkeypatch)
    records,meta=recovery.decode_selected_windows('clip.mov',windows,transform,maximum_bytes=3000)
    assert records is None and meta['mode']=='per_window_memory_fallback'
    assert capture.grabs==capture.reads==0


def test_early_end_and_cancel_release_the_decoder(monkeypatch):
    capture,windows,transform=inputs(monkeypatch,count=3)
    with pytest.raises(ValueError):
        recovery.decode_selected_windows('clip.mov',windows,transform)
    assert capture.released
    capture,windows,transform=inputs(monkeypatch)
    def cancelled():
        if capture.position>=1:
            raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        recovery.decode_selected_windows('clip.mov',windows,transform,checkpoint=cancelled)
    assert capture.released and capture.grabs==1


def test_real_decoded_windows_match_legacy_reader_pixel_for_pixel(tmp_path):
    source=tmp_path/'pixels.avi'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),30,(720,720))
    assert writer.isOpened()
    for frame in range(6):
        image=np.full((720,720,3),20+frame*20,np.uint8)
        cv2.circle(image,(100+frame*12,300),12,(255,40,20),-1)
        writer.write(image)
    writer.release()
    windows=[{'start_frame':2,'end_frame':4,'crop_xyxy':[0,0,600,576]},
             {'start_frame':3,'end_frame':5,'crop_xyxy':[120,100,720,676]}]
    transform={'source_width':720,'source_height':720,'width':720,'height':720}
    records,meta=recovery.decode_selected_windows(source,windows,transform)
    for window in windows:
        legacy=recovery.load_window(source,window,window['crop_xyxy'],transform)
        for original in legacy:
            cached=records[(original['frame'],tuple(window['crop_xyxy']))]
            np.testing.assert_array_equal(original['working'],cached['working'])
            np.testing.assert_array_equal(original['crop'],cached['crop'])
