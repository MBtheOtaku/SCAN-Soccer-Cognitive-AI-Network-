"""Local model boundaries, provenance rejection, and frame/colour conversion."""
import json

import numpy as np
import pytest

from experiments.ball_tracking.cotracker_recovery import (
    CHECKPOINT_SHA256, MODEL_REVISION, SOURCE_REVISION, CoTrackerRecovery)


def provider():
    # Inference-boundary tests use a tiny fake network, not downloaded weights.
    result=object.__new__(CoTrackerRecovery)
    result.model=None
    return result


@pytest.mark.parametrize('change',[{'start_frame':-1},{'seed_frame':13},
    {'seed_local':[float('nan'),100]},{'seed_local':[-1,100]}])
def test_invalid_query_is_rejected_before_loading_a_network(monkeypatch,change):
    p=provider()
    monkeypatch.setattr(p,'_load',lambda:pytest.fail('Invalid input cannot load the optional network'))
    args={'start_frame':10,'seed_frame':12,'seed_local':[100,100],**change}
    with pytest.raises(ValueError):
        p.predict([np.zeros((384,384,3),np.uint8)]*3,**args)


def test_native_bgr_frames_and_absolute_seed_frame_are_converted_correctly(monkeypatch):
    import torch
    p=provider()
    def network(video,*,queries):
        assert tuple(video.shape)==(1,3,3,384,384)
        assert video[0,0,:,0,0].tolist()==[30,20,10]
        assert queries.tolist()==[[[2,100,101]]]
        return (torch.tensor([[[[100.,101.]],[[102.,103.]],[[104.,105.]]]]),
                torch.tensor([[[True],[False],[True]]]))
    p.model=network
    monkeypatch.setattr(p,'_load',lambda:None)
    crop=np.empty((384,384,3),np.uint8)
    crop[:]=[10,20,30]
    estimates=p.predict([crop]*3,start_frame=10,seed_frame=12,seed_local=[100,101])
    assert [e['frame'] for e in estimates]==[10,11,12]
    assert [e['visible'] for e in estimates]==[True,False,True]
    assert estimates[2]['center_local']==[104,105]
    assert all(e['independent_detection'] is False for e in estimates)


@pytest.mark.parametrize('visibility_kind',['float','wrong_shape'])
def test_malformed_network_visibility_is_not_promoted_to_evidence(monkeypatch,visibility_kind):
    import torch
    p=provider()
    vis=torch.ones((1,3,1)) if visibility_kind=='float' else torch.ones((1,3),dtype=torch.bool)
    p.model=lambda *args,**kwargs:(torch.zeros((1,3,1,2)),vis)
    monkeypatch.setattr(p,'_load',lambda:None)
    with pytest.raises(ValueError):
        p.predict([np.zeros((384,384,3),np.uint8)]*3,
                  start_frame=0,seed_frame=0,seed_local=[100,100])


def test_modified_local_source_is_rejected_before_import(tmp_path,monkeypatch):
    import experiments.ball_tracking.cotracker_recovery as module
    record={'model':'facebook/cotracker3','revision':MODEL_REVISION,'commit':SOURCE_REVISION,
            'checkpoint_sha256':CHECKPOINT_SHA256,'source_sha256':{'cotracker/predictor.py':'expected'}}
    (tmp_path/'provenance.json').write_text(json.dumps(record))
    monkeypatch.setattr(module,'file_sha256',lambda p:CHECKPOINT_SHA256 if p.name=='scaled_offline.pth' else 'modified')
    with pytest.raises(ValueError,match='source hash mismatch'):
        CoTrackerRecovery(tmp_path)


def test_provenance_cannot_reference_source_outside_its_directory(tmp_path,monkeypatch):
    import experiments.ball_tracking.cotracker_recovery as module
    record={'model':'facebook/cotracker3','revision':MODEL_REVISION,'commit':SOURCE_REVISION,
            'checkpoint_sha256':CHECKPOINT_SHA256,'source_sha256':{'../../outside.py':'expected'}}
    (tmp_path/'provenance.json').write_text(json.dumps(record))
    monkeypatch.setattr(module,'file_sha256',lambda p:CHECKPOINT_SHA256)
    with pytest.raises(ValueError,match='source hash mismatch'):
        CoTrackerRecovery(tmp_path)
