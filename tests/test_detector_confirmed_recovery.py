"""Learned coordinates can request evidence, never manufacture observations."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.ball_tracking.detector_confirmed_recovery import (
    LearnedRecoverySettings, confirm_window, crop_for_seed, plan_windows, run_recovery,
    detect_small_query_box,plan_motion_windows,FrameAlignmentCache)


def baseline(frames=10):
    return {'width':1280,'height':720,
            'frame_transform':{'source_width':3840,'source_height':2160},
            'path':[{'frame':f,'center':[100+f*2,100] if f<3 else None,
                     'detected':f<3,'confidence':.4 if f<3 else None,
                     'identity_verified':False,'reason':'baseline'} for f in range(frames)]}


class Verifier:
    def __init__(self,status='appearance_changed_not_identity_proof'):
        self.status=status
    def align(self,image):
        return np.eye(3),{}
    def inspect(self,image,candidate,matrix,registration,**kwargs):
        return {'status':self.status,'reference_processing_center':candidate['center']}
    def configuration(self):
        return {'test_verifier':True}


def fixture(frames=10):
    b=baseline(frames)
    w={**plan_windows(b['path'])[0],'crop_xyxy':[200,100,968,676]}
    working=np.zeros((720,1280,3),np.uint8)
    crop=np.zeros((576,768,3),np.uint8)
    records=[{'frame':f,'working':working,'crop':crop} for f in range(frames)]
    estimates=[{'frame':f,'center_local':[101+f*6,200],'visible':True} for f in range(frames)]
    return b,w,records,estimates


def infer(model,image,region):
    x,y=region['query_center_native']
    x-=1  # Current measurement differs from the learned point.
    return [{'center':[x,y],'confidence':.4,'bbox_native_xyxy':[x-3,y-3,x+3,y+3]}],1


def test_planner_uses_only_gap_starts_and_past_real_observations():
    b=baseline()
    windows=plan_windows(b['path'])
    assert windows==[{'gap_start':3,'seed_frame':2,'start_frame':0,'end_frame':9,
                     'seed_center_working':[104,100]}]
    b['annotations']=[{'frame':3,'box':[999,999,1000,1000]}]
    assert plan_windows(b['path'])==windows


def test_recovery_requires_current_detector_measurement_and_uses_its_coordinates():
    b,w,r,e=fixture()
    original=deepcopy(b)
    additions,checks=confirm_window(b,w,r,e,Verifier(),None,infer=infer)
    assert b==original
    assert additions[0]['frame']==3
    assert additions[0]['center']==[106,100]
    assert additions[0]['center'][0] != e[3]['center_local'][0]/3+200/3
    assert additions[0]['support_observation_frames']==[0,1,2,3]
    assert additions[0]['bbox_native_xyxy']==[315,297,321,303]
    assert all(p['identity_verified'] is False and p['detected'] for p in additions)
    assert checks[3]['learned_estimate_is_detection'] is False


def test_reusing_registration_preserves_candidate_checks_and_all_observations():
    b,w,r,e=fixture()
    verifier=Verifier()
    original=confirm_window(b,w,r,e,verifier,None,infer=infer)
    cache=FrameAlignmentCache(verifier)
    cached=confirm_window(b,w,r,e,verifier,None,infer=infer,alignment_cache=cache)
    again=confirm_window(b,w,r,e,verifier,None,infer=infer,alignment_cache=cache)
    assert original==cached==again
    assert cache.configuration()['hits']==len(r)
    assert cache.configuration()['misses']==len(r)
    # A cached registration cannot bypass a new ambiguous ball observation.
    additions,checks=confirm_window(b,w,r,e,verifier,None,
        infer=lambda *args:([],1),alignment_cache=cache)
    assert additions==[] and checks[3]['status']=='no_current_detector_confirmation'


def test_alignment_from_another_verifier_cannot_enter_confirmation():
    b,w,r,e=fixture()
    with pytest.raises(ValueError,match='this run'):
        confirm_window(b,w,r,e,Verifier(),None,infer=infer,
            alignment_cache=FrameAlignmentCache(Verifier()))


def test_confident_model_without_detector_box_cannot_create_an_observation():
    b,w,r,e=fixture()
    additions,checks=confirm_window(b,w,r,e,Verifier(),None,infer=lambda *args:([],1))
    assert additions==[]
    assert checks[3]['status']=='no_current_detector_confirmation'


def test_low_model_visibility_cannot_request_or_select_current_detections():
    b,w,r,e=fixture()
    for p in e:p['visible']=False
    additions,checks=confirm_window(b,w,r,e,Verifier(),None,
        infer=lambda *args:pytest.fail('Visibility must block new detector queries'))
    assert additions==[] and checks[3]['status']=='model_visibility_withheld'


def test_optional_uncertain_search_requires_four_consecutive_actual_detections():
    b,w,r,e=fixture()
    for p in e:p['visible']=False
    e[3]['center_local']=[-100,200]  # Break previous real support.
    additions,checks=confirm_window(b,w,r,e,Verifier(),None,infer=infer,
        settings=LearnedRecoverySettings(search_uncertain_points=True))
    assert [p['frame'] for p in additions]==[7,8,9]
    assert checks[6]['status']=='awaiting_detector_sequence_support'
    assert checks[7]['required_consecutive_detector_observations']==4
    assert all(p['model_visible'] is False and p['model_visibility_role']=='search_only' for p in additions)
    assert all(p['identity_verified'] is False for p in additions)


def test_optional_uncertain_search_without_actual_detections_stays_missing():
    b,w,r,e=fixture()
    for p in e:p['visible']=False
    additions,_=confirm_window(b,w,r,e,Verifier(),None,infer=lambda *args:([],1),
        settings=LearnedRecoverySettings(search_uncertain_points=True))
    assert additions==[]


@pytest.mark.parametrize('status',['resembles_initial_background','background_registration_unavailable',
                                  'seed_area_supplement_withheld','insufficient_patch_texture'])
@pytest.mark.parametrize('uncertain_search',[False,True])
def test_background_registration_and_seed_vetoes_remain(status,uncertain_search):
    b,w,r,e=fixture()
    if uncertain_search:
        for p in e:p['visible']=False
    additions,checks=confirm_window(b,w,r,e,Verifier(status),None,infer=infer,
        settings=LearnedRecoverySettings(search_uncertain_points=uncertain_search))
    assert additions==[] and checks[3]['status']==status


@pytest.mark.parametrize('uncertain_search',[False,True])
def test_competing_balls_are_not_chosen_by_detector_confidence(uncertain_search):
    b,w,r,e=fixture()
    if uncertain_search:
        for p in e:p['visible']=False
    def competing(model,image,region):
        x,y=region['query_center_native']
        return [{'center':[x+dx,y],'confidence':score,
                 'bbox_native_xyxy':[x+dx-2,y-2,x+dx+2,y+2]}
                for dx,score in ((-12,.99),(12,.06))],1
    additions,checks=confirm_window(b,w,r,e,Verifier(),None,infer=competing,
        settings=LearnedRecoverySettings(search_uncertain_points=uncertain_search))
    assert additions==[] and checks[3]['nearby_candidate_count']==2
    assert checks[3]['status']=='ambiguous_detector_candidates'


def test_missing_early_evidence_needs_a_new_three_observation_sequence():
    b,w,r,e=fixture()
    e[3]['visible']=False
    additions,checks=confirm_window(b,w,r,e,Verifier(),None,infer=infer)
    assert [p['frame'] for p in additions]==[6,7,8,9]
    assert checks[4]['status']==checks[5]['status']=='awaiting_detector_sequence_support'


def test_later_real_detections_can_confirm_earlier_measured_boxes_without_filling_gap():
    b,w,r,e=fixture()
    e[3]['visible']=False
    original=deepcopy(b)
    additions,checks=confirm_window(b,w,r,e,Verifier(),None,infer=infer,
        settings=LearnedRecoverySettings(confirm_measured_history=True))
    assert b==original
    assert [p['frame'] for p in additions]==list(range(4,10))
    assert additions[0]['confirmed_at_frame']==6
    assert additions[0]['support_observation_frames']==[4,5,6]
    assert additions[0]['center']==[108,100]  # Actual detector, not fitted coordinates.
    assert checks[3]['status']=='model_visibility_withheld'
    assert checks[4]['status']=='current_detector_observation_confirmed_by_later_frames'
    assert all(p['identity_verified'] is False for p in additions)


def test_two_detections_at_window_end_stay_unconfirmed():
    b,w,r,e=fixture(6)
    e[3]['visible']=False
    additions,_=confirm_window(b,w,r,e,Verifier(),None,infer=infer,
        settings=LearnedRecoverySettings(confirm_measured_history=True))
    assert additions==[]


def test_ambiguity_breaks_history_and_cannot_be_retroactively_filled():
    b,w,r,e=fixture()
    e[3]['visible']=False
    def ambiguous(model,image,region):
        if region['query_center_native'][0]==e[5]['center_local'][0]:
            x,y=region['query_center_native']
            return [{'center':[x+dx,y],'confidence':.8,
                     'bbox_native_xyxy':[x+dx-2,y-2,x+dx+2,y+2]} for dx in (-12,12)],1
        return infer(model,image,region)
    additions,_=confirm_window(b,w,r,e,Verifier(),None,infer=ambiguous,
        settings=LearnedRecoverySettings(confirm_measured_history=True))
    assert [p['frame'] for p in additions]==[6,7,8,9]


def test_inconsistent_old_measurement_does_not_join_a_later_supported_path():
    b,w,r,e=fixture()
    e[3]['visible']=False
    e[4]['center_local'][0]+=180
    additions,_=confirm_window(b,w,r,e,Verifier(),None,infer=infer,
        settings=LearnedRecoverySettings(confirm_measured_history=True))
    assert [p['frame'] for p in additions]==list(range(5,10))


def test_small_crop_is_only_queried_after_no_nearby_normal_candidate():
    b,w,r,e=fixture()
    calls=[]
    def smaller(model,image,query):
        calls.append(list(query))
        return infer(model,image,{'query_center_native':query})
    additions,checks=confirm_window(b,w,r,e,Verifier(),None,infer=lambda *args:([],1),
        small_infer=smaller,settings=LearnedRecoverySettings(small_crop_fallback=True))
    assert [p['frame'] for p in additions]==list(range(3,10))
    assert len(calls)==7 and checks[3]['native_crop_sizes']==[384,256]
    confirm_window(b,w,r,e,Verifier(),None,infer=infer,
        small_infer=lambda *args:pytest.fail('Existing local box must be preserved'),
        settings=LearnedRecoverySettings(small_crop_fallback=True))


def test_ambiguous_normal_crop_cannot_be_resolved_by_trying_another_scale():
    b,w,r,e=fixture()
    def ambiguous(model,image,region):
        x,y=region['query_center_native']
        return [{'center':[x+dx,y],'confidence':.8,
                 'bbox_native_xyxy':[x+dx-2,y-2,x+dx+2,y+2]} for dx in (-12,12)],1
    additions,checks=confirm_window(b,w,r,e,Verifier(),None,infer=ambiguous,
        small_infer=lambda *args:pytest.fail('Ambiguity cannot be bypassed'),
        settings=LearnedRecoverySettings(small_crop_fallback=True))
    assert additions==[] and checks[3]['status']=='ambiguous_detector_candidates'


def test_small_scale_proposals_still_need_background_and_repeated_evidence():
    b,w,r,e=fixture()
    additions,_=confirm_window(b,w,r,e,Verifier('resembles_initial_background'),None,
        infer=lambda *args:([],1),small_infer=lambda m,i,q:infer(m,i,{'query_center_native':q}),
        settings=LearnedRecoverySettings(small_crop_fallback=True,confirm_measured_history=True))
    assert additions==[]


def test_small_crop_preserves_native_edges_and_source_box_coordinates():
    image=np.zeros((576,768,3),np.uint8)
    image[575,767]=[123,45,67]
    class Model:
        def predict(self,crops,**kwargs):
            assert len(crops)==1 and crops[0].shape==(256,256,3)
            assert crops[0][-1,-1].tolist()==[123,45,67]
            assert kwargs=={'imgsz':640,'conf':.05,'classes':[0],'device':'cpu','verbose':False}
            tensor=lambda v:SimpleNamespace(cpu=lambda:SimpleNamespace(numpy=lambda:np.array(v)))
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=tensor([[230,230,250,250]]),conf=tensor([.4])))]
    proposals,count=detect_small_query_box(Model(),image,[767,575])
    assert count==1 and proposals[0]['bbox_native_xyxy']==[742,550,762,570]
    assert proposals[0]['center']==[752,560]


@pytest.mark.parametrize('query',[[float('nan'),100],[-1,100],[768,100]])
def test_small_crop_rejects_invalid_or_outside_queries(query):
    with pytest.raises(ValueError):
        detect_small_query_box(None,np.zeros((576,768,3),np.uint8),query)


def test_long_lost_gap_cannot_bootstrap_a_model_only_identity():
    b,w,r,e=fixture(18)
    for p in e[3:15]:p['visible']=False
    additions,checks=confirm_window(b,w,r,e,Verifier(),None,infer=infer)
    assert additions==[]
    assert checks[15]['status']=='lost_deadline_requires_new_baseline_observation'


def test_accepted_baseline_positions_cannot_be_changed_by_model_disagreement():
    b,w,r,e=fixture()
    b['path'][5].update(detected=True,center=[400,100],confidence=.7)
    original=deepcopy(b)
    additions,checks=confirm_window(b,w,r,e,Verifier(),None,infer=infer)
    assert b==original and all(p['frame']!=5 for p in additions)
    assert checks[5]['status']=='baseline_observation_disagrees_with_learned_estimate'


def test_point_outside_native_crop_cannot_query_detector():
    b,w,r,e=fixture()
    for p in e[3:]:p['center_local']=[-100,200]
    additions,checks=confirm_window(b,w,r,e,Verifier(),None,
        infer=lambda *args:pytest.fail('Out-of-crop point cannot query detector'))
    assert additions==[] and checks[3]['status']=='learned_point_outside_native_context'


@pytest.mark.parametrize('mutation',[lambda e:e.pop(), lambda e:e[4].update(frame=3),
                                   lambda e:e[4].update(visible=1),
                                   lambda e:e[4].update(center_local=[float('nan'),1])])
def test_invalid_estimates_rejected_as_a_whole_window(mutation):
    b,w,r,e=fixture()
    mutation(e)
    with pytest.raises(ValueError,match='Learned estimates'):
        confirm_window(b,w,r,e,Verifier(),None,infer=infer)


def test_expired_runtime_cannot_commit_partial_observations():
    b,w,r,e=fixture()
    with pytest.raises(TimeoutError):
        confirm_window(b,w,r,e,Verifier(),None,infer=infer,deadline=0)


@pytest.mark.parametrize('changes',[{'window_frames':33},{'maximum_windows':True},
    {'minimum_seed_history':2},{'context_frames':3},{'maximum_prediction_gap':13},
    {'detector_gate_working_px':float('inf')},{'maximum_runtime_s':0},{'search_uncertain_points':1},
    {'confirm_measured_history':1},{'small_crop_fallback':'yes'},
    {'motion_guided_crop':1},{'maximum_motion_windows':0}])
def test_invalid_resource_and_distance_settings_rejected(changes):
    with pytest.raises(ValueError):
        LearnedRecoverySettings(**changes)


def test_planner_rejects_predicted_or_missing_coordinates_as_a_seed():
    b=baseline()
    b['path'][2]['center']=None
    with pytest.raises(ValueError,match='observation'):
        plan_windows(b['path'])
    b=baseline()
    b['path'][4]['frame']=5
    with pytest.raises(ValueError,match='contiguous'):
        plan_windows(b['path'])


def test_source_crop_uses_real_pixels_and_stays_inside_portrait_or_landscape():
    for width,height in ((3840,2160),(1080,1920),(720,1280)):
        for seed in ((0,0),(width-1,height-1),(width/2,height/2)):
            a,b,c,d=crop_for_seed(seed,width,height)
            assert 0<=a<c<=width and 0<=b<d<=height
            assert c-a<=768 and d-b<=576


def test_motion_crop_shifts_search_pixels_but_keeps_seed_inside_real_source():
    seed=[1725.5,1529.5]
    centered=crop_for_seed(seed,3840,2160)
    shifted=crop_for_seed(seed,3840,2160,velocity_native=[3.2,-24.5],horizon_frames=25)
    assert centered[1]==1242 and shifted[1]<1100<shifted[3]
    assert seed[1]-shifted[1]<576-18 and seed[0]-shifted[0]>=32
    assert shifted[2]-shifted[0]==768 and shifted[3]-shifted[1]==576
    for velocity in ([1e6,1e6],[-1e6,-1e6]):
        crop=crop_for_seed(seed,3840,2160,velocity_native=velocity,horizon_frames=25)
        assert 32<=seed[0]-crop[0]<=768-32
        assert 32<=seed[1]-crop[1]<=576-32


def test_motion_crop_planner_uses_only_past_measurements_and_is_bounded():
    b=baseline(20)
    for p in b['path'][:3]:p['center']=[200,400-p['frame']*15]
    s=LearnedRecoverySettings(motion_guided_crop=True,maximum_motion_windows=1)
    planned=plan_windows(b['path'],s)
    windows=plan_motion_windows(b['path'],planned,b['frame_transform'],b['width'],b['height'],s)
    assert len(windows)==1 and windows[0]['crop_motion_history_frames']==[0,1,2]
    assert windows[0]['seed_frame']==2 and windows[0]['past_image_velocity_working_px_per_frame']==[0,-15]
    b['path'][10].update(detected=True,center=[999,999],confidence=.9)
    changed=plan_motion_windows(b['path'],planned,b['frame_transform'],b['width'],b['height'],s)
    assert changed[0]['crop_xyxy']==windows[0]['crop_xyxy']
    assert changed[0]['past_image_velocity_working_px_per_frame']==windows[0]['past_image_velocity_working_px_per_frame']
    assert 10 not in changed[0]['new_missing_frames_searched']  # Work coverage uses the frozen missing-frame mask.
    assert plan_motion_windows(b['path'],planned,b['frame_transform'],b['width'],b['height'],LearnedRecoverySettings())==[]


def test_motion_crop_does_not_duplicate_stationary_search_views():
    b=baseline()
    for p in b['path'][:3]:p['center']=[200,400]
    s=LearnedRecoverySettings(motion_guided_crop=True)
    assert plan_motion_windows(b['path'],plan_windows(b['path'],s),b['frame_transform'],1280,720,s)==[]


def test_motion_search_does_not_repeat_the_same_missing_frame_set():
    b=baseline(20)
    for p in b['path'][:3]:p['center']=[200,400-p['frame']*15]
    for p in b['path'][4:7]:p.update(detected=True,confidence=.7,center=[300,400-(p['frame']-4)*15])
    s=LearnedRecoverySettings(motion_guided_crop=True)
    planned=plan_windows(b['path'],s)
    assert [w['seed_frame'] for w in planned]==[2,6]
    motion=plan_motion_windows(b['path'],planned,b['frame_transform'],1280,720,s)
    assert [w['seed_frame'] for w in motion]==[2]
    assert motion[0]['new_missing_frames_searched']==[3]+list(range(7,20))
    planned[0]['end_frame']=10
    motion=plan_motion_windows(b['path'],planned,b['frame_transform'],1280,720,s)
    assert [w['seed_frame'] for w in motion]==[6,2]  # Larger projected shift first.
    assert motion[0]['new_missing_frames_searched']==list(range(7,20))
    assert motion[1]['new_missing_frames_searched']==[3]


def runner_fixture(monkeypatch):
    import cv2
    import experiments.ball_tracking.detector_confirmed_recovery as module
    b,w,records,estimates=fixture()
    b.update(clip='test',source_sha256='same',weights_sha256='same',source_seed=[300,300])
    b['frame_transform'].update(width=1280,height=720)
    for e in estimates:
        e['center_local']=[300+e['frame']*6,288]
    class Capture:
        def read(self):return True,np.zeros((2160,3840,3),np.uint8)
        def release(self):pass
    monkeypatch.setattr(cv2,'VideoCapture',lambda *args:Capture())
    monkeypatch.setattr(module,'file_sha256',lambda *args:'same')
    monkeypatch.setattr(module,'NativeEvidenceVerifier',lambda *args:Verifier())
    monkeypatch.setattr(module,'load_window',lambda *args:records)
    monkeypatch.setattr(module,'decode_selected_windows',lambda *args,**kwargs:(None,{'mode':'test_per_window'}))
    monkeypatch.setattr(module,'detect_native_boxes',infer)
    return b,estimates


def test_unavailable_optional_model_saves_unchanged_usable_baseline(tmp_path,monkeypatch):
    import json
    from experiments.ball_tracking.cotracker_recovery import UnavailableRecovery
    b,_=runner_fixture(monkeypatch)
    original=deepcopy(b)
    result=run_recovery(b,'video.mov',None,'weights.pt',
        UnavailableRecovery(FileNotFoundError('Missing local research checkpoint')),
        tmp_path/'report.json')
    assert b==original and result['path']==b['path']
    assert result['learned_recovery']['windows']==[]
    assert result['learned_recovery']['errors'][0]['stage']=='optional_model_unavailable'
    assert json.loads((tmp_path/'report.json').read_text())['path']==b['path']


@pytest.mark.parametrize('failure',['exception','malformed'])
def test_failed_model_window_cannot_publish_partial_or_fitted_positions(tmp_path,monkeypatch,failure):
    b,estimates=runner_fixture(monkeypatch)
    class Provider:
        def configuration(self):return {'test_provider':True}
        def predict(self,*args,**kwargs):
            if failure=='exception':raise RuntimeError('Local model failed')
            return estimates[:-1]
    result=run_recovery(b,'video.mov',None,'weights.pt',Provider(),tmp_path/'report.json')
    assert result['path']==b['path']
    assert result['learned_recovery']['new_observation_frames']==[]
    assert result['learned_recovery']['windows'][0]['status']=='failed_baseline_preserved'


def test_runner_commits_only_actual_detector_additions_and_keeps_previous_rows(tmp_path,monkeypatch):
    b,estimates=runner_fixture(monkeypatch)
    b['rows']=[{'unchanged_baseline_diagnostic':True}]
    original=deepcopy(b)
    class Provider:
        def configuration(self):return {'test_provider':True}
        def predict(self,*args,**kwargs):return estimates
    result=run_recovery(b,'video.mov',None,'weights.pt',Provider(),tmp_path/'report.json')
    assert b==original and result['rows']==b['rows']
    assert result['path'][:3]==b['path'][:3]
    assert result['learned_recovery']['new_observation_frames']==list(range(3,10))
    assert all(p['proposal_source']=='learned_query_native_detector' for p in result['path'][3:])
    assert all(p['identity_verified'] is False for p in result['path'])
    assert all(p['uses_future_video_frames'] is True for p in result['path'][3:])


def test_failure_after_some_candidate_additions_discards_entire_failed_window(tmp_path,monkeypatch):
    import experiments.ball_tracking.detector_confirmed_recovery as module
    b,estimates=runner_fixture(monkeypatch)
    class FailedVerifier(Verifier):
        def inspect(self,image,candidate,*args,**kwargs):
            if candidate['center'][0]>110:raise RuntimeError('Background registration failed midway')
            return super().inspect(image,candidate,*args,**kwargs)
    class Provider:
        def configuration(self):return {'test_provider':True}
        def predict(self,*args,**kwargs):return estimates
    monkeypatch.setattr(module,'NativeEvidenceVerifier',lambda *args:FailedVerifier())
    result=run_recovery(b,'video.mov',None,'weights.pt',Provider(),tmp_path/'report.json')
    assert result['path']==b['path']
    assert result['learned_recovery']['new_observation_frames']==[]
    assert result['learned_recovery']['windows'][0]['status']=='failed_baseline_preserved'
