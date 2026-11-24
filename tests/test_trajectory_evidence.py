"""Bounded temporal evidence must never turn a prediction into an observation."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.ball_tracking.trajectory_evidence import TrajectoryEvidence, TrajectoryEvidenceSettings
from experiments.ball_tracking.native_evidence_verifier import NativeEvidenceVerifier


def test_three_real_observations_across_one_gap_can_support_current_detection():
    tracker = TrajectoryEvidence()
    assert not tracker.update(0, [[100, 100]])[0]['supported']
    assert not tracker.update(1, [[105, 100]])[0]['supported']
    assert tracker.update(2, []) == []
    result = tracker.update(3, [[115, 100]])[0]
    assert result['supported'] and result['frames'] == [0, 1, 3]
    assert result['bridged_detector_gap'] and result['prediction_residual_px'] == 0


def test_longer_gap_cannot_supply_evidence():
    tracker = TrajectoryEvidence()
    tracker.update(0, [[100, 100]])
    tracker.update(1, [[105, 100]])
    assert tracker.update(2, []) == tracker.update(3, []) == []
    assert not tracker.update(4, [[120, 100]])[0]['supported']


def test_prediction_mismatch_cannot_reuse_history():
    tracker = TrajectoryEvidence()
    tracker.update(0, [[100, 100]])
    tracker.update(1, [[110, 100]])
    tracker.update(2, [])
    result = tracker.update(3, [[95, 100]])[0]
    assert result['frames'] == [3] and not result['supported']


def test_crossing_candidates_cannot_duplicate_support():
    tracker = TrajectoryEvidence()
    tracker.update(0, [[100, 100]])
    tracker.update(1, [[105, 100]])
    result = tracker.update(2, [[109, 100], [111, 100]])
    assert all(not r['supported'] and r['frames'] == [2] for r in result)


def test_hypotheses_for_distinct_objects_stay_separate():
    tracker = TrajectoryEvidence()
    for f in range(3):
        result = tracker.update(f, [[100+f*5, 100], [200+f*5, 100]])
    assert all(r['supported'] and r['frames'] == [0, 1, 2] for r in result)


def test_failed_registration_or_excessive_candidates_resets_support():
    for unavailable in (True, False):
        tracker = TrajectoryEvidence()
        tracker.update(0, [[100, 100]])
        tracker.update(1, [[105, 100]])
        tracker.update(2, [] if unavailable else [[110+i, 100] for i in range(9)], available=not unavailable)
        assert tracker.hypotheses == []
        assert not tracker.update(3, [[115, 100]])[0]['supported']


def test_old_history_expires_even_with_alternating_gaps():
    tracker = TrajectoryEvidence(TrajectoryEvidenceSettings(support_window_frames=3))
    for frame in range(5):
        result = tracker.update(frame, [[100+frame, 100]] if frame % 2 == 0 else [])
    assert result[0]['frames'] == [2, 4] and not result[0]['supported']


@pytest.mark.parametrize('settings', [
    {'confirmation_hits': 2}, {'maximum_missing_frames': True},
    {'maximum_missing_frames': -1}, {'support_window_frames': 2},
    {'maximum_step_px': float('nan')}, {'innovation_radius_px': 0}])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        TrajectoryEvidence(TrajectoryEvidenceSettings(**settings))


def test_invalid_frames_and_points_rejected_without_creating_future_support():
    tracker = TrajectoryEvidence()
    tracker.update(0, [[100, 100]])
    for frame, points in [(2, []), (True, []), (1, [[float('inf'), 100]])]:
        with pytest.raises(ValueError):
            tracker.update(frame, points)


def make_verifier(monkeypatch, enabled):
    image = np.zeros((480, 720, 3), np.uint8)
    verifier = NativeEvidenceVerifier(image, image, (30, 30), trajectory_support=enabled)
    monkeypatch.setattr(verifier, 'align', lambda _: (np.eye(3), {}))
    monkeypatch.setattr(verifier, 'inspect', lambda source, candidate, *args: {
        'status': 'appearance_changed_not_identity_proof',
        'reference_processing_center': candidate['center']})
    return verifier, image


def test_verifier_opt_in_bridges_gap_but_default_still_requires_consecutive_evidence(monkeypatch):
    for enabled in (False, True):
        verifier, image = make_verifier(monkeypatch, enabled)
        for frame in range(4):
            proposals = [] if frame == 2 else [
                {'center': [100+5*frame, 100], 'confidence': .3, 'proposal_source': 'native_detail'}]
            original = deepcopy(proposals)
            result, info = verifier.filter(frame, image, image, proposals)
            assert proposals == original
            if frame == 2:
                assert result == []
        assert len(result) == int(enabled)
        if enabled:
            assert info['trajectory_hypotheses'][0]['native_observation_frames'] == [0, 1, 3]
            assert info['trajectory_added_proposals'] == 1


def test_background_veto_cannot_be_overridden_by_trajectory(monkeypatch):
    verifier, image = make_verifier(monkeypatch, True)
    for f in range(2):
        verifier.filter(f, image, image, [{'center': [100+f*5, 100], 'confidence': .3,
                                         'proposal_source': 'native_detail'}])
    monkeypatch.setattr(verifier, 'inspect', lambda *args: {'status': 'resembles_initial_background'})
    baseline = {'center': [30, 30], 'confidence': .2}
    result, _ = verifier.filter(2, image, image, [baseline,
        {'center': [110, 100], 'confidence': .9, 'proposal_source': 'native_detail'}])
    assert result == [baseline]


def test_real_registration_numpy_coordinates_are_supported(monkeypatch):
    verifier, image = make_verifier(monkeypatch, True)
    monkeypatch.setattr(verifier, 'inspect', lambda source, candidate, *args: {
        'status': 'appearance_changed_not_identity_proof',
        'reference_processing_center': np.asarray(candidate['center'], dtype=np.float64)})
    for f in range(3):
        result, _ = verifier.filter(f, image, image, [{'center': [100+f*5, 100],
            'confidence': .3, 'proposal_source': 'native_detail'}])
    assert len(result) == 1


def test_motion_residual_cannot_remove_existing_consecutive_support(monkeypatch):
    verifier, image = make_verifier(monkeypatch, True)
    for frame,x in enumerate((100,102,114)):
        result, info = verifier.filter(frame,image,image,[{'center':[x,100],
            'confidence':.3,'proposal_source':'native_detail'}])
    assert len(result) == 1
    assert info['hypotheses'][0]['native_observation_frames'] == [0,1,2]
    assert not info['trajectory_hypotheses'][0]['trajectory_evidence']['supported']
    assert info['trajectory_added_proposals'] == 0
