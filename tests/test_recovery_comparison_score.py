import copy

import pytest

from experiments.ball_tracking.score_recovery_comparison import audit_misses, compare, score_report


def inputs():
    m = {"clip_id": "review", "source_sha256": "source",
         "frames": [{"frame": f, "width": 300, "height": 300} for f in range(4)]}
    labels = {"schema": 1, "clip_id": "review", "source_sha256": "source", "annotations": [
        {"frame": 0, "state": "visible", "box": [144, 144, 156, 156]},
        {"frame": 1, "state": "visible", "box": [144, 144, 156, 156]},
        {"frame": 2, "state": "uncertain", "box": None},
        {"frame": 3, "state": "absent", "box": None}]}
    report = {"source_sha256": "source", "source_seed": [10, 10], "weights_sha256": "weights",
              "width": 100, "height": 100,
              "frame_transform": {"source_width": 300, "source_height": 300},
              "path": [{"frame": f, "detected": False, "center": None,
                        "predicted": [50, 50]} for f in range(4)]}
    return m, labels, report


def test_predictions_never_count_as_observations():
    m, labels, report = inputs()
    s = score_report(m, labels, report)
    assert s["correct_observations"] == 0
    assert s["longest_visible_gap_frames"] == 2
    assert s["uncertain_excluded"] == 1


def test_source_scaling_and_frozen_padding():
    m, labels, report = inputs()
    report["path"][0].update(detected=True, center=[50, 50])
    report["path"][1].update(detected=True, center=[54, 50])
    s = score_report(m, labels, report)
    assert s["correct_observations"] == 2
    assert s["strict_box_hits"] == 1
    assert s["false_acceptances"] == 0


def test_false_selection_on_absent_counts_but_uncertain_is_excluded():
    m, labels, report = inputs()
    for point in report["path"]:
        point.update(detected=True, center=[90, 90])
    s = score_report(m, labels, report)
    assert s["false_acceptances"] == 3
    assert s["false_acceptances_absent"] == 1
    assert s["selected_observation_precision"] == 0


@pytest.mark.parametrize("change", ["hash", "incomplete", "duplicate", "geometry"])
def test_rejects_wrong_or_incomplete_references(change):
    m, labels, report = inputs()
    if change == "hash":
        report["source_sha256"] = "other"
    elif change == "incomplete":
        labels["annotations"][0]["state"] = "unreviewed"
        labels["annotations"][0]["box"] = None
    elif change == "duplicate":
        report["path"].append(report["path"][0])
    else:
        m["frames"][0]["width"] = 400
    with pytest.raises(ValueError):
        score_report(m, labels, report)


def test_paired_comparison_requires_same_seed_and_reports_actual_gains():
    m, labels, baseline = inputs()
    candidate = copy.deepcopy(baseline)
    candidate["path"][1].update(detected=True, center=[50, 50])
    s = compare(m, labels, baseline, candidate)
    assert s["improved_frames"] == [1]
    assert s["regressed_frames"] == []
    candidate["source_seed"] = [99, 99]
    with pytest.raises(ValueError, match="same initial seed"):
        compare(m, labels, baseline, candidate)


def logged_check(frame,point=(150,150),proposals=(),searched=True,visible=True):
    return {'frame':frame,'learned_estimate_source':list(point),'model_visible':visible,
            'native_crop_count':1 if searched else 0,'support_observation_frames':[],
            'status':'awaiting_detector_sequence_support',
            'native_proposals':[{'center_source':list(p)} for p in proposals]}


def with_checks(report,checks):
    report['learned_recovery']={'windows':[{'status':'completed','seed_frame':0,'checks':checks}]}
    return report


def test_audit_separates_existing_boxes_from_missing_detector_evidence():
    m,labels,report=inputs()
    with_checks(report,[logged_check(0,proposals=[(150,150)]),logged_check(1)])
    before=copy.deepcopy(report)
    result=audit_misses(m,labels,report)
    assert report==before
    assert [r['category'] for r in result['rows']]==[
        'matching_detector_box_withheld_by_confirmation','no_matching_detector_box_in_logged_search']
    assert result['missed_visible_frames']==2  # Neither absent nor uncertain becomes a visible miss.


def test_audit_distinguishes_drift_no_window_and_visibility_without_query():
    m,labels,report=inputs()
    with_checks(report,[logged_check(0,point=(270,270))])
    assert [r['category'] for r in audit_misses(m,labels,report)['rows']]==[
        'learned_query_missed_reference','no_recorded_recovery_window']
    with_checks(report,[logged_check(0,searched=False,visible=False)])
    assert audit_misses(m,labels,report)['rows'][0]['category']=='matching_learned_point_withheld_before_search'


def test_auxiliary_query_does_not_turn_a_proposal_into_a_scored_observation():
    m,labels,report=inputs()
    with_checks(report,[logged_check(0,searched=False,visible=False)])
    auxiliary=copy.deepcopy(report)
    with_checks(auxiliary,[logged_check(0,proposals=[(150,150)],visible=False)])
    result=audit_misses(m,labels,report,auxiliary_search=auxiliary)
    assert result['rows'][0]['category']=='matching_detector_box_withheld_by_confirmation'
    assert result['rows'][0]['checks'][1]['source']=='auxiliary'
    assert score_report(m,labels,report)['correct_observations']==0
    auxiliary['source_seed']=[999,999]
    with pytest.raises(ValueError,match='same initial seed'):
        audit_misses(m,labels,report,auxiliary_search=auxiliary)


def test_corrupt_logged_coordinates_are_not_silently_classified():
    m,labels,report=inputs()
    with_checks(report,[logged_check(0,point=(float('nan'),150))])
    with pytest.raises(ValueError,match='audit point'):
        audit_misses(m,labels,report)


def test_audit_identifies_missing_image_coverage_separately_from_tracking_drift():
    m,labels,report=inputs()
    with_checks(report,[logged_check(0,point=(270,270))])
    report['learned_recovery']['windows'][0]['crop_xyxy']=[200,200,300,300]
    result=audit_misses(m,labels,report)
    assert result['rows'][0]['category']=='reference_outside_supplied_learned_crop'
    assert result['rows'][0]['checks'][0]['reference_center_in_supplied_crop'] is False
