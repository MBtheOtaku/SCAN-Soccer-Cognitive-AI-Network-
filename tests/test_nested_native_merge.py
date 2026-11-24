import pytest

from experiments.ball_tracking.native_box_detector import merge_boxes


def proposal(box, confidence):
    return {"bbox_native_xyxy": box, "confidence": confidence,
            "center": [(box[0]+box[2])/2, (box[1]+box[3])/2]}


def test_nested_same_object_boxes_merge_only_when_enabled():
    small=proposal([10,10,30,30],.7)
    large=proposal([5,5,35,35],.06)
    assert merge_boxes([large,small]) == [small,large]
    assert merge_boxes([large,small],containment_threshold=.85) == [small]


@pytest.mark.parametrize("other", [[25,10,45,30], [16,0,44,40], [-100,-100,140,140]])
def test_nearby_separate_offset_or_extreme_scale_boxes_stay(other):
    small=proposal([10,10,30,30],.7)
    candidate=proposal(other,.6)
    assert len(merge_boxes([small,candidate],containment_threshold=.85)) == 2


def test_merge_does_not_mutate_detector_proposals():
    box=[10,10,30,30]
    small=proposal(box,.7)
    large=proposal([5,5,35,35],.06)
    result=merge_boxes([large,small],containment_threshold=.85)
    assert result[0] is small and small['bbox_native_xyxy'] is box


@pytest.mark.parametrize('threshold',[-1,0,1.1,float('nan')])
def test_invalid_containment_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        merge_boxes([],containment_threshold=threshold)
