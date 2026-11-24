"""Native crops add real local evidence without deleting or certifying points."""

import copy
import math

import numpy as np
import pytest

from experiments.ball_tracking.native_detail_fallback import augment_native_detail, local_region
from experiments.ball_tracking.probe_native_detail import native_origins
from experiments.ball_tracking import track_unlabelled_video as runner


def transform(scale=3):
    return {"width": 720, "height": 1280, "source_width": round(720 * scale),
            "source_height": round(1280 * scale), "scale_x": 1 / scale,
            "scale_y": 1 / scale, "status": "downsampled"}


def image(scale=3):
    return np.zeros((1280 * scale, 720 * scale, 3), dtype=np.uint8)


@pytest.mark.parametrize("scale,expected_crops", [(1, 1), (1.5, 1), (3, 4)])
@pytest.mark.parametrize("query", [(360, 640), (0, 0), (719, 1279), (-20, 1400)])
def test_local_native_search_is_bounded_and_fits_real_pixels(scale, expected_crops, query):
    t = transform(scale)
    region = local_region(t, query)
    left, top, right, bottom = region["native_xyxy"]
    assert 0 <= left < right <= t["source_width"]
    assert 0 <= top < bottom <= t["source_height"]
    crops = native_origins(region["native_xyxy"])
    assert len(crops) == expected_crops
    assert all(left <= x <= right - 384 and top <= y <= bottom - 384 for x, y in crops)


def test_weak_local_baseline_is_preserved_without_requesting_native_inference():
    baseline = [{"center": [300, 650], "confidence": 0.051}]
    before = copy.deepcopy(baseline)

    def forbidden(*args):
        raise AssertionError("Existing weak local evidence must not be replaced")

    result, info = augment_native_detail(None, image(), transform(), (301, 651), baseline, infer=forbidden)
    assert result == baseline == before
    assert result is not baseline
    assert info["status"] == "not_requested_local_evidence_present"
    assert info["crop_count"] == 0


def test_high_confidence_distant_object_does_not_block_local_search():
    baseline = [{"center": [500, 900], "confidence": 0.99}]
    seen = []

    def native(model, source, region):
        seen.append(source)
        return [{"center": [906, 1956], "confidence": 0.07},
                {"center": [1020, 1950], "confidence": 0.95}], 4

    source = image()
    result, info = augment_native_detail(None, source, transform(), (300, 650), baseline, infer=native)
    assert seen[0] is source
    assert result[0] == baseline[0]
    assert len(result) == 2
    assert result[1]["center"] == [302, 652]
    assert result[1]["native_center"] == [906, 1956]
    assert result[1]["proposal_source"] == "native_detail"
    assert "identity_verified" not in result[1]
    assert len(info["raw_proposals"]) == 2 and len(info["admitted_proposals"]) == 1
    assert info["crop_count"] == 4


def test_boundary_gate_preserves_baseline_and_does_not_merge_nearby_native_objects():
    baseline = [{"center": [340, 650], "confidence": 0.051}]

    def native(*args):
        return [{"center": [996, 1950], "confidence": 0.9},
                {"center": [993, 1950], "confidence": 0.8},
                {"center": [999, 1950], "confidence": 0.7}], 4

    result, info = augment_native_detail(None, image(), transform(), (300, 650), baseline, infer=native)
    assert result[0] == baseline[0]
    # 332 lies on the admission boundary. Box NMS owns duplicate removal;
    # this layer must not discard a different nearby object by centre distance.
    assert [p["center"] for p in result] == [[340, 650], [332, 650], [331, 650]]
    assert len(info["raw_proposals"]) == 3


def test_720_source_is_not_upscaled_or_reinferred():
    result, info = augment_native_detail(None, image(1), transform(1), (300, 650), [],
                                       infer=lambda *args: pytest.fail("No additional detail"))
    assert result == [] and info["status"] == "not_requested_no_additional_source_detail"


def test_opt_in_recovery_search_uses_anchor_without_bypassing_verifier():
    def native(*args):
        return [{"center": [1080, 1950], "confidence": .9},
                {"center": [1200, 1950], "confidence": .99}], 4

    result, info = augment_native_detail(
        None, image(), transform(), (300, 550), [], infer=native,
        recovery_query=(300, 650))
    assert [p["center"] for p in result] == [[360, 650]]
    assert result[0]["proposal_source"] == "native_detail"
    assert "identity_verified" not in result[0]
    assert info["recovery_query"] == [300, 650]


@pytest.mark.parametrize("query", [(math.nan, 1), (1, math.inf), (1,)])
def test_invalid_query_rejected(query):
    with pytest.raises(ValueError):
        local_region(transform(), query)


def test_mismatched_source_dimensions_rejected():
    with pytest.raises(ValueError, match="dimensions disagree"):
        augment_native_detail(None, image(1), transform(), (1, 2), [])


def test_noninteger_source_dimensions_rejected():
    with pytest.raises(ValueError, match="integer image dimensions"):
        local_region({**transform(), "source_width": 2160.0}, (1, 2))


def test_invalid_detector_proposal_is_rejected():
    with pytest.raises(ValueError, match="Invalid native"):
        augment_native_detail(None, image(), transform(), (300, 650), [],
                              infer=lambda *args: ([{"center": [math.nan, 1], "confidence": .4}], 4))


@pytest.mark.parametrize("association,short_edge", [("legacy", 720), ("motion-recovery", None),
                                                   ("motion-recovery", 1080)])
def test_runner_requires_explicit_compatible_offline_mode(tmp_path, association, short_edge):
    with pytest.raises(ValueError, match="requires motion-recovery"):
        runner.run_video({}, None, tmp_path / "weights", tmp_path / "out", 1,
                         association=association, processing_short_edge=short_edge,
                         native_detail_fallback=True)
