"""Background registration and repeated evidence never certify identity."""

import cv2
import numpy as np
import pytest

from experiments.ball_tracking.native_evidence_verifier import NativeEvidenceVerifier, project


def textured():
    rng = np.random.default_rng(31)
    image = rng.integers(20, 170, (480, 320, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def verifier():
    image = textured()
    return NativeEvidenceVerifier(image, image, (30, 30)), image


def registration():
    return {"inlier_current_points": [[200, 200]], "status": "synthetic_registration"}


def test_projective_registration_handles_camera_motion_larger_than_local_gate():
    context, image = verifier()
    matrix = np.array([[1., .02, 60.], [-.01, 1., 14.], [.00002, .00001, 1.]])
    current = cv2.warpPerspective(image, matrix, (320, 480))
    found, meta = context.align(current)
    assert found is not None and meta["inliers"] >= 12
    assert np.linalg.norm(project((180, 220), found) - project((180, 220), matrix)) < 2
    # Registration is image alignment, not a calibrated goal/depth transform.
    assert meta["status"] == "registered_initial_background_not_depth"


def test_blank_images_abstain_from_registration():
    image = np.zeros((480, 320, 3), np.uint8)
    context = NativeEvidenceVerifier(image, image, (30, 30))
    matrix, meta = context.align(image)
    assert matrix is None and meta["status"] == "background_registration_unavailable"


def test_unchanged_background_is_rejected_even_after_exposure_change():
    context, image = verifier()
    current = np.clip(image.astype(float) * .8 + 25, 0, 255).astype(np.uint8)
    result = context.inspect(current, {"center": [200, 200]}, np.eye(3), registration())
    assert result["status"] == "resembles_initial_background"
    assert result["patch_ncc_not_probability"] > .95


def test_new_visible_object_is_evidence_change_without_identity_claim():
    context, image = verifier()
    cv2.circle(image, (200, 200), 5, (240, 240, 240), -1)
    result = context.inspect(image, {"center": [200, 200]}, np.eye(3), registration())
    assert result["status"] == "appearance_changed_not_identity_proof"


def test_initial_target_region_is_left_to_primary_detector():
    context, image = verifier()
    result = context.inspect(image, {"center": [30, 30]}, np.eye(3), registration())
    assert result["status"] == "seed_area_supplement_withheld"


def test_recent_continuity_cannot_bootstrap_supplemental_points_on_near_seed_boot():
    context, image = verifier()
    baseline = {"center": [30, 30], "confidence": .6}
    supplemental = {"center": [31, 31], "confidence": .8, "proposal_source": "native_detail"}
    result, info = context.filter(2, image, image, [baseline, supplemental], accepted_frames=[0, 1])
    assert result == [baseline]
    assert info["checks"][0]["status"] == "seed_area_supplement_withheld"


def test_registration_without_local_support_cannot_admit_extra_proposal():
    context, image = verifier()
    result = context.inspect(image, {"center": [200, 200]}, np.eye(3),
                             {"inlier_current_points": [[0, 0]]})
    assert result["status"] == "insufficient_local_registration_support"


def synthetic_filter(monkeypatch):
    context, image = verifier()
    monkeypatch.setattr(context, "align", lambda _: (np.eye(3), registration()))
    monkeypatch.setattr(context, "inspect", lambda source, candidate, *args: {
        "status": "appearance_changed_not_identity_proof",
        "reference_processing_center": candidate["center"],
    })
    return context, image


def extra(x=200):
    return {"center": [x, 200], "confidence": .5, "proposal_source": "native_detail"}


def test_only_current_third_consistent_observation_is_forwarded(monkeypatch):
    context, image = synthetic_filter(monkeypatch)
    baseline = {"center": [300, 300], "confidence": .8}
    for frame in range(3):
        result, info = context.filter(frame, image, image, [baseline, extra(200 + frame)])
        assert result[0] is baseline
        assert len(result) == (2 if frame == 2 else 1)
    assert info["support_frames"] == [0, 1, 2]
    assert info["status"] == "supported_extra_evidence_not_verified_identity"
    assert "identity_verified" not in result[1]


@pytest.mark.parametrize("middle", [[], [extra(199), extra(201)], [extra(240)]])
def test_gap_ambiguity_or_large_step_cannot_accumulate_false_support(monkeypatch, middle):
    context, image = synthetic_filter(monkeypatch)
    assert context.filter(0, image, image, [extra()])[0] == []
    assert context.filter(1, image, image, middle)[0] == []
    assert context.filter(2, image, image, [extra()])[0] == []


def test_unavailable_registration_preserves_all_baseline_points(monkeypatch):
    context, image = verifier()
    monkeypatch.setattr(context, "align", lambda _: (None, {}))
    baseline = [{"center": [10, 20], "confidence": .1}, {"center": [11, 21], "confidence": .2}]
    result, _ = context.filter(0, image, image, baseline + [extra()])
    assert result == baseline


def test_background_lookalike_never_gains_support():
    context, image = verifier()
    for frame in range(5):
        result, info = context.filter(frame, image, image, [extra()])
        assert result == []
        assert info["support_frames"] == []


def test_skipping_future_frames_is_rejected(monkeypatch):
    context, image = synthetic_filter(monkeypatch)
    context.filter(0, image, image, [])
    with pytest.raises(ValueError, match="contiguous"):
        context.filter(2, image, image, [extra()])


def test_invalid_projective_point_is_rejected():
    with pytest.raises(ValueError):
        project((1, 2), np.zeros((3, 3)))
