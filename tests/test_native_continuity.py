"""Gap repairs can use prior observations while keeping distinct objects."""

import numpy as np
import pytest

from experiments.ball_tracking.native_evidence_verifier import NativeEvidenceVerifier


def setup(monkeypatch):
    image = np.zeros((480, 320, 3), np.uint8)
    verifier = NativeEvidenceVerifier(image, image, (30, 30))
    monkeypatch.setattr(verifier, "align", lambda _: (np.eye(3), {}))
    monkeypatch.setattr(verifier, "inspect", lambda source, candidate, *args: {
        "status": "appearance_changed_not_identity_proof",
        "reference_processing_center": candidate["center"],
    })
    return verifier, image


def point(x):
    return {"center": [x, 200], "confidence": .1, "proposal_source": "native_detail"}


def test_two_prior_accepted_observations_can_support_a_single_gap_repair(monkeypatch):
    verifier, image = setup(monkeypatch)
    result, info = verifier.filter(10, image, image, [point(200)], accepted_frames=[8, 9])
    assert result == [point(200)]
    assert info["support_frames"] == [8, 9, 10]
    assert info["continuity_support_frames"] == [8, 9]
    assert "identity_verified" not in result[0]


def test_missing_prior_observation_does_not_warm_start(monkeypatch):
    verifier, image = setup(monkeypatch)
    result, info = verifier.filter(10, image, image, [point(200)], accepted_frames=[7, 9])
    assert result == [] and info["continuity_support_frames"] == []


@pytest.mark.parametrize("past", [[8, 10], [8, 8, 9], [9, 8], [True, 9], [-2, -1]])
def test_future_duplicate_unsorted_or_fabricated_support_is_rejected(monkeypatch, past):
    verifier, image = setup(monkeypatch)
    with pytest.raises(ValueError, match="past accepted frames"):
        verifier.filter(10, image, image, [point(200)], accepted_frames=past)


def test_nearby_distinct_objects_keep_separate_observation_histories(monkeypatch):
    verifier, image = setup(monkeypatch)
    for frame in range(3):
        result, info = verifier.filter(frame, image, image, [point(200), point(210)])
        assert len(result) == (2 if frame == 2 else 0)
    assert [h["native_observation_frames"] for h in info["hypotheses"]] == [[0, 1, 2], [0, 1, 2]]
    # Candidates reach the association layer; this layer never chooses identity.
    assert result == [point(200), point(210)]


def test_one_old_observation_cannot_supply_two_ambiguous_new_histories(monkeypatch):
    verifier, image = setup(monkeypatch)
    verifier.filter(0, image, image, [point(200)])
    verifier.filter(1, image, image, [point(199), point(201)])
    result, _ = verifier.filter(2, image, image, [point(199), point(201)])
    assert result == []
    result, _ = verifier.filter(3, image, image, [point(199), point(201)])
    # The two locations remain within the assignment ambiguity margin.
    assert result == []


def test_excessive_candidate_ambiguity_preserves_baseline_without_arbitrary_truncation(monkeypatch):
    verifier, image = setup(monkeypatch)
    baseline = {"center": [10, 10], "confidence": .2}
    result, info = verifier.filter(10, image, image, [baseline] + [point(100 + i) for i in range(9)],
                                   accepted_frames=[8, 9])
    assert result == [baseline] and info["hypotheses"] == []


def test_recent_continuity_cannot_override_a_background_veto(monkeypatch):
    verifier, image = setup(monkeypatch)
    monkeypatch.setattr(verifier, "inspect", lambda *args: {"status": "resembles_initial_background"})
    assert verifier.filter(10, image, image, [point(200)], accepted_frames=[8, 9])[0] == []


def test_recent_continuity_cannot_override_failed_registration(monkeypatch):
    verifier, image = setup(monkeypatch)
    monkeypatch.setattr(verifier, "inspect", lambda *args: {"status": "background_registration_unavailable"})
    assert verifier.filter(10, image, image, [point(200)], accepted_frames=[8, 9])[0] == []
