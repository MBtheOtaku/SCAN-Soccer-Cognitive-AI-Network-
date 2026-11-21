"""Uncertainty must never be presented as technical praise or criticism."""
from copy import deepcopy
from dataclasses import MISSING, fields
import json
from types import SimpleNamespace

import pytest

from src.backend.coaching_report import build_coaching_report, technique_grade_available
from src.backend.jev_judgements import JevJudgementResult
from src.backend.llm_coaching import refine_coaching_report_with_llm
from src.backend.situated_state import SituatedState
from src.backend.strike_window import build_strike_window


def state_for():
    state = SituatedState(**{
        f.name: None for f in fields(SituatedState) if f.default is MISSING
    })
    state.perception_usable = True
    state.pose_coverage = 1.0
    state.strike_window = build_strike_window([
        SimpleNamespace(frame_idx=f, pose_confidence=.75,
                        torso_lean_deg=22.3 if f < 96 else 6.2 if f <= 102 else 24.6)
        for f in range(90, 109)
    ], 99, 30, 151)
    return state


def judgement(issue="insufficient_evidence", grade="1", status="ok"):
    return JevJudgementResult(status=status, shadow_mode=True, model="test",
        answers={
            "technique_quality": {"score": 1.15, "probabilities": {grade: .82}},
            "primary_issue": {"choice": issue, "confidence": .99},
            "feedback_warranted": {"noul": .05},
        }, skipped_questions={}, usage=None, error=None)


@pytest.mark.parametrize("grade", ["0", "1", "2", "3"])
def test_insufficient_evidence_withholds_negative_and_positive_grades(grade):
    state, result = state_for(), judgement(grade=grade)
    before = deepcopy(result.to_dict())
    report = build_coaching_report(state, result, None)
    assert report.overall_assessment.startswith("Pose measurements passed")
    assert "22.3" in report.overall_assessment
    assert "6.2" in report.overall_assessment
    assert "24.6" in report.overall_assessment
    assert "not confirmed contact" in report.overall_assessment
    assert "technique grade is withheld" in report.overall_assessment
    assert "82%" not in json.dumps(report.to_dict())
    assert "instability" not in report.overall_assessment
    assert "controlled technical execution" not in report.what_went_well
    # Existing repeatability cue remains a candidate; the controller still
    # decides whether any speech occurs. Measurements never enter speech.
    assert "same approach" in report.spoken_cue
    assert "degrees" not in report.spoken_cue
    assert result.to_dict() == before


def test_local_measurement_limits_do_not_leak_angles():
    state = state_for()
    state.strike_window = {"status": "limited", "reasons": ["Missing pose frames."]}
    report = build_coaching_report(state, judgement(), None)
    assert "Missing pose frames" in report.overall_assessment
    assert "22.3" not in report.overall_assessment


@pytest.mark.parametrize("coverage", [None, float("nan"), float("inf"), -.1, 1.1])
def test_invalid_coverage_is_not_formatted_as_measurement(coverage):
    state = state_for()
    state.pose_coverage = coverage
    state.strike_window = None
    report = build_coaching_report(state, judgement(), None)
    assert "Reliable pose coverage is not available" in report.overall_assessment


def test_observed_outcome_remains_separate_from_withheld_technique_grade():
    state = state_for()
    state.goal_entry_detected = True
    report = build_coaching_report(state, judgement(), None)
    assert "observed entering the goal" in report.outcome_analysis
    assert "technique grade is withheld" in report.overall_assessment


def test_unknown_outcome_and_declared_curl_do_not_create_measurements():
    state = state_for()
    state.training_intention = {"shot_type": "curl"}
    report = build_coaching_report(state, judgement(), None)
    assert "Training intention: curl" in report.overall_assessment
    assert "curl set" in report.next_rep_recommendation
    assert "enough reliable ball" in report.outcome_analysis
    assert state.goal_entry_detected is None


@pytest.mark.parametrize("status", ["error", "skipped"])
def test_failed_jev_cannot_leave_stale_quality_label(status):
    state, result = state_for(), judgement(issue="none", grade="3", status=status)
    report = build_coaching_report(state, result, None)
    assert "technique grade is withheld" in report.overall_assessment
    assert "82%" not in json.dumps(report.to_dict())


def test_language_model_cannot_restore_uncertain_grade(monkeypatch):
    state, result = state_for(), judgement()
    report = build_coaching_report(state, result, None)
    def forbidden(*args, **kwargs):
        raise AssertionError("No language request should be sent for this measurement report")
    monkeypatch.setattr("src.backend.llm_coaching.urlopen", forbidden)
    refined, meta = refine_coaching_report_with_llm(
        situated_state=state, jev_judgements=result.to_dict(),
        deterministic_report=report, api_key="test-placeholder",
    )
    assert refined is report
    assert meta["used_llm"] is False
    assert meta["reason"] == "measurement_report_technique_grade_withheld"
    assert meta["error"] is None


def test_supported_path_still_uses_synthesis_and_keeps_failure_fallback(monkeypatch):
    state, result = state_for(), judgement(issue="none", grade="2")
    report = build_coaching_report(state, result, None)
    assert technique_grade_available(state, result.to_dict())
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"output": [{"content": [{"type": "output_text", "text": json.dumps(report.to_dict())}]}]}).encode()
    def respond(*args, **kwargs):
        calls.append(True)
        return Response()
    monkeypatch.setattr("src.backend.llm_coaching.urlopen", respond)
    refined, meta = refine_coaching_report_with_llm(situated_state=state,
        jev_judgements=result.to_dict(), deterministic_report=report, api_key="test-placeholder")
    assert calls == [True]
    assert refined == report
    assert meta["used_llm"] is True
    def fail(*args, **kwargs): raise TimeoutError("test timeout")
    monkeypatch.setattr("src.backend.llm_coaching.urlopen", fail)
    refined, meta = refine_coaching_report_with_llm(situated_state=state,
        jev_judgements=result.to_dict(), deterministic_report=report, api_key="test-placeholder")
    assert refined is report
    assert meta["used_llm"] is False


def test_unusable_perception_withholds_grade_despite_positive_jev_answer():
    state = state_for()
    state.perception_usable = False
    assert not technique_grade_available(state, judgement(issue="none", grade="3").to_dict())
