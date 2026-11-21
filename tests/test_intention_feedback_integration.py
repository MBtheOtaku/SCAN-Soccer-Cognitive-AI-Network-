"""Task context crosses reasoning layers without creating shot evidence."""

from dataclasses import MISSING, fields
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import sys
import types

import pytest

from src.backend.coaching_report import CoachingReport, build_coaching_report
from src.backend.jev_judgements import JevJudgementResult, _build_jev_state
from src.backend.llm_coaching import _build_evidence_packet, refine_coaching_report_with_llm
from src.backend.situated_state import SituatedState
from src.backend.training_intention import build_intention_feedback, parse_training_intention


def state_for(shot_type, reported_outcome="goal"):
    state = SituatedState(**{
        field.name: None for field in fields(SituatedState) if field.default is MISSING
    })
    state.rep_id = 1
    state.training_intention = parse_training_intention({"shot_type": shot_type, "reported_outcome": reported_outcome})
    state.intention_feedback = build_intention_feedback(state.training_intention)
    return state


@pytest.mark.parametrize("shot_type", ["power", "curl", "placement"])
def test_external_failure_keeps_intention_specific_written_report(monkeypatch, shot_type):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state = state_for(shot_type)
    result = JevJudgementResult(status="error", model="test", answers={}, error="offline",
                                shadow_mode=True, skipped_questions={}, usage=None)
    report = build_coaching_report(state, result, SimpleNamespace())
    assert f"Training intention: {shot_type}." in report.overall_assessment
    assert shot_type in report.next_rep_recommendation
    refined, synthesis = refine_coaching_report_with_llm(
        situated_state=state, jev_judgements=result.to_dict(), deterministic_report=report,
    )
    assert refined == report
    assert synthesis["used_llm"] is False
    # Declared goal is never promoted into machine-observed outcome.
    assert state.goal_entry_detected is None
    assert state.shot_on_target is None
    assert all(metric["value"] is None for metric in state.intention_feedback["metrics"])


def test_declared_context_reaches_jev_and_language_without_changing_policy():
    state = state_for("curl")
    report = CoachingReport("assessment", "observation", None, "unknown", "next rep", None, None)
    packet = _build_evidence_packet(state, {}, report)
    evidence = _build_jev_state(state, None, None)
    assert packet["situated_state"]["training_intention"]["shot_type"] == "curl"
    assert packet["situated_state"]["goal_entry_detected"] is None
    assert evidence["rep"]["intention_feedback"]["reference"]["status"] == "not_comparable"
    assert evidence["rep"]["training_intention"]["reported_outcome"] == "goal"
    assert not any("policy" in key for key in packet)


def test_pose_adapter_never_substitutes_good_coverage_for_metric_ball_evidence(monkeypatch):
    # Load the real adapter without importing the optional MediaPipe runtime.
    monkeypatch.setitem(sys.modules, "mediapipe", types.ModuleType("mediapipe"))
    name = "intention_pose_adapter_test"
    source = Path(__file__).resolve().parents[1] / "src/backend/pose_mvp/scan_pose_mvp_v2.py"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    summary = SimpleNamespace(
        probable_striking_leg="left", pose_coverage=1.0,
        probable_strike_frame=90, probable_strike_time_s=3.0,
        strike_torso_lean_deg=30, strike_left_knee_angle_deg=110, strike_right_knee_angle_deg=140,
    )
    ball = SimpleNamespace(detected=True, mean_confidence=.99, detection_rate=1.0,
                           final_x_norm=.4, final_y_norm=.3)
    goal = SimpleNamespace(shot_on_target=None, goal_entry_detected=None, goal_zone=None)
    intention = {"shot_type": "power", "reported_outcome": "goal"}
    state = module.build_situated_state(summary, ball, goal, training_intention=intention)
    assert state.perception_usable is True
    assert state.task_goal == "practise shooting for ball speed"
    assert state.goal_entry_detected is None
    assert state.strike_time_s == 3.0
    assert all(metric["status"] == "unavailable" for metric in state.intention_feedback["metrics"])
    legacy = module.build_situated_state(summary, ball, goal)
    assert legacy.task_goal == "execute a controlled shot"
    assert legacy.training_intention["shot_type"] == "unspecified"
