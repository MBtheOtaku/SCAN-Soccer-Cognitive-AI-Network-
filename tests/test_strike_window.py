import json
from dataclasses import MISSING, fields
from types import SimpleNamespace

from src.backend.strike_window import build_strike_window, describe_strike_window


def test_synthesis_receives_window_as_object_not_array():
    from src.backend.coaching_report import CoachingReport
    from src.backend.llm_coaching import _build_evidence_packet
    from src.backend.situated_state import SituatedState

    state = SituatedState(**{
        field.name: None for field in fields(SituatedState)
        if field.default is MISSING
    })
    state.strike_window = {"status": "limited", "reasons": ["Missing pose frames."]}
    report = CoachingReport("", "", None, "", "", None, None)
    packet = _build_evidence_packet(state, {}, report)
    serialized = json.loads(json.dumps(packet, allow_nan=False))
    assert serialized["situated_state"]["strike_window"] == state.strike_window


def samples(frames=range(100), confidence=0.9):
    return [SimpleNamespace(
        frame_idx=i,
        pose_confidence=confidence,
        torso_lean_deg=10.0 if i < 47 else 20.0 if i <= 53 else 30.0,
    ) for i in frames]


def test_complete_window_has_three_phases_and_no_diagnosis():
    evidence = build_strike_window(samples(), 50, 30.0, 100)
    assert evidence["status"] == "usable"
    assert evidence["expected_frames"] == 19
    assert evidence["coverage"] == 1.0
    assert [evidence["phases"][p]["torso_median_deg"]
            for p in ("before", "near", "after")] == [10, 20, 30]
    assert "not confirmed contact" in describe_strike_window(evidence)
    json.dumps(evidence, allow_nan=False)


def test_local_gap_is_not_hidden_by_whole_clip_coverage():
    evidence = build_strike_window(samples([i for i in range(100)
                                           if not 46 <= i <= 54]), 50, 30.0, 100)
    assert evidence["status"] == "limited"
    assert evidence["largest_missing_run_frames"] == 9
    assert evidence["phases"]["near"]["torso_median_deg"] is None


def test_clip_boundary_is_flagged():
    evidence = build_strike_window(samples(), 2, 30.0, 100)
    assert evidence["status"] == "limited"
    assert any("beyond the clip" in reason for reason in evidence["reasons"])


def test_low_confidence_and_nonfinite_measurements_are_not_usable():
    assert build_strike_window(samples(confidence=0.4), 50, 30, 100)["status"] == "limited"
    rows = samples()
    for row in rows:
        row.torso_lean_deg = float("nan")
    evidence = build_strike_window(rows, 50, 30, 100)
    assert evidence["status"] == "unavailable"
    json.dumps(evidence, allow_nan=False)


def test_missing_anchor_bad_fps_and_empty_input_are_safe():
    for anchor, fps in ((None, 30), (50, 0), (50, float("nan"))):
        assert build_strike_window(samples(), anchor, fps, 100)["status"] == "unavailable"
    assert build_strike_window([], 50, 30, 100)["status"] == "unavailable"
    assert describe_strike_window(None) is None


def test_duplicate_frames_do_not_inflate_coverage():
    rows = samples()
    evidence = build_strike_window(rows + rows, 50, 30, 100)
    assert evidence["coverage"] == 1.0
    assert evidence["valid_frames"] == 19


def test_frame_rate_changes_window_size():
    evidence = build_strike_window(samples(range(200)), 100, 60, 200)
    assert evidence["expected_frames"] == 37
    assert evidence["half_window_s"] == 0.3
