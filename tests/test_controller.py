"""
test_controller.py
Basic unit tests for the SCAN decision loop. Run with: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from player_state import PlayerState, RepFeatures
from controller import decide


def test_hazard_override():
    state = PlayerState()
    rep = RepFeatures(rep_id=1, timestamp=0.0, is_hazard=True)
    decision = decide(state, rep)
    assert decision.should_speak is True
    assert decision.decision_type == "hazard"


def test_silence_mid_action():
    state = PlayerState()
    rep = RepFeatures(rep_id=1, timestamp=0.0, mid_action=True)
    decision = decide(state, rep)
    assert decision.should_speak is False
    assert decision.decision_type == "silence_protect"


def test_repeated_error_triggers_cue():
    state = PlayerState()
    for i in range(3):
        rep = RepFeatures(rep_id=i, timestamp=float(i) * 10, error_type="poor_first_touch")
        state.add_rep(rep)
    decision = decide(state, RepFeatures(rep_id=3, timestamp=30.0, error_type="poor_first_touch"))
    assert decision.decision_type == "repeated_error"
    assert decision.should_speak is True


def test_default_silence():
    state = PlayerState()
    state.last_intervention_time = 0.0
    rep = RepFeatures(rep_id=1, timestamp=100.0)
    decision = decide(state, rep)
    assert decision.decision_type in ("silence_default", "end_of_rep")
