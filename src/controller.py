"""
controller.py
Priority-ordered decision loop (behavior tree) for SCAN.

Priority order:
  1. Hazard override        - always interrupt, safety-critical
  2. Protect germane load    - stay silent if mid-action or too soon after last cue
  3. Repeated-error summary  - flag a recurring technical error
  4. End-of-rep cue          - lightweight default feedback at rep boundary
  5. Default silence         - withhold feedback (correct output, not a failure)

Each call to decide() returns a Decision object that controller.py's caller
can pass to the LLM feedback module to phrase as natural language.
"""

from dataclasses import dataclass
from typing import Optional

from player_state import PlayerState, RepFeatures

MIN_SECONDS_BETWEEN_INTERVENTIONS = 8.0
REPEATED_ERROR_THRESHOLD = 2


@dataclass
class Decision:
    decision_type: str          # "hazard", "silence_protect", "repeated_error", "end_of_rep", "silence_default"
    should_speak: bool
    reason: str
    error_type: Optional[str] = None


def decide(state: PlayerState, rep: RepFeatures) -> Decision:
    # 1. Hazard override - always speak, regardless of load
    if rep.is_hazard:
        return Decision(
            decision_type="hazard",
            should_speak=True,
            reason="Detected hazardous technique — safety-critical override.",
            error_type=rep.error_type,
        )

    # 2. Protect germane load - don't interrupt mid-action or too soon after last cue
    if rep.mid_action:
        return Decision(
            decision_type="silence_protect",
            should_speak=False,
            reason="Player mid-action; interrupting would add extraneous load.",
        )
    if state.time_since_last_intervention() < MIN_SECONDS_BETWEEN_INTERVENTIONS:
        return Decision(
            decision_type="silence_protect",
            should_speak=False,
            reason="Too soon after last intervention; protecting cognitive load.",
        )

    # 3. Repeated-error summary cue
    if rep.error_type:
        streak = state.repeated_error_streak(rep.error_type)
        if streak >= REPEATED_ERROR_THRESHOLD:
            return Decision(
                decision_type="repeated_error",
                should_speak=True,
                reason=f"Repeated '{rep.error_type}' error ({streak}x) — summary cue warranted.",
                error_type=rep.error_type,
            )

    # 4. End-of-rep cue (lightweight, default check-in)
    if rep.error_type is None and state.total_reps() % 5 == 0:
        return Decision(
            decision_type="end_of_rep",
            should_speak=True,
            reason="Rep milestone reached with no active error; light check-in cue.",
        )

    # 5. Default: stay silent
    return Decision(
        decision_type="silence_default",
        should_speak=False,
        reason="No condition met; silence is the correct output.",
    )
