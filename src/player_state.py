"""
player_state.py
Tracks a player's real-time state during a training session: rep history,
error counts, scan rate, and intervention history. Consumed by controller.py
to decide when/what feedback to give.
"""

from dataclasses import dataclass, field
from time import time
from typing import List, Optional


@dataclass
class RepFeatures:
    """Extracted features for a single repetition (from CV pipeline)."""
    rep_id: int
    timestamp: float
    error_type: Optional[str] = None       # e.g. "poor_first_touch", "off_target_shot"
    is_hazard: bool = False                # e.g. dangerous technique / injury risk
    mid_action: bool = False               # True if player is currently executing a rep
    scan_rate: float = 0.0                 # proxy for visual attention / scanning behavior
    confidence: float = 1.0                # CV detection confidence


@dataclass
class PlayerState:
    """Rolling state object updated after each processed rep."""
    reps: List[RepFeatures] = field(default_factory=list)
    error_counts: dict = field(default_factory=dict)
    intervention_history: List[dict] = field(default_factory=list)
    last_intervention_time: Optional[float] = None

    def add_rep(self, rep: RepFeatures) -> None:
        self.reps.append(rep)
        if rep.error_type:
            self.error_counts[rep.error_type] = self.error_counts.get(rep.error_type, 0) + 1

    def log_intervention(self, decision_type: str, message: str) -> None:
        now = time()
        self.intervention_history.append({
            "timestamp": now,
            "type": decision_type,
            "message": message,
        })
        self.last_intervention_time = now

    def time_since_last_intervention(self) -> float:
        if self.last_intervention_time is None:
            return float("inf")
        return time() - self.last_intervention_time

    def repeated_error_streak(self, error_type: str, window: int = 3) -> int:
        """Count consecutive reps (most recent first) with the same error_type."""
        streak = 0
        for rep in reversed(self.reps[-window:]):
            if rep.error_type == error_type:
                streak += 1
            else:
                break
        return streak

    def total_reps(self) -> int:
        return len(self.reps)

    def total_interventions(self) -> int:
        return len(self.intervention_history)
