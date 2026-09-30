from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class SituatedState:
    """
    Contextual representation of one SCAN training rep.

    This sits between low-level perception and higher-level
    judgment/controller logic.

    The goal is to represent not only what the player's body
    is doing, but what they are doing in the context of the task.
    """

    # -----------------------------------------------------
    # Rep / task context
    # -----------------------------------------------------

    rep_id: int

    drill_type: str
    task_goal: str

    # Example:
    # "approach", "plant", "contact", "follow_through",
    # "rep_complete", "unknown"
    action_phase: str

    # -----------------------------------------------------
    # Perception quality
    # -----------------------------------------------------

    pose_coverage: float

    # Whether SCAN believes the available perception is
    # reliable enough to reason about.
    perception_usable: bool

    # -----------------------------------------------------
    # Strike context
    # -----------------------------------------------------

    striking_leg: Optional[str]
    support_leg: Optional[str]

    strike_frame: Optional[int]
    strike_time_s: Optional[float]

    # -----------------------------------------------------
    # Body state
    # -----------------------------------------------------

    torso_lean_deg: Optional[float]

    left_knee_deg: Optional[float]
    right_knee_deg: Optional[float]

    # -----------------------------------------------------
    # Ball / environment interaction
    # -----------------------------------------------------

    # True when the ball was detected during this rep.
    ball_visible: bool

    # Confidence of the ball tracking system.
    ball_tracking_confidence: Optional[float]

    # Fraction of video frames containing a ball detection.
    ball_detection_rate: Optional[float]

    # Normalized final observed ball position in the video.
    # These are frame-relative for now, not goal-relative.
    ball_final_x_norm: Optional[float]
    ball_final_y_norm: Optional[float]

    # -----------------------------------------------------
    # Shot outcome
    # -----------------------------------------------------

    # These remain unknown until goal tracking is added.
    shot_on_target: Optional[bool]

    goal_entry_detected: Optional[bool]

    # Eventually:
    # "upper_left", "upper_right",
    # "lower_left", "lower_right",
    # "center", "over_bar", etc.
    goal_zone: Optional[str]

    # General confidence in environment/outcome reasoning.
    environment_confidence: Optional[float]
    
    # -----------------------------------------------------
    # Player history
    # -----------------------------------------------------

    repeated_error_type: Optional[str]
    repeated_error_count: int

    seconds_since_last_feedback: Optional[float]

    previous_feedback_type: Optional[str]

    # Later this can be determined from multiple reps.
    recent_improvement: Optional[bool]

    # -----------------------------------------------------
    # Utility
    # -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Convert the situated state into a JSON-friendly dictionary.

        This will also make it easy to send the same state
        to Jev later.
        """
        return asdict(self)