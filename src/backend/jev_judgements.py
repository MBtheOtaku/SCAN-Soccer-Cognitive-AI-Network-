from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.backend.situated_state import SituatedState


# =========================================================
# TYPESAFE / JEV CONFIGURATION
# =========================================================

TYPESAFE_SYSTEMONE_URL = (
    "https://api.typesafe.ai/v1/systemone"
)

DEFAULT_JEV_MODEL = "jev-latest"


# =========================================================
# RESULT TYPE
# =========================================================


@dataclass
class JevJudgmentResult:
    """
    Shadow-mode Jev result.

    These judgments are logged for inspection only.

    They do NOT currently:
    - override the SCAN controller
    - trigger speech
    - generate coaching language
    """

    status: str

    shadow_mode: bool

    model: Optional[str]

    answers: dict[str, Any]

    skipped_questions: dict[str, str]

    usage: Optional[dict[str, Any]]

    error: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


# =========================================================
# STATE CONSTRUCTION
# =========================================================


def _compact_ball_evidence(
    ball_tracking: Optional[
        dict[str, Any]
    ],
) -> Optional[dict[str, Any]]:
    """
    Send Jev the useful ball-tracking summary,
    NOT the entire frame-by-frame trajectory.

    This keeps the reasoning state small and makes
    evidence quality explicit.
    """

    if not ball_tracking:
        return None

    useful_fields = (
        "detected",
        "total_frames",
        "detection_count",
        "detection_rate",
        "reacquired_count",
        "tentative_count",
        "predicted_count",
        "mean_confidence",
        "peak_confidence",
        "first_frame",
        "last_frame",
        "tracking_end_frame",
        "longest_prediction_gap",
    )

    compact = {
        key: ball_tracking.get(key)
        for key in useful_fields
        if key in ball_tracking
    }

    return compact


def _compact_goal_evidence(
    goal_tracking: Optional[
        dict[str, Any]
    ],
) -> Optional[dict[str, Any]]:
    """
    Goal outcome evidence supplied to Jev.

    Keep the goal-relative coordinates because
    they are useful for judging shot placement.
    """

    if not goal_tracking:
        return None

    useful_fields = (
        "goal_calibrated",
        "goal_entry_detected",
        "shot_on_target",
        "entry_x_norm",
        "entry_y_norm",
        "entry_frame",
        "entry_time_s",
        "goal_zone",
        "confidence",
    )

    compact = {
        key: goal_tracking.get(key)
        for key in useful_fields
        if key in goal_tracking
    }

    return compact


def _build_jev_state(
    situated_state: SituatedState,
    ball_tracking: Optional[
        dict[str, Any]
    ],
    goal_tracking: Optional[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    """
    Build the structured state Jev reasons over.

    Jev receives derived SCAN state rather than
    raw video.

    This keeps perception and reasoning separated.
    """

    return {
        "system": "SCAN",
        "task": "evaluate one soccer shooting rep",

        "rep": (
            situated_state.to_dict()
        ),

        "ball_tracking_evidence": (
            _compact_ball_evidence(
                ball_tracking
            )
        ),

        "goal_tracking_evidence": (
            _compact_goal_evidence(
                goal_tracking
            )
        ),

        # Important epistemic rules for our workflow.
        "evidence_rules": [
            (
                "Do not infer measurements or events "
                "that are absent from the supplied state."
            ),
            (
                "Predicted ball trajectory points are "
                "hypotheses and are weaker than visual "
                "or detector observations."
            ),
            (
                "Tentative ball detections are weaker "
                "than detected, visual, or confirmed "
                "reacquired observations."
            ),
            (
                "A successful shot outcome does not "
                "automatically imply good technique."
            ),
            (
                "Good technique does not automatically "
                "imply a successful shot outcome."
            ),
            (
                "Prefer insufficient evidence over a "
                "specific technical diagnosis when the "
                "available perception does not support it."
            ),
        ],
    }


# =========================================================
# QUESTION CONSTRUCTION
# =========================================================


def _outcome_available(
    situated_state: SituatedState,
) -> bool:
    """
    Do we have enough explicit outcome evidence to ask
    Jev to judge shot outcome quality?
    """

    return any(
        value is not None
        for value in (
            situated_state.shot_on_target,
            situated_state.goal_entry_detected,
            situated_state.goal_zone,
        )
    )


def _build_questions(
    situated_state: SituatedState,
) -> tuple[
    dict[str, Any],
    dict[str, str],
]:
    """
    Construct narrow typed judgments.

    Some questions are omitted when SCAN does not
    actually have the evidence required to answer them.
    """

    questions: dict[
        str,
        Any,
    ] = {}

    skipped: dict[
        str,
        str,
    ] = {}

    # -----------------------------------------------------
    # TECHNIQUE QUALITY
    # -----------------------------------------------------

    if situated_state.perception_usable:

        questions[
            "technique_quality"
        ] = {
            "type": "score",

            "instructions": (
                "Rate the quality of the player's "
                "technical execution for this shooting "
                "rep using only the supplied body and "
                "perception evidence. Do not invent "
                "missing biomechanics."
            ),

            "criteria": [
                (
                    "Poor: available evidence supports "
                    "a substantial technical problem "
                    "that likely reduced control."
                ),

                (
                    "Mixed: some execution appears "
                    "functional but there is meaningful "
                    "technical instability or uncertainty."
                ),

                (
                    "Solid: execution is generally "
                    "controlled and the available "
                    "evidence shows no major technical "
                    "problem."
                ),

                (
                    "Strong: available evidence clearly "
                    "supports controlled, repeatable "
                    "technical execution with no "
                    "important correction apparent."
                ),
            ],
        }

    else:

        skipped[
            "technique_quality"
        ] = (
            "Perception was marked unusable."
        )

    # -----------------------------------------------------
    # OUTCOME QUALITY
    # -----------------------------------------------------

    if _outcome_available(
        situated_state
    ):

        questions[
            "outcome_quality"
        ] = {
            "type": "score",

            "instructions": (
                "Rate the quality of the shot outcome "
                "using the supplied ball and goal "
                "evidence. Evaluate the result of the "
                "shot, not the player's body technique."
            ),

            "criteria": [
                (
                    "Poor: the available outcome evidence "
                    "supports an ineffective result such "
                    "as a clear miss or poor placement."
                ),

                (
                    "Acceptable: the shot outcome is "
                    "functional or on target but placement "
                    "is not especially dangerous."
                ),

                (
                    "Strong: the shot produces a good "
                    "goal result or threatening placement, "
                    "such as a side goal zone."
                ),

                (
                    "Excellent: the available evidence "
                    "clearly supports highly effective, "
                    "precise corner-oriented placement."
                ),
            ],
        }

    else:

        skipped[
            "outcome_quality"
        ] = (
            "No explicit shot outcome is available."
        )

    # -----------------------------------------------------
    # PRIMARY ISSUE
    # -----------------------------------------------------

    questions[
        "primary_issue"
    ] = {
        "type": "choice",

        "instructions": (
            "Choose the single most useful primary "
            "coaching issue supported by the current "
            "rep. Choose insufficient_evidence rather "
            "than guessing. Choose none when the rep "
            "does not clearly require correction."
        ),

        "criteria": {
            "none": (
                "No specific correction is clearly "
                "supported by the available evidence."
            ),

            "support_stability": (
                "The support side, plant side, or body "
                "stability through contact is the "
                "clearest issue."
            ),

            "upper_body_control": (
                "Torso or upper-body control is the "
                "clearest technical issue."
            ),

            "strike_timing": (
                "Timing or coordination around the "
                "strike/contact event is the clearest "
                "issue."
            ),

            "accuracy": (
                "Horizontal placement or general shot "
                "accuracy is the clearest issue."
            ),

            "shot_height": (
                "Vertical ball placement or controlling "
                "shot height is the clearest issue."
            ),

            "insufficient_evidence": (
                "SCAN does not currently have enough "
                "reliable evidence to identify a "
                "specific primary issue."
            ),
        },
    }

    # -----------------------------------------------------
    # FEEDBACK WARRANTED
    # -----------------------------------------------------

    questions[
        "feedback_warranted"
    ] = {
        "type": "noul",

        "instructions": (
            "Is there enough reliable evidence in this "
            "rep to justify giving the player a specific "
            "corrective coaching cue?"
        ),

        "criteria": {
            "true": (
                "A specific actionable correction is "
                "supported strongly enough that feedback "
                "would likely help the next repetition."
            ),

            "false": (
                "Evidence is too uncertain, the rep is "
                "already sufficiently strong, or there "
                "is no specific actionable correction "
                "supported by the state."
            ),
        },
    }

    return (
        questions,
        skipped,
    )


# =========================================================
# API CALL
# =========================================================


def run_jev_judgments(
    situated_state: SituatedState,

    ball_tracking: Optional[
        dict[str, Any]
    ] = None,

    goal_tracking: Optional[
        dict[str, Any]
    ] = None,

    api_key: Optional[str] = None,

    model: Optional[str] = None,

    timeout_s: float = 5.0,
) -> JevJudgmentResult:
    """
    Run Jev in SHADOW MODE.

    Failure to reach Jev must never cause the normal
    SCAN video analysis pipeline to fail.
    """

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    api_key = (
        api_key
        or os.getenv(
            "TYPESAFE_API_KEY"
        )
    )

    model = (
        model
        or os.getenv(
            "TYPESAFE_MODEL",
            DEFAULT_JEV_MODEL,
        )
    )

    if not api_key:

        return JevJudgmentResult(
            status="skipped",

            shadow_mode=True,

            model=None,

            answers={},

            skipped_questions={},

            usage=None,

            error=(
                "TYPESAFE_API_KEY is not set."
            ),
        )

    # -----------------------------------------------------
    # Build state + typed questions
    # -----------------------------------------------------

    state = _build_jev_state(
        situated_state=(
            situated_state
        ),

        ball_tracking=(
            ball_tracking
        ),

        goal_tracking=(
            goal_tracking
        ),
    )

    questions, skipped = (
        _build_questions(
            situated_state
        )
    )

    payload = {
        "state": state,
        "model": model,
        "questions": questions,
    }

    request_data = (
        json.dumps(
            payload
        )
        .encode(
            "utf-8"
        )
    )

    request = Request(
        TYPESAFE_SYSTEMONE_URL,

        data=request_data,

        headers={
            "Authorization": (
                f"Bearer {api_key}"
            ),

            "Content-Type": (
                "application/json"
            ),

            "Accept": (
                "application/json"
            ),
        },

        method="POST",
    )

    # -----------------------------------------------------
    # Call Jev
    # -----------------------------------------------------

    try:

        with urlopen(
            request,
            timeout=timeout_s,
        ) as response:

            response_body = (
                response.read()
                .decode(
                    "utf-8"
                )
            )

        data = json.loads(
            response_body
        )

        return JevJudgmentResult(
            status="ok",

            shadow_mode=True,

            model=data.get(
                "model"
            ),

            answers=data.get(
                "answers",
                {},
            ),

            skipped_questions=(
                skipped
            ),

            usage=data.get(
                "usage"
            ),

            error=None,
        )

    # -----------------------------------------------------
    # API returned an error response
    # -----------------------------------------------------

    except HTTPError as exc:

        try:
            error_body = (
                exc.read()
                .decode(
                    "utf-8"
                )
            )

        except Exception:
            error_body = (
                str(exc)
            )

        return JevJudgmentResult(
            status="error",

            shadow_mode=True,

            model=model,

            answers={},

            skipped_questions=(
                skipped
            ),

            usage=None,

            error=(
                f"TypeSafe HTTP "
                f"{exc.code}: "
                f"{error_body}"
            ),
        )

    # -----------------------------------------------------
    # Network failure
    # -----------------------------------------------------

    except URLError as exc:

        return JevJudgmentResult(
            status="error",

            shadow_mode=True,

            model=model,

            answers={},

            skipped_questions=(
                skipped
            ),

            usage=None,

            error=(
                "Could not reach TypeSafe: "
                f"{exc.reason}"
            ),
        )

    # -----------------------------------------------------
    # Anything unexpected
    # -----------------------------------------------------

    except Exception as exc:

        return JevJudgmentResult(
            status="error",

            shadow_mode=True,

            model=model,

            answers={},

            skipped_questions=(
                skipped
            ),

            usage=None,

            error=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )