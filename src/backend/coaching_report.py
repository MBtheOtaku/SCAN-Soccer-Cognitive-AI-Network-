from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from src.backend.situated_state import SituatedState
from src.backend.jev_judgements import JevJudgementResult


# =========================================================
# COACHING REPORT
# =========================================================


@dataclass
class CoachingReport:
    """
    Human-facing interpretation of one SCAN rep.

    This layer sits AFTER perception and Jev reasoning.

    It does not decide whether SCAN is allowed to speak.
    The controller still owns intervention policy.

    The report provides:
    - detailed post-rep analysis for the frontend
    - a concise spoken cue candidate
    """

    overall_assessment: str

    what_went_well: str

    primary_focus: Optional[str]

    outcome_analysis: str

    next_rep_recommendation: str

    spoken_cue: Optional[str]

    confidence_note: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


# =========================================================
# DISPLAY LABELS
# =========================================================


ISSUE_LABELS = {
    "support_stability": "Support-side stability",
    "upper_body_control": "Upper-body control",
    "strike_timing": "Strike timing",
    "accuracy": "Shot accuracy",
    "shot_height": "Shot height",
    "none": "No specific correction",
    "insufficient_evidence": "Insufficient evidence",
}


ISSUE_RECOMMENDATIONS = {
    "support_stability": (
        "On the next rep, keep the support side controlled "
        "through contact and see whether that gives you a "
        "more stable finish."
    ),

    "upper_body_control": (
        "On the next rep, keep the upper body controlled "
        "through contact without forcing the movement."
    ),

    "strike_timing": (
        "On the next rep, focus on arriving at the ball in "
        "rhythm and making clean contact rather than changing "
        "several technical details at once."
    ),

    "accuracy": (
        "On the next rep, keep the same general technique "
        "but place more attention on the intended target area."
    ),

    "shot_height": (
        "On the next rep, keep the approach consistent and "
        "focus on controlling the vertical placement of the ball."
    ),
}


ISSUE_SPOKEN_CUES = {
    "support_stability": (
        "Stay a little stronger through your support side "
        "on the next one."
    ),

    "upper_body_control": (
        "Keep the upper body controlled through contact."
    ),

    "strike_timing": (
        "Stay in rhythm and meet the ball cleanly."
    ),

    "accuracy": (
        "Keep the same approach and focus on your target."
    ),

    "shot_height": (
        "Keep the approach and bring the finish under control."
    ),
}


# =========================================================
# BASIC HELPERS
# =========================================================


def _safe_float(
    value: Any,
) -> Optional[float]:

    try:

        if value is None:
            return None

        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def _percentage(
    value: Optional[float],
) -> Optional[int]:

    if value is None:
        return None

    return int(
        round(
            value * 100
        )
    )


def _get_answer(
    jev_result: JevJudgementResult,
    name: str,
) -> dict[str, Any]:

    answer = (
        jev_result.answers.get(
            name,
            {},
        )
    )

    if isinstance(
        answer,
        dict,
    ):
        return answer

    return {}


def _score_label(
    answer: dict[str, Any],
    labels: list[str],
) -> tuple[
    Optional[str],
    Optional[float],
    Optional[float],
]:
    """
    Convert a Jev score answer into:

        label
        label probability
        expected score
    """

    probabilities = answer.get(
        "probabilities",
        {},
    )

    score = _safe_float(
        answer.get(
            "score"
        )
    )

    if not isinstance(
        probabilities,
        dict,
    ):
        return (
            None,
            None,
            score,
        )

    best_index: Optional[int] = None
    best_probability = -1.0

    for key, value in (
        probabilities.items()
    ):

        try:

            index = int(key)
            probability = float(value)

        except (
            ValueError,
            TypeError,
        ):
            continue

        if (
            0 <= index < len(labels)
            and probability
            > best_probability
        ):

            best_index = index
            best_probability = (
                probability
            )

    if best_index is None:

        return (
            None,
            None,
            score,
        )

    return (
        labels[best_index],
        best_probability,
        score,
    )


def _choice_value(
    answer: dict[str, Any],
) -> tuple[
    Optional[str],
    Optional[float],
]:

    choice = answer.get(
        "choice"
    )

    confidence = _safe_float(
        answer.get(
            "confidence"
        )
    )

    if not isinstance(
        choice,
        str,
    ):
        choice = None

    return (
        choice,
        confidence,
    )


def _noul_probability(
    answer: dict[str, Any],
) -> Optional[float]:

    return _safe_float(
        answer.get(
            "noul"
        )
    )


# =========================================================
# GOAL / OUTCOME HELPERS
# =========================================================


def _goal_zone_phrase(
    zone: Optional[str],
) -> Optional[str]:

    if not zone:
        return None

    phrases = {
        "upper_left": "the upper-left area of the goal",
        "upper_center": "the upper-central area of the goal",
        "upper_right": "the upper-right area of the goal",

        "lower_left": "the lower-left area of the goal",
        "lower_center": "the lower-central area of the goal",
        "lower_right": "the lower-right area of the goal",
    }

    return phrases.get(
        zone,
        zone.replace(
            "_",
            " ",
        ),
    )


def _build_outcome_analysis(
    state: SituatedState,
    outcome_label: Optional[str],
    outcome_probability: Optional[float],
) -> str:

    zone_phrase = (
        _goal_zone_phrase(
            state.goal_zone
        )
    )

    probability_pct = (
        _percentage(
            outcome_probability
        )
    )

    # -----------------------------------------------------
    # Confirmed goal
    # -----------------------------------------------------

    if (
        state.goal_entry_detected
        is True
    ):

        if zone_phrase:

            base = (
                "The ball was observed entering "
                f"{zone_phrase}."
            )

        else:

            base = (
                "The ball was observed entering the goal."
            )

        if outcome_label in {
            "strong",
            "excellent",
        }:

            base += (
                " That supports a high-quality shot outcome "
                "rather than judging the rep from body form alone."
            )

        elif (
            outcome_label
            is not None
        ):

            base += (
                " The finish was successful, although the "
                "overall outcome evidence was more moderate."
            )

        return base

    # -----------------------------------------------------
    # Explicitly on target
    # -----------------------------------------------------

    if state.shot_on_target is True:

        return (
            "The shot was assessed as on target. "
            "The available evidence supports an effective "
            "result, although SCAN does not have a confirmed "
            "goal-entry location for this rep."
        )

    # -----------------------------------------------------
    # Explicit miss
    # -----------------------------------------------------

    if state.shot_on_target is False:

        return (
            "The shot outcome was not on target. "
            "That matters even if parts of the technical "
            "execution looked controlled, because shot quality "
            "depends on both movement and result."
        )

    # -----------------------------------------------------
    # Jev outcome exists but goal evidence is incomplete
    # -----------------------------------------------------

    if outcome_label:

        confidence_phrase = ""

        if probability_pct is not None:

            confidence_phrase = (
                f" with about {probability_pct}% support"
            )

        return (
            "The outcome evidence was judged as "
            f"{outcome_label}{confidence_phrase}, but SCAN "
            "does not have enough confirmed goal information "
            "to describe the exact finish with confidence."
        )

    return (
        "SCAN does not currently have enough reliable ball "
        "and goal evidence to make a specific claim about "
        "the final shot outcome."
    )


# =========================================================
# OVERALL ASSESSMENT
# =========================================================


def _build_overall_assessment(
    technique_label: Optional[str],
    outcome_label: Optional[str],
    primary_issue: Optional[str],
    feedback_probability: Optional[float],
) -> str:

    # -----------------------------------------------------
    # Strong technique + strong outcome
    # -----------------------------------------------------

    if (
        technique_label
        in {
            "solid",
            "strong",
        }

        and

        outcome_label
        in {
            "strong",
            "excellent",
        }
    ):

        if primary_issue in {
            None,
            "none",
        }:

            return (
                "Strong rep overall. The shot produced a "
                "high-quality outcome and the available movement "
                "evidence suggests generally controlled execution, "
                "without a clear technical correction being necessary."
            )

        return (
            "Strong result overall. The shot outcome was effective "
            "and the technique was generally controlled, although "
            "SCAN identified one area that may be worth monitoring "
            "on the next repetition."
        )

    # -----------------------------------------------------
    # Good technique, poor result
    # -----------------------------------------------------

    if (
        technique_label
        in {
            "solid",
            "strong",
        }

        and

        outcome_label
        in {
            "poor",
            "acceptable",
        }
    ):

        return (
            "The movement itself looked reasonably controlled, "
            "but the final shot outcome did not fully match the "
            "quality of the execution. This is a useful rep to "
            "separate technical form from actual effectiveness."
        )

    # -----------------------------------------------------
    # Weak technique, successful result
    # -----------------------------------------------------

    if (
        technique_label
        in {
            "poor",
            "mixed",
        }

        and

        outcome_label
        in {
            "strong",
            "excellent",
        }
    ):

        return (
            "The finish was effective, but the technical evidence "
            "was less convincing. The result was good, while the "
            "underlying execution may still need to become more "
            "stable and repeatable."
        )

    # -----------------------------------------------------
    # Both weak
    # -----------------------------------------------------

    if (
        technique_label
        in {
            "poor",
            "mixed",
        }

        and

        outcome_label
        in {
            "poor",
            "acceptable",
        }
    ):

        return (
            "This rep showed room for improvement in both the "
            "execution and the final result. The most useful next "
            "step is to address one clear issue rather than changing "
            "several things at once."
        )

    # -----------------------------------------------------
    # No reliable outcome
    # -----------------------------------------------------

    if technique_label in {
        "solid",
        "strong",
    }:

        return (
            "The available body-tracking evidence suggests a "
            "generally controlled rep, but SCAN does not have "
            "enough reliable outcome information to judge the "
            "shot as a whole."
        )

    if technique_label in {
        "poor",
        "mixed",
    }:

        return (
            "The technical evidence suggests some instability, "
            "but the available shot-outcome evidence is incomplete. "
            "Any correction should therefore stay conservative."
        )

    # -----------------------------------------------------
    # Complete uncertainty fallback
    # -----------------------------------------------------

    return (
        "SCAN captured the repetition, but the available evidence "
        "is not strong enough to make a detailed overall judgement "
        "without risking over-interpreting the rep."
    )


# =========================================================
# WHAT WENT WELL
# =========================================================


def _build_what_went_well(
    state: SituatedState,
    technique_label: Optional[str],
    outcome_label: Optional[str],
) -> str:

    positive_parts: list[str] = []

    if technique_label == "strong":

        positive_parts.append(
            "The movement evidence strongly supports "
            "controlled technical execution."
        )

    elif technique_label == "solid":

        positive_parts.append(
            "The movement evidence suggests the rep was "
            "generally controlled without a major technical "
            "breakdown."
        )

    if (
        state.goal_entry_detected
        is True
    ):

        zone_phrase = (
            _goal_zone_phrase(
                state.goal_zone
            )
        )

        if zone_phrase:

            positive_parts.append(
                "The finish reached "
                f"{zone_phrase}, giving the rep a strong "
                "practical outcome."
            )

        else:

            positive_parts.append(
                "The shot was observed entering the goal."
            )

    elif (
        outcome_label
        in {
            "strong",
            "excellent",
        }
    ):

        positive_parts.append(
            "The ball and goal evidence supports a "
            "high-quality shot outcome."
        )

    if positive_parts:

        return " ".join(
            positive_parts
        )

    if (
        state.perception_usable
        and state.pose_coverage
        >= 0.70
    ):

        return (
            "SCAN captured enough of the movement to form a "
            "usable representation of the rep, even though no "
            "single strength stood out strongly enough to make "
            "a more specific claim."
        )

    return (
        "There is not enough reliable evidence to identify a "
        "specific strength from this rep."
    )


# =========================================================
# NEXT REP
# =========================================================


def _build_next_rep_recommendation(
    primary_issue: Optional[str],
    feedback_probability: Optional[float],
    outcome_label: Optional[str],
) -> str:

    # -----------------------------------------------------
    # Jev does not think correction is warranted
    # -----------------------------------------------------

    if (
        feedback_probability
        is not None
        and feedback_probability
        < 0.35
    ):

        if outcome_label in {
            "strong",
            "excellent",
        }:

            return (
                "Do not force a technical change on the next rep. "
                "Repeat the same general approach and see whether "
                "you can reproduce the result consistently."
            )

        return (
            "There is not enough evidence to justify changing "
            "your technique from this rep alone. Take another "
            "attempt and look for a repeatable pattern."
        )

    # -----------------------------------------------------
    # Specific issue
    # -----------------------------------------------------

    if (
        primary_issue
        in ISSUE_RECOMMENDATIONS
    ):

        return (
            ISSUE_RECOMMENDATIONS[
                primary_issue
            ]
        )

    # -----------------------------------------------------
    # Insufficient evidence
    # -----------------------------------------------------

    if (
        primary_issue
        == "insufficient_evidence"
    ):

        return (
            "Take another repetition before making a technical "
            "change. SCAN does not yet have enough reliable "
            "evidence to isolate one coaching priority."
        )

    # -----------------------------------------------------
    # No issue
    # -----------------------------------------------------

    return (
        "Focus on reproducing the same movement and outcome "
        "across the next few repetitions rather than searching "
        "for a correction that may not be necessary."
    )


# =========================================================
# SPOKEN CUE
# =========================================================


def _build_spoken_cue(
    primary_issue: Optional[str],
    feedback_probability: Optional[float],
    outcome_label: Optional[str],
) -> Optional[str]:

    # -----------------------------------------------------
    # Low corrective-feedback probability
    # -----------------------------------------------------

    if (
        feedback_probability
        is not None
        and feedback_probability
        < 0.35
    ):

        if outcome_label in {
            "strong",
            "excellent",
        }:

            return (
                "Nice one. That looked controlled — "
                "try to reproduce it."
            )

        return (
            "Take another one. Keep the same approach "
            "and let's see if the pattern repeats."
        )

    # -----------------------------------------------------
    # Specific correction
    # -----------------------------------------------------

    if (
        primary_issue
        in ISSUE_SPOKEN_CUES
    ):

        return (
            ISSUE_SPOKEN_CUES[
                primary_issue
            ]
        )

    # -----------------------------------------------------
    # No correction
    # -----------------------------------------------------

    if primary_issue == "none":

        return (
            "Good rep. Try to reproduce that one."
        )

    # -----------------------------------------------------
    # Uncertain
    # -----------------------------------------------------

    if (
        primary_issue
        == "insufficient_evidence"
    ):

        return None

    return None


# =========================================================
# CONFIDENCE NOTE
# =========================================================


def _build_confidence_note(
    technique_label: Optional[str],
    technique_probability: Optional[float],

    outcome_label: Optional[str],
    outcome_probability: Optional[float],

    issue: Optional[str],
    issue_confidence: Optional[float],

    feedback_probability: Optional[float],
) -> Optional[str]:

    parts: list[str] = []

    technique_pct = (
        _percentage(
            technique_probability
        )
    )

    outcome_pct = (
        _percentage(
            outcome_probability
        )
    )

    issue_pct = (
        _percentage(
            issue_confidence
        )
    )

    feedback_pct = (
        _percentage(
            feedback_probability
        )
    )

    if (
        technique_label
        and technique_pct
        is not None
    ):

        parts.append(
            "Technique was most strongly rated "
            f"{technique_label} "
            f"({technique_pct}% support)."
        )

    if (
        outcome_label
        and outcome_pct
        is not None
    ):

        parts.append(
            "Outcome was most strongly rated "
            f"{outcome_label} "
            f"({outcome_pct}% support)."
        )

    if (
        issue
        and issue_pct
        is not None
    ):

        issue_label = (
            ISSUE_LABELS.get(
                issue,
                issue.replace(
                    "_",
                    " ",
                ),
            )
        )

        parts.append(
            "Primary-focus judgement was "
            f"{issue_label.lower()} "
            f"({issue_pct}% support)."
        )

    if feedback_pct is not None:

        parts.append(
            "Probability that a specific corrective "
            f"cue was warranted: {feedback_pct}%."
        )

    if not parts:
        return None

    return " ".join(
        parts
    )


# =========================================================
# MAIN REPORT BUILDER
# =========================================================


def build_coaching_report(
    situated_state: SituatedState,
    jev_result: JevJudgementResult,
    shot_summary: Any,
) -> CoachingReport:
    """
    Convert SCAN perception + Jev judgements into a
    human-readable coaching report.

    This function is deterministic.

    It does NOT:
    - make another model call
    - invent new biomechanics
    - override the controller
    - decide whether audio is actually played
    """

    # -----------------------------------------------------
    # Read Jev answers
    # -----------------------------------------------------

    technique_answer = (
        _get_answer(
            jev_result,
            "technique_quality",
        )
    )

    outcome_answer = (
        _get_answer(
            jev_result,
            "outcome_quality",
        )
    )

    issue_answer = (
        _get_answer(
            jev_result,
            "primary_issue",
        )
    )

    feedback_answer = (
        _get_answer(
            jev_result,
            "feedback_warranted",
        )
    )

    # -----------------------------------------------------
    # Technique
    # -----------------------------------------------------

    (
        technique_label,
        technique_probability,
        technique_score,
    ) = _score_label(
        technique_answer,

        [
            "poor",
            "mixed",
            "solid",
            "strong",
        ],
    )

    # -----------------------------------------------------
    # Outcome
    # -----------------------------------------------------

    (
        outcome_label,
        outcome_probability,
        outcome_score,
    ) = _score_label(
        outcome_answer,

        [
            "poor",
            "acceptable",
            "strong",
            "excellent",
        ],
    )

    # -----------------------------------------------------
    # Primary issue
    # -----------------------------------------------------

    (
        primary_issue,
        primary_issue_confidence,
    ) = _choice_value(
        issue_answer
    )

    # -----------------------------------------------------
    # Feedback warranted
    # -----------------------------------------------------

    feedback_probability = (
        _noul_probability(
            feedback_answer
        )
    )

    # -----------------------------------------------------
    # If Jev failed, remain conservative.
    # -----------------------------------------------------

    if jev_result.status != "ok":

        primary_issue = (
            "insufficient_evidence"
        )

        primary_issue_confidence = None

        feedback_probability = None

    # -----------------------------------------------------
    # Build report sections
    # -----------------------------------------------------

    overall_assessment = (
        _build_overall_assessment(
            technique_label=(
                technique_label
            ),

            outcome_label=(
                outcome_label
            ),

            primary_issue=(
                primary_issue
            ),

            feedback_probability=(
                feedback_probability
            ),
        )
    )

    what_went_well = (
        _build_what_went_well(
            state=(
                situated_state
            ),

            technique_label=(
                technique_label
            ),

            outcome_label=(
                outcome_label
            ),
        )
    )

    outcome_analysis = (
        _build_outcome_analysis(
            state=(
                situated_state
            ),

            outcome_label=(
                outcome_label
            ),

            outcome_probability=(
                outcome_probability
            ),
        )
    )

    next_rep_recommendation = (
        _build_next_rep_recommendation(
            primary_issue=(
                primary_issue
            ),

            feedback_probability=(
                feedback_probability
            ),

            outcome_label=(
                outcome_label
            ),
        )
    )

    spoken_cue = (
        _build_spoken_cue(
            primary_issue=(
                primary_issue
            ),

            feedback_probability=(
                feedback_probability
            ),

            outcome_label=(
                outcome_label
            ),
        )
    )

    confidence_note = (
        _build_confidence_note(
            technique_label=(
                technique_label
            ),

            technique_probability=(
                technique_probability
            ),

            outcome_label=(
                outcome_label
            ),

            outcome_probability=(
                outcome_probability
            ),

            issue=(
                primary_issue
            ),

            issue_confidence=(
                primary_issue_confidence
            ),

            feedback_probability=(
                feedback_probability
            ),
        )
    )

    # -----------------------------------------------------
    # Human-readable primary focus
    # -----------------------------------------------------

    primary_focus: Optional[str]

    if primary_issue in {
        None,
        "none",
    }:

        primary_focus = None

    elif (
        primary_issue
        == "insufficient_evidence"
    ):

        primary_focus = (
            "Insufficient evidence for a specific correction"
        )

    else:

        primary_focus = (
            ISSUE_LABELS.get(
                primary_issue,

                primary_issue.replace(
                    "_",
                    " ",
                ).title(),
            )
        )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    return CoachingReport(
        overall_assessment=(
            overall_assessment
        ),

        what_went_well=(
            what_went_well
        ),

        primary_focus=(
            primary_focus
        ),

        outcome_analysis=(
            outcome_analysis
        ),

        next_rep_recommendation=(
            next_rep_recommendation
        ),

        spoken_cue=(
            spoken_cue
        ),

        confidence_note=(
            confidence_note
        ),
    )