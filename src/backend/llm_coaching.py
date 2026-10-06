from __future__ import annotations

import json
import os
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.backend.coaching_report import CoachingReport
from src.backend.situated_state import SituatedState


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_COACHING_MODEL = "gpt-6-luna"


# =========================================================
# EVIDENCE PACKET
# =========================================================


def _build_evidence_packet(
    situated_state: SituatedState,
    jev_judgements: dict[str, Any],
    deterministic_report: CoachingReport,
) -> dict[str, Any]:
    """
    Build the ONLY evidence the language model is allowed
    to use when writing the coaching report.

    The model does not receive raw video and must not invent
    observations outside this structured state.
    """

    return {
        "situated_state": {
            "rep_id": situated_state.rep_id,
            "drill_type": situated_state.drill_type,
            "task_goal": situated_state.task_goal,

            "pose_coverage": situated_state.pose_coverage,
            "perception_usable": situated_state.perception_usable,

            "striking_leg": situated_state.striking_leg,
            "support_leg": situated_state.support_leg,

            "strike_frame": situated_state.strike_frame,
            "strike_time_s": situated_state.strike_time_s,

            "ball_visible": situated_state.ball_visible,
            "ball_tracking_confidence": (
                situated_state.ball_tracking_confidence
            ),
            "ball_detection_rate": (
                situated_state.ball_detection_rate
            ),

            "shot_on_target": situated_state.shot_on_target,
            "goal_entry_detected": (
                situated_state.goal_entry_detected
            ),
            "goal_zone": situated_state.goal_zone,
            "environment_confidence": (
                situated_state.environment_confidence
            ),

            "repeated_error_type": (
                situated_state.repeated_error_type
            ),
            "repeated_error_count": (
                situated_state.repeated_error_count
            ),
            "previous_feedback_type": (
                situated_state.previous_feedback_type
            ),
            "recent_improvement": (
                situated_state.recent_improvement
            ),
        },

        "jev_judgements": {
            "status": jev_judgements.get("status"),
            "model": jev_judgements.get("model"),
            "answers": jev_judgements.get("answers", {}),
        },

        "deterministic_report": (
            deterministic_report.to_dict()
        ),
    }


# =========================================================
# STRUCTURED OUTPUT SCHEMA
# =========================================================


COACHING_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_assessment": {
            "type": "string",
        },

        "what_went_well": {
            "type": "string",
        },

        "primary_focus": {
            "type": [
                "string",
                "null",
            ],
        },

        "outcome_analysis": {
            "type": "string",
        },

        "next_rep_recommendation": {
            "type": "string",
        },

        "spoken_cue": {
            "type": [
                "string",
                "null",
            ],
        },

        "confidence_note": {
            "type": [
                "string",
                "null",
            ],
        },
    },

    "required": [
        "overall_assessment",
        "what_went_well",
        "primary_focus",
        "outcome_analysis",
        "next_rep_recommendation",
        "spoken_cue",
        "confidence_note",
    ],

    "additionalProperties": False,
}


# =========================================================
# PROMPT
# =========================================================


COACHING_INSTRUCTIONS = """
You are the language synthesis layer for SCAN,
the Soccer Cognitive AI Network.

You are NOT the perception system.
You are NOT the probabilistic judgement system.
You are NOT allowed to invent observations.

Your job is to turn the provided structured evidence into
natural, thoughtful, concise post-repetition soccer coaching.

SOURCE OF TRUTH
---------------
You may only use information present in:
1. situated_state
2. jev_judgements
3. deterministic_report

The deterministic report represents SCAN's current safe
interpretation of the evidence. You may improve its wording,
connect its ideas, and reduce repetition, but do not contradict
its core conclusion unless the structured Jev evidence clearly
requires it.

CRITICAL RULES
--------------
- Never invent biomechanics.
- Never claim to see something that is not in the evidence.
- Do not convert raw joint angles into a new soccer diagnosis.
- Do not manufacture criticism merely to give feedback.
- Treat Jev judgements as probabilities, not certainty.
- Separate technique quality from shot outcome.
- A successful shot does not prove perfect technique.
- Imperfect technique does not automatically make the outcome poor.
- If evidence is insufficient, explicitly remain uncertain.
- If no primary issue is supported, do not create one.
- If corrective feedback is weakly warranted, prioritize
  repeatability and consistency rather than technique changes.
- Do not describe goalkeeper difficulty, tactical context,
  ball speed, power, spin, contact quality, or placement precision
  unless the supplied evidence directly supports that claim.
- Do not mention measurements simply to sound technical.
- Do not use phrases such as "according to the AI" or
  "the model thinks."
- Speak like a calm, observant soccer coach.
- Do not repeat the same factual observation in more than two sections.
- Prefer interpretation over repetition: explain why evidence matters
  instead of merely restating what happened.

REPORT STYLE
------------
overall_assessment:
2-3 sentences.
Give the main interpretation of the rep.
Connect execution and outcome without repeating later sections.

what_went_well:
1-3 sentences.
Identify the strongest supported positive evidence.
Explain why it matters for the rep.
Do not give next-repetition instructions in this section.
Describe why the positive evidence from this rep is meaningful.

primary_focus:
Short phrase or one sentence.
If there is no justified correction, make that clear.
Do not invent a technical focus.

outcome_analysis:
1-2 sentences.
Discuss only supported ball/goal outcome evidence.

next_rep_recommendation:
1-2 sentences.
Give one clear next-repetition objective.
Prefer testing repeatability when no correction is warranted.
Do not overload the athlete with multiple instructions.

spoken_cue:
One short natural sentence suitable for speech.
Maximum roughly 18 words.
It should prepare the athlete for the next rep.
It must not read the entire report aloud.

confidence_note:
One concise sentence.
Summarize the most useful uncertainty/confidence information
from the Jev probabilities without pretending probabilities
are ground truth.

SECTION DISTINCTNESS
--------------------
Each section must contribute new information or a new interpretation.
Do not simply repeat the same observation using different wording.

Avoid repeatedly mentioning the same goal zone, technique rating,
or overall outcome across several sections.

overall_assessment:
Give the high-level interpretation of the repetition.
Connect execution and outcome, but do not exhaustively list evidence.
This section should answer:
"What kind of rep was this overall?"

what_went_well:
Explain what was encouraging about the rep and why it matters.
Do not merely repeat the outcome or technique label.
When possible, explain what makes the positive evidence useful
for learning, consistency, or future comparison.

primary_focus:
State the single most useful coaching priority.
If no technical correction is justified, clearly say so.
Do not create a weakness just to fill this section.

outcome_analysis:
Discuss only the supported ball and goal outcome.
Do not repeat general comments about technique here.
Keep this section factual and specific.

next_rep_recommendation:
Turn the evidence into one concrete objective for the next repetition.
Do not restate the overall assessment.
If no correction is justified, focus on testing repeatability,
consistency, or whether the same pattern appears again.

confidence_note:
Discuss uncertainty and probabilistic support only.
Do not repeat coaching advice here.
Summarize the most decision-relevant Jev judgements concisely.

COGNITIVE-LOAD PRINCIPLE
------------------------
The written report may explain.
The spoken cue should direct attention.

Good coaching does not require a correction after every rep.
"""


# =========================================================
# RESPONSE PARSING
# =========================================================


def _extract_output_text(
    response_data: dict[str, Any],
) -> str:
    """
    Pull the generated JSON text from a Responses API response.
    """

    output = response_data.get(
        "output",
        [],
    )

    if not isinstance(
        output,
        list,
    ):
        raise ValueError(
            "OpenAI response did not contain an output list."
        )

    for item in output:

        if not isinstance(
            item,
            dict,
        ):
            continue

        content = item.get(
            "content",
            [],
        )

        if not isinstance(
            content,
            list,
        ):
            continue

        for block in content:

            if not isinstance(
                block,
                dict,
            ):
                continue

            if (
                block.get("type")
                == "output_text"
            ):
                text = block.get(
                    "text"
                )

                if isinstance(
                    text,
                    str,
                ):
                    return text

    raise ValueError(
        "OpenAI response contained no output_text."
    )


# =========================================================
# VALIDATION
# =========================================================


def _report_from_dict(
    data: dict[str, Any],
) -> CoachingReport:
    """
    Convert the structured model result into the same
    CoachingReport object already used by SCAN.
    """

    required_strings = [
        "overall_assessment",
        "what_went_well",
        "outcome_analysis",
        "next_rep_recommendation",
    ]

    for key in required_strings:

        value = data.get(
            key
        )

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise ValueError(
                f"Invalid coaching report field: {key}"
            )

    primary_focus = data.get(
        "primary_focus"
    )

    spoken_cue = data.get(
        "spoken_cue"
    )

    confidence_note = data.get(
        "confidence_note"
    )

    for key, value in [
        (
            "primary_focus",
            primary_focus,
        ),
        (
            "spoken_cue",
            spoken_cue,
        ),
        (
            "confidence_note",
            confidence_note,
        ),
    ]:

        if (
            value is not None
            and not isinstance(
                value,
                str,
            )
        ):
            raise ValueError(
                f"Invalid coaching report field: {key}"
            )

    return CoachingReport(
        overall_assessment=(
            data[
                "overall_assessment"
            ].strip()
        ),

        what_went_well=(
            data[
                "what_went_well"
            ].strip()
        ),

        primary_focus=(
            primary_focus.strip()
            if isinstance(
                primary_focus,
                str,
            )
            and primary_focus.strip()
            else None
        ),

        outcome_analysis=(
            data[
                "outcome_analysis"
            ].strip()
        ),

        next_rep_recommendation=(
            data[
                "next_rep_recommendation"
            ].strip()
        ),

        spoken_cue=(
            spoken_cue.strip()
            if isinstance(
                spoken_cue,
                str,
            )
            and spoken_cue.strip()
            else None
        ),

        confidence_note=(
            confidence_note.strip()
            if isinstance(
                confidence_note,
                str,
            )
            and confidence_note.strip()
            else None
        ),
    )


# =========================================================
# MAIN LLM SYNTHESIS
# =========================================================


def refine_coaching_report_with_llm(
    *,
    situated_state: SituatedState,
    jev_judgements: dict[str, Any],
    deterministic_report: CoachingReport,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: float = 12.0,
) -> tuple[
    CoachingReport,
    dict[str, Any],
]:
    """
    Improve SCAN's deterministic CoachingReport using an LLM.

    IMPORTANT:
    The deterministic report is always returned as a fallback
    if the API key is missing, the request fails, the response
    is malformed, or structured validation fails.

    Therefore the LLM can never break the analysis pipeline.
    """

    api_key = (
        api_key
        or os.getenv(
            "OPENAI_API_KEY"
        )
    )

    model = (
        model
        or os.getenv(
            "SCAN_COACHING_MODEL"
        )
        or DEFAULT_COACHING_MODEL
    )

    # -----------------------------------------------------
    # No API key -> deterministic fallback
    # -----------------------------------------------------

    if not api_key:

        return (
            deterministic_report,
            {
                "status": "skipped",
                "model": model,
                "used_llm": False,
                "error": (
                    "OPENAI_API_KEY is not set."
                ),
            },
        )

    evidence = (
        _build_evidence_packet(
            situated_state=(
                situated_state
            ),
            jev_judgements=(
                jev_judgements
            ),
            deterministic_report=(
                deterministic_report
            ),
        )
    )

    payload = {
        "model": model,

        "instructions": (
            COACHING_INSTRUCTIONS
        ),

        "input": (
            "Use the following SCAN evidence to produce "
            "the coaching report.\n\n"
            + json.dumps(
                evidence,
                indent=2,
            )
        ),

        "text": {
            "format": {
                "type": "json_schema",
                "name": "scan_coaching_report",
                "description": (
                    "A grounded post-repetition soccer "
                    "coaching report."
                ),
                "strict": True,
                "schema": (
                    COACHING_REPORT_SCHEMA
                ),
            },
        },

        "max_output_tokens": 900,

        # We do not need the provider to retain these
        # individual training-rep synthesis requests.
        "store": False,
    }

    request = Request(
        OPENAI_RESPONSES_URL,

        data=json.dumps(
            payload
        ).encode(
            "utf-8"
        ),

        headers={
            "Authorization": (
                f"Bearer {api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
        },

        method="POST",
    )

    try:

        with urlopen(
            request,
            timeout=timeout_s,
        ) as response:

            response_data = (
                json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )
            )

        output_text = (
            _extract_output_text(
                response_data
            )
        )

        generated_data = (
            json.loads(
                output_text
            )
        )

        generated_report = (
            _report_from_dict(
                generated_data
            )
        )

        usage = (
            response_data.get(
                "usage"
            )
        )

        return (
            generated_report,
            {
                "status": "ok",
                "model": (
                    response_data.get(
                        "model",
                        model,
                    )
                ),
                "used_llm": True,
                "usage": usage,
                "error": None,
            },
        )

    except HTTPError as exc:

        try:
            error_body = (
                exc.read().decode(
                    "utf-8"
                )
            )

        except Exception:
            error_body = str(
                exc
            )

        return (
            deterministic_report,
            {
                "status": "error",
                "model": model,
                "used_llm": False,
                "error": (
                    f"OpenAI HTTP error "
                    f"{exc.code}: "
                    f"{error_body}"
                ),
            },
        )

    except URLError as exc:

        return (
            deterministic_report,
            {
                "status": "error",
                "model": model,
                "used_llm": False,
                "error": (
                    f"OpenAI connection error: "
                    f"{exc.reason}"
                ),
            },
        )

    except Exception as exc:

        return (
            deterministic_report,
            {
                "status": "error",
                "model": model,
                "used_llm": False,
                "error": (
                    f"Coaching synthesis failed: "
                    f"{exc}"
                ),
            },
        )