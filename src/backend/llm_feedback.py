"""
llm_feedback.py
Converts a Decision object (from controller.py) into a natural-language
coaching cue. This is the module that closes the loop from "silent decision
logic" to "actual spoken/text feedback a player hears."

TODO (Phase 2):
  - Wire up an LLM API call (OpenAI, Anthropic, etc.)
  - Design a prompt template that takes Decision + relevant PlayerState
    context and produces a short, natural coaching cue
  - Add a TTS step if going for spoken feedback
  - Cache/short-circuit for silence_* decision types (no LLM call needed)
"""

from __future__ import annotations

import random
from typing import Optional

from controller import Decision


# ---------------------------------------------------------
# SCAN coaching personality
# ---------------------------------------------------------

# SCAN should sound:
# - calm
# - supportive
# - concise
# - observant
# - conversational
# - like a training companion, not a statistics dashboard


_last_message: Optional[str] = None


def _choose(options: list[str]) -> str:
    """
    Pick a natural variation while avoiding the exact
    same sentence twice in a row.
    """

    global _last_message

    available = [
        option
        for option in options
        if option != _last_message
    ]

    if not available:
        available = options

    message = random.choice(available)

    _last_message = message

    return message


# ---------------------------------------------------------
# Feedback phrasing
# ---------------------------------------------------------

def generate_cue(decision: Decision) -> str:
    """
    Convert a controller Decision into natural coaching language.

    Important:
    This function changes HOW feedback is said.
    It does not decide WHETHER feedback should be given.
    """

    if not decision.should_speak:
        return ""

    # -----------------------------------------------------
    # Hazard
    # -----------------------------------------------------

    if decision.decision_type == "hazard":
        return _choose([
            "Careful there. Reset yourself before the next one.",
            "Easy. Let's reset that technique before you go again.",
            "Hold up for a second. Get yourself set, then we'll go again.",
            "Let's reset that one. I don't want you forcing the movement.",
        ])

    # -----------------------------------------------------
    # Repeated technical error
    # -----------------------------------------------------

    if decision.decision_type == "repeated_error":

        error = decision.error_type or "that movement"

        generic_options = [
            f"I'm seeing {error} come up again. Let's clean that up on the next one.",
            f"Same pattern again with {error}. Focus on fixing that next rep.",
            f"That's showing up a few times now — {error}. Let's make that the focus.",
            f"We're getting the same {error} pattern again. Make one small adjustment and go again.",
        ]

        return _choose(generic_options)

    # -----------------------------------------------------
    # End-of-rep / positive check-in
    # -----------------------------------------------------

    if decision.decision_type == "end_of_rep":
        return _choose([
            "Nice. Keep that rhythm going.",
            "Good work. Stay relaxed and keep building on that.",
            "That's looking more settled. Keep going.",
            "Solid block. Stay composed and go again.",
            "Good. Keep that same feeling on the next one.",
        ])

    return _choose([
        "All right, let's go again.",
        "Good. Reset and take the next one.",
        "Stay composed. Next rep.",
    ])