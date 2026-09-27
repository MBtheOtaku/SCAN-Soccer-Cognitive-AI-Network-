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

from controller import Decision


def generate_cue(decision: Decision) -> str:
    """
    Stub function. Replace with real LLM call for should_speak=True decisions.
    """
    if not decision.should_speak:
        return ""

    # Placeholder deterministic text - swap for LLM-generated phrasing
    templates = {
        "hazard": "Careful — that technique risks injury. Reset your form before the next rep.",
        "repeated_error": f"You've repeated a '{decision.error_type}' error — let's fix that pattern.",
        "end_of_rep": "Good block of reps. Keep that rhythm going.",
    }
    return templates.get(decision.decision_type, "Keep going.")
