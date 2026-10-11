"""Deterministic wording from pressure evidence and a controller decision."""
from src.backend.controller import decide_pressure


def build_pressure_report(evidence: dict) -> dict:
    policy = decide_pressure(evidence)
    event = (evidence.get("events") or [None])[0]
    if event:
        side = event["camera_side"]
        location = f" from camera-{side}" if side != "unclear" else ""
        assessment = f"The selected players' visible gap closed quickly{location} between {event['start_s']:.1f} and {event['end_s']:.1f} seconds. This supports reviewing opponent pressure, not a confirmed collision hazard."
        next_rep = "Review when you noticed the approaching player. Practise scanning before committing to your next action, then choose space that you can verify is open."
    else:
        assessment = "No reliable closing-pressure event was established. This does not prove that no opponent approached."
        next_rep = "Keep both players visible and separated at the start; confirm the athlete's starting camera-side before the next rep."
    return {
        **evidence, "policy": policy, "assessment": assessment,
        "next_rep_recommendation": next_rep,
        "spoken_cue": "Scan the pressure before your next action." if policy["should_speak"] else None,
    }
