"""Training objectives and conservative, intention-specific measurement formulas.

The public parser accepts intention metadata only. ``measurements`` in
``build_intention_feedback`` is reserved for trusted server perception output;
it must never be populated from request form fields or player self-reports.
Current perception need not supply it. A measurement is unavailable unless its
units, coordinate system, dimensions and explicit evidence gates all pass.

These formulas describe ball flight, not biomechanics or coaching policy.
They do not provide a professional score, prove spin, or confirm a goal.
"""

from __future__ import annotations

import json
import math
from typing import Any


SHOT_TYPES = ("unspecified", "power", "curl", "placement")
TARGET_ZONES = (
    "upper_left", "upper_center", "upper_right", "middle_left", "center",
    "middle_right", "lower_left", "lower_center", "lower_right",
)
REPORTED_OUTCOMES = ("goal", "miss", "woodwork", "saved")
MAX_INTENTION_BYTES = 1024
WORLD_COORDINATES = "world_3d_forward_lateral_up"
GOAL_COORDINATES = "goal_plane_horizontal_up"

_INTENTION_KEYS = frozenset(("shot_type", "target_zone", "reported_outcome"))
_QUALITY_TRUE = (
    "identity_verified", "metric_calibrated", "camera_motion_accounted",
    "depth_resolved", "observations_only",
)
_QUALITY_FALSE = ("contains_predictions", "contains_extrapolation")
_POWER_SOURCE = "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0294867"
_REFERENCE_SOURCES = {
    "power": (
        {"title": "Ben Brahim et al. (2023): Different effects of four rest periods after the last weekly training session on young male professional soccer players’ physical performance", "url": _POWER_SOURCE},
        {"title": "Sánchez-Pay et al. (2022): validation of radar ball-speed measurements", "url": "https://www.mdpi.com/1424-8220/22/18/7046"},
    ),
    "curl": (
        {"title": "Asai et al. (2020): Measurements of the Flight Trajectory of a Spinning Soccer Ball and the Magnus Force Acting on It", "url": "https://www.mdpi.com/2504-3900/49/1/88"},
    ),
    "placement": (
        {"title": "Hunter et al. (2018): Modeling the two-dimensional accuracy of soccer kicks", "url": "https://pubmed.ncbi.nlm.nih.gov/29580692/"},
        {"title": "Sørensen et al. (2024): speed and accuracy instructions and target-distance error", "url": "https://www.mdpi.com/2076-3417/14/22/10668"},
    ),
}
_POWER_PROFILE = {
    "metric": "radar_fastest_shot_speed", "mean": 83.18, "sd": 6.61,
    "unit": "km/h", "sample_size": 17,
    "cohort": "Male professionals from one Tunisian league team; age 20.7 ± 1.0 years.",
    "protocol": "Stationary maximal instep kick toward a 1 × 1 m target; fastest of five attempts; one-minute rest; SRA3000 radar 3 m from the stopped ball; 12-hour recovery condition.",
    "statistic": "Cohort mean ± standard deviation of each player's fastest attempt in this condition.",
    "limitation": "Descriptive cohort record, not a professional pass threshold. Radar maxima are not directly comparable with video endpoint transit averages.",
    "source_url": _POWER_SOURCE,
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Training intention contains a duplicate field.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Training intention must not contain non-finite numbers.")


def parse_training_intention(raw: str | dict | None) -> dict:
    """Normalize an optional JSON object; reject unknown or malformed metadata.

    Omitted fields retain legacy defaults. A string is JSON, not a shot-type
    shorthand. Duplicate JSON keys, extra fields and coercions are rejected.
    """
    if raw is None:
        raw = {}
    elif type(raw) is str:
        try:
            if len(raw.encode("utf-8")) > MAX_INTENTION_BYTES:
                raise ValueError("Training intention is too large.")
            raw = json.loads(
                raw, object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
            raise ValueError("Training intention must be a small JSON object.") from exc
    if type(raw) is not dict or any(key not in _INTENTION_KEYS for key in raw):
        raise ValueError("Training intention must contain only supported fields.")

    normalized = {
        "shot_type": raw.get("shot_type", "unspecified"),
        "target_zone": raw.get("target_zone"),
        "reported_outcome": raw.get("reported_outcome"),
    }
    for key, choices, optional in (
        ("shot_type", SHOT_TYPES, False),
        ("target_zone", TARGET_ZONES, True),
        ("reported_outcome", REPORTED_OUTCOMES, True),
    ):
        value = normalized[key]
        if value is None and optional:
            continue
        if type(value) is not str or value not in choices:
            raise ValueError(f"Unsupported training intention {key}.")
    # Also bound dict inputs; enum validation above avoids serializing arbitrary
    # nested or custom objects from a caller.
    if len(json.dumps(normalized).encode("utf-8")) > MAX_INTENTION_BYTES:
        raise ValueError("Training intention is too large.")
    return normalized


def _finite(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _vector(value: Any, dimensions: int) -> tuple[float, ...] | None:
    if type(value) not in (list, tuple) or len(value) != dimensions:
        return None
    if not all(_finite(component) for component in value):
        return None
    return tuple(float(component) for component in value)


def _evidence(packet: Any, required: tuple[str, ...] = ()) -> str | None:
    if type(packet) is not dict:
        return "A calibrated ball-flight measurement is not available for this clip."
    if packet.get("provenance") != "server_perception":
        return "The measurement has no trusted server perception provenance."
    quality = packet.get("quality")
    if type(quality) is not dict:
        return "Explicit measurement quality checks are missing."
    for key in _QUALITY_TRUE + required:
        if quality.get(key) is not True:
            return f"Required evidence check has not passed: {key}."
    for key in _QUALITY_FALSE:
        if quality.get(key) is not False:
            return "Predicted, extrapolated or unspecified trajectory samples are not measurements."
    return None


def _metric(key: str, label: str, unit: str, formula: str,
            value: float | None = None, reason: str | None = None) -> dict:
    if value is not None and not _finite(value):
        value, reason = None, "The calculation did not produce a finite value."
    return {
        "key": key, "label": label,
        "status": "available" if value is not None else "unavailable",
        "value": value, "unit": unit, "formula": formula,
        "reason": reason,
    }


def _release_speed(packet: Any) -> dict:
    reason = _evidence(packet, ("release_timing_verified", "velocity_estimate_validated"))
    vector = None
    if reason is None:
        if packet.get("velocity_unit") != "m/s" or packet.get("coordinate_system") != WORLD_COORDINATES:
            reason = "Release velocity needs metres per second in a calibrated 3D world frame."
        else:
            vector = _vector(packet.get("velocity"), 3)
            if vector is None:
                reason = "Release velocity needs three finite measured components."
    value = math.hypot(*vector) * 3.6 if reason is None else None
    return _metric(
        "release_speed_kmh", "Calibrated ball release speed", "km/h",
        "3.6 * sqrt(v_forward^2 + v_lateral^2 + v_up^2), at independently verified release time",
        value, reason,
    )


def _transit_metrics(packet: Any) -> list[dict]:
    reason = _evidence(packet, ("flight_interval_verified", "timing_verified"))
    elapsed = distance = None
    if reason is None:
        if packet.get("position_unit") != "m" or packet.get("time_unit") != "s" or packet.get("coordinate_system") != WORLD_COORDINATES:
            reason = "Flight transit needs measured 3D metre coordinates and seconds."
        else:
            start = _vector(packet.get("start"), 3)
            end = _vector(packet.get("end"), 3)
            elapsed = packet.get("elapsed")
            if start is None or end is None or not _finite(elapsed) or elapsed <= 0:
                reason = "Flight endpoints must be finite 3D observations with positive elapsed time."
            else:
                distance = math.dist(start, end)
    speed = 3.6 * (distance / elapsed) if reason is None else None
    return [
        _metric("flight_time_s", "Measured flight interval", "s",
                "t_end - t_start over the verified observed flight interval",
                float(elapsed) if reason is None else None, reason),
        _metric("mean_transit_speed_kmh", "Mean endpoint transit speed", "km/h",
                "3.6 * distance_3d(start, end) / (t_end - t_start); magnitude of mean velocity, not mean curved-path speed or release speed",
                speed, reason),
    ]


def _curl_metrics(packet: Any) -> list[dict]:
    reason = _evidence(packet, (
        "release_timing_verified", "flight_interval_verified",
        "velocity_estimate_validated", "trajectory_continuous", "uninterrupted_free_flight",
    ))
    heading = deviation = None
    if reason is None:
        if packet.get("position_unit") != "m" or packet.get("velocity_unit") != "m/s" or packet.get("coordinate_system") != WORLD_COORDINATES:
            reason = "Lateral flight shape needs calibrated 3D metre coordinates and metre-per-second velocities."
        else:
            start = _vector(packet.get("start"), 3)
            end = _vector(packet.get("end"), 3)
            initial = _vector(packet.get("initial_velocity"), 3)
            final = _vector(packet.get("final_velocity"), 3)
            if any(vector is None for vector in (start, end, initial, final)):
                reason = "Lateral flight shape needs finite 3D endpoints and velocity vectors."
            else:
                initial_norm = math.hypot(*initial[:2])
                final_norm = math.hypot(*final[:2])
                if not all(_finite(norm) and norm > 0 for norm in (initial_norm, final_norm)):
                    reason = "Horizontal heading needs finite, nonzero horizontal velocity magnitudes."
                else:
                    # Unit vectors keep the heading calculation well scaled.
                    ax, ay = initial[0] / initial_norm, initial[1] / initial_norm
                    bx, by = final[0] / final_norm, final[1] / final_norm
                    heading = math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by))
                    deviation = ax * (end[1] - start[1]) - ay * (end[0] - start[0])
    return [
        _metric("lateral_heading_change_deg", "Signed horizontal heading change", "deg",
                "degrees(atan2(v0_forward*v1_lateral - v0_lateral*v1_forward, dot(v0_horizontal, v1_horizontal))); principal signed difference in [-180, 180] degrees, not unwrapped total turning; positive toward calibrated lateral axis",
                heading, reason),
        _metric("lateral_deviation_m", "Signed lateral deviation from release heading", "m",
                "cross_2d(unit(v0_horizontal), end_horizontal - start_horizontal); excludes vertical drop",
                deviation, reason),
    ]


def _placement_metric(packet: Any, target_zone: str | None) -> dict:
    reason = _evidence(packet, (
        "goal_plane_intersection_measured", "target_calibrated", "target_predeclared",
    ))
    value = None
    if target_zone is None:
        reason = "Select a target zone before assessing placement against a target."
    elif reason is None:
        if packet.get("position_unit") != "m" or packet.get("coordinate_system") != GOAL_COORDINATES:
            reason = "Placement needs metre coordinates on the calibrated goal plane."
        elif packet.get("target_zone") != target_zone:
            reason = "The measured target does not match the selected target zone."
        else:
            intersection = _vector(packet.get("intersection"), 2)
            target = _vector(packet.get("target"), 2)
            if intersection is None or target is None:
                reason = "The measured plane intersection and target need two finite coordinates each."
            else:
                value = math.dist(intersection, target)
    return _metric(
        "target_distance_m", "Distance from target at measured goal-plane intersection", "m",
        "sqrt((intersection_horizontal - target_horizontal)^2 + (intersection_up - target_up)^2); does not confirm a goal",
        value, reason,
    )


def build_intention_feedback(intention: str | dict | None, measurements: dict | None = None) -> dict:
    """Build deterministic objective-specific observations without pro grading.

    ``measurements`` is a future internal-only mapping with optional
    ``release_speed``, ``transit``, ``curl`` and ``placement`` packets. Every
    packet requires ``provenance='server_perception'`` and a ``quality`` object:
    all ``_QUALITY_TRUE`` flags true, both ``_QUALITY_FALSE`` flags false, plus
    metric-specific checks. Tests provide exact unit/coordinate examples.

    No packet accepts a source pixel trajectory, an estimated strike frame,
    player-reported outcome, or confidence score as a substitute for these
    checks. Existing production callers should leave measurements as ``None``.
    """
    intention = parse_training_intention(intention)
    packets = measurements if type(measurements) is dict else {}
    shot_type = intention["shot_type"]
    focuses = {
        "unspecified": "Choose a training intention to select the relevant measurements.",
        "power": "Ball release speed and measured flight transit; neither is mechanical power, and dip is not a power score.",
        "curl": "Calibrated horizontal heading change and lateral deviation; these do not establish spin or technique quality.",
        "placement": "Distance from the selected target at a measured goal-plane intersection; goal outcome remains separate.",
    }
    metrics = []
    if shot_type == "power":
        metrics = [_release_speed(packets.get("release_speed")), *_transit_metrics(packets.get("transit"))]
    elif shot_type == "curl":
        metrics = _curl_metrics(packets.get("curl"))
    elif shot_type == "placement":
        metrics = [_placement_metric(packets.get("placement"), intention["target_zone"])]

    available = [metric for metric in metrics if metric["status"] == "available"]
    if shot_type == "unspecified":
        summary = "No shot intention was specified; no intention-specific assessment was made."
    elif not available:
        summary = f"{shot_type.capitalize()} intention recorded. The required calibrated evidence is not available for an intention-specific measurement."
    else:
        observations = "; ".join(
            f"{metric['label']}: {metric['value']:.2f} {metric['unit']}"
            for metric in available
        )
        summary = f"{observations}. These observations do not provide a technique rating or professional comparison."

    outcome = intention["reported_outcome"]
    return {
        "shot_type": shot_type,
        "focus": focuses[shot_type],
        "summary": summary,
        "metrics": metrics,
        "reference": {
            "status": "not_comparable",
            "note": "No matched, validated professional reference protocol is configured. Research measurements are task- and setup-specific; no universal professional threshold or rating is applied.",
            "sources": [dict(source) for source in _REFERENCE_SOURCES.get(shot_type, ())],
            "profiles": [dict(_POWER_PROFILE)] if shot_type == "power" else [],
        },
        "reported_outcome": None if outcome is None else {
            "value": outcome, "source": "player_reported", "verified": False,
        },
    }
