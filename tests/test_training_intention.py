import copy
import json
import math

import pytest

from src.backend.training_intention import (
    GOAL_COORDINATES, MAX_INTENTION_BYTES, WORLD_COORDINATES,
    build_intention_feedback, parse_training_intention,
)


def quality(**extra):
    return {
        "identity_verified": True, "metric_calibrated": True,
        "camera_motion_accounted": True, "depth_resolved": True,
        "observations_only": True, "contains_predictions": False,
        "contains_extrapolation": False, **extra,
    }


def packet(**values):
    return {"provenance": "server_perception", **values}


def release():
    return packet(
        velocity=[3, 4, 12], velocity_unit="m/s", coordinate_system=WORLD_COORDINATES,
        quality=quality(release_timing_verified=True, velocity_estimate_validated=True),
    )


def transit():
    return packet(
        start=[0, 0, 0], end=[12, 0, 5], elapsed=0.5,
        position_unit="m", time_unit="s", coordinate_system=WORLD_COORDINATES,
        quality=quality(flight_interval_verified=True, timing_verified=True),
    )


def curl():
    return packet(
        start=[0, 0, 0], end=[10, 2, -4],
        initial_velocity=[10, 0, 1], final_velocity=[math.sqrt(3), 1, -10],
        position_unit="m", velocity_unit="m/s", coordinate_system=WORLD_COORDINATES,
        quality=quality(release_timing_verified=True, flight_interval_verified=True,
                        velocity_estimate_validated=True, trajectory_continuous=True,
                        uninterrupted_free_flight=True),
    )


def placement():
    return packet(
        intersection=[3, 1.5], target=[2.7, 1.1], target_zone="upper_right",
        position_unit="m", coordinate_system=GOAL_COORDINATES,
        quality=quality(goal_plane_intersection_measured=True, target_calibrated=True,
                        target_predeclared=True),
    )


def metrics(feedback):
    return {row["key"]: row for row in feedback["metrics"]}


def test_legacy_defaults_and_partial_json_are_normalized_without_mutation():
    expected = {"shot_type": "unspecified", "target_zone": None, "reported_outcome": None}
    assert parse_training_intention(None) == expected
    assert parse_training_intention("{}") == expected
    original = {"shot_type": "curl"}
    assert parse_training_intention(original) == {**expected, "shot_type": "curl"}
    assert original == {"shot_type": "curl"}


@pytest.mark.parametrize("raw", [
    True, False, 1, 0, float("nan"), [], (), "", "power", '"power"', "null",
    '{"shot_type": true}', '{"shot_type": "power", "shot_type": "curl"}',
    '{"shot_type": NaN}', '{"target_zone": Infinity}',
    {"shot_type": None}, {"shot_type": "Power"}, {"shot_type": "dip"},
    {"target_zone": "top_right"}, {"target_zone": True},
    {"reported_outcome": "scored"}, {"reported_outcome": 1},
    {"measurements": {}}, {"shot_type": {"type": "power"}},
    {"shot_type": ["power"]}, {"reported_outcome": float("inf")},
    " " * (MAX_INTENTION_BYTES + 1) + "{}", {"shot_type": "x" * 10_000},
])
def test_invalid_intention_rejected(raw):
    with pytest.raises(ValueError):
        parse_training_intention(raw)


@pytest.mark.parametrize("shot_type,expected", [
    ("unspecified", set()),
    ("power", {"release_speed_kmh", "flight_time_s", "mean_transit_speed_kmh"}),
    ("curl", {"lateral_heading_change_deg", "lateral_deviation_m"}),
    ("placement", {"target_distance_m"}),
])
def test_each_intention_has_distinct_focus_and_missing_evidence(shot_type, expected):
    feedback = build_intention_feedback({"shot_type": shot_type})
    assert set(metrics(feedback)) == expected
    assert all(row["status"] == "unavailable" and row["value"] is None
               for row in feedback["metrics"])
    assert feedback["focus"]
    assert feedback["reference"]["status"] == "not_comparable"
    json.dumps(feedback, allow_nan=False)


def test_power_distinguishes_initial_speed_transit_and_mechanical_power():
    feedback = build_intention_feedback({"shot_type": "power"}, {
        "release_speed": release(), "transit": transit(),
    })
    values = metrics(feedback)
    assert values["release_speed_kmh"]["value"] == pytest.approx(46.8)
    assert values["mean_transit_speed_kmh"]["value"] == pytest.approx(93.6)
    assert values["flight_time_s"]["value"] == 0.5
    assert "not mean curved-path speed or release speed" in values["mean_transit_speed_kmh"]["formula"]
    assert "mechanical power" in feedback["focus"]
    assert "dip is not a power score" in feedback["focus"]
    assert "professional comparison" in feedback["summary"]


def test_curl_uses_horizontal_flight_geometry_and_signed_change_not_vertical_dip():
    supplied = curl()
    values = metrics(build_intention_feedback({"shot_type": "curl"}, {"curl": supplied}))
    assert values["lateral_heading_change_deg"]["value"] == pytest.approx(30)
    assert values["lateral_deviation_m"]["value"] == pytest.approx(2)
    supplied["final_velocity"][1] = -1
    supplied["end"][1] = -2
    supplied["end"][2] = -200
    values = metrics(build_intention_feedback({"shot_type": "curl"}, {"curl": supplied}))
    assert values["lateral_heading_change_deg"]["value"] == pytest.approx(-30)
    assert values["lateral_deviation_m"]["value"] == pytest.approx(-2)


def test_placement_distance_is_measured_plane_geometry_not_goal_confirmation():
    feedback = build_intention_feedback({
        "shot_type": "placement", "target_zone": "upper_right", "reported_outcome": "goal",
    }, {"placement": placement()})
    value = metrics(feedback)["target_distance_m"]
    assert value["value"] == pytest.approx(0.5)
    assert "does not confirm a goal" in value["formula"]
    assert feedback["reported_outcome"] == {
        "value": "goal", "source": "player_reported", "verified": False,
    }


@pytest.mark.parametrize("flag,value", [
    ("identity_verified", False), ("identity_verified", 1),
    ("metric_calibrated", False), ("camera_motion_accounted", False),
    ("depth_resolved", False), ("observations_only", False),
    ("contains_predictions", True), ("contains_extrapolation", True),
    ("contains_predictions", None), ("release_timing_verified", False),
    ("velocity_estimate_validated", False),
])
def test_quality_checks_cannot_be_replaced_by_high_confidence(flag, value):
    supplied = release()
    supplied["quality"][flag] = value
    supplied["confidence"] = 0.999
    result = metrics(build_intention_feedback({"shot_type": "power"}, {"release_speed": supplied}))
    assert result["release_speed_kmh"]["status"] == "unavailable"


@pytest.mark.parametrize("vector", [
    [1, 2], [1, 2, 3, 4], [True, 2, 3], ["1", 2, 3],
    [float("nan"), 2, 3], [float("inf"), 2, 3], [10 ** 1000, 2, 3],
    None,
])
def test_invalid_velocity_dimensions_or_nonfinite_values_withheld(vector):
    supplied = release()
    supplied["velocity"] = vector
    feedback = build_intention_feedback({"shot_type": "power"}, {"release_speed": supplied})
    assert metrics(feedback)["release_speed_kmh"]["status"] == "unavailable"
    json.dumps(feedback, allow_nan=False)


@pytest.mark.parametrize("change", [
    {"velocity_unit": "km/h"}, {"coordinate_system": "image_2d"},
    {"provenance": "player_reported"}, {"provenance": "client"}, {"quality": {}},
])
def test_wrong_units_coordinates_or_provenance_withheld(change):
    supplied = {**release(), **change}
    assert metrics(build_intention_feedback({"shot_type": "power"}, {
        "release_speed": supplied,
    }))["release_speed_kmh"]["status"] == "unavailable"


@pytest.mark.parametrize("elapsed", [0, -1, True, float("nan"), float("inf")])
def test_transit_requires_finite_positive_measured_time(elapsed):
    supplied = transit()
    supplied["elapsed"] = elapsed
    values = metrics(build_intention_feedback({"shot_type": "power"}, {"transit": supplied}))
    assert values["flight_time_s"]["status"] == "unavailable"
    assert values["mean_transit_speed_kmh"]["status"] == "unavailable"


def test_estimated_strike_frame_is_not_true_release_or_flight_timing():
    supplied = release()
    supplied["quality"].pop("release_timing_verified")
    supplied.update(strike_frame=90, fps=30, strike_time_s=3.0)
    values = metrics(build_intention_feedback({"shot_type": "power"}, {
        "release_speed": supplied, "strike_frame": 90,
    }))
    assert all(row["status"] == "unavailable" for row in values.values())


def test_curl_withholds_raw_image_turns_gaps_and_zero_horizontal_heading():
    variants = [copy.deepcopy(curl()) for _ in range(4)]
    variants[0]["coordinate_system"] = "image_2d"
    variants[1]["quality"]["trajectory_continuous"] = False
    variants[2]["initial_velocity"] = [0, 0, 1]
    variants[3]["final_velocity"] = [0, 0, -1]
    for supplied in variants:
        feedback = build_intention_feedback({"shot_type": "curl"}, {"curl": supplied})
        assert all(row["status"] == "unavailable" for row in feedback["metrics"])


def test_placement_needs_matching_selected_target_and_measured_intersection():
    intention = {"shot_type": "placement", "target_zone": "upper_right"}
    variants = [copy.deepcopy(placement()) for _ in range(5)]
    variants[0]["quality"]["goal_plane_intersection_measured"] = False
    variants[1]["quality"]["target_calibrated"] = False
    variants[2]["target_zone"] = "lower_left"
    variants[3]["intersection"] = [3, 1.5, 0]
    variants[4]["intersection"] = [float("nan"), 1.5]
    for supplied in variants:
        assert metrics(build_intention_feedback(intention, {
            "placement": supplied,
        }))["target_distance_m"]["status"] == "unavailable"
    assert metrics(build_intention_feedback({"shot_type": "placement"}, {
        "placement": placement(),
    }))["target_distance_m"]["status"] == "unavailable"


def test_self_report_never_upgrades_any_intention_measurements():
    for shot_type in ("power", "curl", "placement"):
        feedback = build_intention_feedback({
            "shot_type": shot_type, "target_zone": "upper_right", "reported_outcome": "goal",
        })
        assert all(row["status"] == "unavailable" for row in feedback["metrics"])
        assert feedback["reported_outcome"]["verified"] is False


def test_output_never_contains_nonfinite_calculation_results():
    supplied = transit()
    supplied["start"] = [-1e308, 0, 0]
    supplied["end"] = [1e308, 0, 0]
    feedback = build_intention_feedback({"shot_type": "power"}, {"transit": supplied})
    assert metrics(feedback)["mean_transit_speed_kmh"]["status"] == "unavailable"
    json.dumps(feedback, allow_nan=False)


def test_overflowed_heading_magnitude_cannot_look_like_zero_curl():
    supplied = curl()
    supplied["initial_velocity"] = [1.7e308, 1.7e308, 0]
    feedback = build_intention_feedback({"shot_type": "curl"}, {"curl": supplied})
    assert all(row["status"] == "unavailable" for row in feedback["metrics"])
    json.dumps(feedback, allow_nan=False)


@pytest.mark.parametrize("kind,field,incorrect", [
    ("transit", "position_unit", "px"), ("transit", "time_unit", "frames"),
    ("curl", "position_unit", "px"), ("curl", "velocity_unit", "px/frame"),
    ("placement", "position_unit", "normalized"),
    ("placement", "coordinate_system", "image_2d"),
])
def test_metric_specific_wrong_units_and_coordinate_frames_withheld(kind, field, incorrect):
    supplied = {"transit": transit, "curl": curl, "placement": placement}[kind]()
    supplied[field] = incorrect
    shot_type = "power" if kind == "transit" else kind
    feedback = build_intention_feedback({
        "shot_type": shot_type, "target_zone": "upper_right",
    }, {kind: supplied})
    assert all(row["status"] == "unavailable" for row in feedback["metrics"])


def test_curl_cannot_treat_bounce_or_deflection_as_free_flight_curl():
    supplied = curl()
    supplied["quality"]["uninterrupted_free_flight"] = False
    feedback = build_intention_feedback({"shot_type": "curl"}, {"curl": supplied})
    assert all(row["status"] == "unavailable" for row in feedback["metrics"])


def test_retrospective_target_choice_is_not_predeclared_placement_evidence():
    supplied = placement()
    supplied["quality"]["target_predeclared"] = False
    feedback = build_intention_feedback({
        "shot_type": "placement", "target_zone": "upper_right",
    }, {"placement": supplied})
    assert metrics(feedback)["target_distance_m"]["status"] == "unavailable"


def test_heading_difference_is_principal_signed_delta_not_accumulated_curl():
    supplied = curl()
    supplied["initial_velocity"] = [math.cos(math.radians(170)), math.sin(math.radians(170)), 0]
    supplied["final_velocity"] = [math.cos(math.radians(-170)), math.sin(math.radians(-170)), 0]
    metric = metrics(build_intention_feedback({"shot_type": "curl"}, {
        "curl": supplied,
    }))["lateral_heading_change_deg"]
    assert metric["value"] == pytest.approx(20)
    assert "not unwrapped total turning" in metric["formula"]


def test_professional_cohort_is_reference_context_never_a_video_rating():
    feedback = build_intention_feedback({"shot_type": "power"}, {"release_speed": release()})
    reference = feedback["reference"]
    assert reference["status"] == "not_comparable"
    profile = reference["profiles"][0]
    assert profile["mean"] == 83.18 and profile["sd"] == 6.61
    assert profile["sample_size"] == 17
    assert profile["metric"] == "radar_fastest_shot_speed"
    assert "fastest of five" in profile["protocol"]
    assert "12-hour" in profile["protocol"]
    assert "not a professional pass threshold" in profile["limitation"]
    assert not {"score", "percentile", "ratio", "grade"}.intersection(reference)
    for shot_type in ("curl", "placement"):
        reference = build_intention_feedback({"shot_type": shot_type})["reference"]
        assert reference["profiles"] == []
        assert reference["sources"]


def test_mutating_returned_references_cannot_change_later_feedback():
    feedback = build_intention_feedback({"shot_type": "power"})
    feedback["reference"]["sources"][0]["url"] = "changed"
    feedback["reference"]["profiles"][0]["mean"] = 0
    fresh = build_intention_feedback({"shot_type": "power"})["reference"]
    assert fresh["sources"][0]["url"].startswith("https://journals.plos.org/")
    assert fresh["profiles"][0]["mean"] == 83.18
