from src.backend.opponent_pressure import Person, extract_evidence
from src.backend.pressure_report import build_pressure_report
from src.backend.controller import decide_pressure


def player(x, confidence=.9):
    return Person((x - 20, 100, x + 20, 300), confidence)


def samples(positions):
    return [(i / 5, [player(500), player(x)]) for i, x in enumerate(positions)]


def test_consistent_approach_from_camera_left():
    evidence = extract_evidence(samples([0, 40, 80, 120, 160, 200, 240]), "right")
    assert evidence["status"] == "observed"
    assert evidence["events"][0]["camera_side"] == "left"
    assert evidence["coordinate_frame"] == "camera"
    report = build_pressure_report(evidence)
    assert report["policy"]["should_speak"]
    assert "camera-left" in report["assessment"]
    assert "collision hazard" in report["assessment"]
    assert "cut" not in report["spoken_cue"].lower()


def test_receding_player_does_not_trigger_correction():
    report = build_pressure_report(extract_evidence(samples([220, 180, 140, 100, 60, 20]), "right"))
    assert not report["events"]
    assert report["spoken_cue"] is None


def test_slow_approach_does_not_trigger():
    report = build_pressure_report(extract_evidence(samples([0, 5, 10, 15, 20, 25]), "right"))
    assert report["spoken_cue"] is None


def test_occlusion_breaks_association_instead_of_switching_roles():
    data = samples([0, 50, 100]) + [(0.6, [player(500), player(505)])] + samples([0, 40, 80, 120, 160, 200])
    evidence = extract_evidence(data, "right")
    assert evidence["tracking_interrupted"]
    assert not evidence["events"]


def test_three_people_withhold_role_assignment():
    report = build_pressure_report(extract_evidence([(0, [player(0), player(300), player(600)])], "left"))
    assert report["status"] == "inconclusive"
    assert not report["policy"]["should_speak"]


def test_weak_detections_are_not_treated_as_motion():
    evidence = extract_evidence([(i / 5, [player(500), player(i * 40, .3)]) for i in range(8)], "right")
    assert not evidence["events"]


def test_mid_action_policy_protects_attention():
    evidence = extract_evidence(samples([0, 40, 80, 120, 160, 200, 240]), "right")
    assert not decide_pressure(evidence, mid_action=True)["should_speak"]


def test_sampling_gap_withholds_continuity():
    data = samples([0, 40, 80]) + [(2, [player(500), player(200)])]
    assert extract_evidence(data, "right")["tracking_interrupted"]
