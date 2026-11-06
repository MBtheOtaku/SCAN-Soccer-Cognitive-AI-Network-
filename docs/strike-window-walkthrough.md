# Strike-window evidence: first implementation

Implement these steps in order. This guide adds measurements; it does not validate a biomechanics model. Keep the existing strike detector, tracking, controller, confidence thresholds and goal behaviour.

The window is centred on the **estimated ankle-speed peak**, not confirmed ball contact. All torso angles are unsigned 2D image measurements. Camera motion, perspective and landmark errors can change them. A large angle range alone must never become a diagnosis or an instruction to stay steadier.

## 1. Create `src/backend/strike_window.py`

This module uses only the Python standard library. The data-quality thresholds below are provisional engineering checks, not coaching thresholds or calibrated probabilities.

```python
from __future__ import annotations

import math
from statistics import median
from typing import Protocol, Sequence


class PoseSample(Protocol):
    frame_idx: int
    pose_confidence: float
    torso_lean_deg: float


def build_strike_window(
    metrics: Sequence[PoseSample],
    strike_frame: int | None,
    fps: float,
    total_frames: int,
) -> dict:
    result = {
        "status": "unavailable",
        "anchor": "estimated_ankle_speed_peak_not_confirmed_contact",
        "coordinate_system": "unsigned_2d_torso_angle_to_image_vertical",
        "reasons": [],
        "phases": {},
    }
    if (
        strike_frame is None
        or not math.isfinite(fps)
        or fps <= 0
        or total_frames <= 0
        or not 0 <= strike_frame < total_frames
    ):
        result["reasons"] = ["No valid strike anchor or video timing."]
        return result

    radius = max(2, round(0.30 * fps))
    centre = max(1, round(0.10 * fps))
    start, end = strike_frame - radius, strike_frame + radius
    result.update({
        "strike_frame": strike_frame,
        "requested_start_frame": start,
        "requested_end_frame": end,
        "half_window_s": radius / fps,
    })

    # Use actual frame indices, not positions in the valid-pose list.
    # Never bridge missing observations by interpolating a pose.
    samples = {}
    for sample in metrics:
        if not (max(0, start) <= sample.frame_idx <= min(total_frames - 1, end)):
            continue
        if not (
            math.isfinite(sample.pose_confidence)
            and 0 <= sample.pose_confidence <= 1
            and math.isfinite(sample.torso_lean_deg)
            and 0 <= sample.torso_lean_deg <= 180
        ):
            continue
        previous = samples.get(sample.frame_idx)
        if previous is None or sample.pose_confidence > previous.pose_confidence:
            samples[sample.frame_idx] = sample

    frames = sorted(samples)
    expected = end - start + 1
    coverage = len(frames) / expected
    confidences = [samples[frame].pose_confidence for frame in frames]
    median_confidence = median(confidences) if confidences else None

    # Include missing frames at both edges, not just internal gaps.
    anchors = [start - 1, *frames, end + 1]
    largest_gap = max(b - a - 1 for a, b in zip(anchors, anchors[1:]))
    reasons = result["reasons"]
    if start < 0 or end >= total_frames:
        reasons.append("The requested window extends beyond the clip.")
    if coverage < 0.70:
        reasons.append("Fewer than 70% of requested window frames have valid pose measurements.")
    if median_confidence is None or median_confidence < 0.50:
        reasons.append("Median landmark confidence is below the provisional 0.50 check.")
    if largest_gap > max(2, math.ceil(0.10 * fps)):
        reasons.append("A pose gap exceeds the provisional continuity check.")

    ranges = {
        "before": (start, strike_frame - centre - 1),
        "near": (strike_frame - centre, strike_frame + centre),
        "after": (strike_frame + centre + 1, end),
    }
    for name, (lo, hi) in ranges.items():
        phase = [samples[frame] for frame in frames if lo <= frame <= hi]
        angles = [sample.torso_lean_deg for sample in phase]
        result["phases"][name] = {
            "sample_count": len(phase),
            "expected_frames": max(0, hi - lo + 1),
            "torso_median_deg": median(angles) if angles else None,
            "torso_range_deg": max(angles) - min(angles) if len(angles) >= 2 else None,
        }
        if len(phase) < 2:
            reasons.append(f"Too few observations in the {name} phase.")

    result.update({
        "status": "usable" if not reasons else "limited" if frames else "unavailable",
        "valid_frames": len(frames),
        "expected_frames": expected,
        "coverage": coverage,
        "median_landmark_confidence": median_confidence,
        "min_landmark_confidence": min(confidences) if confidences else None,
        "largest_missing_run_frames": largest_gap,
    })
    return result


def describe_strike_window(evidence: dict | None) -> str | None:
    if not evidence:
        return None
    if evidence.get("status") != "usable":
        reasons = evidence.get("reasons") or ["Insufficient local pose observations."]
        return "Strike-window measurement limits: " + " ".join(reasons)

    phases = evidence["phases"]
    before = phases["before"]["torso_median_deg"]
    near = phases["near"]["torso_median_deg"]
    after = phases["after"]["torso_median_deg"]
    return (
        "Movement observation: median torso angle to image vertical was "
        f"{before:.1f} degrees before, {near:.1f} degrees near, and "
        f"{after:.1f} degrees after the estimated ankle-speed peak. "
        "These are 2D measurements, not confirmed contact timing or a "
        "technique-quality rating; camera motion, perspective and tracking "
        "can affect them."
    )
```

## 2. Update `src/backend/situated_state.py`

Inside `SituatedState`, immediately after `recent_improvement: Optional[bool]`, add this **last dataclass field**, before the Utility section:

```python
    # Optional for compatibility with existing callers and saved states.
    strike_window: Optional[dict] = None
```

Keeping the defaulted field after all required fields preserves existing constructors. `to_dict()` already uses `asdict`, so it automatically includes this evidence for Jev and saved analysis results.

## 3. Update `src/backend/pose_mvp/scan_pose_mvp_v2.py`

Add these imports beside the existing backend imports:

```python
from src.backend.strike_window import (
    build_strike_window,
    describe_strike_window,
)
```

Find this existing block near the end of `analyze()`:

```python
    situated_state = build_situated_state(
        summary,
        ball_summary,
        goal_outcome,
        rep_id=1,
    )
```

Immediately **after** it, before `run_jev_judgements`, insert:

```python
    situated_state.strike_window = build_strike_window(
        metrics=metrics,
        strike_frame=summary.probable_strike_frame,
        fps=fps,
        total_frames=summary.total_frames,
    )
```

Later, find the existing call to `refine_coaching_report_with_llm`. Immediately **after that entire call**, and before the synthesis-status print, add:

```python
    # Add an exact measurement note after synthesis so it survives both
    # external-service failures and language rewriting. Keep it out of speech.
    measurement_note = describe_strike_window(situated_state.strike_window)
    if measurement_note:
        coaching_report.confidence_note = " ".join(
            text for text in (coaching_report.confidence_note, measurement_note)
            if text
        )
```

The existing saved `situated_state` and `coaching_report` already flow through the API to the mobile report. No mobile changes or response-schema changes are needed. This does not alter controller decisions or spoken cues. The existing whole-clip `perception_usable` check still gates Jev's technique question; a usable local window does not override it in this first version.

## 4. Update `src/backend/jev_judgements.py`

In `_build_jev_state()`, locate the `"evidence_rules": [` list. Add these strings to that list:

```python
            "rep.strike_window summarises observations around an estimated "
            "ankle-speed peak; it does not establish actual ball contact.",
            "Use strike-window phase comparisons only when status is usable. "
            "Otherwise explain its reasons and do not diagnose from its angles.",
            "Usable means a provisional local data-quality check passed, "
            "not validated biomechanics, accurate player identity, a calibrated "
            "confidence probability or confirmed contact timing.",
            "Torso angles and angle ranges are unsigned 2D image measurements. "
            "Camera motion, perspective, normal action and landmark errors can "
            "change them. They do not establish forward/backward lean, loss of "
            "balance, support instability or harmful technique.",
            "There is no validated desirable torso-angle range in this state. "
            "Do not invent one or prescribe a correction from angle changes alone. "
            "Missing shot outcome does not itself prove poor technique.",
```

Do not change question probabilities, the choices, or correction thresholds. Jev may still correctly return `insufficient_evidence`.

## 5. Update `src/backend/llm_coaching.py`

In `_build_evidence_packet()`, inside its `"situated_state": { ... }` dictionary, add:

```python
            "strike_window": situated_state.strike_window,
```

In the existing `COACHING_INSTRUCTIONS` text, under `CRITICAL RULES`, add:

```text
- strike_window is descriptive 2D evidence around an estimated ankle-speed
  peak, not confirmed ball contact. Use phase comparisons only if its status
  is usable. Otherwise explain the supplied measurement limitations.
- Changes in torso angles do not by themselves prove instability, poor
  technique, forward/backward lean or a need for correction. There is no
  validated ideal angle range supplied here. Do not invent one.
```

This changes the evidence and wording instructions only; the API request format, model selection and deterministic fallback stay intact.

## 6. Create `tests/test_strike_window.py`

```python
import json
from types import SimpleNamespace

from src.backend.strike_window import build_strike_window, describe_strike_window


def samples(frames=range(100), confidence=0.9):
    return [SimpleNamespace(
        frame_idx=i,
        pose_confidence=confidence,
        torso_lean_deg=10.0 if i < 47 else 20.0 if i <= 53 else 30.0,
    ) for i in frames]


def test_complete_window_has_three_phases_and_no_diagnosis():
    evidence = build_strike_window(samples(), 50, 30.0, 100)
    assert evidence["status"] == "usable"
    assert evidence["expected_frames"] == 19
    assert evidence["coverage"] == 1.0
    assert [evidence["phases"][p]["torso_median_deg"]
            for p in ("before", "near", "after")] == [10, 20, 30]
    assert "not confirmed contact" in describe_strike_window(evidence)
    json.dumps(evidence, allow_nan=False)


def test_local_gap_is_not_hidden_by_whole_clip_coverage():
    evidence = build_strike_window(samples([i for i in range(100)
                                           if not 46 <= i <= 54]), 50, 30.0, 100)
    assert evidence["status"] == "limited"
    assert evidence["largest_missing_run_frames"] == 9
    assert evidence["phases"]["near"]["torso_median_deg"] is None


def test_clip_boundary_is_flagged():
    evidence = build_strike_window(samples(), 2, 30.0, 100)
    assert evidence["status"] == "limited"
    assert any("beyond the clip" in reason for reason in evidence["reasons"])


def test_low_confidence_and_nonfinite_measurements_are_not_usable():
    assert build_strike_window(samples(confidence=0.4), 50, 30, 100)["status"] == "limited"
    rows = samples()
    for row in rows:
        row.torso_lean_deg = float("nan")
    evidence = build_strike_window(rows, 50, 30, 100)
    assert evidence["status"] == "unavailable"
    json.dumps(evidence, allow_nan=False)


def test_missing_anchor_bad_fps_and_empty_input_are_safe():
    for anchor, fps in ((None, 30), (50, 0), (50, float("nan"))):
        assert build_strike_window(samples(), anchor, fps, 100)["status"] == "unavailable"
    assert build_strike_window([], 50, 30, 100)["status"] == "unavailable"
    assert describe_strike_window(None) is None


def test_duplicate_frames_do_not_inflate_coverage():
    rows = samples()
    evidence = build_strike_window(rows + rows, 50, 30, 100)
    assert evidence["coverage"] == 1.0
    assert evidence["valid_frames"] == 19


def test_frame_rate_changes_window_size():
    evidence = build_strike_window(samples(range(200)), 100, 60, 200)
    assert evidence["expected_frames"] == 37
    assert evidence["half_window_s"] == 0.3
```

## 7. Run checks from SCAN's root

Use your existing working backend Python environment. No new dependencies are required for the feature.

```powershell
python -m pytest tests/test_strike_window.py tests/test_controller.py -q
python -m compileall -q src/backend/strike_window.py src/backend/situated_state.py src/backend/pose_mvp/scan_pose_mvp_v2.py src/backend/jev_judgements.py src/backend/llm_coaching.py
```

Then stop the backend with Ctrl+C and restart it in the **same terminal containing your API-key environment variables**:

```powershell
python -m uvicorn src.backend.api.main:app --host 127.0.0.1 --port 8000
```

Expo can stay running because the mobile code has not changed.

## 8. Compare the same saved reps

Retry analysis on three existing reps: one clear view, one with missing pose observations, and one with an awkward camera angle. Keep the previous results for comparison.

For each new analysis, inspect `data/output/<new-analysis-id>/scan_shot_summary.json`:

- `situated_state.strike_window` must contain the phases, quality checks and reasons.
- A usable window must produce a factual angle summary under Confidence and limitations, even if an external service fails.
- Limited windows must explain their measurement limits without presenting a confident angle comparison.
- `jev_judgements.status` and `coaching_synthesis.status` distinguish successful services from fallback operation.
- Shot outcome must remain unknown without goal evidence. No new spoken cue should be produced just because a torso angle changes.

Review the annotated video at `probable_strike_time_s`: does the estimated event actually align with the athlete's strike, and is the pose tracking the intended player? A clean data window cannot establish either fact. If the anchor is wrong, fixing contact-event estimation is a separate follow-up; do not tune coaching thresholds to compensate.

This first version gives the report more factual detail. It does **not** promise more corrections: a trustworthy pipeline can still conclude that no specific correction is justified. Save broader biomechanics validation, camera compensation, mobile goal calibration and cross-rep comparisons for later.
