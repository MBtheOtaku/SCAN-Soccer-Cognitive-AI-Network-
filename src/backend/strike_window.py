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