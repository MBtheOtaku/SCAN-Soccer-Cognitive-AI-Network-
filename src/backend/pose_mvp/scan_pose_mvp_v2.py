"""
SCAN Pose MVP v2
================
Video -> cropped pose estimation -> skeleton overlay -> per-frame metrics ->
probable strike event -> conservative prototype feedback.

This is an engineering prototype, not a validated soccer biomechanics model.
Thresholds are intentionally labeled as provisional and should later be replaced
with evidence-based or learned rules.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from src.backend.situated_state import SituatedState
from src.backend.ball_tracking import (
    BallTrackSummary,
    track_ball,
)
from src.backend.goal_tracking import (
    GoalCalibration,
    GoalOutcome,
    analyze_goal_outcome,
)
import cv2
import numpy as np
import subprocess
import random 

# Load MediaPipe dynamically so static analysis does not require its optional
# package metadata to be installed in the editor's Python environment.
mp = importlib.import_module("mediapipe")

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

SKELETON_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


@dataclass
class ROI:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class FrameMetrics:
    frame_idx: int
    timestamp_s: float
    pose_confidence: float
    left_knee_angle_deg: float
    right_knee_angle_deg: float
    torso_lean_deg: float
    stance_ratio: float
    shoulder_width_px: float
    hip_mid_x: float
    hip_mid_y: float
    left_ankle_x: float
    left_ankle_y: float
    right_ankle_x: float
    right_ankle_y: float


@dataclass
class ShotSummary:
    total_frames: int
    valid_pose_frames: int
    pose_coverage: float
    probable_striking_leg: Optional[str]
    probable_strike_frame: Optional[int]
    probable_strike_time_s: Optional[float]
    strike_peak_relative_ankle_speed_px_s: Optional[float]
    strike_torso_lean_deg: Optional[float]
    strike_left_knee_angle_deg: Optional[float]
    strike_right_knee_angle_deg: Optional[float]
    prototype_feedback_code: Optional[str]
    prototype_feedback: str


def parse_roi(text: Optional[str], width: int, height: int) -> ROI:
    if not text:
        return ROI(0, 0, width, height)
    try:
        x1, y1, x2, y2 = [int(v.strip()) for v in text.split(",")]
    except Exception as exc:
        raise ValueError("ROI must be x1,y1,x2,y2") from exc
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))

    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    if x2 - x1 < 50 or y2 - y1 < 50:
        raise ValueError("ROI is too small")

    return ROI(x1, y1, x2, y2)


def visibility(landmarks, idx: int) -> float:
    lm = landmarks[idx]
    vals = []
    if hasattr(lm, "visibility"):
        vals.append(float(lm.visibility))
    if hasattr(lm, "presence"):
        vals.append(float(lm.presence))
    return min(vals) if vals else 1.0


def px(landmarks, idx: int, roi: ROI) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([
        roi.x1 + float(lm.x) * roi.width,
        roi.y1 + float(lm.y) * roi.height,
    ], dtype=np.float64)


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return float("nan")
    cosv = float(np.dot(ba, bc) / denom)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def torso_lean_deg(landmarks, roi: ROI) -> float:
    shoulder_mid = (px(landmarks, LEFT_SHOULDER, roi) + px(landmarks, RIGHT_SHOULDER, roi)) / 2
    hip_mid = (px(landmarks, LEFT_HIP, roi) + px(landmarks, RIGHT_HIP, roi)) / 2
    torso = shoulder_mid - hip_mid
    denom = np.linalg.norm(torso)
    if denom < 1e-9:
        return float("nan")
    vertical = np.array([0.0, -1.0])
    cosv = float(np.dot(torso, vertical) / denom)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def compute_metrics(landmarks, roi: ROI, frame_idx: int, timestamp_s: float) -> Optional[FrameMetrics]:
    required = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
                LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    conf = min(visibility(landmarks, i) for i in required)
    if conf < 0.35:
        return None

    ls, rs = px(landmarks, LEFT_SHOULDER, roi), px(landmarks, RIGHT_SHOULDER, roi)
    lh, rh = px(landmarks, LEFT_HIP, roi), px(landmarks, RIGHT_HIP, roi)
    lk, rk = px(landmarks, LEFT_KNEE, roi), px(landmarks, RIGHT_KNEE, roi)
    la, ra = px(landmarks, LEFT_ANKLE, roi), px(landmarks, RIGHT_ANKLE, roi)

    shoulder_width = float(np.linalg.norm(ls - rs))
    stance = float(np.linalg.norm(la - ra) / shoulder_width) if shoulder_width > 1e-6 else float("nan")
    hip_mid = (lh + rh) / 2

    m = FrameMetrics(
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        pose_confidence=conf,
        left_knee_angle_deg=angle_deg(lh, lk, la),
        right_knee_angle_deg=angle_deg(rh, rk, ra),
        torso_lean_deg=torso_lean_deg(landmarks, roi),
        stance_ratio=stance,
        shoulder_width_px=shoulder_width,
        hip_mid_x=float(hip_mid[0]),
        hip_mid_y=float(hip_mid[1]),
        left_ankle_x=float(la[0]), left_ankle_y=float(la[1]),
        right_ankle_x=float(ra[0]), right_ankle_y=float(ra[1]),
    )
    nums = [m.left_knee_angle_deg, m.right_knee_angle_deg, m.torso_lean_deg,
            m.stance_ratio, m.shoulder_width_px]
    return m if all(np.isfinite(v) for v in nums) else None


def smooth1d(x: np.ndarray, window: int = 5) -> np.ndarray:
    if len(x) < window or window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[:len(x)]


def estimate_strike(metrics: Sequence[FrameMetrics], fps: float) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    """
    Engineering heuristic: estimate the striking leg as the ankle with the largest
    velocity relative to the player's hip midpoint. This removes much of the forward
    running motion and emphasizes leg swing.
    """
    if len(metrics) < 8:
        return None, None, None

    # Metrics may have gaps, so use timestamps rather than assuming adjacent frames.
    t = np.array([m.timestamp_s for m in metrics], dtype=float)
    hip = np.array([[m.hip_mid_x, m.hip_mid_y] for m in metrics])
    left = np.array([[m.left_ankle_x, m.left_ankle_y] for m in metrics]) - hip
    right = np.array([[m.right_ankle_x, m.right_ankle_y] for m in metrics]) - hip

    for arr in (left, right):
        arr[:, 0] = smooth1d(arr[:, 0], 5)
        arr[:, 1] = smooth1d(arr[:, 1], 5)

    dt = np.gradient(t)
    dt[dt < 1e-4] = 1.0 / max(fps, 1.0)
    lv = np.sqrt(np.gradient(left[:, 0])**2 + np.gradient(left[:, 1])**2) / dt
    rv = np.sqrt(np.gradient(right[:, 0])**2 + np.gradient(right[:, 1])**2) / dt
    lv = smooth1d(lv, 5)
    rv = smooth1d(rv, 5)

    li, ri = int(np.argmax(lv)), int(np.argmax(rv))
    if lv[li] >= rv[ri]:
        return "left", li, float(lv[li])
    return "right", ri, float(rv[ri])


def prototype_feedback(metrics: Sequence[FrameMetrics], strike_idx: Optional[int], striking_leg: Optional[str]) -> Tuple[Optional[str], str]:
    if not metrics:
        return (
        None,
        "A reliable player pose could not be established "
        "for this video."
        )

    if strike_idx is None or striking_leg is None:
        return (
            None,
            "Pose was tracked, but a strike event "
            "could not be estimated reliably."
        )

    # Use a small strike window instead of medians across the entire approach/follow-through.
    lo, hi = max(0, strike_idx - 3), min(len(metrics), strike_idx + 4)
    window = metrics[lo:hi]
    lean_span = max(m.torso_lean_deg for m in window) - min(m.torso_lean_deg for m in window)

    # Provisional engineering threshold. It is intentionally not presented as validated biomechanics.
    if lean_span > 18.0:
        return "upper_body_variability", "Prototype cue: keep your upper body a little steadier through the strike."

    plant_leg = "right" if striking_leg == "left" else "left"

    options = [
        (
            f"Nice strike. Try to keep your {plant_leg} side "
            "a little steadier through contact."
        ),
        (
            f"Good connection there. Stay a bit stronger over "
            f"that {plant_leg} side as you hit through it."
        ),
        (
            f"That looked solid. Keep the {plant_leg} side "
            "composed through contact."
        ),
        (
            f"Good rep. Just stay a little more stable on the "
            f"{plant_leg} side as you strike."
        ),
        (
            f"Nice one. Hold that {plant_leg} side steady "
            "through the ball."
        ),
    ]

    return (
        "strike_detected",
        random.choice(options),
    )

def build_summary(metrics: Sequence[FrameMetrics], total_frames: int, fps: float) -> ShotSummary:
    coverage = (len(metrics) / total_frames) if total_frames else 0.0
    leg, idx, peak = estimate_strike(metrics, fps)
    if idx is None:
        code, feedback = prototype_feedback(metrics, None, None)
        return ShotSummary(total_frames, len(metrics), coverage, None, None, None, None,
                           None, None, None, code, feedback)

    m = metrics[idx]
    code, feedback = prototype_feedback(metrics, idx, leg)
    return ShotSummary(
        total_frames=total_frames,
        valid_pose_frames=len(metrics),
        pose_coverage=coverage,
        probable_striking_leg=leg,
        probable_strike_frame=m.frame_idx,
        probable_strike_time_s=m.timestamp_s,
        strike_peak_relative_ankle_speed_px_s=peak,
        strike_torso_lean_deg=m.torso_lean_deg,
        strike_left_knee_angle_deg=m.left_knee_angle_deg,
        strike_right_knee_angle_deg=m.right_knee_angle_deg,
        prototype_feedback_code=code,
        prototype_feedback=feedback,
    )

def build_situated_state(
    summary: ShotSummary,
    ball_summary: BallTrackSummary,
    goal_outcome: GoalOutcome,
    rep_id: int = 1,
) -> SituatedState:
    """
    Convert the completed pose-analysis result into a contextual
    representation of the player's shooting rep.

    This is the bridge between low-level perception and later
    probabilistic / cognitive reasoning.
    """

    # -----------------------------------------------------
    # Determine support leg
    # -----------------------------------------------------

    support_leg: Optional[str] = None

    if summary.probable_striking_leg == "left":
        support_leg = "right"

    elif summary.probable_striking_leg == "right":
        support_leg = "left"

    # -----------------------------------------------------
    # Perception quality
    # -----------------------------------------------------

    # Provisional threshold.
    # Later this should be validated rather than hard-coded.
    perception_usable = (
        summary.pose_coverage >= 0.60
    )

    # -----------------------------------------------------
    # Build contextual state
    # -----------------------------------------------------

    return SituatedState(
        rep_id=rep_id,

        drill_type="shooting",
        task_goal="execute a controlled shot",

        # The current MVP analyzes the whole uploaded rep,
        # so by this point the action has completed.
        action_phase="rep_complete",

        pose_coverage=summary.pose_coverage,
        perception_usable=perception_usable,

        striking_leg=summary.probable_striking_leg,
        support_leg=support_leg,

        strike_frame=summary.probable_strike_frame,
        strike_time_s=summary.probable_strike_time_s,

        torso_lean_deg=summary.strike_torso_lean_deg,
        left_knee_deg=summary.strike_left_knee_angle_deg,
        right_knee_deg=summary.strike_right_knee_angle_deg,

        # Ball / environment evidence
        ball_visible=ball_summary.detected,

        ball_tracking_confidence=(
            ball_summary.mean_confidence
        ),

        ball_detection_rate=(
            ball_summary.detection_rate
        ),

        ball_final_x_norm=(
            ball_summary.final_x_norm
        ),

        ball_final_y_norm=(
            ball_summary.final_y_norm
        ),

        shot_on_target=(
            goal_outcome.shot_on_target
        ),

        goal_entry_detected=(
            goal_outcome.goal_entry_detected
        ),

        goal_zone=(
            goal_outcome.goal_zone
        ),

        environment_confidence=(
            ball_summary.mean_confidence
            if ball_summary.detected
            else None
        ),

        # PlayerState history will populate these later.
        repeated_error_type=None,
        repeated_error_count=0,

        seconds_since_last_feedback=None,
        previous_feedback_type=None,
        recent_improvement=None,
    )

def draw_pose(frame: np.ndarray, landmarks, roi: ROI) -> None:
    points: Dict[int, Tuple[int, int]] = {}
    for idx in range(len(landmarks)):
        if visibility(landmarks, idx) < 0.25:
            continue
        p = px(landmarks, idx, roi)
        points[idx] = (int(round(p[0])), int(round(p[1])))

    for a, b in SKELETON_CONNECTIONS:
        if a in points and b in points:
            cv2.line(frame, points[a], points[b], (255, 255, 255), 2, cv2.LINE_AA)
    for idx in [11,12,23,24,25,26,27,28,29,30,31,32]:
        if idx in points:
            cv2.circle(frame, points[idx], 4, (255,255,255), -1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, metric: Optional[FrameMetrics], roi: ROI) -> None:
    if roi.x1 or roi.y1 or roi.x2 != frame.shape[1] or roi.y2 != frame.shape[0]:
        cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), (255,255,255), 1)
    lines = ["SCAN Pose MVP v2"]
    if metric is None:
        lines.append("pose: uncertain")
    else:
        lines += [
            f"pose conf: {metric.pose_confidence:.2f}",
            f"L knee: {metric.left_knee_angle_deg:.1f} deg",
            f"R knee: {metric.right_knee_angle_deg:.1f} deg",
            f"torso lean: {metric.torso_lean_deg:.1f} deg",
        ]
    y = 28
    for s in lines:
        cv2.putText(frame, s, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, s, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 1, cv2.LINE_AA)
        y += 26


def save_csv(metrics: Sequence[FrameMetrics], path: Path) -> None:
    fields = list(FrameMetrics.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in metrics:
            w.writerow(asdict(m))


def detect_player_roi(
    video_path: Path,
    model_path: Path,
    samples: int = 20,
    padding: float = 0.35,
) -> ROI:
    """
    Estimate one fixed player ROI by independently sampling frames
    across the video.

    This calibration pass uses IMAGE mode because sampled frames are
    not processed sequentially.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = (
        mp.tasks.vision.PoseLandmarkerOptions
    )
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.25,
        min_pose_presence_confidence=0.25,
        output_segmentation_masks=False,
    )

    boxes = []

    sample_indices = np.linspace(
        0,
        max(frame_count - 1, 0),
        min(samples, frame_count),
        dtype=int,
    )

    core_landmarks = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE,
    ]

    try:
        with PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            for frame_index in sample_indices:
                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(frame_index),
                )

                ok, frame = cap.read()

                if not ok:
                    continue

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                # IMAGE mode: each sampled frame is independent.
                result = landmarker.detect(mp_image)

                if not result.pose_landmarks:
                    continue

                landmarks = result.pose_landmarks[0]

                # Reject weak partial detections.
                reliable_core = [
                    idx
                    for idx in core_landmarks
                    if visibility(landmarks, idx) >= 0.25
                ]

                if len(reliable_core) < 6:
                    continue

                visible = [
                    lm
                    for lm in landmarks
                    if getattr(
                        lm,
                        "visibility",
                        1.0,
                    ) >= 0.25
                ]

                if not visible:
                    continue

                xs = [
                    max(
                        0.0,
                        min(width, lm.x * width),
                    )
                    for lm in visible
                ]

                ys = [
                    max(
                        0.0,
                        min(height, lm.y * height),
                    )
                    for lm in visible
                ]

                boxes.append(
                    (
                        min(xs),
                        min(ys),
                        max(xs),
                        max(ys),
                    )
                )

    finally:
        cap.release()

    if not boxes:
        raise RuntimeError(
            "Automatic player calibration failed: "
            "no reliable full-body pose detected."
        )

    # Movement envelope across the entire sampled rep.
    box_array = np.array(boxes, dtype=float)

    x1 = np.percentile(box_array[:, 0], 10)
    y1 = np.percentile(box_array[:, 1], 10)

    x2 = np.percentile(box_array[:, 2], 90)
    y2 = np.percentile(box_array[:, 3], 90)

    box_width = x2 - x1
    box_height = y2 - y1

    x_padding = max(
        box_width * padding,
        width * 0.05,
    )

    y_padding = max(
        box_height * padding,
        height * 0.05,
    )

    x1 -= x_padding
    x2 += x_padding
    y1 -= y_padding
    y2 += y_padding

    # -----------------------------------------------------
    # Minimum crop size
    # Prevent tiny partial-body ROIs.
    # -----------------------------------------------------

    minimum_width = width * 0.35
    minimum_height = height * 0.50

    current_width = x2 - x1
    current_height = y2 - y1

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    if current_width < minimum_width:
        half_width = minimum_width / 2
        x1 = center_x - half_width
        x2 = center_x + half_width

    if current_height < minimum_height:
        half_height = minimum_height / 2
        y1 = center_y - half_height
        y2 = center_y + half_height

    # Clamp final ROI to video dimensions.
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))

    if x2 - x1 < 50 or y2 - y1 < 50:
        raise RuntimeError(
            "Automatic player ROI was too small."
        )

    return ROI(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def analyze(
    video: Path,
    model: Path,
    output: Path,
    csv_path: Path,
    json_path: Path,
    roi_text: Optional[str],
) -> ShotSummary:

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------------------------------------------------------
    # ROI calibration
    # ---------------------------------------------------------

    if roi_text:
        # Manual ROI still supported for debugging/testing.
        roi = parse_roi(
            roi_text,
            width,
            height,
        )

        print(
            f"Using manual ROI: "
            f"{roi.x1},{roi.y1},{roi.x2},{roi.y2}"
        )

    else:
        print(
            "No ROI provided. "
            "Running automatic player calibration..."
        )

        try:
            roi = detect_player_roi(
                video,
                model,
            )

            print(
                f"Automatic ROI: "
                f"{roi.x1},{roi.y1},{roi.x2},{roi.y2}"
            )

        except RuntimeError as exc:
            print(
                f"Automatic calibration failed ({exc}). "
                "Falling back to full frame."
            )

            roi = ROI(
                x1=0,
                y1=0,
                x2=width,
                y2=height,
            )

    # ---------------------------------------------------------
    # Temporary OpenCV output
    # ---------------------------------------------------------

    temp_output = output.with_name(
        f"{output.stem}_temp.mp4"
    )

    writer = cv2.VideoWriter(
        str(temp_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create {temp_output}"
        )

    # ---------------------------------------------------------
    # MediaPipe setup
    # ---------------------------------------------------------

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = (
        mp.tasks.vision.PoseLandmarkerOptions
    )
    RunningMode = mp.tasks.vision.RunningMode

    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(model)
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
        output_segmentation_masks=False,
    )

    metrics: List[FrameMetrics] = []
    frame_idx = 0

    # ---------------------------------------------------------
    # Frame-by-frame pose analysis
    # ---------------------------------------------------------

    try:
        with PoseLandmarker.create_from_options(
            opts
        ) as landmarker:

            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                crop = frame[
                    roi.y1:roi.y2,
                    roi.x1:roi.x2,
                ]

                rgb = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = int(
                    round(frame_idx / fps * 1000)
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                fm = None

                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]

                    fm = compute_metrics(
                        lms,
                        roi,
                        frame_idx,
                        frame_idx / fps,
                    )

                    draw_pose(
                        frame,
                        lms,
                        roi,
                    )

                    if fm is not None:
                        metrics.append(fm)

                draw_hud(
                    frame,
                    fm,
                    roi,
                )

                writer.write(frame)

                frame_idx += 1

    finally:
        cap.release()
        writer.release()

    # ---------------------------------------------------------
    # Convert OpenCV output to browser-compatible H.264
    # ---------------------------------------------------------

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_output),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output),
        ],
        check=True,
    )

    # Remove temporary mp4v file after conversion.
    temp_output.unlink(missing_ok=True)

    # ---------------------------------------------------------
    # Save analysis artifacts
    # ---------------------------------------------------------

    save_csv(
        metrics,
        csv_path,
    )

    summary = build_summary(
        metrics,
        frame_idx,
        fps,
    )

    # ---------------------------------------------------------
    # Ball detection / tracking
    # ---------------------------------------------------------

    ball_result = track_ball(
        video,
    )

    ball_summary = ball_result.summary

    # ---------------------------------------------------------
    # Goal analysis
    # ---------------------------------------------------------

    goal_calibration_path = Path(
        "data/goal_calibration.json"
    )

    if goal_calibration_path.exists():

        goal_calibration = (
            GoalCalibration.load(
                goal_calibration_path
            )
        )

        goal_outcome = (
            analyze_goal_outcome(
                detections=(
                    ball_result.detections
                ),

                strike_frame=(
                    summary.probable_strike_frame
                ),

                calibration=(
                    goal_calibration
                ),

                frame_width=width,
                frame_height=height,
            )
        )

    else:

        print(
            "No goal calibration found. "
            "Goal analysis skipped."
        )

        goal_outcome = GoalOutcome(
            goal_calibrated=False,

            goal_entry_detected=None,
            shot_on_target=None,

            entry_x_norm=None,
            entry_y_norm=None,

            entry_frame=None,
            entry_time_s=None,

            goal_zone=None,
            confidence=None,
        )

    print(
        "Ball tracking: "
        f"detected={ball_summary.detected}, "
        f"detections={ball_summary.detection_count}, "
        f"rate={ball_summary.detection_rate:.2f}"
    )

    # ---------------------------------------------------------
    # Situated state
    # ---------------------------------------------------------

    situated_state = build_situated_state(
        summary,
        ball_summary,
        goal_outcome,
        rep_id=1,
    )

    # ---------------------------------------------------------
    # Combined analysis payload
    # ---------------------------------------------------------

    analysis_payload = asdict(summary)

    analysis_payload["ball_tracking"] = {
        **ball_summary.to_dict(),

        "trajectory": [
            asdict(detection)
            for detection in ball_result.detections
        ],
    }

    analysis_payload["goal_tracking"] = (
        goal_outcome.to_dict()
    )

    analysis_payload["situated_state"] = (
        situated_state.to_dict()
    )

    json_path.write_text(
        json.dumps(
            analysis_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return summary

def main() -> None:
    p = argparse.ArgumentParser(description="SCAN pose/strike MVP")
    p.add_argument("video", type=Path)
    p.add_argument("--model", type=Path, default=Path("pose_landmarker_full.task"))
    p.add_argument("--output", type=Path, default=Path("scan_annotated_v2.mp4"))
    p.add_argument("--csv", type=Path, default=Path("scan_pose_metrics_v2.csv"))
    p.add_argument("--json", type=Path, default=Path("scan_shot_summary.json"))
    p.add_argument("--roi", type=str, default=None, help="x1,y1,x2,y2; omit for automatic player calibration",)
    a = p.parse_args()
    if not a.video.exists():
        raise FileNotFoundError(a.video)
    if not a.model.exists():
        raise FileNotFoundError(f"Missing pose model: {a.model}")
    s = analyze(a.video, a.model, a.output, a.csv, a.json, a.roi)
    print(json.dumps(asdict(s), indent=2))
    print(f"\nAnnotated video: {a.output}\nMetrics CSV: {a.csv}\nShot JSON: {a.json}")
    print("\nPrototype only: feedback rules are not validated soccer biomechanics.")


if __name__ == "__main__":
    main()
