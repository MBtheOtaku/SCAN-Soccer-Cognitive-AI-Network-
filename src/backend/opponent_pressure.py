"""Optional two-player, post-rep image-space pressure evidence.

No distance in metres, collision prediction, player-relative direction or escape
route is inferred. Roles are assigned by the user's initial camera-side selection.
Stop association at occlusion/ambiguity rather than manufacture a trajectory.
"""
from dataclasses import dataclass
from math import hypot, isfinite
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Person:
    box: tuple[float, float, float, float]
    confidence: float

    @property
    def foot(self):
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)

    @property
    def height(self):
        return self.box[3] - self.box[1]


def overlap(a: Person, b: Person) -> float:
    x1 = max(a.box[0], b.box[0])
    y1 = max(a.box[1], b.box[1])
    x2 = min(a.box[2], b.box[2])
    y2 = min(a.box[3], b.box[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    areas = [(p.box[2] - p.box[0]) * p.height for p in (a, b)]
    return intersection / max(sum(areas) - intersection, 1)


def distance(a: Person, b: Person) -> float:
    return hypot(a.foot[0] - b.foot[0], a.foot[1] - b.foot[1])


def extract_evidence(samples: list[tuple[float, list[Person]]], athlete_side: str) -> dict:
    if athlete_side not in {"left", "right"}:
        raise ValueError("Select the athlete's initial camera-left or camera-right position.")
    rows = []
    previous = None
    interrupted = False
    for timestamp, people in samples:
        people = [p for p in people if p.confidence >= .6 and p.height > 1 and all(isfinite(v) for v in p.box)]
        if len(people) != 2 or overlap(*people) > .12:
            interrupted = True
            break
        if previous is None:
            people.sort(key=lambda p: p.foot[0])
            athlete, opponent = people if athlete_side == "left" else people[::-1]
            if abs(athlete.foot[0] - opponent.foot[0]) < .3 * athlete.height:
                interrupted = True
                break
        else:
            pa, po = previous
            direct = distance(pa, people[0]) + distance(po, people[1])
            swapped = distance(pa, people[1]) + distance(po, people[0])
            scale = (pa.height + po.height) / 2
            if abs(direct - swapped) < .3 * scale:
                interrupted = True
                break
            athlete, opponent = people if direct < swapped else people[::-1]
            # A large jump, long sampling gap or scale discontinuity breaks identity.
            if timestamp - rows[-1][0] > .35 or max(distance(pa, athlete), distance(po, opponent)) > .65 * scale:
                interrupted = True
                break
            if any(not .7 < current.height / old.height < 1.4 for old, current in ((pa, athlete), (po, opponent))):
                interrupted = True
                break
        height = (athlete.height + opponent.height) / 2
        gap = distance(athlete, opponent) / height
        rows.append((timestamp, gap, athlete, opponent))
        previous = athlete, opponent

    events = []
    # Evaluate one-second windows, rather than classify an entire clip by endpoints.
    for end in range(3, len(rows)):
        candidates = [i for i in range(end) if .6 <= rows[end][0] - rows[i][0] <= 1.2]
        if not candidates:
            continue
        start = candidates[0]
        window = rows[start:end + 1]
        elapsed = window[-1][0] - window[0][0]
        drop = window[0][1] - window[-1][1]
        changes = [window[i][1] - window[i + 1][1] for i in range(len(window) - 1)]
        consistency = sum(change > 0 for change in changes) / len(changes)
        # Engineering thresholds in projected body-height units, not safety thresholds.
        if drop < .5 or drop / elapsed < .75 or consistency < .75 or window[-1][1] > 1.8:
            continue
        _, _, a, o = window[0]
        relative_x = o.foot[0] - a.foot[0]
        side = "left" if relative_x < -.25 * a.height else "right" if relative_x > .25 * a.height else "unclear"
        events.append({
            "start_s": round(window[0][0], 2), "end_s": round(window[-1][0], 2),
            "camera_side": side, "gap_drop_body_heights": round(drop, 3),
            "closing_rate_body_heights_per_s": round(drop / elapsed, 3),
            "mean_detection_confidence": round(sum((r[2].confidence + r[3].confidence) / 2 for r in window) / len(window), 3),
            "samples": len(window),
        })
        break  # One assessment/cue per rep.
    return {
        "status": "observed" if events else "inconclusive" if interrupted or len(rows) < 4 else "no_supported_approach",
        "coordinate_frame": "camera", "athlete_start_side": athlete_side,
        "tracked_samples": len(rows), "tracking_interrupted": interrupted,
        "events": events,
        "confidence_note": "Prototype closing thresholds are unvalidated. Image-space motion only; detector confidence is not a collision probability. Perspective, camera movement and occlusion limit interpretation. Camera-left/right is not the athlete's left/right. No cut direction or clear escape route is established.",
    }


def analyze_pressure(video: Path, athlete_side: str, model_path: Path, detector: Callable | None = None) -> dict:
    import cv2
    if detector is None:
        if not model_path.is_file():
            raise FileNotFoundError("Local person detector model is unavailable.")
        from ultralytics import YOLO
        model = YOLO(str(model_path))  # Local file only; never download a missing model.

        def detector(frame):
            result = model.predict(frame, classes=[0], conf=.6, verbose=False)[0]
            return [Person(tuple(box), float(conf)) for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]

    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError("Could not decode the pressure clip.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not isfinite(fps) or fps <= 0:
            raise ValueError("Clip timing is unavailable.")
        stride = max(1, round(fps / 5))
        samples = []
        # Bound optional analysis to the first 20 seconds / 100 samples.
        frame_idx = 0
        while len(samples) < 100 and frame_idx / fps <= 20:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                samples.append((frame_idx / fps, detector(frame)))
            frame_idx += 1
        return extract_evidence(samples, athlete_side)
    finally:
        cap.release()
