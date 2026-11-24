"""Causal image-plane motion support for intermittent detector observations.

This is a short-horizon search/association prior, not calibrated ballistics.
It never manufactures an observation, fills a hidden frame, or predicts a goal.
Positions must already share a registered reference-image coordinate system.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TrajectoryEvidenceSettings:
    confirmation_hits: int = 3
    support_window_frames: int = 5
    maximum_missing_frames: int = 1
    maximum_step_px: float = 12.0
    innovation_radius_px: float = 8.0
    ambiguity_margin_px: float = 2.0
    maximum_hypotheses: int = 8


class TrajectoryEvidence:
    """Match unique detector hits with bounded velocity prediction over one gap."""

    def __init__(self, settings=None):
        self.settings = settings or TrajectoryEvidenceSettings()
        s = self.settings
        if (any(type(v) is not int or v < 1 for v in
                (s.confirmation_hits, s.support_window_frames, s.maximum_hypotheses))
                or type(s.maximum_missing_frames) is not int or s.maximum_missing_frames < 0
                or s.confirmation_hits < 3 or s.support_window_frames < s.confirmation_hits
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in (s.maximum_step_px, s.innovation_radius_px, s.ambiguity_margin_px))):
            raise ValueError("Invalid trajectory evidence settings")
        self.hypotheses = []
        self.last_frame = None

    def configuration(self):
        return {**asdict(self.settings), "model": "bounded_constant_image_velocity",
                "predictions_are_observations": False,
                "certainty": "empirical_motion_support_not_identity_or_physics_proof"}

    def reset(self):
        self.hypotheses = []

    def update(self, frame, points, *, available=True):
        if type(frame) is not int or frame < 0 or (self.last_frame is not None and frame != self.last_frame + 1):
            raise ValueError("Trajectory evidence requires contiguous nonnegative frames")
        if type(available) is not bool:
            raise ValueError("Expected boolean registration availability")
        for point in points:
            if len(point) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
                raise ValueError("Expected finite reference-image points")
        self.last_frame = frame
        s = self.settings
        if not available or len(points) > s.maximum_hypotheses:
            self.reset()
            return []
        previous = []
        for history in self.hypotheses:
            history = [(f, p) for f, p in history if frame - f < s.support_window_frames]
            if history and frame - history[-1][0] <= s.maximum_missing_frames + 1:
                previous.append(history)

        def cost(point, history):
            last_frame, last = history[-1]
            dt = frame - last_frame
            if math.dist(point, last) > s.maximum_step_px * dt:
                return math.inf
            if len(history) < 2:
                return math.dist(point, last)
            old_frame, old = history[-2]
            velocity = [(last[i] - old[i]) / (last_frame - old_frame) for i in range(2)]
            predicted = [last[i] + velocity[i] * dt for i in range(2)]
            distance = math.dist(point, predicted)
            return distance if distance <= s.innovation_radius_px * math.sqrt(dt) else math.inf

        costs = [[cost(point, history) for history in previous] for point in points]

        def unique_best(entries):
            ranked = sorted((v, i) for i, v in enumerate(entries) if math.isfinite(v))
            if not ranked or len(ranked) > 1 and ranked[1][0] - ranked[0][0] <= s.ambiguity_margin_px:
                return None
            return ranked[0][1]

        results, updated, used = [], [], set()
        for i, point in enumerate(points):
            match = unique_best(costs[i])
            assigned = match is not None and unique_best([row[match] for row in costs]) == i
            if assigned:
                history = previous[match]
                used.add(match)
            else:
                history = []
            history = history + [(frame, tuple(point))]
            updated.append(history)
            results.append({"frames": [f for f, _ in history],
                            "supported": len(history) >= s.confirmation_hits,
                            "bridged_detector_gap": any(b[0] - a[0] > 1 for a, b in zip(history, history[1:])),
                            "prediction_residual_px": costs[i][match] if assigned else None})
        # An empty detector frame can retain a hypothesis, never output a point.
        updated.extend(history for j, history in enumerate(previous)
                       if j not in used and frame - history[-1][0] <= s.maximum_missing_frames)
        # Never choose a subset of competing hypotheses by arbitrary ordering.
        self.hypotheses = updated if len(updated) <= s.maximum_hypotheses else []
        if not self.hypotheses:
            for result in results:
                result["supported"] = False
        return results
