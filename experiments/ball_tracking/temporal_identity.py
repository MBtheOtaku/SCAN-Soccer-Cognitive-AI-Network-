"""Conservative, causal identity pilot; no labels or outcome reasoning.

Keep a main motion estimate plus at most two tentative branches. A far proposal
cannot redirect the main estimate on a single frame. All accepted detections
remain provisional: motion consistency is not proof of target identity.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class IdentitySettings:
    core_min_px: float = 18.0
    core_max_px: float = 32.0
    core_speed_gain: float = 0.75
    broad_min_px: float = 95.0
    confidence_weight: float = 0.25
    confirmation_hits: int = 2
    minimum_branch_motion_px: float = 6.0
    maximum_branches: int = 2
    maximum_branch_age: int = 4
    maximum_prediction_gap: int = 12
    branch_padding_px: float = 16.0


def estimate_camera_translation(previous, current) -> tuple[tuple[float, float], dict]:
    """Estimate translation only; failed alignment supplies no correction."""
    import cv2
    import numpy as np

    if previous.shape != current.shape:
        raise ValueError("Camera alignment requires matching frame sizes")
    height, width = current.shape[:2]
    ratio = min(1.0, 360.0 / width)
    size = (round(width * ratio), round(height * ratio))
    def grey(frame):
        return cv2.GaussianBlur(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size),
                                (3, 3), 0).astype(np.float32)
    shift, response = cv2.phaseCorrelate(grey(previous), grey(current),
                                        cv2.createHanningWindow(size, cv2.CV_32F))
    if (not all(math.isfinite(v) for v in (*shift, response)) or response < 0.2
            or math.hypot(*shift) / ratio > min(width, height) * 0.08):
        return (0.0, 0.0), {"status": "unavailable_no_correction", "response_not_probability":
                            response if math.isfinite(response) else None}
    correction = (shift[0] / ratio, shift[1] / ratio)
    return correction, {"status": "estimated_translation_only", "shift_px": list(correction),
                         "response_not_probability": response}


@dataclass
class _Branch:
    center: tuple[float, float]
    velocity: tuple[float, float]
    motion: tuple[float, float]
    hits: int
    age: int
    cost: float
    candidate: dict


class TemporalIdentity:
    """One manual seed; only current/past proposals and camera motion are read."""

    def __init__(self, seed, *, scale: float = 1.0, settings: IdentitySettings | None = None):
        if len(seed) != 2 or not all(math.isfinite(v) for v in seed) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Expected a finite seed and positive image scale")
        self.center = tuple(map(float, seed))
        self.last_observed = self.center
        self.velocity = (0.0, 0.0)
        self.scale = scale
        self.settings = settings or IdentitySettings()
        self.misses = 0
        self.branches: list[_Branch] = []
        self.last_frame = None

    def configuration(self):
        return {**asdict(self.settings), "image_scale": self.scale,
                "identity_verified": False, "status": "offline_motion_identity_pilot"}

    def predict(self, camera_shift=(0.0, 0.0)):
        velocity = self.velocity if self.misses < self.settings.maximum_prediction_gap else (0.0, 0.0)
        return tuple(self.center[i] + velocity[i] + camera_shift[i] for i in range(2))

    def _radius(self, velocity):
        s = self.settings
        return max(s.core_min_px * self.scale,
                   min(s.core_max_px * self.scale, 12 * self.scale + s.core_speed_gain * math.hypot(*velocity)))

    def _cost(self, candidate, predicted, radius):
        return (math.dist(candidate["center"], predicted) / radius
                - self.settings.confidence_weight * math.log(max(candidate["confidence"], 1e-6)))

    def update(self, row: dict, camera_shift=(0.0, 0.0)) -> dict:
        frame = row["frame"]
        if self.last_frame is not None and frame != self.last_frame + 1:
            raise ValueError("Temporal identity requires contiguous frames")
        if not all(math.isfinite(v) for v in camera_shift):
            raise ValueError("Camera shift must be finite")
        candidates = row.get("candidates", [])
        for candidate in candidates:
            if (len(candidate["center"]) != 2 or not all(math.isfinite(v) for v in candidate["center"])
                    or not math.isfinite(candidate["confidence"]) or not 0 < candidate["confidence"] <= 1):
                raise ValueError("Invalid detector candidate")
        self.last_frame = frame
        predicted = self.predict(camera_shift)
        self.last_observed = tuple(self.last_observed[i] + camera_shift[i] for i in range(2))
        s = self.settings
        radius = self._radius(self.velocity)
        broad = max(s.broad_min_px * self.scale, 2 * math.hypot(*self.velocity) + 40 * self.scale)
        branch_searches = [(tuple(b.center[i] + b.velocity[i] + camera_shift[i] for i in range(2)),
                            self._radius(b.velocity) + s.branch_padding_px * self.scale) for b in self.branches]
        eligible = [c for c in candidates if math.dist(c["center"], predicted) <= broad
                    or any(math.dist(c["center"], target) <= gate for target, gate in branch_searches)]
        core = sorted((c for c in eligible if math.dist(c["center"], predicted) <= radius),
                      key=lambda c: self._cost(c, predicted, radius))
        chosen, reason = None, "no_supported_candidate"
        if self.misses >= s.maximum_prediction_gap:
            self.branches = []
            reason = "lost_requires_new_seed"
        else:
            ambiguous = (len(core) > 1 and self._cost(core[1], predicted, radius)
                         - self._cost(core[0], predicted, radius) < 0.2
                         and math.dist(core[0]["center"], core[1]["center"]) > 12 * self.scale)
            if core and not ambiguous and self.misses <= 1:
                chosen, reason = core[0], "near_prediction_provisional"
                self.branches = []
            else:
                updated = []
                for branch in self.branches:
                    target = tuple(branch.center[i] + branch.velocity[i] + camera_shift[i] for i in range(2))
                    branch_radius = self._radius(branch.velocity) + s.branch_padding_px * self.scale
                    matches = [c for c in eligible if math.dist(c["center"], target) <= branch_radius]
                    if not matches or branch.age >= s.maximum_branch_age:
                        continue
                    candidate = min(matches, key=lambda c: self._cost(c, target, branch_radius))
                    step = tuple(candidate["center"][i] - branch.center[i] - camera_shift[i] for i in range(2))
                    dot = sum(step[i] * branch.velocity[i] for i in range(2))
                    if dot < -0.25 * math.hypot(*step) * math.hypot(*branch.velocity):
                        continue
                    updated.append(_Branch(tuple(candidate["center"]),
                        tuple(0.65 * step[i] + 0.35 * branch.velocity[i] for i in range(2)),
                        tuple(branch.motion[i] + step[i] for i in range(2)), branch.hits + 1,
                        branch.age + 1, branch.cost + self._cost(candidate, target, branch_radius), candidate))
                for candidate in eligible:
                    if any(math.dist(candidate["center"], branch.center) < 12 * self.scale for branch in updated):
                        continue
                    velocity = tuple((candidate["center"][i] - self.last_observed[i]) / (self.misses + 1) for i in range(2))
                    updated.append(_Branch(tuple(candidate["center"]), velocity, (0.0, 0.0), 1, 1,
                                           self._cost(candidate, predicted, broad), candidate))
                updated.sort(key=lambda b: b.cost / b.hits)
                self.branches = updated[:s.maximum_branches]
                confirmed = [b for b in self.branches if b.hits >= s.confirmation_hits
                             and math.hypot(*b.motion) >= s.minimum_branch_motion_px * self.scale]
                # A close two-way ambiguity is unresolved even after a short run.
                if confirmed and not ambiguous:
                    winner = min(confirmed, key=lambda b: b.cost / b.hits)
                    chosen, reason = winner.candidate, "temporally_supported_provisional"
                    self.branches = []
                else:
                    reason = "ambiguous_candidates" if ambiguous else "awaiting_temporal_support" if eligible else "no_candidate"
        if chosen is not None:
            center = tuple(chosen["center"])
            measured = tuple((center[i] - self.last_observed[i]) / (self.misses + 1) for i in range(2))
            proposed = tuple(0.65 * measured[i] + 0.35 * self.velocity[i] for i in range(2))
            delta = tuple(proposed[i] - self.velocity[i] for i in range(2))
            limit = (95 * self.scale if math.hypot(*self.velocity) < 8 * self.scale else
                     max(12 * self.scale, 0.75 * math.hypot(*self.velocity)))
            factor = min(1.0, limit / max(math.hypot(*delta), 1e-9))
            self.velocity = tuple(self.velocity[i] + factor * delta[i] for i in range(2))
            self.center = self.last_observed = center
            self.misses = 0
        else:
            self.center = predicted
            self.misses += 1
        return {"clip": row["clip"], "frame": frame, "split": row["split"],
                "center": list(chosen["center"]) if chosen else None,
                "confidence": chosen["confidence"] if chosen else None,
                "detected": chosen is not None, "predicted": list(predicted),
                "gate_px": radius, "broad_gate_px": broad, "candidate_count": len(candidates),
                "eligible_count": len(eligible), "reason": reason, "missing_run": self.misses,
                "tentative_branches": len(self.branches), "identity_verified": False}
