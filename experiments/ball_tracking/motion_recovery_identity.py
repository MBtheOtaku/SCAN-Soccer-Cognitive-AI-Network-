"""Causal, opt-in motion release after a local slow-ball recovery.

Accepted local movement earns a bounded flight corridor. Distant tentative
proposals never earn release. Repeated off-target background proposals are
excluded from that corridor; neither rule verifies ball identity.
"""

from collections import deque
from dataclasses import asdict, dataclass
import math

from experiments.ball_tracking.recovery_identity import RecoveryIdentity


@dataclass(frozen=True)
class MotionRecoverySettings:
    minimum_step_px: float = 12.0
    direction_cosine: float = 0.8
    confirmation_steps: int = 3
    stationary_radius_px: float = 4.0
    stationary_separation_px: float = 48.0
    stationary_prior_hits: int = 3
    stationary_history_frames: int = 8


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / max(math.hypot(*a) * math.hypot(*b), 1e-9)


class MotionRecoveryIdentity(RecoveryIdentity):
    """Release only after a contiguous, unambiguous accepted local launch."""

    def __init__(self, seed, *, scale=1.0, settings=None, recovery_settings=None,
                 motion_settings=None):
        super().__init__(seed, scale=scale, settings=settings,
                         recovery_settings=recovery_settings)
        self.motion_settings = motion_settings or MotionRecoverySettings()
        s = self.motion_settings
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 3 for v in
                (s.confirmation_steps, s.stationary_prior_hits, s.stationary_history_frames))
                or s.stationary_prior_hits > s.stationary_history_frames
                or not math.isfinite(s.direction_cosine) or not 0 < s.direction_cosine <= 1
                or any(not math.isfinite(v) or v <= 0 for v in
                       (s.minimum_step_px, s.stationary_radius_px, s.stationary_separation_px))):
            raise ValueError("Invalid motion-recovery settings")
        self._release_active = False
        self._motion_steps = []
        self._last_detected_frame = None
        # All retained positions are advanced into the current image coordinates.
        self._background = deque()

    def configuration(self):
        return {**super().configuration(),
                "status": "offline_motion_recovery_pilot",
                "certainty": "provisional_motion_consistency_not_verified_identity",
                "after_recovery": "local_guard_with_accepted_motion_release",
                "motion_settings": asdict(self.motion_settings)}

    def _relatch(self):
        self._release_active = False
        self._local_lock = True
        self._motion_steps = []
        self.branches = []

    def update(self, row, camera_shift=(0.0, 0.0)):
        frame = row["frame"]
        if self.last_frame is not None and frame != self.last_frame + 1:
            raise ValueError("Temporal identity requires contiguous frames")
        raw = row.get("candidates", [])
        if not all(math.isfinite(v) for v in camera_shift):
            raise ValueError("Camera shift must be finite")
        for candidate in raw:
            if (len(candidate["center"]) != 2
                    or not all(math.isfinite(v) for v in candidate["center"])
                    or not math.isfinite(candidate["confidence"])
                    or not 0 < candidate["confidence"] <= 1):
                raise ValueError("Invalid detector candidate")
        s = self.motion_settings
        aligned = row.get("camera_motion", {}).get("status") == "estimated_translation_only"
        anchor = tuple(self.last_observed[i] + camera_shift[i] for i in range(2))
        old_velocity, old_misses = self.velocity, self.misses
        self._background = deque((f, [tuple(p[i] + camera_shift[i] for i in range(2))
                                     for p in points])
                                 for f, points in self._background
                                 if frame - f <= s.stationary_history_frames)
        reason = "not_armed"
        if not aligned:
            self._background.clear()
            self._motion_steps = []
            if self._release_active:
                self._relatch()
                reason = "camera_alignment_unavailable"

        vetoed = []
        association_row = row
        if self._release_active:
            predicted = self.predict(camera_shift)
            corridor = self._radius(self.velocity) + self.settings.branch_padding_px * self.scale
            allowed = []
            for candidate in raw:
                center = candidate["center"]
                if math.dist(center, predicted) > corridor:
                    continue
                stationary_hits = sum(any(math.dist(center, p) <= s.stationary_radius_px * self.scale
                                          for p in points) for _, points in self._background)
                if stationary_hits >= s.stationary_prior_hits:
                    vetoed.append(list(center))
                else:
                    allowed.append(candidate)
            association_row = {**row, "candidates": allowed}
            reason = "bounded_motion_corridor"

        result = super().update(association_row, camera_shift)
        result["candidate_count"] = len(raw)
        result["stationary_veto_count"] = len(vetoed)
        result["stationary_veto_centers"] = vetoed
        result["motion_release_support_frames"] = []

        if self._release_active:
            if result["detected"]:
                step = tuple((result["center"][i] - anchor[i]) / (old_misses + 1)
                             for i in range(2))
                if (result["reason"] == "slow_recovery_provisional"
                        or math.hypot(*step) < s.minimum_step_px * self.scale
                        or _cosine(step, old_velocity) < s.direction_cosine):
                    self._relatch()
                    reason = "observed_slowing_or_direction_change"
            elif result["reason"] in ("ambiguous_candidates", "lost_requires_new_seed"):
                self._relatch()
                reason = result["reason"]
        elif self._local_lock:
            # A far proposal, a prediction, or a detection after a gap cannot
            # contribute a step. Only accepted contiguous local observations do.
            if (result["detected"] and aligned
                    and result["reason"] != "slow_recovery_provisional"
                    and self._last_detected_frame == frame - 1):
                step = tuple(result["center"][i] - anchor[i] for i in range(2))
                competitors = [c for c in raw
                               if math.dist(c["center"], result["center"])
                               <= self.recovery_settings.anchor_radius_px * self.scale]
                coherent = (not self._motion_steps
                            or _cosine(step, self._motion_steps[-1][1]) >= s.direction_cosine)
                if (len(competitors) == 1 and coherent
                        and math.hypot(*step) >= s.minimum_step_px * self.scale):
                    self._motion_steps.append((frame, step))
                    self._motion_steps = self._motion_steps[-s.confirmation_steps:]
                else:
                    self._motion_steps = []
                if len(self._motion_steps) == s.confirmation_steps:
                    self._release_active = True
                    self._local_lock = False
                    self.branches = []
                    result["motion_release_support_frames"] = [f for f, _ in self._motion_steps]
                    reason = "accepted_local_motion_supported"
            else:
                self._motion_steps = []

        # Background memory records only off-target proposals. It has no labels,
        # future frames, or target-position updates, and expires automatically.
        target = result["center"] if result["detected"] else anchor
        if aligned:
            off_target = [tuple(c["center"]) for c in raw
                          if math.dist(c["center"], target)
                          >= s.stationary_separation_px * self.scale]
            self._background.append((frame, off_target))
        if result["detected"]:
            self._last_detected_frame = frame
        result.update(motion_release_active=self._release_active,
                      motion_release_steps=len(self._motion_steps),
                      motion_release_reason=reason)
        return result
