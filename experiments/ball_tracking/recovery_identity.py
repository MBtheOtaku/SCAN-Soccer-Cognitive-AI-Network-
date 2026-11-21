"""Optional local slow/rebound recovery, never verified target identity.

The original temporal rule remains unchanged. This hypothesis can recover only
near a recent observation after a slow track, with repeated detector evidence
and available camera alignment. It never fills past or current detector gaps.
"""

from dataclasses import asdict, dataclass
import math

from experiments.ball_tracking.temporal_identity import TemporalIdentity


@dataclass(frozen=True)
class RecoverySettings:
    minimum_accepted_observations: int = 5
    anchor_radius_px: float = 32.0
    maximum_previous_speed_px: float = 12.0
    maximum_step_speed_px: float = 8.0
    confirmation_hits: int = 3
    support_window_frames: int = 7
    maximum_absent_between_hits: int = 3


class RecoveryIdentity(TemporalIdentity):
    """Add a bounded stop/rebound hypothesis after the parent rule abstains."""

    def __init__(self, seed, *, scale=1.0, settings=None, recovery_settings=None):
        super().__init__(seed, scale=scale, settings=settings)
        self.recovery_settings = recovery_settings or RecoverySettings()
        s = self.recovery_settings
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in
                (s.minimum_accepted_observations, s.confirmation_hits, s.support_window_frames))
                or isinstance(s.maximum_absent_between_hits, bool)
                or not isinstance(s.maximum_absent_between_hits, int)
                or s.maximum_absent_between_hits < 0
                or s.confirmation_hits < 2 or s.confirmation_hits > s.support_window_frames
                or any(not math.isfinite(v) or v <= 0 for v in
                       (s.anchor_radius_px, s.maximum_previous_speed_px, s.maximum_step_speed_px))):
            raise ValueError("Invalid slow-recovery settings")
        self.accepted_observations = 0
        self._local_lock = False
        # Positions are advanced by camera translation into the current image.
        self._support: list[tuple[int, tuple[float, float]]] = []

    def configuration(self):
        return {**super().configuration(),
                "status": "offline_temporal_slow_recovery_pilot",
                "recovery_settings": asdict(self.recovery_settings),
                "after_recovery": "gate_to_recent_observation_until_new_seed"}

    def update(self, row, camera_shift=(0.0, 0.0)):
        raw_candidates = row.get("candidates", [])
        local_row = row
        if self._local_lock:
            anchor = tuple(self.last_observed[i] + camera_shift[i] for i in range(2))
            for candidate in raw_candidates:
                if (len(candidate["center"]) != 2
                        or not all(math.isfinite(v) for v in candidate["center"])
                        or not math.isfinite(candidate["confidence"])
                        or not 0 < candidate["confidence"] <= 1):
                    raise ValueError("Invalid detector candidate")
            allowed = [c for c in raw_candidates if math.dist(c["center"], anchor)
                       <= self.recovery_settings.anchor_radius_px * self.scale]
            local_row = {**row, "candidates": allowed}
        result = super().update(local_row, camera_shift)
        result["candidate_count"] = len(raw_candidates)
        result["post_recovery_local_gate"] = self._local_lock
        result["recovery_support_hits"] = 0
        if result["detected"]:
            self.accepted_observations += 1
            self._support = []
            return result

        frame = row["frame"]
        s = self.recovery_settings
        self._support = [(f, tuple(p[i] + camera_shift[i] for i in range(2)))
                         for f, p in self._support
                         if frame - f < s.support_window_frames]
        if (self._support and frame - self._support[-1][0]
                > s.maximum_absent_between_hits + 1):
            self._support = []

        aligned = row.get("camera_motion", {}).get("status") == "estimated_translation_only"
        if (not aligned
                or self.accepted_observations < s.minimum_accepted_observations
                or self.misses > self.settings.maximum_prediction_gap
                or math.hypot(*self.velocity) > s.maximum_previous_speed_px * self.scale):
            self._support = []
            return result

        anchor = self.last_observed
        nearby = [c for c in row.get("candidates", [])
                  if math.dist(c["center"], anchor) <= s.anchor_radius_px * self.scale]
        if len(nearby) > 1:
            # Do not choose one of two nearby objects by detector score alone.
            self._support = []
            result["recovery_reason"] = "ambiguous_near_last_observation"
            return result
        if not nearby:
            result["recovery_support_hits"] = len(self._support)
            return result

        candidate = nearby[0]
        center = tuple(candidate["center"])
        if self._support:
            previous_frame, previous = self._support[-1]
            speed = math.dist(center, previous) / (frame - previous_frame)
            if speed > s.maximum_step_speed_px * self.scale:
                self._support = []
        self._support.append((frame, center))
        result["recovery_support_hits"] = len(self._support)
        if len(self._support) < s.confirmation_hits:
            return result

        support_frames = [f for f, _ in self._support]
        first_frame, first = self._support[0]
        self.velocity = tuple((center[i] - first[i]) / (frame - first_frame) for i in range(2))
        self.center = self.last_observed = center
        self.misses = 0
        self.branches = []
        self.accepted_observations += 1
        self._local_lock = True
        self._support = []
        result.update(center=list(center), confidence=candidate["confidence"], detected=True,
                      reason="slow_recovery_provisional", missing_run=0, tentative_branches=0,
                      recovery_anchor_px=list(anchor), recovery_support_frames=support_frames)
        return result
