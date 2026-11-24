"""Offline bounded alternative to a drifting flight prediction.

Only current detector evidence can be selected. No interpolation, outcome
inference, future labels or automatic recovery after the lost deadline.
"""

import math

from experiments.ball_tracking.motion_recovery_identity import MotionRecoveryIdentity


class DriftRecoveryIdentity(MotionRecoveryIdentity):
    """Require three consecutive, unambiguous moving observations to recover."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drift_support = []

    def configuration(self):
        return {**super().configuration(), "status": "offline_drift_recovery_pilot",
                "drift_recovery": {"anchor_radius_px": 96, "maximum_step_px": 24,
                                   "minimum_motion_px": 6, "confirmation_hits": 3,
                                   "maximum_velocity_residual_px": 8}}

    def update(self, row, camera_shift=(0.0, 0.0)):
        # Parent validates inputs and preserves its established selection rules.
        result = super().update(row, camera_shift)
        self._drift_support = [
            (f, tuple(p[i] + camera_shift[i] for i in range(2)))
            for f, p in self._drift_support]
        result["drift_support_frames"] = []
        aligned = row.get("camera_motion", {}).get("status") == "estimated_translation_only"
        if (result["detected"] or not aligned or self.accepted_observations < 5
                or self.misses > self.settings.maximum_prediction_gap
                or result["reason"] == "ambiguous_candidates"):
            self._drift_support = []
            return result
        # A recent accepted observation bounds the search even as prediction
        # drifts. Persistent off-target background memory remains a veto.
        candidates = []
        for candidate in row.get("candidates", []):
            point = candidate["center"]
            if math.dist(point, self.last_observed) > 96 * self.scale:
                continue
            hits = sum(any(math.dist(point, p) <= self.motion_settings.stationary_radius_px * self.scale
                           for p in points) for _, points in self._background)
            if hits < self.motion_settings.stationary_prior_hits:
                candidates.append(candidate)
        # Do not resolve competing objects by confidence alone.
        if len(candidates) != 1:
            self._drift_support = []
            return result
        candidate = candidates[0]
        point = tuple(candidate["center"])
        if self._drift_support:
            f, previous = self._drift_support[-1]
            if f != row["frame"] - 1 or math.dist(point, previous) > 24 * self.scale:
                self._drift_support = []
        self._drift_support.append((row["frame"], point))
        self._drift_support = self._drift_support[-3:]
        result["drift_support_frames"] = [f for f, _ in self._drift_support]
        if len(self._drift_support) != 3:
            return result
        first, middle, last = [p for _, p in self._drift_support]
        residual = math.dist(last, tuple(2 * middle[i] - first[i] for i in range(2)))
        if math.dist(first, last) < 6 * self.scale or residual > 8 * self.scale:
            return result
        support = [f for f, _ in self._drift_support]
        self.velocity = tuple((last[i] - first[i]) / 2 for i in range(2))
        self.center = self.last_observed = last
        self.misses = 0
        self.branches = []
        self._support = []
        self._drift_support = []
        self._relatch()
        self._last_detected_frame = row["frame"]
        self.accepted_observations += 1
        result.update(center=list(last), confidence=candidate["confidence"], detected=True,
                      reason="drift_recovery_provisional", missing_run=0,
                      tentative_branches=0, drift_support_frames=support,
                      motion_release_active=False, post_recovery_local_gate=True,
                      identity_verified=False)
        return result
