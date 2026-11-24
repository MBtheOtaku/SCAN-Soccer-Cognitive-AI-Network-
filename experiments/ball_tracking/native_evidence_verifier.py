"""Causal checks for extra native proposals, not proof of ball identity.

The initial image supplies background context. SIFT
matches and RANSAC align that context to the current image; this is an image
registration aid, never a goal-plane, depth or outcome measurement.
"""

import math

import cv2
import numpy as np


SETTINGS = {
    "status": "offline_native_evidence_verification_pilot",
    "features": 1500, "descriptor_ratio": .7, "ransac_threshold_px": 3.0,
    "minimum_inliers": 12, "minimum_inlier_ratio": .5,
    "maximum_median_reprojection_px": 1.5,
    "maximum_nearest_inlier_px": 160.0,
    "static_patch_ncc_threshold": .85,
    "native_patch_size": 17, "native_background_window": 35,
    "seed_background_exclusion_px": 48.0,
    "seed_region_policy": "primary_detector_only_no_supplemental_bootstrap",
    "confirmation_frames": 3, "maximum_reference_step_px": 12.0,
    "hypothesis_match_ambiguity_px": 2.0, "maximum_hypotheses": 8,
    "certainty": "provisional_image_evidence_not_verified_identity",
}


def project(point, matrix):
    value = matrix @ np.array([*point, 1.0], dtype=float)
    if not np.isfinite(value).all() or abs(value[2]) < 1e-9:
        raise ValueError("Invalid projective point")
    return value[:2] / value[2]


class NativeEvidenceVerifier:
    def __init__(self, initial_source, initial_working, source_seed, *, trajectory_support=False):
        if type(trajectory_support) is not bool:
            raise ValueError("Expected boolean trajectory support")
        from experiments.ball_tracking.trajectory_evidence import TrajectoryEvidence
        self.trajectory = TrajectoryEvidence() if trajectory_support else None
        self.source = initial_source.copy()
        self.height, self.width = initial_working.shape[:2]
        self.sx = initial_source.shape[1] / self.width
        self.sy = initial_source.shape[0] / self.height
        self.seed = np.array([source_seed[0] / self.sx, source_seed[1] / self.sy])
        self.sift = cv2.SIFT_create(nfeatures=SETTINGS["features"])
        mask = np.full((self.height, self.width), 255, np.uint8)
        cv2.circle(mask, tuple(round(v) for v in self.seed),
                   round(SETTINGS["seed_background_exclusion_px"]), 0, -1)
        self.reference_keys, self.reference_descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(initial_working, cv2.COLOR_BGR2GRAY), mask)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.hypotheses = []
        self.last_frame = None

    def configuration(self):
        return {**SETTINGS,
                "reference": "first_processed_source_image_only",
                "baseline_proposals": "preserved_without_filtering",
                "trajectory_support": self.trajectory.configuration() if self.trajectory else None}

    def align(self, current):
        if current.shape[:2] != (self.height, self.width):
            raise ValueError("Background registration requires matching processing dimensions")
        keys, descriptors = self.sift.detectAndCompute(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), None)
        meta = {"status": "background_registration_unavailable"}
        if self.reference_descriptors is None or descriptors is None or len(descriptors) < 2:
            return None, meta
        pairs = self.matcher.knnMatch(self.reference_descriptors, descriptors, k=2)
        good = sorted((m for pair in pairs if len(pair) == 2 for m, n in [pair]
                       if m.distance < SETTINGS["descriptor_ratio"] * n.distance),
                      key=lambda m: m.distance)
        unique, seen = [], set()
        for match in good:
            if match.trainIdx not in seen:
                unique.append(match)
                seen.add(match.trainIdx)
        meta["matches"] = len(unique)
        if len(unique) < SETTINGS["minimum_inliers"]:
            return None, meta
        a = np.float32([self.reference_keys[m.queryIdx].pt for m in unique])
        b = np.float32([keys[m.trainIdx].pt for m in unique])
        matrix, mask = cv2.findHomography(a, b, cv2.RANSAC, SETTINGS["ransac_threshold_px"])
        if matrix is None or mask is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-9:
            return None, meta
        inliers = mask.ravel().astype(bool)
        if not inliers.any():
            return None, meta
        predicted = cv2.perspectiveTransform(a.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        errors = np.linalg.norm(predicted - b, axis=1)
        meta.update(inliers=int(inliers.sum()), inlier_ratio=float(inliers.mean()),
                    median_reprojection_px=float(np.median(errors[inliers])))
        if (meta["inliers"] < SETTINGS["minimum_inliers"]
                or meta["inlier_ratio"] < SETTINGS["minimum_inlier_ratio"]
                or meta["median_reprojection_px"] > SETTINGS["maximum_median_reprojection_px"]):
            return None, meta
        meta.update(status="registered_initial_background_not_depth", matrix=matrix.tolist(),
                    inlier_current_points=b[inliers].tolist())
        return matrix, meta

    def inspect(self, source, candidate, matrix, registration, *, origin=(0, 0)):
        point = candidate["center"]
        if matrix is None:
            return {"status": "background_registration_unavailable"}
        reference_point = project(point, np.linalg.inv(matrix))
        x, y = point[0] * self.sx, point[1] * self.sy
        center = (float(x - origin[0]), float(y - origin[1]))
        if min(center[0], center[1], source.shape[1] - center[0], source.shape[0] - center[1]) < 18:
            return {"status": "insufficient_patch_context"}
        patch = cv2.getRectSubPix(source, (17, 17), center)
        detail = {"reference_processing_center": reference_point.tolist()}
        if math.dist(reference_point, self.seed) <= SETTINGS["seed_background_exclusion_px"]:
            # A boot can cover the ball here. Preserve the primary detector's
            # behaviour rather than bootstrap extra target observations.
            return {**detail, "status": "seed_area_supplement_withheld"}
        inliers = registration["inlier_current_points"]
        if min((math.dist(point, p) for p in inliers), default=math.inf) > SETTINGS["maximum_nearest_inlier_px"]:
            return {**detail, "status": "insufficient_local_registration_support"}
        scale = np.diag([self.sx, self.sy, 1.0])
        inverse = np.linalg.inv(scale @ matrix @ np.linalg.inv(scale))
        xx, yy = np.meshgrid(np.arange(-17, 18, dtype=np.float32) + x,
                             np.arange(-17, 18, dtype=np.float32) + y)
        grid = np.stack([xx, yy], axis=-1).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(grid, inverse).reshape(35, 35, 2)
        if (mapped[:, :, 0].min() < 0 or mapped[:, :, 1].min() < 0
                or mapped[:, :, 0].max() >= self.source.shape[1] - 1
                or mapped[:, :, 1].max() >= self.source.shape[0] - 1):
            return {**detail, "status": "reference_patch_outside_image"}
        grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if float(grey.std()) < 5:
            return {**detail, "status": "insufficient_patch_texture"}
        background = cv2.remap(self.source, mapped[:, :, 0], mapped[:, :, 1], cv2.INTER_LINEAR)
        scores = cv2.matchTemplate(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY), grey,
                                   cv2.TM_CCOEFF_NORMED)
        score = float(cv2.minMaxLoc(scores)[1])
        if not math.isfinite(score):
            return {**detail, "status": "invalid_patch_similarity"}
        return {**detail, "patch_ncc_not_probability": score,
                "status": ("resembles_initial_background" if score >= SETTINGS["static_patch_ncc_threshold"]
                           else "appearance_changed_not_identity_proof")}

    def reset_support(self):
        self.hypotheses = []
        if self.trajectory is not None:
            self.trajectory.reset()

    def filter(self, frame, source, working, proposals, *, accepted_frames=()):
        if self.last_frame is not None and frame != self.last_frame + 1:
            raise ValueError("Native evidence verification requires contiguous frames")
        if (any(isinstance(f, bool) or not isinstance(f, int) or f < 0 or f >= frame for f in accepted_frames)
                or list(accepted_frames) != sorted(set(accepted_frames))):
            raise ValueError("Continuity support must contain distinct past accepted frames")
        self.last_frame = frame
        baseline = [p for p in proposals if p.get("proposal_source") != "native_detail"]
        extra = [p for p in proposals if p.get("proposal_source") == "native_detail"]
        warm = list(accepted_frames)[-2:] == [frame - 2, frame - 1]
        info = {"status": "no_extra_proposals", "checks": [], "support_frames": [],
                "continuity_support_frames": list(accepted_frames)[-2:] if warm else [],
                "hypotheses": []}
        if not extra:
            self.hypotheses = []
            if self.trajectory is None:
                self.reset_support()
            else:
                self.trajectory.update(frame, [])
            return list(proposals), info
        matrix, registration = self.align(working)
        info.update(status="extra_native_evidence_checked", background_registration=registration)
        usable = []
        for candidate in extra:
            result = self.inspect(source, candidate, matrix, registration)
            info["checks"].append({"center": candidate["center"], **result})
            if result["status"] == "appearance_changed_not_identity_proof":
                usable.append((candidate, result["reference_processing_center"]))
        if not usable or len(usable) > SETTINGS["maximum_hypotheses"]:
            self.reset_support()
            if self.trajectory is not None:
                self.trajectory.update(frame, [], available=False)
            info["status"] = "no_bounded_extra_native_evidence"
            return baseline, info
        trajectory_forwarded = []
        if self.trajectory is not None:
            support = self.trajectory.update(frame, [[float(v) for v in point] for _, point in usable])
            info["trajectory_hypotheses"] = []
            for (candidate, _), evidence in zip(usable, support, strict=True):
                supported = evidence["supported"]
                info["trajectory_hypotheses"].append({"center": candidate["center"],
                                            "native_observation_frames": evidence["frames"],
                                            "forwarded_to_association": supported,
                                            "trajectory_evidence": evidence})
                if supported:
                    trajectory_forwarded.append(candidate)
        previous = [h for h in self.hypotheses if h["last_frame"] == frame - 1]
        points = [p for _, p in usable]
        margin = SETTINGS["hypothesis_match_ambiguity_px"]
        hypotheses, forwarded = [], []
        for index, (candidate, point) in enumerate(usable):
            matches = sorted((math.dist(point, h["point"]), i) for i, h in enumerate(previous))
            frames = []
            if (matches and matches[0][0] <= SETTINGS["maximum_reference_step_px"]
                    and (len(matches) == 1 or matches[1][0] - matches[0][0] > margin)):
                old = previous[matches[0][1]]
                reverse = sorted((math.dist(p, old["point"]), i) for i, p in enumerate(points))
                if (reverse[0][1] == index
                        and (len(reverse) == 1 or reverse[1][0] - reverse[0][0] > margin)):
                    frames = old["frames"]
            frames = (frames + [frame])[-SETTINGS["confirmation_frames"]:]
            hypotheses.append({"point": point, "frames": frames, "last_frame": frame})
            supported = warm or len(frames) == SETTINGS["confirmation_frames"]
            info["hypotheses"].append({"center": candidate["center"],
                                        "native_observation_frames": frames,
                                        "forwarded_to_association": supported})
            if supported:
                forwarded.append(candidate)
        self.hypotheses = hypotheses
        if len(hypotheses) == 1:
            info["support_frames"] = list(accepted_frames)[-2:] + [frame] if warm else hypotheses[0]["frames"]
        # Preserve the established verifier. The motion model is additional
        # evidence only; it cannot reject a supported existing hypothesis.
        trajectory_added = [candidate for candidate in trajectory_forwarded
                            if all(candidate is not old for old in forwarded)]
        forwarded.extend(trajectory_added)
        if self.trajectory is not None:
            info["trajectory_added_proposals"] = len(trajectory_added)
        info["status"] = ("supported_extra_evidence_not_verified_identity" if forwarded
                          else "awaiting_repeated_extra_evidence")
        # Several plausible objects stay separate. The association layer owns
        # target selection and its ambiguity checks; persistence is not identity.
        return baseline + forwarded, info
