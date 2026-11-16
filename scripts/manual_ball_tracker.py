"""Conservative manually seeded template baseline. No detector or coaching policy."""
import math

import cv2
import numpy as np


class ManualBallTracker:
    """Fixed seed appearance plus local motion search; identity is NOT guaranteed.

    Thresholds are provisional quality heuristics, not probabilities. Failure is
    latched until a human constructs a new tracker with a fresh seed. Never searches
    the whole frame or uses later manual labels to reacquire.
    """
    def __init__(self, frame, box, frame_index=0, min_score=0.65, margin=0.08,
                 confirm_frames=3, search_radius=None):
        h, w = frame.shape[:2]
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError("Seed needs four finite original-image coordinates.")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("Seed box must be inside the image.")
        x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
        if min(x2-x1, y2-y1) < 4:
            raise ValueError("Ball patch is too small for this baseline (minimum four pixels).")
        if not (0 < min_score < 1 and 0 < margin < 1) or confirm_frames < 2:
            raise ValueError("Invalid template thresholds or confirmation count.")
        if search_radius is not None and (not math.isfinite(search_radius) or search_radius <= 0):
            raise ValueError("Search radius must be positive and finite.")
        self.template = self.gray(frame)[y1:y2, x1:x2].copy()
        if float(self.template.std()) < 5:
            raise ValueError("Seed has too little image texture; select a clearer ball frame.")
        self.shape = (h, w)
        self.seed_size = (x2-x1, y2-y1)
        self.box = [float(v) for v in (x1, y1, x2, y2)]
        self.history = [self.center(self.box)]
        self.last_frame = frame_index
        self.streak = 1
        self.confirm_frames = confirm_frames
        self.min_score, self.margin = min_score, margin
        self.radius = search_radius or max(32, max(self.seed_size)*4)
        self.failure = None

    @staticmethod
    def gray(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    @staticmethod
    def center(box):
        return ((box[0]+box[2])/2, (box[1]+box[3])/2)

    def stop(self, status, reason, score=None, alternative_score=None):
        self.failure = {"status": status, "reason": reason, "box": None,
                        "match_score": score, "alternative_score": alternative_score}
        return dict(self.failure)

    def update(self, frame, frame_index):
        if frame_index != self.last_frame + 1:
            raise ValueError("Track consecutive decoded frames only.")
        self.last_frame = frame_index
        if frame.shape[:2] != self.shape:
            raise ValueError("Image dimensions changed; select the ball again.")
        if self.failure is not None:
            return {**self.failure, "reason": "Tracking stopped; manual re-selection required."}
        cx, cy = self.history[-1]
        if len(self.history) > 1:
            cx += self.history[-1][0]-self.history[-2][0]
            cy += self.history[-1][1]-self.history[-2][1]
        h, w = self.shape
        left, top = max(0, math.floor(cx-self.radius)), max(0, math.floor(cy-self.radius))
        right, bottom = min(w, math.ceil(cx+self.radius)), min(h, math.ceil(cy+self.radius))
        search = self.gray(frame)[top:bottom, left:right]
        candidates = []
        current_w, current_h = self.box[2]-self.box[0], self.box[3]-self.box[1]
        for factor in (0.8, 1.0, 1.25):
            tw, th = round(current_w*factor), round(current_h*factor)
            if (min(tw, th) < 4 or tw > search.shape[1] or th > search.shape[0]
                    or not 0.5 <= tw/self.seed_size[0] <= 2
                    or not 0.5 <= th/self.seed_size[1] <= 2):
                continue
            template = cv2.resize(self.template, (tw, th))
            scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            scores = np.where(np.isfinite(scores), scores, -1).astype(np.float32)
            # Keep several spatial peaks per scale; an alternative is not hidden
            # merely because it ranks below the best match.
            for _ in range(4):
                _, score, _, position = cv2.minMaxLoc(scores)
                if score < self.min_score-self.margin:
                    break
                px, py = position
                box = [left+px, top+py, left+px+tw, top+py+th]
                candidates.append({"box": box, "score": float(score)})
                sx, sy = max(1, tw//2), max(1, th//2)
                scores[max(0, py-sy):py+sy+1, max(0, px-sx):px+sx+1] = -1
        if not candidates:
            return self.stop("lost", "No adequate local appearance match.")
        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]
        # Deduplicate scale/position proposals for the same object. This is a
        # heuristic and can merge two extremely close balls; no identity guarantee.
        separation = max(4, min(best["box"][2]-best["box"][0], best["box"][3]-best["box"][1])*0.8)
        alternatives = [c for c in candidates[1:] if math.dist(
            self.center(c["box"]), self.center(best["box"])) > separation]
        alternative = alternatives[0]["score"] if alternatives else None
        if best["score"] < self.min_score:
            return self.stop("lost", "Local appearance match is too weak.", best["score"], alternative)
        if alternative is not None and best["score"]-alternative < self.margin:
            return self.stop("ambiguous", "Competing local appearance matches; target identity uncertain.",
                             best["score"], alternative)
        self.box = [float(v) for v in best["box"]]
        self.history = (self.history + [self.center(self.box)])[-2:]
        self.streak += 1
        return {"status": "tracking" if self.streak >= self.confirm_frames else "calibrating",
                "reason": "Local seed-appearance match; identity unverified.", "box": self.box.copy(),
                "match_score": best["score"], "alternative_score": alternative}
