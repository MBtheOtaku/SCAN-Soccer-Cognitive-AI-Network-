"""Experimental image-space association, not contact or shot confirmation."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Candidate:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    appearance: tuple[float, ...] = ()

    @property
    def centre(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def size(self):
        return math.sqrt((self.x2 - self.x1) * (self.y2 - self.y1))

    def valid(self):
        values = (self.x1, self.y1, self.x2, self.y2, self.confidence, *self.appearance)
        return (all(math.isfinite(v) for v in values)
                and self.x2 > self.x1 and self.y2 > self.y1
                and 0 <= self.confidence <= 1
                and all(v >= 0 for v in self.appearance))


def iou(a, b):
    intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0, min(a.y2, b.y2) - max(a.y1, b.y1))
    union = a.size ** 2 + b.size ** 2 - intersection
    return intersection / union if union > 0 else 0


def deduplicate(candidates, threshold=0.35):
    kept = []
    for candidate in sorted((c for c in candidates if c.valid()),
                            key=lambda c: c.confidence, reverse=True):
        if all(iou(candidate, previous) < threshold for previous in kept):
            kept.append(candidate)
    return kept


class TemporalBall:
    """All thresholds are provisional. 'observed' means accepted detector evidence,
    not verified identity. Camera compensation is not included in this version.
    """
    def __init__(self, seed=None, confirm_frames=3, max_missing=3):
        if confirm_frames < 2 or max_missing < 0:
            raise ValueError("Require at least two confirmation frames and nonnegative gap.")
        if seed is not None and (len(seed) != 2 or not all(math.isfinite(v) for v in seed)):
            raise ValueError("Seed must be a finite image-coordinate pair.")
        self.seed = seed
        self.confirm_frames = confirm_frames
        self.max_missing = max_missing
        self.pending = []
        self.history = []
        self.last_frame = None
        self.last_time = None
        self.active = False
        self.segment = 0

    def update(self, frame, timestamp, candidates):
        if (not math.isfinite(timestamp) or timestamp < 0
                or (self.last_frame is not None and frame != self.last_frame + 1)
                or (self.last_time is not None and timestamp <= self.last_time)):
            raise ValueError("Process consecutive frames with increasing finite timestamps.")
        self.last_frame, self.last_time = frame, timestamp
        candidates = [c for c in deduplicate(candidates) if c.confidence >= 0.15]
        result = {"frame": frame, "timestamp_s": timestamp,
                  "status": "lost", "observation": None,
                  "prediction": None, "segment": self.segment,
                  "reason": "No supported ball initialization."}

        # A prediction is a separate field and never appended as a measurement.
        reference = self.history[-1] if self.active else self.pending[-1] if self.pending else None
        predicted = None
        if reference:
            _, last_t, last = reference
            predicted = last.centre
            if self.active and len(self.history) >= 2:
                _, previous_t, previous = self.history[-2]
                dt = last_t - previous_t
                if dt > 0:
                    elapsed = timestamp - last_t
                    predicted = tuple(last.centre[j] + (last.centre[j] - previous.centre[j])
                                      * elapsed / dt for j in (0, 1))
            result["prediction"] = {"x": predicted[0], "y": predicted[1]}
        elif self.seed is not None:
            predicted = self.seed

        ranked = []
        for candidate in candidates:
            if reference:
                last = reference[2]
                ratio = candidate.size / last.size
                missing = frame - reference[0]
                gate = max(20.0, 3.0 * last.size) * (1 + 0.35 * max(0, missing - 1))
                distance = math.dist(candidate.centre, predicted)
                if distance > gate or not 0.4 <= ratio <= 2.5:
                    continue
                overlap = 1.0
                if last.appearance and candidate.appearance:
                    if len(last.appearance) != len(candidate.appearance):
                        continue
                    overlap = sum(min(a, b) for a, b in zip(last.appearance, candidate.appearance))
                    if overlap < 0.25:
                        continue
                cost = distance / gate + 0.2 * abs(math.log(ratio)) + 0.3 * (1 - overlap)
                ranked.append((cost - 0.1 * candidate.confidence, candidate))
            elif self.seed is not None:
                distance = math.dist(candidate.centre, self.seed)
                if distance <= max(24.0, 2.0 * candidate.size):
                    ranked.append((distance / max(24.0, 2.0 * candidate.size), candidate))
            elif candidate.confidence >= 0.25:
                ranked.append((1 - candidate.confidence, candidate))

        ranked.sort(key=lambda pair: pair[0])
        # At automatic startup require a unique eligible candidate, rather than
        # selecting the highest score from several unrelated objects.
        ambiguous = (len(ranked) > 1 and
                     (reference is None and self.seed is None
                      or ranked[1][0] - ranked[0][0] < 0.15))
        if ranked and not ambiguous:
            chosen = ranked[0][1]
            if not self.active:
                self.pending.append((frame, timestamp, chosen))
                if len(self.pending) < self.confirm_frames:
                    result.update(status="tentative", observation=asdict(chosen),
                                  reason="Waiting for consecutive detector matches.")
                    return result
                self.history = self.pending[-2:]
                self.pending = []
                self.active = True
                self.segment += 1
                self.seed = None
            else:
                self.history = (self.history + [(frame, timestamp, chosen)])[-2:]
            result.update(status="observed", observation=asdict(chosen),
                          segment=self.segment, reason="Detector match passed provisional association checks.")
            return result

        self.pending = []
        if self.active and frame - self.history[-1][0] <= self.max_missing:
            result.update(status="uncertain" if ambiguous else "predicted",
                          reason="Ambiguous detector matches." if ambiguous else "No detector match; prediction only.")
        else:
            self.active = False
            self.history = []
            result.update(status="uncertain" if ambiguous else "lost", prediction=None,
                          reason="Ambiguous initialization." if ambiguous else "Track lost or no supported initialization.")
        return result
