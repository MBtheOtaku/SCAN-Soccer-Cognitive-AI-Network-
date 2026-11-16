"""Current-frame appearance evidence for the hand-seeded offline ball pilot.

This is a comparison with the visible seed patch, not an identity classifier.
Blur, occlusion, scale change, and lighting can make a real ball score poorly.
"""

import cv2
import numpy as np


def seed_template(frame, box):
    width = max(5, int(box[2] - box[0]))
    height = max(5, int(box[3] - box[1]))
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return cv2.getRectSubPix(frame, (width, height), center)


def appearance_evidence(template, frame, center):
    height, width = template.shape[:2]
    patch = cv2.getRectSubPix(frame, (width, height),
                              (float(center[0]), float(center[1])))
    a = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    correlation = float(cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED)[0, 0])
    hsv_a = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [18, 8], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a, alpha=1, norm_type=cv2.NORM_L1)
    cv2.normalize(hist_b, hist_b, alpha=1, norm_type=cv2.NORM_L1)
    distance = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))
    if not np.isfinite(correlation):
        correlation = 0.0
    return {"correlation": correlation, "hist_distance": distance}


def appearance_penalty(evidence):
    """Soft cost; never turn a low template score into a positive identification."""
    return (8 * max(0, 0.30 - evidence["correlation"]) +
            4 * max(0, evidence["hist_distance"] - 0.65))
