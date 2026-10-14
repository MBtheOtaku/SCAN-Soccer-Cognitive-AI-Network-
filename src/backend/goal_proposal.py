"""Goal-frame geometry proposals. Scores rank candidates, never imply certainty."""
import cv2
import numpy as np
from itertools import product


def propose_goals(image: np.ndarray) -> list[dict]:
    h0, w0 = image.shape[:2]
    scale = min(1., 960 / w0)
    frame = cv2.resize(image, (round(w0 * scale), round(h0 * scale)))
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 145), (179, 85, 255))
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 130)
    edges = cv2.bitwise_and(edges, cv2.dilate(white, np.ones((3, 3), np.uint8)))
    # Two segment lengths reduce sensitivity to fragmented crossbar edges.
    passes = [cv2.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength=length, maxLineGap=20)
              for length in (20, max(24, int(w * .025)))]
    lines = [result.reshape(-1, 4) for result in passes if result is not None]
    if not lines:
        return []
    horizontal, vertical = [], []
    for x1, y1, x2, y2 in np.concatenate(lines):
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = x2 - x1, y2 - y1
        if .14 * w < dx < .9 * w and abs(dy) < .04 * dx and .08 * h < min(y1, y2) < .75 * h:
            horizontal.append((float(x1), float(y1), float(x2), float(y2)))
        if abs(dy) > .06 * h and abs(dx) < .25 * abs(dy):
            vertical.append((float((x1 + x2) / 2), float(min(y1, y2)), float(max(y1, y2))))
    # Thick white posts can have fragmented edges against a net. Search their
    # white interior too, without extending them across dark/unobserved gaps.
    post_lines = cv2.HoughLinesP(white, 1, np.pi / 180, 40, minLineLength=max(20, int(.07 * h)), maxLineGap=5)
    if post_lines is not None:
        for x1, y1, x2, y2 in post_lines.reshape(-1, 4):
            if abs(y2 - y1) > .06 * h and abs(x2 - x1) < .15 * abs(y2 - y1):
                vertical.append((float((x1 + x2) / 2), float(min(y1, y2)), float(max(y1, y2))))
    candidates = []
    for x1, y1, x2, y2 in horizontal:
        length = x2 - x1
        tolerance = max(8, length * .055)
        groups = []
        for x, top in ((x1, y1), (x2, y2)):
            column_posts = []
            start = int(top) + 4
            stop = min(h, int(top + length * .65))
            for column in range(max(1, int(x - tolerance)), min(w - 1, int(x + tolerance))):
                support = (white[start:stop, column - 1:column + 2] > 0).mean(axis=1)
                if len(support) < 9:
                    continue
                smooth = np.convolve(support, np.ones(9) / 9, mode="valid")
                if smooth[0] < .8:
                    continue
                drops = np.flatnonzero(smooth < .75)
                if not len(drops):
                    continue  # No observed post endpoint; do not invent one.
                end = start + int(drops[0]) + 4
                if end - top > .06 * h:
                    column_posts.append((float(column), top, float(end)))
            posts = [(vx, vy, by) for vx, vy, by in vertical
                     if abs(vx - x) < tolerance and abs(vy - top) < max(15, .06 * h) and by - top > .06 * h]
            posts.extend(column_posts)
            if not posts:
                break
            groups.append(sorted(set(posts), key=lambda p: abs(p[0] - x))[:40])
        if len(groups) != 2:
            continue
        left, right = min(product(*groups), key=lambda pair:
                          2 * abs(pair[0][2] - pair[1][2]) +
                          2 * (abs(pair[0][0] - x1) + abs(pair[1][0] - x2)) +
                          abs(pair[0][1] - y1) + abs(pair[1][1] - y2))
        lx, _, lb = left
        rx, _, rb = right
        height = ((lb - y1) + (rb - y2)) / 2
        if not .15 < height / length < .8 or abs(lb - rb) > height * .35:
            continue
        points = np.array([[lx, y1], [rx, y2], [rx, rb], [lx, lb]], np.float32)
        if not cv2.isContourConvex(points) or np.any(points < 0) or np.any(points[:, 1] >= .9 * h):
            continue
        interior = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(interior, points.astype(np.int32), 255)
        grass = cv2.inRange(hsv, (25, 20, 30), (95, 255, 255))
        grass_fraction = np.count_nonzero(cv2.bitwise_and(grass, interior)) / max(1, np.count_nonzero(interior))
        # Avoid interpreting pitch markings as a goal mouth. May abstain when the
        # goal is backed by grass; confirmation/manual fallback remains available.
        if grass_fraction > .65:
            continue
        normalized = points / [w, h]
        candidates.append({"corners": normalized.tolist(), "rank_score": round(float(length / w + height / h - .3 * grass_fraction - 3 * abs(lb - rb) / h), 3)})
    selected = []
    for candidate in sorted(candidates, key=lambda c: c["rank_score"], reverse=True):
        if any(np.mean(np.linalg.norm(np.array(candidate["corners"]) - np.array(c["corners"]), axis=1)) < .025 for c in selected):
            continue
        selected.append(candidate)
    return selected[:4]
