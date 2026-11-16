"""Offline full-frame sliding-window ball detection with global duplicate removal."""

import math
import time

from scripts.finetune_ball import iou


def tile_origins(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= 0 or tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("Image size and tile size must be positive; overlap must be smaller than a tile.")
    if length <= tile_size:
        return [0]
    last = length - tile_size
    positions = list(range(0, last + 1, tile_size - overlap))
    if positions[-1] != last:
        positions.append(last)
    return positions


def suppress_duplicates(candidates: list[dict], nms_iou: float, max_detections: int) -> list[dict]:
    if not 0 <= nms_iou <= 1 or max_detections <= 0:
        raise ValueError("Invalid duplicate threshold or detection limit.")
    retained = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if all(iou(candidate["box"], previous["box"]) <= nms_iou for previous in retained):
            retained.append(candidate)
            if len(retained) == max_detections:
                break
    return retained


def predict_sliced(image, model, *, tile_size=512, overlap=128, imgsz=960,
                   confidence=0.25, device="cpu", nms_iou=0.5, max_detections=100):
    """Run the same detector on overlapping tiles; return original-image boxes."""
    if image is None or len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Expected a decoded BGR image.")
    if not 0 < confidence < 1 or imgsz <= 0:
        raise ValueError("Invalid confidence or model image size.")
    height, width = image.shape[:2]
    xs = tile_origins(width, tile_size, overlap)
    ys = tile_origins(height, tile_size, overlap)
    candidates, tiles = [], []
    started = time.perf_counter()
    for y in ys:
        for x in xs:
            crop = image[y:min(y + tile_size, height), x:min(x + tile_size, width)].copy()
            crop_height, crop_width = crop.shape[:2]
            result = model.predict(crop, device=device, imgsz=imgsz, conf=confidence,
                                   iou=nms_iou, max_det=max_detections, classes=[0],
                                   rect=False, verbose=False)[0]
            tiles.append([x, y, x + crop_width, y + crop_height])
            for box, score in zip(result.boxes.xyxy.cpu().tolist(),
                                  result.boxes.conf.cpu().tolist()):
                if not all(math.isfinite(v) for v in (*box, score)):
                    continue
                left, top, right, bottom = (
                    max(0.0, min(float(box[0]), crop_width)) + x,
                    max(0.0, min(float(box[1]), crop_height)) + y,
                    max(0.0, min(float(box[2]), crop_width)) + x,
                    max(0.0, min(float(box[3]), crop_height)) + y,
                )
                if right <= left or bottom <= top:
                    continue
                candidates.append({"box": [left, top, right, bottom],
                                   "confidence": float(score), "tile_origin": [x, y]})
    predictions = suppress_duplicates(candidates, nms_iou, max_detections)
    return {"tiles": tiles, "raw_candidates": candidates, "predictions": predictions,
            "elapsed_seconds": time.perf_counter() - started}
