"""Native tiles with box-overlap merging instead of merging nearby objects."""

import math

from experiments.ball_tracking.probe_native_detail import native_origins


NMS_IOU = .5
MAX_BATCH = 16


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def merge_boxes(proposals, threshold=NMS_IOU):
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Expected an IoU threshold between zero and one")
    retained = []
    for proposal in sorted(proposals, key=lambda p: -p["confidence"]):
        box = proposal["bbox_native_xyxy"]
        if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                or box[2] <= box[0] or box[3] <= box[1]
                or not math.isfinite(proposal["confidence"])
                or not 0 < proposal["confidence"] <= 1):
            raise ValueError("Invalid native detection box")
        if all(box_iou(box, other["bbox_native_xyxy"]) < threshold for other in retained):
            retained.append(proposal)
    return retained


def detect_native_boxes(model, image, region):
    """Use the same real 384px tiles, weights, input size and score threshold."""
    origins = native_origins(region["native_xyxy"])
    query = region.get("query_center_native")
    if query is not None:
        if len(query) != 2 or not all(math.isfinite(v) for v in query):
            raise ValueError("Expected a finite native query centre")
        left, top, right, bottom = region["native_xyxy"]
        central = (max(left, min(right - 384, round(query[0] - 192))),
                   max(top, min(bottom - 384, round(query[1] - 192))))
        # A regular grid can straddle the query without a query-centred tile.
        origins = [central] + [origin for origin in origins if origin != central]
    proposals = []
    for start in range(0, len(origins), MAX_BATCH):
        batch = origins[start:start + MAX_BATCH]
        crops = [image[y:y + 384, x:x + 384] for x, y in batch]
        if any(crop.shape[:2] != (384, 384) for crop in crops):
            raise ValueError("Native crop exceeds source image")
        results = model.predict(crops, imgsz=640, conf=.05, classes=[0],
                                device="cpu", verbose=False)
        for (left, top), result in zip(batch, results, strict=True):
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            scores = result.boxes.conf.cpu().numpy().tolist()
            for box, confidence in zip(boxes, scores, strict=True):
                absolute = [box[0] + left, box[1] + top, box[2] + left, box[3] + top]
                proposals.append({"center": [(absolute[0] + absolute[2]) / 2,
                                              (absolute[1] + absolute[3]) / 2],
                                  "confidence": confidence, "bbox_native_xyxy": absolute})
    return merge_boxes(proposals), len(origins)
