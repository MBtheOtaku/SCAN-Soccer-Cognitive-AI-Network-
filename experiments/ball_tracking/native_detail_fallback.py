"""Optional causal native-detail proposals; never resets or verifies identity.

The reduced-image detector remains the primary source. Only a local evidence
gap requests native crops, and only proposals near the causal query can enter
the unchanged association layer. No reference positions or colours are read.
"""

import math
import time

from experiments.ball_tracking.native_box_detector import detect_native_boxes


SETTINGS = {
    "status": "offline_local_native_detail_pilot",
    "processing_short_edge": 720,
    "local_evidence_radius_px": 32.0,
    "native_admission_radius_px": 32.0,
    "native_context_half_extent_px": 96.0,
    "native_box_nms_iou": .5,
    "merge_rule": "box_overlap_not_centre_distance",
    "include_query_centred_tile": True,
    "native_crop_size": 384,
    "native_max_stride": 256,
    "native_max_batch": 16,
    "certainty": "detector_proposals_only_unchanged_identity_checks",
}


def local_region(transform, query):
    """A small matched scene window with at least one real 384px crop."""
    width, height = transform["width"], transform["height"]
    native_width, native_height = transform["source_width"], transform["source_height"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 384
           for v in (width, height, native_width, native_height)):
        raise ValueError("Expected integer image dimensions of at least 384 pixels")
    if min(width, height) != SETTINGS["processing_short_edge"]:
        raise ValueError("Native detail fallback requires 720-short-edge processing")
    if (native_width < width or native_height < height
            or len(query) != 2 or not all(math.isfinite(v) for v in query)):
        raise ValueError("Expected finite query and source without upscaling")
    sx, sy = native_width / width, native_height / height
    bounds = []
    for center, scale, limit in zip(query, (sx, sy), (native_width, native_height)):
        half = max(192, math.ceil(SETTINGS["native_context_half_extent_px"] * scale))
        length = min(limit, 2 * half)
        low = max(0, min(limit - length, round(center * scale - half)))
        bounds.append((low, low + length))
    return {"native_xyxy": [bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]],
            "source_per_processing_xy": [sx, sy],
            "query_center_native": [query[0] * sx, query[1] * sy]}


def augment_native_detail(model, source_image, transform, query, baseline, *, infer=None,
                          recovery_query=None, merge_nested_native=False):
    """Preserve baseline proposals, adding bounded evidence before association."""
    if type(merge_nested_native) is not bool:
        raise ValueError("Expected boolean nested native merging")
    # Recovery is explicitly opt-in; retain the original crop budget.
    search_query = query if recovery_query is None else recovery_query
    region = local_region(transform, search_query)
    if source_image.shape[:2] != (transform["source_height"], transform["source_width"]):
        raise ValueError("Source image dimensions disagree with frame transform")
    for proposal in baseline:
        if (len(proposal["center"]) != 2
                or not all(math.isfinite(v) for v in proposal["center"])
                or not math.isfinite(proposal["confidence"])
                or not 0 < proposal["confidence"] <= 1):
            raise ValueError("Invalid baseline proposal")
    result = list(baseline)
    info = {"status": "not_requested_local_evidence_present", "crop_count": 0,
            "query_center_processing": list(query), "native_region": region,
            "raw_proposals": [], "admitted_proposals": [], "elapsed_s": 0.0,
            "recovery_query": list(recovery_query) if recovery_query is not None else None}
    if recovery_query is None and any(math.dist(p["center"], query) <= SETTINGS["local_evidence_radius_px"]
           for p in baseline):
        return result, info
    sx, sy = region["source_per_processing_xy"]
    if sx == sy == 1.0:
        info["status"] = "not_requested_no_additional_source_detail"
        return result, info
    started = time.perf_counter()
    proposals, count = (infer or detect_native_boxes)(model, source_image, region)
    info["nested_merge_enabled"] = merge_nested_native
    info["proposals_before_nested_merge"] = len(proposals)
    if merge_nested_native:
        from experiments.ball_tracking.native_box_detector import merge_boxes
        proposals = merge_boxes(proposals, containment_threshold=.85)
    info["nested_duplicates_removed"] = info["proposals_before_nested_merge"] - len(proposals)
    info.update(status="native_local_search", crop_count=count,
                elapsed_s=time.perf_counter() - started)
    for proposal in proposals:
        native = proposal["center"]
        confidence = proposal["confidence"]
        if (len(native) != 2 or not all(math.isfinite(v) for v in native)
                or not math.isfinite(confidence) or not 0 < confidence <= 1):
            raise ValueError("Invalid native proposal")
        point = [native[0] / sx, native[1] / sy]
        candidate = {"center": point, "confidence": confidence,
                     "proposal_source": "native_detail", "native_center": list(native)}
        if "bbox_native_xyxy" in proposal:
            candidate["native_bbox_xyxy"] = proposal["bbox_native_xyxy"]
        info["raw_proposals"].append(candidate)
        if (not 0 <= point[0] < transform["width"]
                or not 0 <= point[1] < transform["height"]
                or math.dist(point, search_query) > (SETTINGS["native_admission_radius_px"]
                                                      if recovery_query is None else 96.0)):
            continue
        result.append(candidate)
        info["admitted_proposals"].append(candidate)
    return result, info
