"""Offline detector-confirmed learned recovery after automatically chosen gaps.

Keep the frozen baseline intact. A learned point can request native pixels;
only a current detector box with background and temporal support can add an
observation. No labels, outcomes or coaching are consumed or produced.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

from scripts.ball_dataset import file_sha256
from experiments.ball_tracking.native_box_detector import detect_native_boxes, merge_boxes
from experiments.ball_tracking.native_evidence_verifier import NativeEvidenceVerifier
from experiments.ball_tracking.track_unlabelled_video import prepare_frame
from experiments.ball_tracking.trajectory_evidence import TrajectoryEvidence


@dataclass(frozen=True)
class LearnedRecoverySettings:
    window_frames: int = 28
    context_frames: int = 2
    minimum_seed_history: int = 3
    maximum_windows: int = 4
    maximum_prediction_gap: int = 12
    detector_gate_working_px: float = 6.0
    maximum_runtime_s: float = 240.0
    search_uncertain_points: bool = False

    def __post_init__(self):
        if type(self.search_uncertain_points) is not bool:
            raise ValueError("Expected a boolean uncertain-point search option")
        for value, low, high in ((self.window_frames, 5, 32), (self.context_frames, 2, 4),
                                 (self.minimum_seed_history, 3, 12), (self.maximum_windows, 1, 8),
                                 (self.maximum_prediction_gap, 1, 12)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid learned-recovery frame/window bounds")
        if self.minimum_seed_history < self.context_frames + 1:
            raise ValueError("Seed history must cover the real context observations")
        if (type(self.detector_gate_working_px) not in (int, float)
                or not math.isfinite(self.detector_gate_working_px)
                or not 0 < self.detector_gate_working_px <= 12
                or type(self.maximum_runtime_s) not in (int, float)
                or not math.isfinite(self.maximum_runtime_s) or self.maximum_runtime_s <= 0):
            raise ValueError("Invalid learned-recovery distance/runtime bound")


def plan_windows(path, settings=None):
    """Gap starts and strictly prior detector observations choose every query."""
    s = settings or LearnedRecoverySettings()
    if not path:
        raise ValueError("Expected a nonempty baseline path")
    streak, result, last_frame = 0, [], None
    for index, point in enumerate(path):
        frame = point['frame']
        if (type(frame) is not int or frame < 0 or type(point['detected']) is not bool
                or last_frame is not None and frame != last_frame + 1):
            raise ValueError("Baseline frames must be contiguous with boolean observed states")
        last_frame = frame
        if point['detected']:
            center = point['center']
            if (not isinstance(center, (list, tuple)) or len(center) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in center)):
                raise ValueError("Invalid baseline detector observation")
            streak += 1
            continue
        if streak >= s.minimum_seed_history:
            seed_frame = frame - 1
            start = seed_frame - s.context_frames
            end = min(path[-1]['frame'], start + s.window_frames - 1)
            if end > seed_frame:
                result.append({'gap_start': frame, 'seed_frame': seed_frame,
                               'start_frame': start, 'end_frame': end,
                               'seed_center_working': list(path[index-1]['center'])})
        streak = 0
    return result


def crop_for_seed(seed_native, source_width, source_height):
    if min(source_width, source_height) < 384:
        raise ValueError("Native recovery needs source dimensions of at least 384 pixels")
    width, height = min(768, source_width), min(576, source_height)
    left = max(0, min(source_width-width, round(seed_native[0]-width/2)))
    top = max(0, min(source_height-height, round(seed_native[1]-height/2)))
    return [left, top, left+width, top+height]


def load_window(source, window, crop, transform):
    import cv2
    cap = cv2.VideoCapture(str(source))
    records = []
    try:
        for frame in range(window['end_frame']+1):
            ok, image = cap.read()
            if not ok:
                raise ValueError("Source ended before the requested recovery window")
            if image.shape[:2] != (transform['source_height'], transform['source_width']):
                raise ValueError("Decoded source geometry changed")
            if frame < window['start_frame']:
                continue
            working, _ = prepare_frame(image, 720)
            left, top, right, bottom = crop
            records.append({'frame': frame, 'working': working,
                            'crop': image[top:bottom, left:right].copy()})
    finally:
        cap.release()
    return records


def confirm_window(baseline, window, records, estimates, verifier, detector, *, settings=None,
                   infer=None, deadline=None):
    """Return only additional measured observations, never fitted coordinates."""
    s = settings or LearnedRecoverySettings()
    expected = list(range(window['start_frame'], window['end_frame']+1))
    if ([r['frame'] for r in records] != expected or [e['frame'] for e in estimates] != expected
            or any(type(e['visible']) is not bool or len(e['center_local']) != 2
                   or any(type(v) not in (int,float) or not math.isfinite(v) for v in e['center_local'])
                   for e in estimates)):
        raise ValueError("Learned estimates must match decoded frames and have finite points/boolean visibility")
    by_frame = {p['frame']: p for p in baseline['path']}
    transform = baseline['frame_transform']
    dimensions = (baseline['width'],baseline['height'],transform['source_width'],transform['source_height'])
    if (any(type(d) is not int or d < 384 for d in dimensions)
            or transform['source_width'] < baseline['width']
            or transform['source_height'] < baseline['height']):
        raise ValueError("Expected original source dimensions without upscaling")
    sx, sy = transform['source_width']/baseline['width'], transform['source_height']/baseline['height']
    origin = window['crop_xyxy'][:2]
    support = TrajectoryEvidence()
    additions, rows = [], []
    last_anchor = window['seed_frame']
    for record, estimate in zip(records, estimates, strict=True):
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("Learned recovery exceeded its cooperative runtime limit")
        frame = record['frame']
        point = by_frame[frame]
        local = estimate['center_local']
        source_point = [local[i]+origin[i] for i in range(2)]
        learned = [source_point[0]/sx, source_point[1]/sy]
        row = {'frame': frame, 'baseline_detected': point['detected'],
               'learned_estimate_source': source_point, 'model_visible': estimate['visible'],
               'learned_estimate_is_detection': False, 'status': 'not_confirmed',
               'native_proposals': [], 'support_observation_frames': []}
        if point['detected']:
            last_anchor = frame
        before_seed = frame < window['seed_frame']
        # The pre-query context contains genuine baseline observations only;
        # the model's forced query visibility cannot create extra observations.
        if not before_seed and not estimate['visible'] and not s.search_uncertain_points:
            row['status'] = 'model_visibility_withheld'
            support.update(frame, [], available=False)
            rows.append(row)
            continue
        if not point['detected'] and frame-last_anchor > s.maximum_prediction_gap:
            row['status'] = 'lost_deadline_requires_new_baseline_observation'
            support.update(frame, [], available=False)
            rows.append(row)
            continue
        h, w = record['crop'].shape[:2]
        if not before_seed and not (18 <= local[0] < w-18 and 18 <= local[1] < h-18):
            row['status'] = 'learned_point_outside_native_context'
            support.update(frame, [], available=False)
            rows.append(row)
            continue
        candidate = None
        if point['detected']:
            if before_seed or math.dist(point['center'], learned) <= s.detector_gate_working_px:
                candidate = {'center': list(point['center']), 'confidence': point['confidence'],
                             'proposal_source': 'existing_baseline_detector'}
            else:
                row['status'] = 'baseline_observation_disagrees_with_learned_estimate'
        else:
            left = max(0, min(w-384, round(local[0]-192)))
            top = max(0, min(h-384, round(local[1]-192)))
            region = {'native_xyxy': [left,top,left+384,top+384], 'query_center_native': list(local)}
            proposals, count = (infer or detect_native_boxes)(detector, record['crop'], region)
            proposals = merge_boxes(proposals, containment_threshold=.85)
            nearby = []
            for p in proposals:
                absolute = [p['center'][i]+origin[i] for i in range(2)]
                working = [absolute[0]/sx, absolute[1]/sy]
                row['native_proposals'].append({**p, 'center_source': absolute})
                if math.dist(working, learned) <= s.detector_gate_working_px:
                    nearby.append({'center': working, 'confidence': p['confidence'],
                                   'bbox_native_xyxy': [v+origin[i%2] for i,v in enumerate(p['bbox_native_xyxy'])],
                                   'proposal_source': 'learned_query_native_detector'})
            row['native_crop_count'] = count
            row['nearby_candidate_count'] = len(nearby)
            if len(nearby) == 1:
                candidate = nearby[0]
            else:
                row['status'] = 'ambiguous_detector_candidates' if nearby else 'no_current_detector_confirmation'
        if candidate is None:
            support.update(frame, [], available=False)
            rows.append(row)
            continue
        matrix, registration = verifier.align(record['working'])
        check = verifier.inspect(record['crop'], candidate, matrix, registration, origin=tuple(origin))
        row['background_check'] = check
        row['current_detector_center_working'] = candidate['center']
        if check['status'] != 'appearance_changed_not_identity_proof':
            support.update(frame, [], available=False)
            row['status'] = check['status']
            rows.append(row)
            continue
        evidence = support.update(frame, [[float(v) for v in check['reference_processing_center']]])[0]
        row['support_observation_frames'] = evidence['frames']
        row['status'] = 'baseline_observation_preserved' if point['detected'] else 'awaiting_detector_sequence_support'
        # Optional uncertain points request pixels only. Four consecutive real
        # detector observations are stronger evidence than a model visibility
        # flag; neither this agreement nor the flag proves target identity.
        uncertain_search = not estimate['visible'] and s.search_uncertain_points
        sequence_ok = evidence['supported'] and (not uncertain_search or
            evidence['frames'][-4:] == list(range(frame-3,frame+1)))
        row['required_consecutive_detector_observations'] = 4 if uncertain_search else None
        if not point['detected'] and sequence_ok:
            observed = {**deepcopy(point), 'center': list(candidate['center']),
                        'confidence': candidate['confidence'], 'detected': True, 'missing_run': 0,
                        'reason': 'detector_confirmed_learned_recovery_provisional',
                        'proposal_source': candidate['proposal_source'], 'identity_verified': False,
                        'support_observation_frames': list(evidence['frames']),
                        'bbox_native_xyxy': candidate['bbox_native_xyxy'],
                        'learned_estimate_source': source_point, 'model_visible': estimate['visible'],
                        'model_visibility_role': 'search_only' if uncertain_search else 'required_visible',
                        'uses_future_video_frames': True}
            additions.append(observed)
            last_anchor = frame
            row['status'] = 'current_detector_observation_recovered'
        rows.append(row)
    return additions, rows


def run_recovery(baseline, source, detector, detector_weights, provider, destination, *,
                 settings=None, baseline_path=None):
    """Post-process a frozen run; optional model failure leaves baseline usable."""
    import cv2
    s = settings or LearnedRecoverySettings()
    source, weights, destination = Path(source).resolve(), Path(detector_weights).resolve(), Path(destination)
    if destination.exists():
        raise FileExistsError("Preserve earlier results; choose a new report path")
    if file_sha256(source) != baseline['source_sha256'] or file_sha256(weights) != baseline['weights_sha256']:
        raise ValueError("Source/weights differ from the frozen baseline")
    transform = baseline['frame_transform']
    dimensions = (baseline['width'],baseline['height'],transform['source_width'],transform['source_height'])
    if (any(type(d) is not int or d < 384 for d in dimensions)
            or transform['source_width'] < baseline['width']
            or transform['source_height'] < baseline['height']):
        raise ValueError("Expected original source dimensions without upscaling")
    if (min(baseline['width'],baseline['height']) != 720
            or (transform['width'],transform['height']) != (baseline['width'],baseline['height'])
            or baseline.get('learned_recovery') is not None
            or max(transform['source_width'],transform['source_height']) > 3840
            or transform['source_width']*transform['source_height'] > 3840*2160
            or len(baseline['path']) > 1800):
        raise ValueError("Expected an unaugmented bounded 720-short-edge baseline")
    windows = plan_windows(baseline['path'],s)
    if any(p['detected'] and (not 0 <= p['center'][0] < baseline['width']
                             or not 0 <= p['center'][1] < baseline['height']
                             or type(p['confidence']) not in (int,float)
                             or not math.isfinite(p['confidence']) or not 0 < p['confidence'] <= 1)
           for p in baseline['path']):
        raise ValueError("Invalid source baseline observation")
    result = deepcopy(baseline)
    started = time.perf_counter()
    detail = {'status': 'offline_optional_detector_confirmed_learned_recovery',
              'settings': asdict(s), 'window_selection': 'baseline_gap_starts_with_three_prior_observations',
              'uses_future_video_frames': True, 'labels_used_for_inference': False,
              'model': provider.configuration(), 'eligible_windows': windows,
              'windows': [], 'errors': [], 'baseline_path': str(baseline_path) if baseline_path else None,
              'baseline_report_sha256': file_sha256(Path(baseline_path)) if baseline_path else None,
              'baseline_path_preserved': True, 'new_observation_frames': [],
              'identity_verified': False}
    cap = cv2.VideoCapture(str(source))
    try:
        ok, initial = cap.read()
    finally:
        cap.release()
    if not ok or initial.shape[:2] != (transform['source_height'],transform['source_width']):
        raise ValueError("Source preview geometry differs from baseline")
    working, _ = prepare_frame(initial,720)
    verifier = NativeEvidenceVerifier(initial,working,baseline['source_seed'])
    detail['background_verifier'] = verifier.configuration()
    del initial, working
    if detail['model'].get('status') == 'unavailable':
        detail['errors'].append({**detail['model']['error'],'stage':'optional_model_unavailable'})
        windows = []
    by_frame = {p['frame']:i for i,p in enumerate(result['path'])}
    sx, sy = transform['source_width']/baseline['width'], transform['source_height']/baseline['height']
    for index, planned in enumerate(windows[:s.maximum_windows]):
        if time.perf_counter()-started > s.maximum_runtime_s:
            detail['errors'].append({'type':'TimeoutError','stage':'between_windows'})
            break
        window = deepcopy(planned)
        seed = [window['seed_center_working'][0]*sx,window['seed_center_working'][1]*sy]
        crop = crop_for_seed(seed,transform['source_width'],transform['source_height'])
        window.update(seed_center_source=seed,crop_xyxy=crop,status='running')
        print(f"{baseline['clip']}: learned recovery window {index+1}, seed {window['seed_frame']}, frames {window['start_frame']}–{window['end_frame']}",flush=True)
        begin = time.perf_counter()
        try:
            records = load_window(source,window,crop,transform)
            window['decode_s'] = time.perf_counter()-begin
            model_start = time.perf_counter()
            estimates = provider.predict([r['crop'] for r in records],
                start_frame=window['start_frame'],seed_frame=window['seed_frame'],
                seed_local=[seed[0]-crop[0],seed[1]-crop[1]])
            window['learned_inference_s'] = time.perf_counter()-model_start
            additions, checks = confirm_window(baseline,window,records,estimates,verifier,detector,
                settings=s,deadline=started+s.maximum_runtime_s)
            window['model_estimates'] = estimates
            window['checks'] = checks
            added = []
            for observed in additions:
                target = by_frame[observed['frame']]
                if not result['path'][target]['detected']:
                    result['path'][target] = observed
                    added.append(observed['frame'])
            window.update(status='completed',added_observation_frames=added)
            print(f"{baseline['clip']}: recovered current detector observations {added}",flush=True)
        except Exception as exc:
            error = {'type':type(exc).__name__,'message':str(exc)[:240], 'stage':'recovery_window'}
            window.update(status='failed_baseline_preserved',error=error)
            detail['errors'].append(error)
        window['elapsed_s'] = time.perf_counter()-begin
        detail['windows'].append(window)
    if file_sha256(source) != baseline['source_sha256']:
        raise ValueError("Source changed during recovery")
    detail['elapsed_s'] = time.perf_counter()-started
    detail['new_observation_frames'] = [p['frame'] for a,p in zip(baseline['path'],result['path'],strict=True)
                                         if p['detected'] and not a['detected']]
    detail['implementation_sha256'] = {name:file_sha256(Path(__file__).with_name(name)) for name in
        ('detector_confirmed_recovery.py','cotracker_recovery.py','native_box_detector.py',
         'native_evidence_verifier.py','trajectory_evidence.py')}
    result.update(status='offline_detector_confirmed_learned_recovery_not_production',learned_recovery=detail,
                  selected_detection_frames=sum(p['detected'] for p in result['path']),
                  frames_without_selected_detection=sum(not p['detected'] for p in result['path']))
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--source',type=Path,help='Defaults to the frozen baseline source path')
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--cotracker-root',type=Path,required=True)
    parser.add_argument('--research-runtime',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=6)
    parser.add_argument('--search-uncertain-points',action='store_true',
        help='Offline ablation: uncertain points request pixels; require four consecutive detector observations')
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error('--threads must be between 1 and 16')
    from experiments.ball_tracking.cotracker_recovery import CoTrackerRecovery, UnavailableRecovery
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(args.threads)
    baseline = json.loads(args.baseline.read_text(encoding='utf-8-sig'))
    try:
        provider = CoTrackerRecovery(args.cotracker_root,runtime_directory=args.research_runtime,threads=args.threads)
    except (OSError,ValueError,KeyError) as exc:
        provider = UnavailableRecovery(exc)
    model = YOLO(str(args.weights))
    if model.task != 'detect' or model.names != {0:'ball'}:
        raise ValueError('Expected the frozen one-class ball detector')
    result = run_recovery(baseline,args.source or baseline['source_name'],model,args.weights,
                          provider,args.output,baseline_path=args.baseline,
                          settings=LearnedRecoverySettings(search_uncertain_points=args.search_uncertain_points))
    print(f"Saved {args.output}: {len(result['learned_recovery']['new_observation_frames'])} additional detector observations (not accuracy)")


if __name__ == '__main__':
    main()
