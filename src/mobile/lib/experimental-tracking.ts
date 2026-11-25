import type {} from 'expo/types';
import { File } from 'expo-file-system';
import { fetch } from 'expo/fetch';
import type { Rep } from './reps';

export type Position = { x: number; y: number };
export type ContentRect = { x: number; y: number; width: number; height: number };
export type TrackingPreview = {
  tracking_id: string; status: 'ready'; preview_url: string;
  width: number; height: number; preview_width: number; preview_height: number;
  frame_index: number; time_s: number; fps: number; duration_s: number;
  recovery_available?: boolean;
};
export type TrackingStatus = {
  tracking_id: string;
  status: 'ready' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
  processed_frames: number; total_frames: number; result_url: string | null; error: string | null;
  phase?: 'baseline' | 'recovery' | 'completed';
  recovery_processed_windows?: number; recovery_total_windows?: number;
};
export type TrackingPoint = {
  frame: number; time_s: number; state: 'observed_candidate' | 'predicted' | 'lost';
  end_time_s?: number;
  center: Position | null; prediction: Position | null; identity_verified: false; reason: string;
  recovered?: boolean;
};
export type TrackingResult = {
  tracking_id: string; status: 'completed'; width: number; height: number; fps: number;
  start_frame: number; seed: Position; points: TrackingPoint[];
  timing_mode?: 'presentation_timestamps' | 'reported_fps_approximate';
  processing?: { model_load_s: number; baseline_s: number; recovery_s: number; total_s: number };
  recovery?: { requested: boolean; status: 'not_requested' | 'completed' | 'partial' | 'unavailable'; added_observations: number; warning: string | null };
};

const ID = /^[a-f0-9]{32}$/;
const MAX_POINTS = 30_000;
const WAIT_LIMIT_MS = 20 * 60_000;
const statuses: TrackingStatus['status'][] = ['ready', 'queued', 'running', 'completed', 'failed', 'cancelled', 'interrupted'];
const object = (value: unknown): Record<string, unknown> | null => value !== null && typeof value === 'object' && !Array.isArray(value)
  ? value as Record<string, unknown> : null;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const positive = (value: unknown): value is number => finite(value) && value > 0;
const integer = (value: unknown): value is number => finite(value) && Number.isInteger(value) && value >= 0;
const message = (value: unknown): string | null => typeof value === 'string' && value.trim() ? value.trim().slice(0, 500) : null;

function id(value: unknown): string {
  if (typeof value !== 'string' || !ID.test(value)) throw new Error('SCAN returned an invalid tracking job.');
  return value;
}

export function trackingOrigin(setting = process.env.EXPO_PUBLIC_SCAN_API_URL): string {
  if (!setting?.trim()) throw new Error('Set EXPO_PUBLIC_SCAN_API_URL in .env.local and restart Metro.');
  const url = new URL(setting.trim());
  if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash || url.pathname !== '/') {
    throw new Error('Use the HTTPS backend origin without a path or credentials.');
  }
  return url.origin;
}

// Never follow arbitrary server-provided URLs or send a local video elsewhere.
export function trackingArtifactPath(value: unknown, jobId: string, kind: 'preview' | 'result'): string {
  const expected = `/experimental-tracking/${id(jobId)}/${kind}`;
  if (value !== expected) throw new Error('SCAN returned a tracking artifact outside this job.');
  return expected;
}

export function previewImageUrl(preview: TrackingPreview): string {
  return trackingOrigin() + trackingArtifactPath(preview.preview_url, preview.tracking_id, 'preview');
}

function position(value: unknown): Position | null {
  const point = object(value);
  return point && finite(point.x) && finite(point.y) && point.x >= 0 && point.x < 1 && point.y >= 0 && point.y < 1
    ? { x: point.x, y: point.y } : null;
}

export function parseTrackingPreview(value: unknown): TrackingPreview {
  const data = object(value);
  if (!data || data.status !== 'ready' || !positive(data.width) || !positive(data.height) ||
      !positive(data.preview_width) || !positive(data.preview_height) || !positive(data.fps) ||
      data.fps > 240 || !positive(data.duration_s) || !integer(data.frame_index) || !finite(data.time_s) || data.time_s < 0) {
    throw new Error('SCAN returned an incomplete tracking preview.');
  }
  const jobId = id(data.tracking_id);
  if (Math.abs(data.time_s - data.frame_index / data.fps) > 1 / data.fps || data.time_s >= data.duration_s ||
      Math.abs(data.width / data.height - data.preview_width / data.preview_height) > .01) {
    throw new Error('The tracking preview has inconsistent orientation or timing.');
  }
  return { tracking_id: jobId, status: 'ready', preview_url: trackingArtifactPath(data.preview_url, jobId, 'preview'),
    width: data.width, height: data.height, preview_width: data.preview_width, preview_height: data.preview_height,
    frame_index: data.frame_index, time_s: data.time_s, fps: data.fps, duration_s: data.duration_s,
    ...(typeof data.recovery_available === 'boolean' ? { recovery_available: data.recovery_available } : {}) };
}

export function parseTrackingStatus(value: unknown, expectedId: string): TrackingStatus {
  const data = object(value);
  if (!data || data.tracking_id !== id(expectedId) || !statuses.includes(data.status as TrackingStatus['status']) ||
      !integer(data.processed_frames) || !integer(data.total_frames) || data.processed_frames > data.total_frames) {
    throw new Error('SCAN returned an invalid tracking status.');
  }
  const resultUrl = data.result_url === null || data.result_url === undefined ? null
    : trackingArtifactPath(data.result_url, expectedId, 'result');
  if (data.status === 'completed' && !resultUrl) throw new Error('The completed tracking result is unavailable.');
  const phase = data.phase;
  if (phase !== undefined && !['baseline', 'recovery', 'completed'].includes(phase as string)) throw new Error('Invalid tracking stage.');
  const windows = data.recovery_total_windows, processed = data.recovery_processed_windows;
  if ((windows !== undefined || processed !== undefined) &&
      (!integer(windows) || !integer(processed) || processed > windows)) throw new Error('Invalid recovery progress.');
  return { tracking_id: expectedId, status: data.status as TrackingStatus['status'], processed_frames: data.processed_frames,
    total_frames: data.total_frames, result_url: resultUrl, error: message(data.error),
    ...(phase !== undefined ? { phase: phase as TrackingStatus['phase'] } : {}),
    ...(integer(windows) && integer(processed) ? { recovery_total_windows: windows, recovery_processed_windows: processed } : {}) };
}

export function parseTrackingResult(value: unknown, expectedId: string): TrackingResult {
  const data = object(value), seed = position(data?.seed);
  if (!data || data.tracking_id !== id(expectedId) || data.status !== 'completed' || !positive(data.width) ||
      !positive(data.height) || !positive(data.fps) || data.fps > 240 || !integer(data.start_frame) || !seed ||
      !Array.isArray(data.points) || data.points.length > MAX_POINTS) {
    throw new Error('SCAN returned an invalid tracking result.');
  }
  const fps = data.fps;
  const timingMode = data.timing_mode;
  if (timingMode !== undefined && !['presentation_timestamps', 'reported_fps_approximate'].includes(timingMode as string)) {
    throw new Error('Invalid tracking playback timing.');
  }
  const presentationTiming = timingMode === 'presentation_timestamps';
  let previous = data.start_frame - 1;
  let previousTime = -1, previousEnd = -1;
  const points: TrackingPoint[] = data.points.map(raw => {
    const point = object(raw);
    if (!point || !integer(point.frame) || point.frame <= previous || !finite(point.time_s) || point.time_s < 0 ||
        (presentationTiming
          ? point.time_s <= previousTime || !finite(point.end_time_s) || point.end_time_s <= point.time_s ||
            point.end_time_s > 31 || point.time_s < previousEnd
          : Math.abs(point.time_s - point.frame / fps) > 1 / fps) || point.identity_verified !== false ||
        typeof point.state !== 'string' || !['observed_candidate', 'predicted', 'lost'].includes(point.state)) {
      throw new Error('The tracking result contains an invalid frame.');
    }
    previous = point.frame;
    previousTime = point.time_s;
    if (presentationTiming) previousEnd = point.end_time_s as number;
    const center = position(point.center), prediction = position(point.prediction);
    if (point.state === 'observed_candidate' && !center) throw new Error('A tracking observation has no valid location.');
    // A lost frame never acquires a marker from another frame or its prediction.
    return { frame: point.frame, time_s: point.time_s, state: point.state as TrackingPoint['state'],
      center: point.state === 'observed_candidate' ? center : null,
      prediction: point.state === 'predicted' ? prediction : null,
      identity_verified: false, reason: message(point.reason) || 'Ball identity remains unverified.',
      ...(presentationTiming ? { end_time_s: point.end_time_s as number } : {}),
      ...(typeof point.recovered === 'boolean' ? { recovered: point.state === 'observed_candidate' && point.recovered } : {}) };
  });
  let processing: TrackingResult['processing'];
  if (data.processing != null) {
    const timing = object(data.processing);
    if (!timing || ![timing.model_load_s, timing.baseline_s, timing.recovery_s, timing.total_s].every(v => finite(v) && v >= 0)) {
      throw new Error('Invalid tracking processing times.');
    }
    processing = { model_load_s: timing.model_load_s as number, baseline_s: timing.baseline_s as number,
      recovery_s: timing.recovery_s as number, total_s: timing.total_s as number };
  }
  let recovery: TrackingResult['recovery'];
  if (data.recovery != null) {
    const detail = object(data.recovery);
    if (!detail || typeof detail.requested !== 'boolean' || !integer(detail.added_observations) ||
        !['not_requested', 'completed', 'partial', 'unavailable'].includes(detail.status as string)) throw new Error('Invalid recovery result.');
    recovery = { requested: detail.requested, status: detail.status as NonNullable<TrackingResult['recovery']>['status'],
      added_observations: detail.added_observations, warning: message(detail.warning) };
  }
  return { tracking_id: expectedId, status: 'completed', width: data.width, height: data.height,
    fps, start_frame: data.start_frame, seed, points,
    ...(timingMode !== undefined ? { timing_mode: timingMode as TrackingResult['timing_mode'] } : {}),
    ...(processing ? { processing } : {}), ...(recovery ? { recovery } : {}) };
}

export function containRect(containerWidth: number, containerHeight: number, imageWidth: number, imageHeight: number): ContentRect | null {
  if (![containerWidth, containerHeight, imageWidth, imageHeight].every(positive)) return null;
  const scale = Math.min(containerWidth / imageWidth, containerHeight / imageHeight);
  const width = imageWidth * scale, height = imageHeight * scale;
  return { x: (containerWidth - width) / 2, y: (containerHeight - height) / 2, width, height };
}

export function tapToSeed(x: number, y: number, rect: ContentRect | null): Position | null {
  if (!rect || !finite(x) || !finite(y) || x < rect.x || y < rect.y || x >= rect.x + rect.width || y >= rect.y + rect.height) return null;
  return { x: (x - rect.x) / rect.width, y: (y - rect.y) / rect.height };
}

// Only the exact current frame is eligible. No nearest-point snapping, future
// selection, interpolation or carrying a point across missing observations.
export function pointAtTime(points: readonly TrackingPoint[], time: number, fps: number): TrackingPoint | null {
  if (!finite(time) || time < 0 || !positive(fps)) return null;
  if (points[0]?.end_time_s !== undefined) {
    // Each measured source-frame interval is explicit. Never extrapolate a
    // position, bridge an omitted frame or move the track forward by a guessed offset.
    let low = 0, high = points.length - 1, selected: TrackingPoint | null = null;
    while (low <= high) {
      const middle = Math.floor((low + high) / 2), point = points[middle];
      if (point.time_s <= time) { selected = point; low = middle + 1; }
      else high = middle - 1;
    }
    return selected?.end_time_s !== undefined && time < selected.end_time_s ? selected : null;
  }
  const frame = Math.floor(time * fps + 1e-7);
  let low = 0, high = points.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2), point = points[middle];
    if (point.frame === frame) return point.time_s <= time + 1e-6 ? point : null;
    if (point.frame < frame) low = middle + 1; else high = middle - 1;
  }
  return null;
}

async function request(path: string, options: RequestInit, signal: AbortSignal, timeoutMs: number): Promise<unknown> {
  if (!/^\/experimental-tracking\/(?:preview|[a-f0-9]{32}(?:\/(?:start|result|cancel))?)$/.test(path)) throw new Error('Invalid tracking request.');
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal.addEventListener('abort', cancel, { once: true });
  if (signal.aborted) controller.abort();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const response = await fetch(trackingOrigin() + path, { ...options, signal: controller.signal,
      headers: { Accept: 'application/json', ...options.headers } });
    const data: unknown = await response.json().catch(() => null);
    if (!response.ok) throw new Error(message(object(data)?.detail) || `Tracking returned HTTP ${response.status}.`);
    if (!object(data)) throw new Error('Expected SCAN tracking JSON. Check the backend address and restart the updated backend.');
    return data;
  } catch (error) {
    if (signal.aborted) throw new Error('Stopped waiting. A job already started on the laptop continues there.');
    if (timedOut) throw new Error('Tracking request timed out. Check status before starting another job.');
    if (error instanceof TypeError && /network|fetch/i.test(error.message)) {
      throw new Error('Cannot reach SCAN tracking. Check Tailscale, internet access and the laptop backend.');
    }
    throw error;
  } finally {
    clearTimeout(timer); signal.removeEventListener('abort', cancel);
  }
}

export async function prepareTracking(rep: Rep, signal: AbortSignal): Promise<TrackingPreview> {
  const file = new File(rep.uri);
  if (!file.exists || !positive(file.size)) throw new Error('This original clip is unavailable. Try playing it first.');
  const form = new FormData();
  form.append('video', file, rep.name);
  return parseTrackingPreview(await request('/experimental-tracking/preview', { method: 'POST', body: form }, signal, 180_000));
}

export async function startTracking(jobId: string, seed: Position, signal: AbortSignal, recoverMisses = false): Promise<TrackingStatus> {
  if (!position(seed)) throw new Error('Tap the ball inside the preview first.');
  return parseTrackingStatus(await request(`/experimental-tracking/${id(jobId)}/start`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ seed, ...(recoverMisses ? { recover_misses: true } : {}) }),
  }, signal, 20_000), jobId);
}

export async function trackingStatus(jobId: string, signal: AbortSignal): Promise<TrackingStatus> {
  return parseTrackingStatus(await request(`/experimental-tracking/${id(jobId)}`, {}, signal, 15_000), jobId);
}

export async function cancelTracking(jobId: string, signal: AbortSignal): Promise<TrackingStatus> {
  return parseTrackingStatus(await request(`/experimental-tracking/${id(jobId)}/cancel`, { method: 'POST' }, signal, 15_000), jobId);
}

export async function trackingResult(jobId: string, signal: AbortSignal): Promise<TrackingResult> {
  return parseTrackingResult(await request(`/experimental-tracking/${id(jobId)}/result`, {}, signal, 30_000), jobId);
}

function pause(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const cancel = () => { clearTimeout(timer); reject(new Error('Stopped waiting. The laptop job continues.')); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', cancel); resolve(); }, ms);
    signal.addEventListener('abort', cancel, { once: true });
    if (signal.aborted) cancel();
  });
}

export async function waitForTracking(jobId: string, signal: AbortSignal, onStatus: (status: TrackingStatus) => void): Promise<TrackingResult> {
  const deadline = Date.now() + WAIT_LIMIT_MS;
  while (Date.now() < deadline) {
    const status = await trackingStatus(jobId, signal);
    onStatus(status);
    if (status.status === 'completed') return trackingResult(jobId, signal);
    if (['failed', 'cancelled', 'interrupted'].includes(status.status)) throw new Error(status.error || `The tracking job ${status.status}.`);
    if (status.status === 'ready') throw new Error('The preview is ready. Tap the ball, then start tracking.');
    await pause(1500, signal);
  }
  throw new Error('Stopped waiting after 20 minutes. The laptop may still be working; use Check status to reconnect.');
}
