import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useEventListener } from 'expo';
import { useFocusEffect } from 'expo-router';
import { useVideoPlayer, VideoView, type VideoPlayer } from 'expo-video';
import { AppState, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { Action, BallProgress } from './controls';
import {
  containRect, pointAtTime, prepareTracking, previewImageUrl, startTracking, tapToSeed,
  waitForTracking, type ContentRect, type Position, type TrackingPreview,
  type TrackingResult, type TrackingStatus,
} from '../lib/experimental-tracking';
import type { Rep } from '../lib/reps';
import { ui } from '../lib/theme';

// SDK 58 VideoPlayer is a native mutable SharedObject. currentTime is its
// documented exact-seek API, rather than React render state.
function seekPaused(player: VideoPlayer, time: number) {
  player.pause();
  player.currentTime = time;
}

function Marker({ point, rect, kind }: { point: Position; rect: ContentRect; kind: 'candidate' | 'prediction' | 'seed' }) {
  const size = kind === 'seed' ? 22 : 18;
  return <View pointerEvents="none" style={[s.marker, kind === 'candidate' ? s.candidate : kind === 'seed' ? s.seed : s.prediction,
    { width: size, height: size, borderRadius: size / 2, left: rect.x + point.x * rect.width - size / 2,
      top: rect.y + point.y * rect.height - size / 2 }]} />;
}

// One original video/player, with a separate initial-frame picker. Tracking
// points are visual research output and never enter the coaching request.
export function ExperimentalReplay({ rep }: { rep: Rep }) {
  const player = useVideoPlayer(rep.uri, current => {
    current.muted = false;
    current.volume = 1;
    current.staysActiveInBackground = false;
    current.timeUpdateEventInterval = 1 / 30;
  });
  const [playbackStatus, setPlaybackStatus] = useState(player.status);
  const [playbackError, setPlaybackError] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const [preview, setPreview] = useState<TrackingPreview | null>(null);
  const [seed, setSeed] = useState<Position | null>(null);
  const [job, setJob] = useState<TrackingStatus | null>(null);
  const [result, setResult] = useState<TrackingResult | null>(null);
  const [busy, setBusy] = useState<'upload' | 'tracking' | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [videoLayout, setVideoLayout] = useState({ width: 0, height: 0 });
  const [previewLayout, setPreviewLayout] = useState({ width: 0, height: 0 });
  const operation = useRef<AbortController | null>(null);
  const focused = useRef(true);

  const stopWaiting = useCallback(() => {
    const pending = operation.current;
    operation.current = null;
    pending?.abort();
    setBusy(null);
    if (pending) setNotice('Stopped waiting. A job already started on the laptop continues. Use Check status to reconnect while this screen is open.');
  }, []);

  useFocusEffect(useCallback(() => {
    focused.current = true;
    return () => { focused.current = false; player.pause(); stopWaiting(); };
  }, [player, stopWaiting]));

  useEffect(() => {
    const listener = AppState.addEventListener('change', state => {
      if (state !== 'active') { player.pause(); stopWaiting(); }
    });
    return () => listener.remove();
  }, [player, stopWaiting]);

  useEventListener(player, 'statusChange', event => {
    setPlaybackStatus(event.status); setPlaybackError(event.error?.message ?? null);
  });
  useEventListener(player, 'timeUpdate', event => { setTime(event.currentTime); });

  const videoRect = useMemo(() => result
    ? containRect(videoLayout.width, videoLayout.height, result.width, result.height) : null,
  [videoLayout, result]);
  const previewRect = useMemo(() => preview
    ? containRect(previewLayout.width, previewLayout.height, preview.width, preview.height) : null,
  [previewLayout, preview]);
  const point = result ? pointAtTime(result.points, time, result.fps) : null;
  const visibleOverlay = Boolean(result && showOverlay && !fullscreen);
  const marker = point?.state === 'observed_candidate' ? point.center : point?.state === 'predicted' ? point.prediction : null;
  const overlayLabel = !point ? 'No tracking observation for this frame'
    : point.state === 'observed_candidate' ? 'Observed candidate · Identity unverified'
      : point.state === 'predicted' ? 'Prediction only · Ball not observed' : 'Lost · No ball marker';

  async function run(kind: 'upload' | 'tracking', task: (signal: AbortSignal) => Promise<void>) {
    if (operation.current || !focused.current || AppState.currentState !== 'active') return;
    const controller = new AbortController();
    operation.current = controller;
    setBusy(kind); setError(null); setNotice(null);
    try { await task(controller.signal); }
    catch (failure) {
      if (focused.current && operation.current === controller) {
        setError(failure instanceof Error ? failure.message : 'Tracking could not finish. Your original clip is safe.');
      }
    } finally {
      if (operation.current === controller) {
        operation.current = null;
        if (focused.current) setBusy(null);
      }
    }
  }

  function upload() {
    player.pause();
    void run('upload', async signal => {
      setPreview(null); setJob(null); setSeed(null); setResult(null); setImageLoaded(false);
      const prepared = await prepareTracking(rep, signal);
      if (signal.aborted || !focused.current) return;
      setPreview(prepared);
      setJob({ tracking_id: prepared.tracking_id, status: 'ready', processed_frames: 0,
        total_frames: Math.ceil(prepared.duration_s * prepared.fps), result_url: null, error: null });
    });
  }

  async function collect(prepared: TrackingPreview, signal: AbortSignal) {
    const completed = await waitForTracking(prepared.tracking_id, signal, status => {
      if (!signal.aborted && focused.current) setJob(status);
    });
    if (signal.aborted || !focused.current) return;
    if (completed.width !== prepared.width || completed.height !== prepared.height ||
        Math.abs(completed.fps - prepared.fps) > .001 || completed.start_frame !== prepared.frame_index ||
        (seed && (Math.abs(completed.seed.x - seed.x) > .00001 || Math.abs(completed.seed.y - seed.y) > .00001))) {
      throw new Error('This tracking result does not match the initial frame and ball mark.');
    }
    seekPaused(player, completed.start_frame / completed.fps);
    setTime(completed.start_frame / completed.fps); setResult(completed); setShowOverlay(true);
    setNotice('Tracking review ready. Press play above to inspect the original clip with the overlay.');
  }

  function start() {
    if (!preview || !seed || job?.status !== 'ready') return;
    const prepared = preview, selected = seed;
    player.pause();
    void run('tracking', async signal => {
      // Keep the known job id even if the start response is interrupted. A
      // status check reconnects without uploading or starting another job.
      setJob(current => current ? { ...current, status: 'queued' } : current);
      const status = await startTracking(prepared.tracking_id, selected, signal);
      if (signal.aborted || !focused.current) return;
      setJob(status);
      await collect(prepared, signal);
    });
  }

  function reconnect() {
    if (!preview) return;
    const prepared = preview;
    void run('tracking', signal => collect(prepared, signal));
  }

  return <>
    <View style={s.videoFrame} onLayout={event => setVideoLayout(event.nativeEvent.layout)}>
      <VideoView player={player} style={s.video} nativeControls contentFit="contain"
        fullscreenOptions={{ enable: !visibleOverlay }} allowsPictureInPicture={false}
        onFullscreenEnter={() => setFullscreen(true)} onFullscreenExit={() => setFullscreen(false)} />
      {visibleOverlay && videoRect && marker && <Marker point={marker} rect={videoRect}
        kind={point?.state === 'observed_candidate' ? 'candidate' : 'prediction'} />}
    </View>
    {(playbackStatus === 'loading' || playbackStatus === 'idle') && <BallProgress label="Opening your clip…" />}
    {playbackStatus === 'error' && <Text style={ui.error}>{playbackError || 'This clip could not be played.'}</Text>}
    {result && <View style={{ gap: 8 }}>
      <Text style={ui.small}>{showOverlay ? overlayLabel : 'Tracking overlay hidden'}</Text>
      <Text style={ui.small}>Yellow ring: observed candidate. Purple dashed ring: prediction. No marker during loss. Playback alignment uses reported FPS and is approximate.</Text>
      <Action secondary title={showOverlay ? 'Hide tracking overlay' : 'Show tracking overlay'} onPress={() => setShowOverlay(current => !current)} />
    </View>}
    <View style={ui.card}>
      <Text style={ui.badge}>EXPERIMENTAL BALL TRACKING</Text>
      <Text style={ui.heading}>Mark once. Review the track.</Text>
      <Text style={ui.body}>For stationary shots, send this original clip to your laptop, tap the ball on its initial frame, then inspect the tracker over your replay.</Text>
      <Text style={ui.small}>This is a visual experiment. Candidate positions and predictions do not verify ball identity or feed coaching, speed, curl or goal outcomes.</Text>
      {!preview && !busy && <Action title="Load initial frame" onPress={upload} />}
      {busy === 'upload' && <BallProgress label="Sending original clip and preparing its initial frame…" />}
      {preview && job?.status === 'ready' && <>
        <Text style={ui.small}>Tap the centre of the ball below at {preview.time_s.toFixed(2)} s. The cyan ring is your initial mark, not a detection. Tap again to adjust before starting.</Text>
        <Pressable accessibilityRole="button" accessibilityLabel="Tap the ball in the initial video frame"
          disabled={Boolean(busy) || !imageLoaded} style={s.previewFrame}
          onLayout={event => setPreviewLayout(event.nativeEvent.layout)}
          onPress={event => {
            const selected = tapToSeed(event.nativeEvent.locationX, event.nativeEvent.locationY, previewRect);
            if (selected) { setSeed(selected); setError(null); }
            else setError('Tap inside the image. The black border is outside the video.');
          }}>
          <Image pointerEvents="none" source={{ uri: previewImageUrl(preview) }} style={s.preview} resizeMode="contain"
            onLoad={() => setImageLoaded(true)} onError={() => { setImageLoaded(false); setError('The initial frame could not load. Check your connection, then load a new preview.'); }} />
          {seed && previewRect && <Marker point={seed} rect={previewRect} kind="seed" />}
        </Pressable>
        {!imageLoaded && !error && <Text style={ui.small}>Opening the initial frame…</Text>}
        <Action title="Run experimental tracking" disabled={!seed || !imageLoaded || Boolean(busy)} onPress={start} />
      </>}
      {busy === 'tracking' && <BallProgress
        progress={job?.total_frames ? job.processed_frames / job.total_frames : undefined}
        label={job?.status === 'queued' ? 'Waiting for the laptop tracker…' : 'Tracking on your laptop…'} />}
      {job && !busy && job.status !== 'ready' && !result && <>
        <Text style={ui.small}>Tracking job: {job.status}. Checking status reuses this job and does not upload the video again.</Text>
        <Action secondary title="Check status" onPress={reconnect} />
      </>}
      {busy && <Action secondary title="Stop waiting" onPress={stopWaiting} />}
      {preview && !busy && <Action secondary title="Load a new initial frame" onPress={upload} />}
      {notice && <Text style={ui.small} accessibilityLiveRegion="polite">{notice}</Text>}
      {error && <Text style={ui.error} accessibilityRole="alert">{error}</Text>}
      <Text style={ui.small}>Keep the laptop backend and Tailscale connected. Processing can take several minutes. Leaving or backgrounding this screen pauses replay and stops waiting; an already started laptop job continues. Your original recording stays on this phone.</Text>
    </View>
  </>;
}

const s = StyleSheet.create({
  videoFrame: { backgroundColor: '#000', borderRadius: 22, overflow: 'hidden' },
  video: { width: '100%', aspectRatio: 3 / 4 },
  previewFrame: { width: '100%', aspectRatio: 3 / 4, backgroundColor: '#000', overflow: 'hidden', borderRadius: 12 },
  preview: { width: '100%', height: '100%' },
  marker: { position: 'absolute', borderWidth: 2.5, backgroundColor: 'transparent' },
  candidate: { borderColor: '#FFE050' },
  prediction: { borderColor: '#D99CFF', borderStyle: 'dashed' },
  seed: { borderColor: '#64E5F2', borderWidth: 2 },
});
