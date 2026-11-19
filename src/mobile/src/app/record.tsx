import { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, AppState, BackHandler, Linking, Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { router, Stack, useFocusEffect } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useKeepAwake } from 'expo-keep-awake';
import { Action, BallProgress } from '../../components/controls';
import { useReps } from '../../lib/rep-store';
import { colors, ui } from '../../lib/theme';
import { startCountdown } from '../../lib/countdown';

type Phase = 'ready' | 'countdown' | 'recording' | 'saving' | 'saved';
const captureProfiles = {
  '720p': { label: '720p', bitrate: 6_000_000 },
  '1080p': { label: '1080p', bitrate: 12_000_000 },
  '2160p': { label: '4K', bitrate: 35_000_000 },
} as const;
type CaptureQuality = keyof typeof captureProfiles;

function KeepScreenAwake() {
  useKeepAwake();
  return null;
}

export default function Record() {
  const [permission, requestPermission] = useCameraPermissions();
  const camera = useRef<CameraView>(null);
  const lock = useRef(false);
  const recording = useRef(false);
  const began = useRef(0);
  const countdown = useRef<ReturnType<typeof startCountdown> | null>(null);
  const active = useRef(false);
  const cameraGeneration = useRef(0);
  const [generation, setGeneration] = useState(0);
  const [quality, setQuality] = useState<CaptureQuality>('1080p');
  const [stabilization, setStabilization] = useState<'auto' | 'off'>('auto');
  const [showCameraSettings, setShowCameraSettings] = useState(false);
  const [repSeconds, setRepSeconds] = useState(5);
  const [delaySeconds, setDelaySeconds] = useState(3);
  const [remaining, setRemaining] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [phase, setPhase] = useState<Phase>('ready');
  const [cameraReady, setCameraReady] = useState(false);
  const [foreground, setForeground] = useState(AppState.currentState === 'active');
  const [error, setError] = useState<string | null>(null);
  const [focused, setFocused] = useState(false);
  const store = useReps();
  const busy = phase === 'countdown' || phase === 'recording' || phase === 'saving' || store.saving;

  function configureCamera(nextQuality: CaptureQuality, nextStabilization: 'auto' | 'off') {
    if (lock.current || busy || (nextQuality === quality && nextStabilization === stabilization)) return;
    // Reconfigure only between reps. Wait for the new native camera to be ready.
    cameraGeneration.current += 1;
    setGeneration(cameraGeneration.current);
    setCameraReady(false); setError(null); setPhase('ready');
    setQuality(nextQuality); setStabilization(nextStabilization);
  }

  useFocusEffect(useCallback(() => {
    active.current = AppState.currentState === 'active'; setFocused(true);
    return () => {
      active.current = false; countdown.current?.cancel(); setFocused(false); setCameraReady(false);
    };
  }, []));
  useEffect(() => {
    if (!focused || !busy) return;
    const listener = BackHandler.addEventListener('hardwareBackPress', () => {
      Alert.alert('Keep this rep open', 'Cancel the countdown, or stop recording and wait for the rep to finish saving before leaving.');
      return true;
    });
    return () => listener.remove();
  }, [focused, busy]);
  useEffect(() => {
    const listener = AppState.addEventListener('change', state => {
      active.current = state === 'active';
      if (state !== 'active') countdown.current?.cancel();
      if (state !== 'active' && recording.current) {
        try { camera.current?.stopRecording(); } catch { /* The OS may already have interrupted it. */ }
      }
      setForeground(state === 'active');
      if (state !== 'active') setCameraReady(false);
    });
    return () => listener.remove();
  }, []);
  useEffect(() => {
    if (phase !== 'recording') return;
    const timer = setInterval(() => setElapsed(Math.min(repSeconds, (Date.now() - began.current) / 1000)), 100);
    return () => clearInterval(timer);
  }, [phase, repSeconds]);

  async function record() {
    if (lock.current || busy || store.pending || store.loading || !cameraReady || !camera.current || !foreground || !focused) return;
    lock.current = true; setElapsed(0); setError(null); setShowCameraSettings(false);
    try {
      if (delaySeconds) {
        setPhase('countdown');
        const waiting = startCountdown(delaySeconds, setRemaining);
        countdown.current = waiting;
        const completed = await waiting.promise;
        countdown.current = null;
        if (!completed || !active.current || !camera.current) { setPhase('ready'); return; }
      }
      if (!active.current || !camera.current) { setPhase('ready'); return; }
      recording.current = true; began.current = Date.now(); setPhase('recording');
      const clip = await camera.current.recordAsync({ maxDuration: repSeconds,
        // Expo requires an explicit iOS codec for videoBitrate to take effect.
        ...(Platform.OS === 'ios' ? { codec: 'avc1' as const } : {}) });
      recording.current = false; setPhase('saving');
      if (!clip?.uri) throw new Error('The recording was interrupted before a clip was returned.');
      await store.save(clip.uri);
      setPhase('saved');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not finish this recording.'); setPhase('ready');
    } finally { recording.current = false; lock.current = false; }
  }
  function stop() {
    try { camera.current?.stopRecording(); }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not stop the recording.'); }
  }
  async function retry() {
    try { await store.retry(); setError(null); setPhase('saved'); }
    catch { /* The store exposes the error and preserves the pending clip. */ }
  }
  const status = phase === 'countdown' ? 'GET READY' : phase === 'recording' ? 'RECORDING' : phase === 'saving' || store.saving ? 'SAVING' :
    phase === 'saved' ? 'SAVED' : cameraReady ? 'READY' : 'CAMERA SETUP';

  return <SafeAreaView style={ui.screen} edges={['bottom', 'left', 'right']}>
    {focused && foreground && <KeepScreenAwake />}
    <Stack.Screen options={{ gestureEnabled: !busy, headerBackVisible: false,
      headerLeft: () => <Pressable accessibilityRole="button" accessibilityLabel="Back to home"
        accessibilityState={{ disabled: busy }} disabled={busy} hitSlop={10}
        onPress={() => router.canGoBack() ? router.back() : router.replace('/')}
        style={{ minHeight: 44, minWidth: 44, justifyContent: 'center', opacity: busy ? .35 : 1 }}>
        <Text style={{ color: colors.green, fontSize: 27 }}>‹</Text>
      </Pressable> }} />
    <ScrollView contentContainerStyle={[ui.content, { gap: 18 }]}>
      <View style={ui.row}><Text style={ui.eyebrow}>ONE REP AT A TIME</Text>
        <View style={s.status}><View style={[s.dot, phase === 'recording' && { backgroundColor: '#FF8A76' }]} /><Text style={ui.badge}>{status}</Text></View></View>
      <Text style={[ui.title, { fontSize: 29, lineHeight: 34 }]}>Find your frame.</Text>
      <View style={s.cameraFrame}>
        {permission?.granted && foreground && focused ? <CameraView key={generation} ref={camera} style={StyleSheet.absoluteFill}
          facing="back" mode="video" mute videoQuality={quality} videoBitrate={captureProfiles[quality].bitrate}
          videoStabilizationMode={stabilization} zoom={0}
          onCameraReady={() => { if (generation === cameraGeneration.current) setCameraReady(true); }}
          onMountError={e => { if (generation === cameraGeneration.current) { setCameraReady(false); setError(e.message); } }} /> : <View style={s.cameraPlaceholder}>
          <Text style={ui.heading}>{permission?.granted ? 'Camera paused' : 'Your pitch. Your perspective.'}</Text>
          <Text style={[ui.body, { textAlign: 'center' }]}>Position your phone so your full movement stays in frame.</Text>
        </View>}
        <View pointerEvents="none" style={s.topTag}><Text style={s.tagText}>{repSeconds} SEC · {captureProfiles[quality].label} REQUESTED · VIDEO ONLY</Text></View>
        <View pointerEvents="none" style={[s.corner, s.topLeft]} /><View pointerEvents="none" style={[s.corner, s.topRight]} />
        <View pointerEvents="none" style={[s.corner, s.bottomLeft]} /><View pointerEvents="none" style={[s.corner, s.bottomRight]} />
        {phase === 'countdown' && <View pointerEvents="none" style={s.countdown} accessibilityLiveRegion="polite">
          <Text style={s.countdownNumber}>{remaining || 'GO'}</Text><Text style={ui.heading}>Get into position</Text>
        </View>}
      </View>
      <View style={ui.card}>
        <Text style={ui.badge}>RECORDING LENGTH</Text>
        <Choices values={[5, 10, 15]} selected={repSeconds} disabled={busy}
          onSelect={value => { setRepSeconds(value); setPhase('ready'); }} label="Recording length" />
        <Text style={ui.badge}>START COUNTDOWN</Text>
        <Choices values={[0, 3, 5]} selected={delaySeconds} disabled={busy}
          onSelect={setDelaySeconds} label="Start countdown" />
      </View>
      <View style={ui.card}>
        <Pressable accessibilityRole="button" accessibilityLabel="Camera settings"
          accessibilityState={{ expanded: showCameraSettings, disabled: busy }} disabled={busy}
          onPress={() => setShowCameraSettings(value => !value)} style={[ui.row, { minHeight: 44 }]}>
          <View style={{ gap: 6, flex: 1 }}><Text style={ui.badge}>CAMERA SETTINGS</Text>
            <Text style={ui.body}>{captureProfiles[quality].label} · Stabilization {stabilization === 'auto' ? 'Auto' : 'Off'}</Text></View>
          <Text style={ui.heading}>{showCameraSettings ? '−' : '+'}</Text>
        </Pressable>
        {showCameraSettings && <><Text style={ui.badge}>RECORDING QUALITY</Text>
        <View style={ui.row}>{(Object.keys(captureProfiles) as CaptureQuality[]).map(value => <Pressable key={value}
          accessibilityRole="button" accessibilityLabel={`Recording quality: ${captureProfiles[value].label}`}
          accessibilityState={{ selected: quality === value, disabled: busy }} disabled={busy}
          onPress={() => configureCamera(value, stabilization)}
          style={[s.choice, quality === value && s.selected, busy && { opacity: .45 }]}>
          <Text style={[ui.heading, quality === value && { color: colors.dark }]}>{captureProfiles[value].label}</Text>
        </Pressable>)}</View>
        <Text style={ui.small}>Start with 1080p. 4K can keep more detail but uses more storage and takes longer to upload. Your phone uses a supported resolution.</Text>
        <Text style={ui.badge}>IMAGE STABILIZATION</Text>
        <View style={ui.row}>{(['auto', 'off'] as const).map(value => <Pressable key={value}
          accessibilityRole="button" accessibilityLabel={`Image stabilization: ${value}`}
          accessibilityState={{ selected: stabilization === value, disabled: busy }} disabled={busy}
          onPress={() => configureCamera(quality, value)}
          style={[s.choice, stabilization === value && s.selected, busy && { opacity: .45 }]}>
          <Text style={[ui.heading, stabilization === value && { color: colors.dark }]}>{value === 'auto' ? 'Auto' : 'Off'}</Text>
        </Pressable>)}</View>
        <Text style={ui.small}>Auto helps with camera shake. Try Off on a fixed tripod. Use good light and keep the ball, your full movement, and the goal visible.</Text></>}
      </View>
      {!permission?.granted ? <Action title={permission?.canAskAgain === false ? 'Open camera settings' : 'Enable camera'}
        onPress={() => { void (permission?.canAskAgain === false ? Linking.openSettings() : requestPermission()).catch(() => setError('Camera access could not be requested. Open Settings to try again.')); }} /> :
        phase === 'countdown' ? <Action title="Cancel countdown" secondary onPress={() => countdown.current?.cancel()} /> :
        phase === 'recording' ? <Action title="Stop & save rep" onPress={stop} /> :
          <Action title={busy ? 'Saving your rep…' : phase === 'saved' ? 'Record another rep' : `Record ${repSeconds}-second rep`}
            disabled={busy || store.loading || store.pending || !cameraReady || !foreground}
            onPress={() => void record()} />}
      {phase === 'countdown' ? <Text style={ui.body}>Recording starts after the countdown. Keep SCAN open.</Text> :
        phase === 'recording' ? <BallProgress progress={elapsed / repSeconds} label={`Rep timer · ${elapsed.toFixed(1)} / ${repSeconds} sec`} /> :
        busy ? <BallProgress label="Saving your rep on this phone…" /> : phase === 'saved' ?
          <BallProgress progress={1} label="Saved on this phone" /> :
          <Text style={ui.body}>{repSeconds}-second capture{delaySeconds ? ` after a ${delaySeconds}-second countdown` : ', starting immediately'}. Keep your phone steady and your full movement visible.</Text>}
      {(error || store.error) && <Text style={ui.error} accessibilityRole="alert">{error || store.error}</Text>}
      {error && quality !== '720p' && !busy && !store.pending && <Action title={`Try ${quality === '2160p' ? '1080p' : '720p'} recording`} secondary
        onPress={() => configureCamera(quality === '2160p' ? '1080p' : '720p', stabilization)} />}
      {store.pending && !busy && <Action title="Retry save" secondary onPress={() => void retry()} />}
      {!busy && <Action title="View saved reps" secondary onPress={() => router.push('/library')} />}
      <Text style={ui.small}>{busy ? 'Keep SCAN open until your rep is saved.' : 'Clips stay on this phone. No upload or analysis is running.'}</Text>
    </ScrollView>
  </SafeAreaView>;
}

function Choices({ values, selected, disabled, onSelect, label }: {
  values: number[]; selected: number; disabled: boolean; onSelect: (value: number) => void; label: string;
}) {
  return <View style={ui.row}>{values.map(value => <Pressable key={value} accessibilityRole="button"
    accessibilityLabel={`${label}: ${value ? `${value} seconds` : 'Off'}`}
    accessibilityState={{ selected: selected === value, disabled }} disabled={disabled}
    onPress={() => onSelect(value)} style={[s.choice, selected === value && s.selected, disabled && { opacity: .45 }]}>
    <Text style={[ui.heading, selected === value && { color: colors.dark }]}>{value ? `${value}s` : 'Off'}</Text>
  </Pressable>)}</View>;
}

const s = StyleSheet.create({
  choice: { flex: 1, alignItems: 'center', borderRadius: 12, paddingVertical: 12, backgroundColor: colors.raised },
  selected: { backgroundColor: colors.green },
  countdown: { position: 'absolute', top: 0, bottom: 0, left: 0, right: 0,
    alignItems: 'center', justifyContent: 'center', backgroundColor: '#070C0999' },
  countdownNumber: { color: colors.green, fontSize: 80, fontWeight: '800', fontVariant: ['tabular-nums'] },
  status: { flexDirection: 'row', gap: 7, alignItems: 'center' }, dot: { width: 6, height: 6, borderRadius: 3, backgroundColor: colors.green },
  cameraFrame: { width: '100%', aspectRatio: 4 / 3, borderRadius: 24, overflow: 'hidden', backgroundColor: colors.panel,
    borderWidth: 1, borderColor: colors.line },
  cameraPlaceholder: { flex: 1, padding: 30, alignItems: 'center', justifyContent: 'center', gap: 14 },
  topTag: { position: 'absolute', top: 14, left: 14, backgroundColor: '#070C09BB', borderRadius: 8, padding: 8 },
  tagText: { color: '#F2F7EF', fontSize: 10, fontWeight: '700', letterSpacing: 1 },
  corner: { position: 'absolute', width: 18, height: 18, borderColor: colors.green },
  topLeft: { top: 58, left: 16, borderTopWidth: 2, borderLeftWidth: 2 },
  topRight: { top: 58, right: 16, borderTopWidth: 2, borderRightWidth: 2 },
  bottomLeft: { bottom: 16, left: 16, borderBottomWidth: 2, borderLeftWidth: 2 },
  bottomRight: { bottom: 16, right: 16, borderBottomWidth: 2, borderRightWidth: 2 },
});
