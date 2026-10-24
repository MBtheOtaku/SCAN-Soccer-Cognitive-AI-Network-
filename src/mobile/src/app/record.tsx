import { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, AppState, BackHandler, Linking, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { router, Stack, useFocusEffect } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useKeepAwake } from 'expo-keep-awake';
import { Action, BallProgress } from '../../components/controls';
import { useReps } from '../../lib/rep-store';
import { colors, ui } from '../../lib/theme';

const REP_SECONDS = 5;
type Phase = 'ready' | 'recording' | 'saving' | 'saved';

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
  const [elapsed, setElapsed] = useState(0);
  const [phase, setPhase] = useState<Phase>('ready');
  const [cameraReady, setCameraReady] = useState(false);
  const [foreground, setForeground] = useState(AppState.currentState === 'active');
  const [error, setError] = useState<string | null>(null);
  const [focused, setFocused] = useState(false);
  const store = useReps();
  const busy = phase === 'recording' || phase === 'saving' || store.saving;

  useFocusEffect(useCallback(() => {
    setFocused(true);
    return () => { setFocused(false); setCameraReady(false); };
  }, []));
  useEffect(() => {
    if (!focused || !busy) return;
    const listener = BackHandler.addEventListener('hardwareBackPress', () => {
      Alert.alert('Keep this rep open', 'Stop the recording and wait for it to finish saving before leaving.');
      return true;
    });
    return () => listener.remove();
  }, [focused, busy]);
  useEffect(() => {
    const listener = AppState.addEventListener('change', state => {
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
    const timer = setInterval(() => setElapsed(Math.min(REP_SECONDS, (Date.now() - began.current) / 1000)), 100);
    return () => clearInterval(timer);
  }, [phase]);

  async function record() {
    if (lock.current || busy || store.pending || store.loading || !cameraReady || !camera.current || !foreground || !focused) return;
    lock.current = true; recording.current = true; began.current = Date.now();
    setElapsed(0); setPhase('recording'); setError(null);
    try {
      const clip = await camera.current.recordAsync({ maxDuration: REP_SECONDS });
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
  const status = phase === 'recording' ? 'RECORDING' : phase === 'saving' || store.saving ? 'SAVING' :
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
        {permission?.granted && foreground && focused ? <CameraView ref={camera} style={StyleSheet.absoluteFill}
          facing="back" mode="video" mute videoQuality="720p" onCameraReady={() => setCameraReady(true)}
          onMountError={e => { setCameraReady(false); setError(e.message); }} /> : <View style={s.cameraPlaceholder}>
          <Text style={ui.heading}>{permission?.granted ? 'Camera paused' : 'Your pitch. Your perspective.'}</Text>
          <Text style={[ui.body, { textAlign: 'center' }]}>Position your phone so your full movement stays in frame.</Text>
        </View>}
        <View pointerEvents="none" style={s.topTag}><Text style={s.tagText}>5 SEC · VIDEO ONLY</Text></View>
        <View pointerEvents="none" style={[s.corner, s.topLeft]} /><View pointerEvents="none" style={[s.corner, s.topRight]} />
        <View pointerEvents="none" style={[s.corner, s.bottomLeft]} /><View pointerEvents="none" style={[s.corner, s.bottomRight]} />
      </View>
      {!permission?.granted ? <Action title={permission?.canAskAgain === false ? 'Open camera settings' : 'Enable camera'}
        onPress={() => { void (permission?.canAskAgain === false ? Linking.openSettings() : requestPermission()).catch(() => setError('Camera access could not be requested. Open Settings to try again.')); }} /> :
        phase === 'recording' ? <Action title="Stop & save rep" onPress={stop} /> :
          <Action title={busy ? 'Saving your rep…' : phase === 'saved' ? 'Record another rep' : 'Record 5-second rep'}
            disabled={busy || store.loading || store.pending || !cameraReady || !foreground}
            onPress={() => void record()} />}
      {phase === 'recording' ? <BallProgress progress={elapsed / REP_SECONDS} label={`Rep timer · ${elapsed.toFixed(1)} / ${REP_SECONDS} sec`} /> :
        busy ? <BallProgress label="Saving your rep on this phone…" /> : phase === 'saved' ?
          <BallProgress progress={1} label="Saved on this phone" /> :
          <Text style={ui.body}>A five-second capture. Keep your phone steady and your full movement visible.</Text>}
      {(error || store.error) && <Text style={ui.error} accessibilityRole="alert">{error || store.error}</Text>}
      {store.pending && !busy && <Action title="Retry save" secondary onPress={() => void retry()} />}
      {!busy && <Action title="View saved reps" secondary onPress={() => router.push('/library')} />}
      <Text style={ui.small}>{busy ? 'Keep SCAN open until your rep is saved.' : 'Clips stay on this phone. No upload or analysis is running.'}</Text>
    </ScrollView>
  </SafeAreaView>;
}

const s = StyleSheet.create({
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
