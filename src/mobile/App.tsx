import { useEffect, useRef, useState } from 'react';
import { AppState, Button, Image, Linking, Modal, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useVideoPlayer, VideoView } from 'expo-video';
import { useKeepAwake } from 'expo-keep-awake';
import { hasPendingSave, listReps, recoverSave, saveRep, type Rep } from './lib/reps';

function Playback({ rep, close }: { rep: Rep; close: () => void }) {
  const player = useVideoPlayer(rep.uri);
  return <Modal onRequestClose={close}>
    <SafeAreaView style={s.screen}>
      <Text style={s.heading}>SAVED REP</Text>
      <VideoView player={player} style={s.camera} nativeControls contentFit="contain" />
      <Button title="CLOSE PLAYBACK" onPress={close} color="#288a43" />
    </SafeAreaView>
  </Modal>;
}

function Recorder() {
  useKeepAwake();
  const [permission, requestPermission] = useCameraPermissions();
  const camera = useRef<CameraView>(null);
  const busy = useRef(false);
  const recording = useRef(false);
  const unsaved = useRef<string | null>(null);
  const [phase, setPhase] = useState<'loading' | 'ready' | 'recording' | 'saving'>('loading');
  const [cameraReady, setCameraReady] = useState(false);
  const [foreground, setForeground] = useState(AppState.currentState === 'active');
  const [message, setMessage] = useState('Loading saved reps…');
  const [pending, setPending] = useState(false);
  const [reps, setReps] = useState<Rep[]>([]);
  const [selected, setSelected] = useState<Rep | null>(null);
  const errorText = (e: unknown) => e instanceof Error ? e.message : 'Could not complete the recording.';

  useEffect(() => {
    let mounted = true;
    void (async () => {
      try { await recoverSave(); }
      catch (e) { if (mounted) setMessage(errorText(e)); }
      try {
        const saved = await listReps();
        const waiting = await hasPendingSave();
        if (mounted) {
          setReps(saved); setPending(waiting);
          if (!waiting) setMessage('Ready. Recordings stay on this phone.');
        }
      } catch (e) { if (mounted) { setMessage(errorText(e)); setPending(true); } }
      finally { if (mounted) setPhase('ready'); }
    })();
    const subscription = AppState.addEventListener('change', state => {
      setForeground(state === 'active'); setCameraReady(false);
      if (state !== 'active' && recording.current) {
        try { camera.current?.stopRecording(); } catch { /* Native interruption may already have stopped it. */ }
      }
    });
    return () => { mounted = false; subscription.remove(); };
  }, []);

  async function refresh() {
    setReps(await listReps());
    setPending(await hasPendingSave());
  }
  async function retry() {
    if (busy.current) return;
    busy.current = true; setPhase('saving');
    try {
      if (await hasPendingSave()) await recoverSave();
      else if (unsaved.current) await saveRep(unsaved.current);
      unsaved.current = null;
      await refresh(); setMessage('Saved on this phone.');
    } catch (e) { setMessage(errorText(e)); setPending(true); }
    finally { busy.current = false; setPhase('ready'); }
  }
  async function record() {
    if (busy.current || pending || !cameraReady || !camera.current) return;
    busy.current = true; recording.current = true; setPhase('recording');
    setMessage('Recording up to 5 seconds…');
    try {
      const clip = await camera.current.recordAsync({ maxDuration: 5 });
      recording.current = false; setPhase('saving');
      if (!clip?.uri) throw new Error('Recording was interrupted before a clip was returned.');
      unsaved.current = clip.uri;
      setMessage('Saving on this phone…');
      await saveRep(clip.uri);
      unsaved.current = null;
      await refresh(); setMessage('Saved on this phone. Tap the rep below to play it.');
    } catch (e) {
      setMessage(errorText(e));
      setPending(Boolean(unsaved.current));
    } finally { recording.current = false; busy.current = false; setPhase('ready'); }
  }
  function stopRecording() {
    try { camera.current?.stopRecording(); }
    catch (e) { setMessage(errorText(e)); }
  }

  return <SafeAreaView style={s.screen}>
    <ScrollView contentContainerStyle={s.content}>
      <Image source={require('./assets/scan-icon.png')} style={s.logo} />
      <Text style={s.heading}>SCAN · LOCAL REPS</Text>
      <Text style={s.text}>Record here. Review anytime. Analysis will come later.</Text>
      {!permission?.granted ? <>
        <Text style={s.text}>Allow camera access to record a rep. Microphone audio is off.</Text>
        <Button title={permission?.canAskAgain === false ? 'OPEN SETTINGS' : 'ALLOW CAMERA'}
          onPress={() => { void (permission?.canAskAgain === false ? Linking.openSettings() : requestPermission()); }} />
      </> : foreground && !selected ?
        <CameraView ref={camera} style={s.camera} facing="back" mode="video" mute videoQuality="720p"
          onCameraReady={() => setCameraReady(true)}
          onMountError={e => { setCameraReady(false); setMessage(e.message); }} /> : null}
      <Text style={s.text} accessibilityLiveRegion="polite">{message}</Text>
      {phase === 'recording' ? <Button title="STOP AND SAVE" onPress={stopRecording} color="#288a43" /> :
        <Button title={phase === 'saving' ? 'SAVING…' : 'RECORD 5-SECOND REP'}
          disabled={phase !== 'ready' || !cameraReady || !foreground || pending}
          onPress={() => void record()} color="#288a43" />}
      {pending && <Button title="RETRY SAVE" disabled={phase !== 'ready'} onPress={() => void retry()} />}
      <Text style={s.heading}>SAVED ON THIS PHONE · {reps.length}</Text>
      {reps.map(rep => <View key={rep.name} style={s.row}>
        <Button title={new Date(Number(rep.name.split('-')[1])).toLocaleString()}
          disabled={phase !== 'ready'} color="#288a43"
          onPress={() => { setCameraReady(false); setSelected(rep); }} />
      </View>)}
      <Text style={s.text}>Wait for “Saved on this phone” before closing. Uninstalling the app removes these recordings.</Text>
    </ScrollView>
    {selected && <Playback rep={selected} close={() => setSelected(null)} />}
  </SafeAreaView>;
}

export default function App() {
  return <SafeAreaProvider><Recorder /></SafeAreaProvider>;
}
const s = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#050806' },
  content: { padding: 20, gap: 18 },
  heading: { color: '#77ff92', fontSize: 20, fontWeight: '800', marginTop: 12 },
  text: { color: '#f2f7f3', fontSize: 15, lineHeight: 22 },
  logo: { width: 76, height: 76, borderRadius: 18 },
  camera: { width: '100%', height: 360, borderRadius: 16, overflow: 'hidden' },
  row: { borderWidth: 1, borderColor: '#173d21', borderRadius: 8, padding: 8 },
});