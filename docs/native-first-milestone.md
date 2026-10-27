# SCAN native milestone 1 — implement it yourself

Prepared 19 October 2026. This is a code walkthrough, not an installed or device-tested app.
Historical walkthrough: the existing mobile project was upgraded to SDK 58 on
26 October 2026 to match the phone's Expo Go. Do not rerun this SDK 57 scaffold
over the existing app. Use the current checked-in package.json/package-lock.json
and [current UI/testing guide](native-ui.md) for continued development.
The existing Next.js/FastAPI app is unchanged. Create a separate Expo React Native client.

Goal: open the app, record up to five seconds without internet, save a completed rep,
reopen the app and play it from the saved list. No uploads or analysis in this milestone.
Use a physical iPhone for camera testing.

## 1. Create the mobile project

In PowerShell, from SCAN's root:

```powershell
cd "C:\Users\Matthew Bui\OneDrive\SCAN"
npx create-expo-app@latest src/mobile --template blank-typescript@sdk-57
cd src/mobile
npx expo install expo-camera expo-file-system expo-video react-native-safe-area-context expo-keep-awake
New-Item -ItemType Directory -Force lib | Out-Null
Copy-Item ../../docs/assets/scan-native-icon-v1.png assets/scan-icon.png
```

SDK 57 is pinned for physical-iPhone Expo Go compatibility during the SDK 58
transition (verified 19 October 2026). Do not use the unversioned blank template
for this walkthrough. Use `expo install` so dependency versions match the scaffold's Expo SDK. Keep its
generated package.json, index.ts and tsconfig.json. If Expo Go reports an unsupported
SDK, stop and match the project to the SDK supported by your installed Expo Go; do
not independently downgrade React Native. Expo's SDK transition guidance changes.

File layout after you paste the following code:

```text
src/mobile/
  App.tsx          replace the starter screen
  lib/reps.ts      create: local storage
  app.json        edit app identity and permissions
  eas.json        create later for standalone installation
  assets/scan-icon.png
```

## 2. Create src/mobile/lib/reps.ts

Camera recordings initially live in cache. This module copies them into the app's
Documents directory, verifies byte size and only then exposes them as saved reps.
An on-disk job lets RETRY SAVE or a later launch retry an interrupted copy. Paths
inside cache are stored relative to the current app container.

```ts
import * as FS from 'expo-file-system/legacy';

export type Rep = { name: string; uri: string };
type Job = { name: string; cachePath: string; bytes: number };

function root() {
  if (!FS.documentDirectory) throw new Error('Local storage is unavailable.');
  return FS.documentDirectory + 'scan-reps/';
}
async function prepare() {
  await FS.makeDirectoryAsync(root(), { intermediates: true });
}
async function size(uri: string) {
  const info = await FS.getInfoAsync(uri);
  return info.exists && !info.isDirectory ? info.size : 0;
}
export async function hasPendingSave() {
  await prepare();
  return (await FS.getInfoAsync(root() + 'pending.json')).exists;
}
export async function recoverSave() {
  await prepare();
  const journal = root() + 'pending.json';
  if (!(await FS.getInfoAsync(journal)).exists) return;
  const job: Job = JSON.parse(await FS.readAsStringAsync(journal));
  if (!/^rep-\d+-[a-z0-9]+\.(mov|mp4)$/.test(job.name) ||
      typeof job.cachePath !== 'string' || job.cachePath.includes('..') ||
      !Number.isFinite(job.bytes) || job.bytes <= 0 || !FS.cacheDirectory) {
    throw new Error('The pending recording needs storage recovery. Keep the app installed.');
  }
  const target = root() + job.name;
  const staging = target + '.partial';
  if (await size(target) !== job.bytes) {
    if (await size(staging) !== job.bytes) {
      const source = FS.cacheDirectory + job.cachePath;
      if (await size(source) !== job.bytes) {
        throw new Error('The unfinished recording is unavailable. Previously saved reps are still kept.');
      }
      await FS.deleteAsync(staging, { idempotent: true });
      await FS.copyAsync({ from: source, to: staging });
      if (await size(staging) !== job.bytes) throw new Error('Save incomplete. Free storage, then retry.');
    }
    await FS.moveAsync({ from: staging, to: target });
  }
  await FS.deleteAsync(journal, { idempotent: true });
  // Delete only the temporary source after the verified final file exists.
  await FS.deleteAsync(FS.cacheDirectory + job.cachePath, { idempotent: true }).catch(() => {});
}
export async function saveRep(uri: string) {
  await prepare();
  if (await hasPendingSave()) throw new Error('Retry the previous save first.');
  const bytes = await size(uri);
  if (!bytes || !FS.cacheDirectory || !uri.startsWith(FS.cacheDirectory)) {
    throw new Error('The camera did not return a usable local recording.');
  }
  const ext = uri.toLowerCase().endsWith('.mov') ? 'mov' : 'mp4';
  const job: Job = {
    name: `rep-${Date.now()}-${Math.random().toString(36).slice(2, 10)}.${ext}`,
    cachePath: uri.slice(FS.cacheDirectory.length), bytes,
  };
  const temporaryJournal = root() + 'pending.tmp';
  await FS.writeAsStringAsync(temporaryJournal, JSON.stringify(job));
  await FS.moveAsync({ from: temporaryJournal, to: root() + 'pending.json' });
  await recoverSave();
}
export async function listReps(): Promise<Rep[]> {
  await prepare();
  const names = (await FS.readDirectoryAsync(root()))
    .filter(name => /^rep-\d+-[a-z0-9]+\.(mov|mp4)$/.test(name)).sort().reverse();
  return names.map(name => ({ name, uri: root() + name }));
}
```

The explicit `/legacy` import uses Expo's documented asynchronous file API. It
avoids loading videos into JavaScript memory. This is a starter storage layer;
do not claim protection against a force-kill during recording or before the save
journal is written. Cache can be purged; only the final Documents copy is saved.

## 3. Replace src/mobile/App.tsx

```tsx
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
```

## 4. Edit app.json

Keep the scaffold's other fields. Inside its `expo` object, set these fields. Merge
the plugin into an existing plugins list if one is present; do not create a second
`expo` object. The bundle identifier below is a proposed unique identifier; Apple
will validate availability when you build.

```json
  

This is a portrait prototype. Landscape pitch framing, exports and deletion can be
follow-up tasks. iOS icon processing should be verified during the native build;
the supplied source PNG is square, opaque and high resolution.

## 5. Open it today for development

Install Expo Go from Apple's App Store. Create a free Expo account and use the same
account on the phone and CLI if required by your Expo Go version.

From `src/mobile`:

```powershell
npx tsc --noEmit
npx expo-doctor
npx expo login
npx expo start --go
```

Scan the QR code using your iPhone camera. This opens native React Native screens
inside Expo Go. Initially the development server must be reachable. If LAN access
is blocked, `npx expo start --go --tunnel` may help; accept its tunnel dependency
only if you want that temporary development service. A tunnel needs internet.

Expo Go is an early test host, with its own icon and storage. It is not the
standalone SCAN installation and must not be used to claim laptop-free cold launch.
Saved reps in Expo Go will not automatically migrate to your standalone SCAN app.

## 6. Install a private standalone app (after Apple membership)

For the Windows/EAS path, iOS ad hoc distribution requires paid Apple Developer
membership. Apple's listed fee is US$99/year, with local currency where available.
Membership purchase and signing login are steps you perform yourself. A Mac/Xcode
personal-team route has different limitations and is not this Windows walkthrough.

Create `src/mobile/eas.json`:

```json
{
  "build": {
    "preview": {
      "distribution": "internal",
      "developmentClient": false,
      "ios": { "simulator": false }
    }
  }
}
```

Run from `src/mobile` only when ready to create an Expo cloud project and upload
the source for a build. Review what files EAS will include: keep API keys, .env,
certificates, personal videos and large backend model files out of the archive.
Do not put backend secrets in app configuration or EXPO_PUBLIC variables.

```powershell
npx eas-cli@latest login
npx eas-cli@latest build:configure
npx eas-cli@latest device:create
npx eas-cli@latest build --platform ios --profile preview
```

Register/select only your iPhone when asked about devices. The preview build embeds
JavaScript and assets; it does not require Metro or your laptop to launch. Follow
the installation link on your registered iPhone. The icon is SCAN's own icon.
In Expo project settings, disable unauthenticated access to internal build links.
Only provisioned iPhones can install this ad hoc build; this is not backend login
security. Provisioning expires and may require renewal/rebuilding. Build queue,
account limits and Apple device registration can prevent same-day installation.

There is no cloud analysis in milestone 1 and no video upload code. Device backup
settings may still back up app documents. Uninstalling the app deletes its local
data; reinstalling is not a persistence test.

## 7. Acceptance test on the installed preview build

1. Launch SCAN and grant camera permission. Record a rep and wait for Saved.
2. Play it from the saved list, then close playback and record another rep.
3. Force-close only after Saved, reopen, and play both reps.
4. Shut down the laptop. Enable airplane mode, explicitly turn Wi-Fi off, and
   cold-launch SCAN from its own icon. Record, save, reopen and play a third rep.
5. Deny camera permission, then allow it through Settings; saved playback should
   remain available. Test STOP AND SAVE as well as the five-second automatic stop.
6. Background or lock the phone mid-recording. A shortened clip may be returned,
   or the OS may interrupt recording. The app must not claim a missing clip is saved.
7. A save error must show RETRY SAVE, not success. Previously saved reps must remain.

Byte-size validation checks copy completion, not video decodability: playback on
your actual iPhone is part of acceptance. An unexpected kill before Saved can lose
an in-progress recording; this milestone does not promise crash-proof capture.

## Sources and icon

- [Expo project setup](https://docs.expo.dev/get-started/create-a-project/)
- [Expo Camera: recordings start in cache](https://docs.expo.dev/versions/latest/sdk/camera/)
- [Expo FileSystem legacy: cache and Documents semantics](https://docs.expo.dev/versions/latest/sdk/filesystem-legacy/)
- [Expo Video](https://docs.expo.dev/versions/latest/sdk/video/)
- [Expo internal distribution](https://docs.expo.dev/build/internal-distribution/)
- [Apple enrollment](https://developer.apple.com/help/account/membership/program-enrollment/)

Icon asset: `docs/assets/scan-native-icon-v1.png`, generated with the built-in image tool.
Prompt: “Use case: logo-brand. Create a polished square 1024x1024 opaque iPhone app icon for SCAN, Soccer Cognitive AI Network. Near-black forest green background #050806, bright mint green #77ff92 with subtle off-white accents, matching a modern soccer training app. One bold circular emblem centred with generous 16% safe margins. The circle is divided precisely vertically down the middle: LEFT HALF a clearly recognizable soccer ball with bold pentagon/hexagon panel geometry; RIGHT HALF the front-facing right half of a friendly, composed AI soccer coach's head, geometric human-like face with a single eye and subtle circuit traces at the temple. The two halves join into ONE cohesive circular symbol. Athletic, intelligent, calm, confident. Clean flat graphic logo with crisp thick shapes, limited detail legible at small app-icon sizes. No text, no letters, no watermark, no mockup, no phone frame. Fill the square background edge to edge, no pre-rounded corners. Not a menacing robot, no glowing lens flares.”
