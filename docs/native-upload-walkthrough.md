# Native rep upload — file-by-file walkthrough

Prepared 3 November 2026 for the current Expo SDK 58 app and FastAPI API.
You will make the app edits below. Creating this guide does not connect the app
automatically. No new packages or backend changes are needed.

Finish line: open a saved rep, check the backend, tap Analyse rep, and read the
backend's written report. The local video is kept on success, failure and retry.
Budget: about 15 minutes for connection setup, 50 for code, and 40 for testing.

## 1. Verify the backend separately

Keep Tailscale connected on laptop and phone. In your backend terminal, using
the Python environment that already runs SCAN:

```powershell
cd 'C:\Users\Matthew Bui\OneDrive\SCAN'
python -m uvicorn src.backend.api.main:app --host 127.0.0.1 --port 8000
```

In another PowerShell window:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' serve status
```

Check that HTTPS port 443 forwards to `http://127.0.0.1:8000`. If that forward is
missing, restore it with:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' serve --bg --https=443 http://127.0.0.1:8000
```

On phone Safari, open:

```text
https://laptop-rbgvktra.tail744b42.ts.net/health
```

Expected: `{"status":"healthy","service":"SCAN API"}`. Fix this connection
before adding upload code. Backend HTTPS uses port 443 (omitted from the URL).
Expo's HTTPS preview uses port 8443. They are different services.

## 2. Create the environment file

Create `C:\Users\Matthew Bui\OneDrive\SCAN\src\mobile\.env.local`.
If it already exists, add this line without replacing its other contents:

```dotenv
EXPO_PUBLIC_SCAN_API_URL=https://laptop-rbgvktra.tail744b42.ts.net
```

Use the backend origin without `/health`, `/api/scan`, or the Expo preview port.
This setting is a public address bundled into the app. Never put API keys in an
`EXPO_PUBLIC_` variable. The mobile `.gitignore` already ignores `.env*.local`.
The address remains private to your Tailscale network; this is not public hosting.

## 3. Create the backend client

Create `C:\Users\Matthew Bui\OneDrive\SCAN\src\mobile\lib\backend.ts`:

```ts
import type {} from 'expo/types';
import * as FS from 'expo-file-system/legacy';
import { File } from 'expo-file-system';
import { fetch } from 'expo/fetch';
import type { Rep } from './reps';

export type ReportSection = { label: string; text: string };
export type AnalysisResult = {
  id: string;
  sections: ReportSection[];
  fallback: string | null;
};

function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
}

function text(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function origin() {
  const setting = process.env.EXPO_PUBLIC_SCAN_API_URL?.trim();
  if (!setting) throw new Error('Set EXPO_PUBLIC_SCAN_API_URL in .env.local and restart Metro.');
  const url = new URL(setting);
  if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash ||
      (url.pathname !== '/' && url.pathname !== '')) {
    throw new Error('Use the HTTPS backend origin without a path or credentials.');
  }
  return url.origin;
}

async function request(
  path: string, options: RequestInit, signal: AbortSignal, timeoutMs: number,
): Promise<Record<string, unknown>> {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal.addEventListener('abort', cancel, { once: true });
  if (signal.aborted) controller.abort();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const response = await fetch(origin() + path, {
      ...options, signal: controller.signal,
      headers: { Accept: 'application/json' },
    });
    const data = object(await response.json().catch(() => null));
    if (!response.ok) {
      const detail = text(data?.detail);
      throw new Error(detail ? detail.slice(0, 400) : `SCAN returned HTTP ${response.status}.`);
    }
    if (!data) throw new Error('Expected SCAN JSON. Check the backend address and port.');
    return data;
  } catch (error) {
    if (timedOut) throw new Error('Request timed out. Your clip is safe; the server may still finish processing it.');
    if (signal.aborted) throw new Error('Stopped waiting. Your clip is safe; the server may still finish processing it.');
    if (error instanceof TypeError && /network|fetch/i.test(error.message)) {
      throw new Error('Cannot reach SCAN. Check Tailscale, internet access and the laptop backend.');
    }
    throw error;
  } finally {
    clearTimeout(timer);
    signal.removeEventListener('abort', cancel);
  }
}

export async function checkBackend(signal: AbortSignal) {
  const data = await request('/health', {}, signal, 15_000);
  if (data.status !== 'healthy' || data.service !== 'SCAN API') {
    throw new Error('This address did not return the SCAN backend health response.');
  }
}

export async function analyseRep(rep: Rep, signal: AbortSignal): Promise<AnalysisResult> {
  const info = await FS.getInfoAsync(rep.uri);
  if (!info.exists || info.isDirectory || info.size <= 0) {
    throw new Error('This local recording is unavailable. Try playing it first.');
  }
  const form = new FormData();
  // SDK 58's Expo fetch needs a File/Blob, not a legacy { uri, name, type } object.
  form.append('video', new File(rep.uri), rep.name);
  form.append('goal_setup', 'null');
  // Let fetch supply the multipart boundary; do not set Content-Type yourself.
  const data = await request('/analyze', { method: 'POST', body: form }, signal, 600_000);
  const id = text(data.analysis_id);
  const summary = object(data.summary);
  if (!id || !summary || !object(data.artifacts)) {
    throw new Error('SCAN returned an incomplete analysis response. Your local clip is unchanged.');
  }
  const report = object(data.coaching_report);
  const fields = [
    ['overall_assessment', 'Assessment'],
    ['what_went_well', 'What went well'],
    ['primary_focus', 'Focus'],
    ['outcome_analysis', 'Outcome'],
    ['next_rep_recommendation', 'Next rep'],
    ['confidence_note', 'Confidence and limitations'],
  ] as const;
  const sections: ReportSection[] = [];
  for (const [key, label] of fields) {
    const value = text(report?.[key]);
    if (value) sections.push({ label, text: value });
  }
  return {
    id, sections,
    fallback: sections.length ? null : text(summary.prototype_feedback),
  };
}
```

What this file does: verifies the saved file, sends multipart field `video` to
the existing `/analyze` endpoint, validates the response, and selects only written
report fields actually returned by the backend. It never deletes the recording.
It does not use your notes as model evidence or generate new coaching language.

## 4. Create the upload/report panel

Create `C:\Users\Matthew Bui\OneDrive\SCAN\src\mobile\components\rep-analysis.tsx`:

```tsx
import { useEffect, useRef, useState } from 'react';
import { Text, View } from 'react-native';
import { Action, BallProgress } from './controls';
import { analyseRep, checkBackend, type AnalysisResult } from '../lib/backend';
import type { Rep } from '../lib/reps';
import { ui } from '../lib/theme';

export function RepAnalysis({ rep }: { rep: Rep }) {
  const [phase, setPhase] = useState<'idle' | 'checking' | 'analysing'>('idle');
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const pending = useRef<AbortController | null>(null);
  const mounted = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; pending.current?.abort(); };
  }, []);

  async function run(analyse: boolean) {
    if (pending.current) return;
    const controller = new AbortController();
    pending.current = controller;
    setError(null); setStatus(null); setPhase('checking');
    if (analyse) setResult(null);
    try {
      await checkBackend(controller.signal);
      if (controller.signal.aborted || !mounted.current) return;
      if (analyse) {
        setPhase('analysing');
        const response = await analyseRep(rep, controller.signal);
        if (!controller.signal.aborted && mounted.current) setResult(response);
      } else setStatus('Connected to SCAN.');
    } catch (e) {
      if (mounted.current) setError(e instanceof Error ? e.message : 'The request failed. Your clip is safe.');
    } finally {
      pending.current = null;
      if (mounted.current) setPhase('idle');
    }
  }

  const busy = phase !== 'idle';
  return <View style={ui.card}>
    <Text style={ui.badge}>BACKEND ANALYSIS</Text>
    <Text style={ui.body}>Analyse this saved video on your laptop. Keep SCAN open while waiting.</Text>
    <Text style={ui.small}>Goal calibration is not connected to native uploads yet, so shot outcome remains unknown. Opponent-pressure review is off for this first version.</Text>
    <Action title="Check backend connection" secondary disabled={busy}
      onPress={() => { void run(false); }} />
    <Action title={result ? 'Analyse again' : error ? 'Retry analysis' : 'Analyse rep'} disabled={busy}
      onPress={() => { void run(true); }} />
    {busy && <>
      <BallProgress label={phase === 'checking' ? 'Checking backend…' : 'Sending video and analysing…'} />
      <Action title="Stop waiting" secondary onPress={() => pending.current?.abort()} />
    </>}
    {status && <Text accessibilityLiveRegion="polite" style={ui.body}>{status}</Text>}
    {error && <Text accessibilityRole="alert" style={ui.error}>{error}</Text>}
    {result && <View style={{ gap: 16 }}>
      <Text style={ui.heading}>Session report</Text>
      <Text style={ui.small}>Analysis ID: {result.id}</Text>
      {result.sections.map(section => <View key={section.label} style={{ gap: 6 }}>
        <Text style={ui.heading}>{section.label}</Text>
        <Text style={ui.body}>{section.text}</Text>
      </View>)}
      {!result.sections.length && <Text style={ui.body}>{result.fallback || 'Analysis completed, but no written coaching report was returned.'}</Text>}
    </View>}
    <Text style={ui.small}>Your original stays on this phone. Stop waiting or leaving this screen does not stop a job already running on the server.</Text>
  </View>;
}
```

This panel makes no request until you tap a button. Analysis first checks the
backend, then uploads. It prevents double taps and displays an indeterminate
indicator because the current API does not report processing percentages.
It displays written output only; it does not add spoken cues or change policy.

## 5. Connect the panel to playback

Edit `C:\Users\Matthew Bui\OneDrive\SCAN\src\mobile\src\app\rep\[name].tsx`.
The square brackets are literally part of the filename.

Add this import with the other component imports:

```tsx
import { RepAnalysis } from '../../../components/rep-analysis';
```

Find this existing line:

```tsx
<RepExport key={`export-${rep.name}`} rep={rep} />
```

Directly beneath it, add:

```tsx
<RepAnalysis key={`analysis-${rep.name}`} rep={rep} />
```

In the existing SAVED ON THIS PHONE card, replace this sentence:

```tsx
<Text style={ui.body}>Review your movement at your own pace. Coaching analysis is not connected yet.</Text>
```

with:

```tsx
<Text style={ui.body}>Review your movement at your own pace, or use backend analysis above.</Text>
```

Keep the existing player, title editor, export control and navigation. You do
not need to edit `_layout.tsx`, `App.tsx`, `reps.ts`, or the recording screen.

## 6. Check your edits

Stop Metro with Ctrl+C if it is running. In the mobile terminal:

```powershell
cd 'C:\Users\Matthew Bui\OneDrive\SCAN\src\mobile'
npx tsc --noEmit
npx expo lint
```

Fix any reported errors before restarting. These checks need no package installs.

## 7. Restart the preview

Use the same mobile terminal:

```powershell
$env:EXPO_PACKAGER_PROXY_URL = 'https://laptop-rbgvktra.tail744b42.ts.net:8443'
npm run start:stable -- --localhost
```

That variable configures Expo, while `.env.local` configures SCAN's backend URL.
They intentionally use different ports. Fully close Expo Go and reopen:

```text
exps://laptop-rbgvktra.tail744b42.ts.net:8443
```

Make sure private HTTPS 8443 still forwards to `http://localhost:8081` in
`tailscale serve status`. If missing, recreate it:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' serve --bg --https=8443 http://localhost:8081
```

## 8. Test one complete loop

1. Open Saved Reps and replay a clip to confirm it exists.
2. Tap Check backend connection. Expect Connected to SCAN.
3. Tap Analyse rep once. Wait for the report. Processing can take minutes.
4. Check the backend terminal for `POST /analyze` and its response status.
5. Compare the report with what is visible in the video. A non-training clip may
   produce limited evidence or an analysis error; that is not an upload failure.
6. Stop the backend with Ctrl+C. Tap Analyse again and confirm a readable error,
   then replay your original clip to confirm it is preserved.
7. Restart the backend and explicitly retry. There is no automatic retry loop.
8. Try Stop waiting, then verify the clip remains playable. The server can still
   finish that request; a retry can start an additional analysis job.
9. Finally repeat the connection/upload check with phone Wi-Fi off and mobile
   data/Tailscale on. The laptop must remain awake and connected.

### Scope of this first version

- Reports live in this playback panel's memory. Closing/reopening it loses the
  displayed report; it does not lose the local video. Persisting reports is a
  separate follow-up, not a reason to upload repeatedly by default.
- Keep the app foregrounded. There is no background upload queue or resumable job.
- Requests time out after 15 seconds for health and 10 minutes for analysis.
  Client cancellation/timeout does not cancel server computation.
- No native goal calibration, pressure-side selector, annotated-video playback,
  automatic audio, or hosted server is added in this milestone.
- The private Tailscale connection is the current access control. This does not
  make the API ready for anonymous public deployment.

## Documentation used

- https://docs.expo.dev/versions/v58.0.0/sdk/filesystem-legacy/
- https://docs.expo.dev/versions/v58.0.0/sdk/filesystem/#uploading-files-using-expofetch
- https://reactnative.dev/docs/network
- Existing `src/backend/api/main.py` and the website's report rendering.
