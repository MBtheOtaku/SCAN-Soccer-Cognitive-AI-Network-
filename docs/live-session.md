# Live Session v0.1

Goal setup: [automatic goal proposals, confirmation and manual adjustment](goal-setup.md).

Optional post-rep opponent-pressure review and a Bluetooth speaker goal beacon are
documented in [the pressure/beacon prototype guide](opponent-pressure.md).

Upload-video analysis and Session Report remain available. Live Session uses the same
`POST /analyze` endpoint, with no streaming or on-device ML. Enable the camera, position
the phone with the player's full movement visible, then press START REP. Recording
lasts 4.5 seconds and uploads automatically. Keep the browser foregrounded and the
phone awake. READY permits another repetition; the last report stays visible.

## Laptop smoke test (PowerShell)

From the repository root, using the Python environment with the existing dependencies:

```powershell
python -m uvicorn src.backend.api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd src/frontend
npm run dev -- --hostname 0.0.0.0
```

Open `http://localhost:3000` on the laptop. Browser API requests use `/api/scan`,
forwarded by Next to `http://127.0.0.1:8000`. Check an existing MP4/MOV upload and its Session Report,
then open Live Session, enable the camera and record two repetitions.

## Phone setup: same-origin API and trusted HTTPS

The browser now uses `/api/scan` on the frontend's own origin for goal setup,
analysis and result videos. Next forwards requests to `http://127.0.0.1:8000`.
Old `NEXT_PUBLIC_API_BASE` values are ignored and can be removed. Backend CORS
origins do not need updating for this proxied workflow. Restart Next after updating.

Start the backend from the repository root, using the Python environment with
SCAN's dependencies (the backend no longer needs HTTPS certificates):

```powershell
python -m uvicorn src.backend.api.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in a separate terminal:

```powershell
cd src/frontend
npm run dev -- --hostname 0.0.0.0 --experimental-https --experimental-https-key "$env:USERPROFILE\scan-certs\scan-key.pem" --experimental-https-cert "$env:USERPROFILE\scan-certs\scan.pem"
```

Open `https://<laptop-address>:3000/api/scan/health` on the phone to verify the
complete connection, then open `https://<laptop-address>:3000` to use SCAN.
Allow frontend port 3000 through the appropriate Windows Firewall network profile.
The phone does not need access to backend port 8000.

### What this fixes, and what still needs testing

Changing networks no longer requires editing the browser's backend address or
backend CORS settings. A server-only `SCAN_BACKEND_ORIGIN` can override loopback
if the backend deliberately runs elsewhere. Use an HTTP loopback backend by
default; do not disable TLS verification to connect to an untrusted HTTPS backend.

The phone still needs a connection to the laptop and a trusted frontend HTTPS
address. Connecting the laptop to the iPhone's Personal Hotspot is a candidate
for pitch use, but test phone-to-laptop access before relying on it. No venue
Wi-Fi is required if that connection works. External AI services still require
internet access; existing deterministic fallbacks remain unchanged.

An IP-based frontend URL and certificate can still change across networks.
A stable laptop hostname included in the certificate may help only if it resolves
from the phone on that network; this change does not configure hostname discovery
or install certificates. On iPhone, the issuing root certificate must be installed
and fully trusted. Never share the CA private key.

If Next reports a development-origin warning, set `SCAN_DEV_HOSTS` to the exact
frontend hostname or IP (no scheme or port), then restart Next. This is a Next
development setting, separate from backend routing. Do not use a blanket wildcard.

Smoke-test goal preview, a normal upload, result-video seeking and a live rep on
the phone. The local HTTPS/hotspot path requires a hands-on device test.

## Behaviour and checks

- OPEN LIVE SESSION should change to CLOSE LIVE SESSION and reveal ENABLE CAMERA.
  Camera permission is requested only after ENABLE CAMERA. If OPEN never changes,
  check the frontend terminal for blocked dev asset requests, set `SCAN_DEV_HOSTS`
  to the hostname you open on the phone, restart Next, then reload Safari. For the
  LAN example: `$env:SCAN_DEV_HOSTS="192.168.1.50"` in PowerShell.

- Rear camera is preferred; recording uses video only and negotiates MP4/WebM.
  The backend now accepts WebM; decoding still uses its existing OpenCV installation.
  Unsupported codecs produce an analysis error, rather than a new CV pipeline.
- Speech follows the shared VOICE setting and uses `coaching_report.spoken_cue`.
  An explicit null/empty report cue preserves silence. Prototype feedback is used
  only when the report is absent. Some mobile browsers restrict asynchronous speech;
  REPLAY CUE provides a user-gesture fallback and the text remains visible.
- Camera denial, disconnection, unsupported recording, empty clips and API failure
  show retryable errors. Analysis requests time out after three minutes. Closing
  Live Session stops camera tracks and discards pending client work; aborting a
  request does not stop analysis already running on the backend.
- No rep history is sent between requests in v0.1; this retains existing endpoint
  semantics. There is no automatic rep detection or intervention during action.
- Test permission denial/retry, VOICE OFF, repeated reps, closing mid-recording,
  backend offline/retry, existing upload, and a phone MP4/WebM clip on your actual
  device. Codec support, certificate trust and mobile speech require hardware checks.

Frontend checks: `npm run lint`, `npx tsc --noEmit`, `npm run build`.
Backend checks: `python -m pytest tests` with `PYTHONPATH=src/backend` for the
existing controller tests.
