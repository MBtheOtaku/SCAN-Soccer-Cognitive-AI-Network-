# Live Session v0.1

Upload-video analysis and Session Report remain available. Live Session uses the same
`POST /analyze` endpoint, with no streaming or on-device ML. Enable the camera, position
the phone with the player's full movement visible, then press START REP. Recording
lasts 4.5 seconds and uploads automatically. Keep the browser foregrounded and the
phone awake. READY permits another repetition; the last report stays visible.

## Laptop smoke test (PowerShell)

From the repository root, using the Python environment with the existing dependencies:

```powershell
python -m uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000
```

In another terminal:

```powershell
cd src/frontend
npm run dev -- --hostname 0.0.0.0
```

Open `http://localhost:3000` on the laptop. The API defaults to
`http://127.0.0.1:8000`. Check an existing MP4/MOV upload and its Session Report,
then open Live Session, enable the camera and record two repetitions.

## Phone test on the same Wi-Fi: trusted HTTPS

Plain `http://<laptop-ip>:3000` does **not** provide a secure camera context.
HTTPS on the frontend also requires HTTPS on the API to avoid mixed-content blocking.
Use a locally trusted certificate setup (for example mkcert), or trusted HTTPS
tunnels to both servers. This feature does not set up certificates or tunnels.

For a local certificate setup:

1. Find the laptop's Wi-Fi IPv4 address with `ipconfig` (example `192.168.1.50`).
2. Install mkcert, run `mkcert -install`, then generate a certificate outside the
   repo for that IP and localhost: `mkcert 192.168.1.50 localhost 127.0.0.1`.
   Use your actual address. Install and trust mkcert's **root certificate** on the
   phone as well; on iOS enable full trust in Certificate Trust Settings. Never
   copy the CA private key to the phone or commit certificate keys.
3. In `src/frontend/.env.local`, set:

   ```dotenv
   NEXT_PUBLIC_API_BASE=https://192.168.1.50:8000
   ```

   This is the API URL only. Never prefix OpenAI or TypeSafe secrets with
   `NEXT_PUBLIC_`. Restart Next after changing this value; production builds embed it.
   Next's development JavaScript allowlist automatically includes this API hostname
   for the same-laptop setup. If the frontend uses a different hostname (for example
   a separate HTTPS tunnel), set `SCAN_DEV_HOSTS` to that frontend hostname before
   starting Next. This accepts comma-separated hostnames, without schemes or ports.
4. Start the backend from the repository root, substituting your IP and absolute
   certificate/key paths:

   ```powershell
   $env:SCAN_CORS_ORIGINS="https://192.168.1.50:3000"
   python -m uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile C:/certs/scan-key.pem --ssl-certfile C:/certs/scan.pem
   ```

5. Start Next from `src/frontend` with the same trusted certificate:

   ```powershell
   npm run dev -- --hostname 0.0.0.0 --experimental-https --experimental-https-key C:/certs/scan-key.pem --experimental-https-cert C:/certs/scan.pem
   ```

6. Allow inbound TCP 3000 and 8000 for the private Wi-Fi network in Windows
   Firewall if needed. On the phone open `https://192.168.1.50:8000/health` first,
   verify no certificate warning, then open `https://192.168.1.50:3000`.
   Enable camera access and record two reps without refreshing.

The bare `--experimental-https` flag generates a localhost certificate; it alone
does not establish trust on a phone or cover the laptop's LAN IP. With HTTPS tunnels,
set `NEXT_PUBLIC_API_BASE` to the API tunnel URL and `SCAN_CORS_ORIGINS` to the exact
frontend tunnel origin. CORS accepts comma-separated additional origins, without a
trailing slash, and keeps the existing localhost defaults. There is no wildcard.
The backend has no authentication: use a trusted development network; public tunnels
expose the analysis service and generated outputs unless access controls are added.

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
