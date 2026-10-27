# SCAN from the pitch, with the laptop at home

This setup uses a private Tailscale connection between your own Windows laptop
and iPhone. The laptop stays awake, plugged in and online at home. The phone uses
mobile data at the pitch. Home Wi-Fi IP changes no longer determine these URLs.

There are two separate services:

- Expo/Metro opens the current native preview. The native app currently records
  and replays local reps; it does not submit analysis requests yet.
- FastAPI runs SCAN analysis. Giving it a remote address prepares it for native
  integration; it does not add analysis buttons to the native app automatically.

## 1. Connect both devices

Install Windows Tailscale from https://tailscale.com/download and its iPhone app
from the App Store. Sign in to the same personal account on both devices. Enable
the iPhone VPN connection when prompted. Keep this network limited to your devices.

Open a new PowerShell window after installing. These commands use the default
Windows installation path; if that path is missing, finish the installer first.

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' status
& 'C:\Program Files\Tailscale\tailscale.exe' ip -4
```

The second command returns the laptop's private Tailscale address, typically
starting with `100.`. This differs from its home Wi-Fi address.

## 2. Make the backend privately reachable

In the existing backend terminal, from SCAN's root, with the working Python
environment already active:

```powershell
Set-Location 'C:\Users\Matthew Bui\OneDrive\SCAN'
python -m uvicorn src.backend.api.main:app --host 127.0.0.1 --port 8000
```

Leave it running. In a separate PowerShell window (Administrator if Tailscale
requests it):

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/health'
& 'C:\Program Files\Tailscale\tailscale.exe' serve --bg --https=443 http://127.0.0.1:8000
& 'C:\Program Files\Tailscale\tailscale.exe' serve status
```

If Serve prints an HTTPS permission link, open it, enable HTTPS for your network
and rerun the command. Use **Serve** for private access. Do not enable public
Funnel sharing. Keep FastAPI bound to `127.0.0.1`; no router port forwarding or
home-IP certificate regeneration is needed.

Copy the exact `https://...ts.net` address printed by Serve. On the phone, turn
Wi-Fi off, leave mobile data and Tailscale on, and visit that address with `/health`
added. Expected JSON includes `"status":"healthy"` and `"service":"SCAN API"`.

This is the first acceptance check. A successful laptop-only check does not prove
remote phone access.

## 3. Open the existing Expo preview from mobile data

Stop the current Metro terminal with Ctrl+C before restarting. In a separate
PowerShell window, forward a second private port for Metro:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' serve --bg --http=8082 http://localhost:8081
& 'C:\Program Files\Tailscale\tailscale.exe' serve status
```

Backend HTTPS port 443 and Metro port 8082 are separate. HTTP on the private
Tailscale network travels inside its encrypted connection.
Use `localhost` for this Metro target: on this Windows setup, Expo's `--localhost`
listener uses IPv6 `::1`. Forwarding to `127.0.0.1:8081` instead produces a 502.

In the mobile terminal:

```powershell
Set-Location 'C:\Users\Matthew Bui\OneDrive\SCAN\src\mobile'
$scanTailStatus = & 'C:\Program Files\Tailscale\tailscale.exe' status --json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $scanTailStatus.BackendState -ne 'Running' -or !$scanTailStatus.Self.DNSName) { throw 'Connect laptop Tailscale first.' }
$scanTailHost = $scanTailStatus.Self.DNSName.TrimEnd('.')
$env:EXPO_PACKAGER_PROXY_URL = "http://${scanTailHost}:8082"
npm run start:stable -- --localhost
```

Wait for Metro to be ready. It now prints the remote `exp://<laptop>.<tailnet>.ts.net:8082` address.
Use the hostname shown by Serve, not its `100.x` address: the HTTP Serve route is
matched by hostname. The IP URL can return 404 even when Tailscale is connected.
On the iPhone, still on mobile data with Tailscale connected, open that exact URL
in Safari and accept Open in Expo Go. Use the same Expo account on CLI and phone
if Expo Go requests sign-in.

First test `http://<laptop>.<tailnet>.ts.net:8082/status` in phone Safari if Expo does not open. It
should report `packager-status:running`. A 502 means Serve matched the hostname
but could not reach Metro; check that Metro is running on port 8081. If the status
check works but Expo does not, inspect
Metro's error and confirm Expo Go/SDK compatibility rather than changing home IPs.

Force-close Expo Go and reopen the remote link while Wi-Fi is off. Record a rep,
wait for Saved, then open and replay it. This proves the preview can start away
from home, rather than merely retaining a previously loaded screen.

Stable preview keeps automatic file watching disabled for the OneDrive issue.
After code edits, stop and restart Metro. Keep both Metro and the backend terminal
running at home for this development setup.

To return this terminal to the usual home Wi-Fi preview, stop Metro and run:

```powershell
Remove-Item Env:EXPO_PACKAGER_PROXY_URL -ErrorAction SilentlyContinue
npm run start:stable
```

## 4. Native analysis is the next coding milestone

Following the project's preference for implementing larger features yourself,
make the remote health check pass before adding the native upload feature.
The existing API contract is:

- `GET /health` for connection status.
- `POST /analyze` with multipart field `video` containing the saved MOV/MP4.
- Optional `athlete_start_side` is `left` or `right` in camera-image coordinates.
- Confirmed goal setup additionally needs `goal_setup` and `goal_reference`.
  Omit these until native calibration exists; shot outcome must remain unknown.
- Response includes `summary`, `coaching_report`, `jev_judgements`,
  `pressure_report`, and relative `/outputs/...` artifact paths. Resolve artifact
  URLs against the same HTTPS backend origin.

Suggested files for the subsequent walkthrough:

- `src/mobile/lib/backend.ts`: validate and save the private HTTPS backend origin,
  health check, multipart upload, timeout, and useful failure messages.
- `src/mobile/src/app/rep/[name].tsx`: Analyse button for the selected saved rep.
- `src/mobile/src/app/analysis/[name].tsx`: waiting, result and retry states.

Keep local clips on failed uploads, allow explicit retry and show an indeterminate
analysis indicator. Do not invent a 0–100 analysis percentage or infer coaching
from a transport response. No cloud API keys belong in the phone app.

## Availability and rollback

The phone needs internet to reach home. If mobile service drops, remote analysis
is unavailable; local recording/replay can continue while the app is loaded.
An installed standalone build with bundled code is a later milestone and removes
the need for Metro, but remote analysis will still need the home backend online.

In Windows Power settings, prevent sleep while plugged in for pitch sessions.
Closing the lid may still sleep the laptop; leave it open unless configured.
Serve's background setting persists, but it does not restart Python or Metro after
a laptop reboot. Start those again before leaving.

To disable only the forwards created here:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' serve --http=8082 off
& 'C:\Program Files\Tailscale\tailscale.exe' serve --https=443 off
```

## References

- https://tailscale.com/docs/features/tailscale-serve
- https://tailscale.com/docs/reference/tailscale-cli/serve
- https://docs.expo.dev/more/expo-cli/ (EXPO_PACKAGER_PROXY_URL)

Initially prepared against SDK 57; the mobile app was upgraded to SDK 58 on
26 October 2026 to match the phone's Expo Go. The private HTTP route on 8082 remains
configured. A private HTTPS preview route was also configured for debugging:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' serve --bg --https=8443 http://localhost:8081
$env:EXPO_PACKAGER_PROXY_URL = 'https://laptop-rbgvktra.tail744b42.ts.net:8443'
npm run start:stable -- --localhost
```

Run the npm command from `src/mobile`. For HTTPS use
`exps://laptop-rbgvktra.tail744b42.ts.net:8443` on the phone, with the `s` included.
Use your own Serve hostname for another laptop. An HTTP request on HTTPS port 8443
returns 400; changing ports does not resolve an Expo SDK mismatch. Keep Expo Go
installed to retain its local clips. The SDK 58 dependencies, TypeScript, lint,
iOS bundle and Expo Doctor checks passed; final phone verification remains.
