# Native SCAN screens

The mobile app now enters through `src/mobile/index.ts`, which imports
`expo-router/entry`. The screen routes live in `src/mobile/src/app/`:

- `index.tsx`: Home, local rep counts and navigation.
- `record.tsx`: camera permission, optional countdown, 5/10/15-second recording, saving and retry.
- `library.tsx`: previously saved recordings with readable dates and play actions.
- `rep/[name].tsx`: local playback resolved from a saved file name.

The previous `App.tsx` is kept as reference and is no longer the entry screen.
Do not paste the old single-screen guide over the new routes.

`lib/reps.ts` and its Documents folder are unchanged. `lib/rep-store.tsx` keeps
the saved list and an unfinished save alive across navigation. No backend
connection, new storage format, deletion or automatic upload was added.

## Progress and transitions

The football moves along a green line. During capture, 0–100% represents elapsed
time in the selected rep window, not analysis quality, upload progress or bytes
saved. Early stop moves directly into saving. Saving/loading durations are unknown,
so those states animate the ball without showing a percentage. A completed save
shows 100% only after the existing save and refresh operations finish.

Stack navigation uses native slide transitions. Reduce Motion disables slides and
continuous progress animation. Screen-reader labels describe buttons and progress.

The camera is mounted only while the Record route is focused and foregrounded.
Leaving the route is blocked while recording or saving. Recording stops when the
app is backgrounded; previously saved reps remain available in the library.

## Run and check on iPhone

Stop all Metro terminals before dependency installation. Start again after the
installation and checks finish:

```powershell
cd "C:\Users\Matthew Bui\OneDrive\SCAN\src\mobile"
npx expo start --go --clear
```

1. Home shows the original saved-rep count. Confirm that earlier clips still play.
2. Start a rep. Allow camera permission or open Settings if permission is denied.
3. Record for five seconds: timer progresses, saving has no invented percentage,
   and success appears only after the clip is saved.
4. Test early STOP & SAVE, then record another rep without leaving the screen.
5. Try back/swipe during recording and saving: the route must remain open.
6. Open Saved Reps, play a clip, close playback, then return to Record. The camera
   should restart and the Record button should become available again.
7. Background the app mid-recording. Confirm interruption is handled without
   claiming that a missing clip was saved.
8. Force-close after Saved, reopen and confirm all completed clips remain listed.
9. Enable iPhone Reduce Motion and test navigation and progress indicators.

Expo Go still needs the development server for loading the project. These UI
changes do not create a standalone installation or host the analysis backend.

## Saved-rep titles

Current runtime: Expo SDK 58, upgraded on 26 October 2026 to match the installed
iPhone Expo Go. The SDK 57 configuration snapshot and prior dependencies are
retained under `.expo/` and excluded from Metro. No phone recordings or storage
formats were changed by the upgrade. TypeScript, lint, iOS export and Expo Doctor
(20/20) passed; the phone still needs the final runtime check.

Open Saved Reps, choose a clip and tap **Add title & notes** (or **Edit title & notes**).
Save a title of up to 80 characters and optional notes of up to 500 characters.
Notes are the athlete's own observations, not SCAN coaching conclusions.
The title appears in playback and the library,
alongside the original recording date. Cancel discards edits; saving a blank
title restores the default label. Titles are stored locally beside the clips;
the video filenames and playback paths stay unchanged.

Phone check: rename an older clip, return to the library, close/reopen Expo Go
and confirm the title persists and the video still plays. Then check Cancel,
changing a title a second time, and clearing it. Restart stable Metro after edits
before testing.

## Countdown, recording length and export

On Record, choose a 5-, 10- or 15-second capture and an Off, 3- or 5-second
countdown. Defaults are 5-second recording and 3-second countdown for each visit.
Choices are locked during countdown, recording and saving. Cancel countdown
returns to Ready without making a clip. Backgrounding or leaving the camera screen
cancels any pending countdown; returning must never start a delayed recording.
Early Stop & save still works. The progress percentage measures recording time only.

On a saved rep's playback page, **Export video** opens the native share sheet
using `expo-sharing` (included in Expo Go for SDK 58). Choose Save Video or
Save to Files when available. This exports the original MOV/MP4, keeps the
original in SCAN, and does not embed titles/notes in the exported file. Closing
the share sheet is not reported as a successful export.

Phone checks after restarting Metro:

1. Test all three durations, including early Stop & save and a second rep.
2. Cancel a countdown, then confirm no recording appears after waiting.
3. Background during a countdown; return and confirm Ready rather than Recording.
4. Add/edit/clear notes, cancel an edit, and reopen the app to check persistence.
   Confirm older title-only reps and videos still work.
5. Export to Files and play the copy; also cancel export and confirm no false
   success message. Confirm SCAN's original clip remains playable.

## Windows TreeFS watcher crash

If Metro exits with `TreeFS: Failed to make parent directory entry`, that failure
is in Metro's file-change index, before screen navigation. Clearing caches alone
has not consistently prevented it in this workspace. The exact filesystem trigger
has not been confirmed; OneDrive is a possible contributor.

Use the opt-in stable preview to bypass the watcher:

```powershell
npm run start:stable
```

This uses Expo CLI's CI mode to disable file watching. It prints
LAN URLs for Expo Go because CI mode has no interactive QR menu. Choose the Wi-Fi
address reachable from your phone. Fully close/reopen Expo Go after restarting.

Fast Refresh is disabled. Stop and restart this command after source edits;
reloading the phone alone can reuse the old bundle. The regular `npm start`
command remains available. Stable mode preserves the cache to avoid a full rebuild
on every restart; use `npm run start:stable -- --clear` only to reset a stale cache.
A working checkout outside OneDrive is the preferred
next environment test if you want normal live editing; no checkout was moved and
no synced files were deleted for this workaround.
