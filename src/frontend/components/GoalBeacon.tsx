"use client";
import { useEffect, useState } from "react";
import { closeBeacon, pulseBeacon, stopBeacon, unlockBeacon } from "../lib/beacon";
import AudioSoundCheck from "./AudioSoundCheck";

export default function GoalBeacon({ ready, voiceEnabled, testReady, testing, onTestingChange }: {
  ready: boolean; voiceEnabled: boolean; testReady: boolean; testing: boolean; onTestingChange: (active: boolean) => void;
}) {
  const [enabled, setEnabled] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    // The sound check owns the shared tone while testing; do not cut off its pulse.
    if (testing) return;
    if (!enabled || !ready || !voiceEnabled) { stopBeacon(); return; }
    const timer = window.setInterval(pulseBeacon, 2000);
    return () => { window.clearInterval(timer); };
  }, [enabled, ready, voiceEnabled, testing]);
  useEffect(() => () => closeBeacon(), []);

  return <div className="goal-beacon">
    <button className="secondary-button" disabled={!voiceEnabled || !ready || testing} aria-pressed={enabled} onClick={async () => {
      if (enabled) { setEnabled(false); stopBeacon(); return; }
      try { await unlockBeacon(); setEnabled(true); setError(null); pulseBeacon(); }
      catch { setError("Beacon audio could not start. Try enabling audio again in Safari."); }
    }}>{enabled ? "STOP GOAL BEACON" : "START GOAL BEACON"}</button>
    <p>Optional goal reference tone. Place your Bluetooth speaker near the target if you want the sound to come from there. No virtual left/right direction. Paused during recording, analysis and feedback; follows VOICE ON/OFF.</p>
    {error && <p role="alert">{error}</p>}
    <AudioSoundCheck disabled={!voiceEnabled || !testReady} onStart={() => { setEnabled(false); stopBeacon(); }} onActiveChange={onTestingChange} />
  </div>;
}
