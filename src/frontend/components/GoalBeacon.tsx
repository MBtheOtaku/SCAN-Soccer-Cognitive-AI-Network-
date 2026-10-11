"use client";
import { useEffect, useState } from "react";
import { closeBeacon, pulseBeacon, stopBeacon, unlockBeacon } from "../lib/beacon";

export default function GoalBeacon({ ready, voiceEnabled }: { ready: boolean; voiceEnabled: boolean }) {
  const [enabled, setEnabled] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled || !ready || !voiceEnabled) { stopBeacon(); return; }
    const timer = window.setInterval(pulseBeacon, 2000);
    return () => { window.clearInterval(timer); stopBeacon(); };
  }, [enabled, ready, voiceEnabled]);
  useEffect(() => () => closeBeacon(), []);

  return <div className="goal-beacon">
    <button className="secondary-button" disabled={!voiceEnabled || !ready} aria-pressed={enabled} onClick={async () => {
      if (enabled) { setEnabled(false); stopBeacon(); return; }
      try { await unlockBeacon(); setEnabled(true); setError(null); pulseBeacon(); }
      catch { setError("Beacon audio could not start. Try enabling audio again in Safari."); }
    }}>{enabled ? "STOP GOAL BEACON" : "START GOAL BEACON"}</button>
    <p>Optional goal reference tone. Place your Bluetooth speaker near the target if you want the sound to come from there. No virtual left/right direction. Paused during recording, analysis and feedback; follows VOICE ON/OFF.</p>
    {error && <p role="alert">{error}</p>}
  </div>;
}
