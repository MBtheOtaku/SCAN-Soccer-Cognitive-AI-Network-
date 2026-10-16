"use client";
import { useEffect, useRef, useState } from "react";
import { pulseBeacon, stopBeacon, unlockBeacon } from "../lib/beacon";
import { speak, stopSpeaking } from "../lib/speech";
import { createSoundCheck, type SoundCheckMode } from "../lib/soundCheck";

export default function AudioSoundCheck({ disabled, onStart, onActiveChange }: {
  disabled: boolean; onStart: () => void; onActiveChange: (active: boolean) => void;
}) {
  const [active, setActive] = useState(false);
  const [message, setMessage] = useState("Connect your speaker, then choose a test. Adjust volume on your phone or speaker.");
  const check = useRef<ReturnType<typeof createSoundCheck> | null>(null);
  function stop() {
    check.current?.stop(); setActive(false); onActiveChange(false);
    setMessage("Sound check stopped.");
  }
  useEffect(() => {
    const cancel = () => {
      check.current?.stop(); setActive(false); onActiveChange(false);
    };
    const hidden = () => { if (document.hidden) { cancel(); setMessage("Sound check stopped while the page was hidden."); } };
    if (disabled) cancel();
    document.addEventListener("visibilitychange", hidden);
    return () => { document.removeEventListener("visibilitychange", hidden); check.current?.stop(); onActiveChange(false); };
  }, [disabled, onActiveChange]);

  function start(mode: SoundCheckMode) {
    if (disabled || active) return;
    if (mode !== "beacon" && !("speechSynthesis" in window)) {
      setMessage("Voice playback is unavailable in this browser. You can still test the beacon."); return;
    }
    onStart();
    check.current?.stop();
    check.current = createSoundCheck({
      unlock: unlockBeacon, pulse: pulseBeacon, speak, stopTone: stopBeacon, stopVoice: stopSpeaking,
      status: (text, playing) => { setMessage(text); setActive(playing); onActiveChange(playing); },
    });
    void check.current.start(mode);
  }

  return <div className="sound-check">
    <h3 className="panel-label">AUDIO SOUND CHECK</h3>
    <p>Test your speaker before training. The sequence plays three beacon pulses, then a sample voice cue.</p>
    <div className="live-controls">
      <button className="secondary-button" disabled={disabled || active} onClick={() => start("beacon")}>TEST TARGET BEACON</button>
      <button className="secondary-button" disabled={disabled || active} onClick={() => start("voice")}>TEST COACHING VOICE</button>
      <button className="secondary-button" disabled={disabled || active} onClick={() => start("sequence")}>TEST SEQUENCE</button>
      <button className="secondary-button" disabled={!active} onClick={stop}>STOP AUDIO</button>
    </div>
    <p role="status" aria-live="polite">{disabled ? "Available with VOICE ON, between repetitions." : message}</p>
    <p className="pressure-note">Sample only: “Sound check. Ready for your next repetition.” Audio uses your device’s current output; connect Bluetooth in your device settings.</p>
  </div>;
}
