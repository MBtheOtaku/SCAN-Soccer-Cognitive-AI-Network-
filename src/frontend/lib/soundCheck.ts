// User-triggered setup samples only; no training events or coaching policy.
export type SoundCheckMode = "beacon" | "voice" | "sequence";
export const SOUND_CHECK_CUE = "Sound check. Ready for your next repetition.";

export function createSoundCheck(audio: {
  unlock: () => Promise<void>;
  pulse: () => void;
  speak: (text: string, options: { onEnd: () => void; onError: () => void }) => void;
  stopTone: () => void;
  stopVoice: () => void;
  status: (message: string, active: boolean) => void;
}) {
  let generation = 0;
  let active = false;
  const timers = new Set<ReturnType<typeof setTimeout>>();
  function stop() {
    generation++;
    for (const timer of timers) clearTimeout(timer);
    timers.clear();
    if (active) { active = false; audio.stopTone(); audio.stopVoice(); }
  }
  async function start(mode: SoundCheckMode) {
    stop();
    active = true;
    const token = generation;
    audio.stopTone(); audio.stopVoice();
    audio.status("Starting sound check…", true);
    const current = () => active && token === generation;
    const finish = (message: string) => {
      if (!current()) return;
      stop(); audio.status(message, false);
    };
    const later = (callback: () => void, ms: number) => {
      const timer = setTimeout(() => { timers.delete(timer); if (current()) callback(); }, ms);
      timers.add(timer);
    };
    const voice = () => {
      audio.status("Playing sample voice cue…", true);
      try { audio.speak(SOUND_CHECK_CUE, {
        onEnd: () => finish("Sound check finished. Did you hear the sample clearly?"),
        onError: () => finish("Voice could not play. Try TEST COACHING VOICE directly."),
      }); } catch { finish("Voice could not play. Try TEST COACHING VOICE directly."); return; }
      later(() => finish("Voice did not finish. Check audio permissions and try again."), 15000);
    };
    try {
      if (mode === "voice") { voice(); return; }
      later(() => finish("Audio did not start or finish. Check audio permissions and retry."), 20000);
      await audio.unlock();
      if (!current()) return;
      audio.status("Playing three target beacon pulses…", true);
      audio.pulse();
      later(audio.pulse, 600);
      later(audio.pulse, 1200);
      later(() => mode === "sequence" ? voice() : finish("Beacon test finished. Did you hear three pulses?"), 2000);
    } catch {
      finish("Audio could not start. Check browser audio permissions and retry.");
    }
  }
  return { start, stop };
}
