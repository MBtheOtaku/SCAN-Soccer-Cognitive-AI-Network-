// A mono reference pulse. Physical direction comes from speaker placement only.
let context: AudioContext | null = null;
const playing = new Set<OscillatorNode>();

export async function unlockBeacon() {
  if (!context || context.state === "closed") context = new AudioContext();
  await context.resume();
}

export function pulseBeacon() {
  if (!context || context.state !== "running" || window.speechSynthesis?.speaking) return;
  const oscillator = context.createOscillator();
  const gain = context.createGain();
  oscillator.frequency.value = 660;
  const now = context.currentTime;
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(.08, now + .015);
  gain.gain.linearRampToValueAtTime(0, now + .16);
  oscillator.connect(gain);
  gain.connect(context.destination);
  playing.add(oscillator);
  oscillator.onended = () => { playing.delete(oscillator); oscillator.disconnect(); gain.disconnect(); };
  oscillator.start(now);
  oscillator.stop(now + .17);
}

export function stopBeacon() {
  for (const oscillator of playing) {
    try { oscillator.stop(); } catch { /* Already ended. */ }
  }
  playing.clear();
}

export function closeBeacon() {
  stopBeacon();
  if (context && context.state !== "closed") void context.close().catch(() => {});
  context = null;
}
