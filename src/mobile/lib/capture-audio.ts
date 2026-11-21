export type CaptureAudioMode = 'sound' | 'video-only';

// Camera preview may be silent while permission is pending; recording must not silently fall back.
export function captureAudio(mode: CaptureAudioMode, microphone: { granted: boolean } | null) {
  const withSound = mode === 'sound' && microphone?.granted === true;
  return {
    canRecord: mode === 'video-only' || withSound,
    mute: !withSound,
    label: withSound ? 'SOUND ON' : mode === 'video-only' ? 'VIDEO ONLY' : 'MIC NEEDED',
  };
}
