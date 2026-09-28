let selectedVoice: SpeechSynthesisVoice | null = null;

const preferredBritishVoices = [
  "Microsoft Ryan",
  "Microsoft George",
  "Google UK English Male",
  "Daniel",
];

function findBritishVoice(): SpeechSynthesisVoice | null {
  if (typeof window === "undefined") {
    return null;
  }

  const voices = window.speechSynthesis.getVoices();

  // Try known British male voices first.
  for (const preferredName of preferredBritishVoices) {
    const match = voices.find((voice) =>
      voice.name
        .toLowerCase()
        .includes(preferredName.toLowerCase())
    );

    if (match) {
      return match;
    }
  }

  // Otherwise use any British English voice.
  const britishVoice = voices.find(
    (voice) =>
      voice.lang.toLowerCase() === "en-gb" ||
      voice.lang.toLowerCase().startsWith("en-gb")
  );

  return britishVoice ?? voices[0] ?? null;
}

export function initializeSpeech() {
  if (typeof window === "undefined") {
    return;
  }

  selectedVoice = findBritishVoice();

  window.speechSynthesis.onvoiceschanged = () => {
    selectedVoice = findBritishVoice();
  };
}

export function speak(
  text: string,
  options?: {
    rate?: number;
    pitch?: number;
    volume?: number;
  }
) {
  if (
    typeof window === "undefined" ||
    !("speechSynthesis" in window)
  ) {
    return;
  }

  // Stop overlapping announcements.
  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);

  utterance.lang = "en-GB";

  if (selectedVoice) {
    utterance.voice = selectedVoice;
  }

  // Slightly measured delivery gives it the AI-assistant feel.
  utterance.rate = options?.rate ?? 0.92;
  utterance.pitch = options?.pitch ?? 0.9;
  utterance.volume = options?.volume ?? 0.9;

  window.speechSynthesis.speak(utterance);
}

export function stopSpeaking() {
  if (typeof window !== "undefined") {
    window.speechSynthesis.cancel();
  }
}