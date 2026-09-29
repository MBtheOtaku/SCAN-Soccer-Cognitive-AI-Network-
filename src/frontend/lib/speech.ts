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


// ---------------------------------------------------------
// SCAN startup dialogue
// ---------------------------------------------------------

const greetings = {
  morning: [
    "Good morning, my friend. Ready to get some work in?",
    "Morning. Good to see you again. Let's get after it.",
    "Good morning. Hope you're feeling sharp today.",
    "Morning, my friend. Whenever you're ready, I'm ready.",
    "Good morning. Let's see what you've got today.",
  ],

  afternoon: [
    "Good afternoon, my friend. Ready for another session?",
    "Afternoon. Good to have you back.",
    "Good afternoon. Let's get some quality work in.",
    "Welcome back. Whenever you're ready, let's begin.",
    "Good afternoon, my friend. Let's see how you're moving today.",
  ],

  evening: [
    "Good evening, my friend. Let's make this session count.",
    "Evening. Good to see you back.",
    "Good evening. Ready when you are.",
    "Welcome back, my friend. Let's get to work.",
    "Evening. Let's finish the day with some good work.",
  ],
};


// ---------------------------------------------------------
// Random dialogue selection
// ---------------------------------------------------------

function pickNonRepeating(
  options: string[],
  storageKey: string
): string {
  const previous =
    typeof window !== "undefined"
      ? window.sessionStorage.getItem(storageKey)
      : null;

  /*
    Avoid immediately repeating the previous line.
    If there is only one option, just use it.
  */
  const available =
    options.length > 1
      ? options.filter((option) => option !== previous)
      : options;

  const choice =
    available[
      Math.floor(Math.random() * available.length)
    ];

  if (typeof window !== "undefined") {
    window.sessionStorage.setItem(
      storageKey,
      choice
    );
  }

  return choice;
}


// ---------------------------------------------------------
// Startup greeting
// ---------------------------------------------------------

export function getStartupGreeting(): string {
  const hour = new Date().getHours();

  // Morning: midnight → 11:59 AM
  if (hour < 12) {
    return pickNonRepeating(
      greetings.morning,
      "scan-last-morning-greeting"
    );
  }

  // Afternoon: noon → 5:59 PM
  if (hour < 18) {
    return pickNonRepeating(
      greetings.afternoon,
      "scan-last-afternoon-greeting"
    );
  }

  // Evening: 6 PM → midnight
  return pickNonRepeating(
    greetings.evening,
    "scan-last-evening-greeting"
  );
}


// ---------------------------------------------------------
// Speech initialization
// ---------------------------------------------------------

export function initializeSpeech() {
  if (typeof window === "undefined") {
    return;
  }

  selectedVoice = findBritishVoice();

  /*
    Some browsers load their available voices asynchronously.
    When that list changes, search for the preferred voice again.
  */
  window.speechSynthesis.onvoiceschanged = () => {
    selectedVoice = findBritishVoice();
  };
}


// ---------------------------------------------------------
// Speak
// ---------------------------------------------------------

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

  const utterance =
    new SpeechSynthesisUtterance(text);

  utterance.lang = "en-GB";

  if (selectedVoice) {
    utterance.voice = selectedVoice;
  }

  /*
    Slightly slower and lower pitched than the defaults
    to give SCAN a calm assistant/coach delivery.
  */
  utterance.rate = options?.rate ?? 0.92;
  utterance.pitch = options?.pitch ?? 0.9;
  utterance.volume = options?.volume ?? 0.9;

  window.speechSynthesis.speak(utterance);
}


// ---------------------------------------------------------
// Stop speech
// ---------------------------------------------------------

export function stopSpeaking() {
  if (typeof window !== "undefined") {
    window.speechSynthesis.cancel();
  }
}