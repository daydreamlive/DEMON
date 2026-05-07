// Single source of truth for keyboard shortcuts. The hook reads these to
// drive behavior; the config modal reads them to render the Keyboard tab.

export interface KeyBinding {
  combo: string;
  description: string;
}

export const KEY_BINDINGS: KeyBinding[] = [
  { combo: "A + ▲▼", description: "Remix strength (denoise)" },
  { combo: "G + ▲▼", description: "Structure strength (hint)" },
  { combo: "B + ▲▼", description: "Prompt blend (A↔B)" },
  { combo: "Enter", description: "Send prompt" },
  { combo: "⌘/Ctrl + Enter", description: "Send prompt (while in textarea)" },
  { combo: "E + ▲▼", description: "Engine: feedback" },
  { combo: "H + ▲▼", description: "Engine: shift" },
  { combo: "N + ▲▼", description: "Engine: nshare" },
  { combo: "D + ▲▼", description: "Engine: ode" },
  { combo: "W + ▲▼", description: "DCW low" },
  { combo: "Y + ▲▼", description: "DCW high" },
  { combo: "T", description: "Toggle DCW on/off" },
  { combo: "Shift + T", description: "Cycle DCW mode" },
  { combo: "Shift + W", description: "Cycle DCW wavelet" },
  { combo: "0…7 + ▲▼", description: "Channel gains G0…G7" },
  { combo: "Shift + 1…6 + ▲▼", description: "Channels (CH13…CH56)" },
  { combo: "F", description: "Randomize seed" },
  { combo: "R", description: "Record / stop audio" },
  { combo: "Space", description: "Pause / resume" },
  { combo: "K", description: "Toggle kiosk mode" },
  { combo: "O", description: "Toggle Advanced Controls drawer" },
  { combo: "Esc", description: "Toggle Advanced Controls drawer" },
];
