"use client";

import { useMidiStore } from "@/store/useMidiStore";

// Skeuomorphic "MIDI in" jack toggle. Sits in the right-side tools
// cluster (HeroMacros' hero-macros-tools) under the Full Controls
// button. A red LED with a label — lit means MIDI is on, dark means
// off. Click flips the store's ``enabled`` flag; ``useMidi`` watches
// that and only calls ``navigator.requestMIDIAccess`` when it's true,
// so the browser permission prompt fires only on an explicit
// user-initiated enable.
//
// Hidden on touch-only devices via CSS — phones have nothing to gain
// from MIDI and the toggle would just clutter the chrome.

export function MidiInToggle() {
  const enabled = useMidiStore((s) => s.enabled);
  const status = useMidiStore((s) => s.status);
  const setEnabled = useMidiStore((s) => s.setEnabled);

  // Three visual states for the LED: off (dark) / on-armed
  // (steady red — access granted, listening) / on-denied (steady
  // amber — user enabled but browser rejected / Web MIDI N/A).
  // Status tone is the existing channel for "warn" / "off" → mirror
  // that into the LED color rather than introducing a new field.
  const ledState: "off" | "on" | "warn" =
    !enabled ? "off"
      : status.tone === "warn" || status.tone === "off" ? "warn"
      : "on";

  const title = enabled
    ? `MIDI in: ${status.message}. Click to disable.`
    : "MIDI in: off. Click to enable (the browser will ask for permission).";

  return (
    <button
      type="button"
      className={`midi-in-toggle midi-in-toggle--${ledState}`}
      onClick={() => setEnabled(!enabled)}
      aria-pressed={enabled}
      aria-label={enabled ? "Disable MIDI in" : "Enable MIDI in"}
      title={title}
    >
      <span className="midi-in-toggle-label">MIDI in</span>
      <span className="midi-in-toggle-led" aria-hidden="true" />
    </button>
  );
}
