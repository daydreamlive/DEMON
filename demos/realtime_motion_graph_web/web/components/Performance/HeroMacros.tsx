"use client";

import { useSessionStore } from "@/store/useSessionStore";

import { Knob } from "./Knob";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";

// Permanent 3-knob row that sits at the bottom-center of the canvas,
// above the closed drawer handle, between AudioSourceCrate (left) and
// RecordButton (right). This is the "performance palette" — the three
// macros every musician reaches for, always visible so the main canvas
// reads as an instrument the moment the user lands.
//
// Knobs are smaller (~52 px caps) than the drawer's 64 px CORE knobs
// so this row feels like a focused subset, not a duplicate of the
// drawer. The set mirrors mobile's <LiteControls/> — proven minimal
// palette: DENOISE / STRUCTURE / FEEDBACK.
//
// Visibility:
//  - Hidden while the session is idle (no point dialing a remix in
//    before there's a stream).
//  - Hidden when the Full Controls drawer is open (mutually exclusive
//    with the CORE knobs — same params, same actions, just one place
//    at a time). CSS handles the drawer-open hide via body.drawer-open.
//  - Hidden below 768 px viewport (mobile gets LiteControls instead).
//    CSS handles the breakpoint.

const HERO_PARAMS = ["denoise", "hint_strength", "feedback"] as const;

export function HeroMacros() {
  const status = useSessionStore((s) => s.status);
  const started = status !== "idle";
  if (!started) return null;
  return (
    <div className="hero-macros" data-hero-macros>
      {HERO_PARAMS.map((p) => (
        <Knob
          key={p}
          param={p}
          label={defaultLabelFor(p)}
          kbd={kbdHintFor(p)}
        />
      ))}
    </div>
  );
}
