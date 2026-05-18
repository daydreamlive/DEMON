"use client";

import { useState, type ReactElement } from "react";

// Tab strip for the slide-up Full Controls drawer. Combo
// segmented-control + icon-led tabs — a single bordered hardware
// shell containing six beveled cells, each with a small monoline
// icon above its label. The active cell sits recessed (inset
// shadow); inactive cells stay flush. Vocabulary mirrors the
// HeroMacros bay, the MasterPanel, and the knob caps — same
// skeumorphic dimensionality used everywhere else this drawer
// touches.
//
// Six tabs:
//   CORE   — MIX / TRACK / TIMBRE / FEEDBACK / BASS / TREBLE (knobs)
//   MOD    — SHIFT / N.SHARE / JITTER (knobs) + DCW config
//   VOICE  — the 14 latent channels (V1–V8 + M1–M6, faders)
//   PROMPT — prompts, key, time signature, seed
//   LIB    — LoRAs
//   CONFIG — session controls: track/key/sig, transport, MIDI, prefs

export const DRAWER_TABS = ["core", "mod", "voice", "prompt", "lib", "config"] as const;
export type DrawerTab = (typeof DRAWER_TABS)[number];

const TAB_LABELS: Record<DrawerTab, string> = {
  core: "Core",
  mod: "Mod",
  voice: "Voice",
  prompt: "Prompt",
  lib: "Lib",
  config: "Config",
};

// Monoline 16x16 icons — same vocabulary as the halo menu (1.4px
// stroke, round caps/joins, no fill) so the hand-drawn feel reads
// consistent across the chrome.
const TAB_ICONS: Record<DrawerTab, ReactElement> = {
  // CORE — knob cap with indicator notch
  core: (
    <>
      <circle cx="8" cy="8" r="5.2" />
      <line x1="8" y1="3.2" x2="8" y2="5.6" />
    </>
  ),
  // MOD — sine wave / LFO squiggle
  mod: (
    <path d="M2 8 Q 4.5 3.5 7 8 T 12 8 T 14 8" />
  ),
  // VOICE — vertical fader bank (three caps at different heights)
  voice: (
    <>
      <line x1="4" y1="2.5" x2="4" y2="13.5" />
      <line x1="8" y1="2.5" x2="8" y2="13.5" />
      <line x1="12" y1="2.5" x2="12" y2="13.5" />
      <rect x="2.5" y="6" width="3" height="2" rx="0.4" />
      <rect x="6.5" y="9.5" width="3" height="2" rx="0.4" />
      <rect x="10.5" y="4.5" width="3" height="2" rx="0.4" />
    </>
  ),
  // PROMPT — speech bubble with two lines
  prompt: (
    <>
      <path d="M2.5 3.5h11a1 1 0 0 1 1 1v6a1 1 0 0 1-1 1H9.5l-3 2.5v-2.5H2.5a1 1 0 0 1-1-1v-6a1 1 0 0 1 1-1z" />
      <line x1="5" y1="7" x2="11" y2="7" />
      <line x1="5" y1="9.5" x2="9" y2="9.5" />
    </>
  ),
  // LIB — cassette tape (matches the deprecated top-left button's
  // language — the LoRA library lives here now)
  lib: (
    <>
      <rect x="2" y="4" width="12" height="8" rx="1.2" />
      <circle cx="6" cy="9" r="1.4" />
      <circle cx="10" cy="9" r="1.4" />
    </>
  ),
  // CONFIG — gear / cog
  config: (
    <>
      <circle cx="8" cy="8" r="2.2" />
      <path d="M8 1.8v1.6 M8 12.6v1.6 M14.2 8h-1.6 M3.4 8H1.8 M12.4 3.6l-1.1 1.1 M4.7 11.3l-1.1 1.1 M12.4 12.4l-1.1-1.1 M4.7 4.7L3.6 3.6" />
    </>
  ),
};

interface Props {
  active: DrawerTab;
  onChange: (tab: DrawerTab) => void;
}

export function DrawerTabs({ active, onChange }: Props) {
  return (
    <div className="drawer-tabs" role="tablist" aria-label="Full controls">
      {DRAWER_TABS.map((t) => (
        <button
          key={t}
          type="button"
          role="tab"
          aria-selected={active === t}
          className={`drawer-tab${active === t ? " drawer-tab--active" : ""}`}
          onClick={() => onChange(t)}
        >
          <svg
            className="drawer-tab-icon"
            viewBox="0 0 16 16"
            width={16}
            height={16}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.4}
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            {TAB_ICONS[t]}
          </svg>
          <span className="drawer-tab-label">{TAB_LABELS[t]}</span>
        </button>
      ))}
    </div>
  );
}

export function useDrawerTab(initial: DrawerTab = "core") {
  return useState<DrawerTab>(initial);
}
