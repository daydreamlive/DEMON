"use client";

import { useState, type ReactElement } from "react";

// Tab strip for the Full Controls panel. Segmented hardware shell with
// six primary cells, monoline icon above each label. Active cell sits
// recessed via inset shadow + brighter foreground (a pressed hardware
// button).
//
// Body tabs (left to right):
//   CORE     — denoise / structure / timbre / seed (knobs) + track
//              picker + upload + the two reference-track pickers
//   STYLES   — prompts (top) + LoRA library (bottom). The two
//              "what should this sound like" surfaces share one tab.
//              Sits next to CORE because it carries the same
//              "where is this song heading" mental model as the
//              reference tracks above it.
//   MOD      — engine internals (shift / feedback / feedback_depth) +
//              DCW + CFG sub-tiles for the expert surfaces.
//   CHANNELS — the 14 latent channels (faders)
//   SAVED    — saved sessions
//   CONFIG   — session controls: key/sig, transport, MIDI, prefs

export const DRAWER_TABS = ["core", "styles", "mod", "voice", "saved", "config"] as const;
export type DrawerTab = (typeof DRAWER_TABS)[number];

const TAB_LABELS: Record<DrawerTab, string> = {
  core: "Core",
  mod: "Mod",
  voice: "Channels",
  styles: "Styles",
  saved: "Saved",
  config: "Config",
};

// Monoline 16x16 icons — same vocabulary as the halo menu (1.4px
// stroke, round caps/joins, no fill).
const TAB_ICONS: Record<DrawerTab, ReactElement> = {
  core: (
    <>
      <circle cx="8" cy="8" r="5.2" />
      <line x1="8" y1="3.2" x2="8" y2="5.6" />
    </>
  ),
  mod: <path d="M2 8 Q 4.5 3.5 7 8 T 12 8 T 14 8" />,
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
  // Styles tab — speech-bubble (prompts) over a cassette frame (LoRAs):
  // the two surfaces this tab combines, stacked vertically inside one
  // 16x16 monoline glyph.
  styles: (
    <>
      <path d="M2.5 2.5h11a1 1 0 0 1 1 1v3.5a1 1 0 0 1-1 1H9l-2.5 2v-2H2.5a1 1 0 0 1-1-1v-3.5a1 1 0 0 1 1-1z" />
      <line x1="4.5" y1="4.5" x2="11.5" y2="4.5" />
      <line x1="4.5" y1="6" x2="9.5" y2="6" />
      <rect x="2" y="10.5" width="12" height="3.5" rx="0.6" />
      <circle cx="5" cy="12.25" r="0.6" />
      <circle cx="11" cy="12.25" r="0.6" />
    </>
  ),
  saved: (
    <>
      <path d="M3.5 2.5h6.5l3 3v8a1 1 0 0 1-1 1H3.5a1 1 0 0 1-1-1v-10a1 1 0 0 1 1-1z" />
      <path d="M5.5 7.5h5 M5.5 10h5" />
    </>
  ),
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
      <div className="drawer-tabs-row drawer-tabs-row--primary">
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
    </div>
  );
}

export function useDrawerTab(initial: DrawerTab = "core") {
  return useState<DrawerTab>(initial);
}
