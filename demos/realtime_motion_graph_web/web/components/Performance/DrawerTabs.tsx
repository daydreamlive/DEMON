"use client";

import { useState } from "react";

// Tab strip for the slide-up Full Controls drawer. Modeled on the
// MAIN / GRAINS / FILTER / PITCH / SPACE / ARP / EFFECTS / EXTRAS /
// MATRIX header that Sound Particles GrainDust uses — sectioned by
// function with a clear hierarchy from "things you touch every session"
// to "things you configure once."
//
// Six tabs:
//   CORE   — MIX / TRACK / TIMBRE / FEEDBACK / BASS / TREBLE (knobs)
//   MOD    — SHIFT / N.SHARE / JITTER (knobs) + DCW config
//   VOICE  — the 14 latent channels (V1–V8 + M1–M6, faders)
//   PROMPT — prompts, key, time signature, seed
//   LIB    — LoRAs
//   CONFIG — session controls: track/key/sig, transport, MIDI, prefs
//
// Same IA on desktop + mobile so muscle memory carries between viewports.

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
          {TAB_LABELS[t]}
        </button>
      ))}
    </div>
  );
}

export function useDrawerTab(initial: DrawerTab = "core") {
  return useState<DrawerTab>(initial);
}
