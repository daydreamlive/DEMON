"use client";

import { SliderGroup } from "./SliderGroup";

// Generic mixer tile that wraps a row of sliders. Replaces the dynamic
// buildChannelTile() helper from app.js. `params` is the list of slider
// names (must exist in SLIDER_META).

interface Props {
  label: string;
  params: { param: string; label: string; max?: number }[];
}

const DISPLAY_NAMES: Record<string, string> = {
  noise_share: "nshare",
  ode_noise: "ode",
  hint_strength: "structure strength",
  dcw_scaler: "DCW low",
  dcw_high_scaler: "DCW high",
};

// Map slider param → keyboard hint shown beneath the slider. Mirrors the
// chord layout in hooks/useKeyboardShortcuts.ts; if you change one, change
// the other.
const KBD_FOR_PARAM: Record<string, string> = {
  denoise: "A + ▲▼",
  hint_strength: "G + ▲▼",
  feedback: "E + ▲▼",
  shift: "H + ▲▼",
  noise_share: "N + ▲▼",
  ode_noise: "D + ▲▼",
  ch_g0: "0 + ▲▼",
  ch_g1: "1 + ▲▼",
  ch_g2: "2 + ▲▼",
  ch_g3: "3 + ▲▼",
  ch_g4: "4 + ▲▼",
  ch_g5: "5 + ▲▼",
  ch_g6: "6 + ▲▼",
  ch_g7: "7 + ▲▼",
  ch13: "⇧1 + ▲▼",
  ch14: "⇧2 + ▲▼",
  ch19: "⇧3 + ▲▼",
  ch23: "⇧4 + ▲▼",
  ch29: "⇧5 + ▲▼",
  ch56: "⇧6 + ▲▼",
  dcw_scaler: "W + ▲▼",
  dcw_high_scaler: "Y + ▲▼",
};

export function kbdHintFor(param: string): string | undefined {
  return KBD_FOR_PARAM[param];
}

export function SliderTile({ label, params }: Props) {
  return (
    <div className="mixer-tile" data-tile={label.toLowerCase().replace(/ /g, "-")}>
      <div className="mixer-tile-label">{label}</div>
      <div className="mixer-channels">
        {params.map(({ param, label: pLabel, max }) => (
          <SliderGroup
            key={param}
            param={param}
            label={pLabel}
            max={max}
            kbd={KBD_FOR_PARAM[param]}
          />
        ))}
      </div>
    </div>
  );
}

export function defaultLabelFor(param: string): string {
  return DISPLAY_NAMES[param] ?? param.replace(/_/g, " ");
}
