"use client";

import { useEffect, useState } from "react";

import { useConfig } from "@/lib/config";

import { SliderGroup } from "./SliderGroup";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";

// localStorage key for the dismissable experimental-feature notice.
const VOICE_WARNING_DISMISSED_KEY = "demon:voiceWarningDismissed";

// The CHANNELS tab body — the model's internal latent channels. These
// 14 latent-space sliders are the closest thing this app has to a
// synth's voice/dimension controls.
//
// Per-channel ranges come from `useConfig().channel_ranges`. Real
// bounds vary per channel (e.g. ch_g4 → [0, 2.5], ch_g0 → [0, 2.2])
// and some channels are tagged `reverse: true` because they sound
// better when turned down — the slider widget translates between its
// visual rail and the actual bounds via lib/sliderMapping, with
// unity=1.0 anchoring defaults at the rail midpoint so the whole
// bank lines up visually regardless of per-channel caps.
//
// Merges the prior ChannelGainsTile + ChannelsTile into one tile so
// the CHANNELS tab has a single coherent surface instead of two
// adjacent tiles.

const VOICES = ["ch_g0", "ch_g1", "ch_g2", "ch_g3", "ch_g4", "ch_g5", "ch_g6", "ch_g7"];
const MORPH = ["ch13", "ch14", "ch19", "ch23", "ch29", "ch56"];

export function VoiceTile() {
  const ranges = useConfig().channel_ranges;
  // Experimental-feature notice — dismissable, and the dismissal sticks
  // across reloads. Read after mount (not in the useState initializer)
  // so a localStorage read can't break SSR hydration.
  const [warningDismissed, setWarningDismissed] = useState(false);
  useEffect(() => {
    try {
      setWarningDismissed(
        localStorage.getItem(VOICE_WARNING_DISMISSED_KEY) === "1",
      );
    } catch {}
  }, []);
  const dismissWarning = () => {
    try {
      localStorage.setItem(VOICE_WARNING_DISMISSED_KEY, "1");
    } catch {}
    setWarningDismissed(true);
  };
  return (
    <div className="mixer-tile mixer-tile--voice" data-tile="voice">
      {!warningDismissed && (
        <div className="voice-tile-warning" role="note">
          <button
            type="button"
            className="voice-tile-warning-close"
            onClick={dismissWarning}
            aria-label="Dismiss experimental-feature notice"
          >
            ×
          </button>
          <div className="voice-tile-warning-title">Experimental feature</div>
          <p className="voice-tile-warning-body">
            These are not traditional audio channels and gains. They
            manipulate different dimensions of the model&apos;s latent
            space, and produce results ranging from nuanced and beautiful
            to abrupt and discordant. Use at your own risk.
          </p>
        </div>
      )}
      <div className="voice-sections-row">
        <div className="voice-section">
          <div className="voice-section-label">channel highlights</div>
          <div className="mixer-channels">
            {MORPH.map((p) => {
              const r = ranges[p];
              return (
                <SliderGroup
                  key={p}
                  param={p}
                  label={defaultLabelFor(p)}
                  min={r?.min}
                  max={r?.max}
                  reverse={r?.reverse}
                  unity={1.0}
                  kbd={kbdHintFor(p)}
                />
              );
            })}
          </div>
        </div>
        <div className="voice-section-divider" aria-hidden="true" />
        <div className="voice-section">
          <div className="voice-section-label">channel groups</div>
          <div className="mixer-channels">
            {VOICES.map((p) => {
              const r = ranges[p];
              return (
                <SliderGroup
                  key={p}
                  param={p}
                  label={defaultLabelFor(p)}
                  min={r?.min}
                  max={r?.max}
                  reverse={r?.reverse}
                  unity={1.0}
                  kbd={kbdHintFor(p)}
                />
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
