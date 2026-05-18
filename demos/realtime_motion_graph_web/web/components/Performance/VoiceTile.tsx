"use client";

import { SliderGroup } from "./SliderGroup";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";

// The VOICE tab body — the model's internal voice channels. A synth has
// multiple "voices" (independent dimensions of its character), and
// these 14 latent-space sliders are the closest thing this app has to
// that concept. Naming honestly: V1–V8 for the always-available
// internal voices, M1–M6 for hand-tuned morph channels known to
// produce noticeable perceptual change.
//
// The perceptual mapping of each channel isn't standardized — we don't
// pretend V3 is "drums" — so the section copy frames this as
// exploration: sweep to discover what each channel does for your source.
//
// Merges the prior ChannelGainsTile + ChannelsTile into one tile so the
// VOICE tab has a single coherent surface instead of two adjacent tiles.

const VOICES = ["ch_g0", "ch_g1", "ch_g2", "ch_g3", "ch_g4", "ch_g5", "ch_g6", "ch_g7"];
const MORPH = ["ch13", "ch14", "ch19", "ch23", "ch29", "ch56"];

export function VoiceTile() {
  return (
    <div className="mixer-tile mixer-tile--voice" data-tile="voice">
      <div className="mixer-tile-label">Voice</div>
      <p className="voice-tile-blurb">
        Internal voice channels of the model. Each one shapes a different
        dimension of the output — frequency, dynamics, transients. Their
        exact perceptual mappings are still being charted; sweep to discover
        what each does for your source.
      </p>
      <div className="voice-sections-row">
        <div className="voice-section">
          <div className="voice-section-label">Internal voices</div>
          <div className="mixer-channels">
            {VOICES.map((p) => (
              <SliderGroup
                key={p}
                param={p}
                label={defaultLabelFor(p)}
                kbd={kbdHintFor(p)}
              />
            ))}
          </div>
        </div>
        <div className="voice-section-divider" aria-hidden="true" />
        <div className="voice-section">
          <div className="voice-section-label">Tuned morph</div>
          <div className="mixer-channels">
            {MORPH.map((p) => (
              <SliderGroup
                key={p}
                param={p}
                label={defaultLabelFor(p)}
                kbd={kbdHintFor(p)}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
