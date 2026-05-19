"use client";

import { SliderGroup } from "./SliderGroup";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";

// The CHANNELS tab body — the model's internal latent channels. These
// 14 latent-space sliders are the closest thing this app has to a
// synth's voice/dimension controls. Naming honestly: V1–V8 for the
// always-available internal latent channels, M1–M6 for hand-tuned
// morph latents known to produce noticeable perceptual change.
//
// The perceptual mapping of each channel isn't standardized — we don't
// pretend V3 is "drums" — so the section copy frames this as
// exploration: sweep to discover what each channel does for your
// source.
//
// Merges the prior ChannelGainsTile + ChannelsTile into one tile so
// the CHANNELS tab has a single coherent surface instead of two
// adjacent tiles.

const VOICES = ["ch_g0", "ch_g1", "ch_g2", "ch_g3", "ch_g4", "ch_g5", "ch_g6", "ch_g7"];
const MORPH = ["ch13", "ch14", "ch19", "ch23", "ch29", "ch56"];

export function VoiceTile() {
  return (
    <div className="mixer-tile mixer-tile--voice" data-tile="voice">
      <div className="voice-tile-warning" role="note">
        <div className="voice-tile-warning-title">Experimental feature</div>
        <p className="voice-tile-warning-body">
          These are not traditional audio channels and gains. They
          manipulate different dimensions of the model&apos;s latent
          space, and produce results ranging from nuanced and beautiful
          to abrupt and discordant. Use at your own risk.
        </p>
      </div>
      <div className="voice-sections-row">
        <div className="voice-section">
          <div className="voice-section-label">Channels</div>
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
          <div className="voice-section-label">Channel gains</div>
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
