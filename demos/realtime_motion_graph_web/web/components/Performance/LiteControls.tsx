"use client";

import { LiteTrackCarousel } from "./LiteTrackCarousel";
import { RecordToggle } from "./RecordToggle";
import { SliderGroup } from "./SliderGroup";

interface Props {
  onOpenAllControls: () => void;
}

// Mobile-first "Lite" mixer. Curated for mid-performance, no-typing-required
// use: three primary sliders (remix, structure, feedback), the audio-track
// carousel (with upload chip), and the record toggle. Prompt entry and seed
// randomization live in the "All controls" sheet — they're not meant to
// happen inside a performance.
export function LiteControls({ onOpenAllControls }: Props) {
  return (
    <div className="lite-controls">
      <div className="lite-row lite-row--main">
        <SliderGroup param="denoise" label="remix" />
        <SliderGroup param="hint_strength" label="structure" />
        <SliderGroup param="feedback" label="feedback" />
      </div>

      <LiteTrackCarousel />

      <div className="lite-row lite-row--actions">
        <RecordToggle />
        <button
          type="button"
          className="lite-all-controls"
          onClick={onOpenAllControls}
        >
          All controls
          <span className="lite-all-controls-arrow" aria-hidden="true">
            →
          </span>
        </button>
      </div>
    </div>
  );
}
