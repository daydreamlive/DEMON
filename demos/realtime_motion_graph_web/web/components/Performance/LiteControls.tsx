"use client";

import { LiteTrackCarousel } from "./LiteTrackCarousel";
import { RecordToggle } from "./RecordToggle";
import { SliderGroup } from "./SliderGroup";

interface Props {
  onOpenAllControls: () => void;
  /** Render a small pulsing dot next to "All controls" — typically
   *  toggled by the host when there are unsaved session tweaks. DEMON
   *  doesn't ship session save (auth + /api/sessions live in
   *  demon-public-demo), so the prop stays optional. */
  unsavedDot?: boolean;
}

// Mobile-first "Lite" mixer. Curated for mid-performance, no-typing-required
// use: three primary sliders (denoise, structure, feedback), the audio-track
// carousel (with upload chip), and the record toggle. Prompt entry and seed
// randomization live in the "All controls" sheet — they're not meant to
// happen inside a performance.
//
// Renders as fixed bottom chrome on mobile (the AdvancedDrawer mobile
// branch mounts this directly, not inside a slide-up drawer). Click
// "All controls" → opens <MobileFullSheet/>, the full 7-tab equivalent.
export function LiteControls({ onOpenAllControls, unsavedDot }: Props) {
  return (
    <div className="lite-controls">
      <div className="lite-row lite-row--main">
        <SliderGroup param="denoise" label="denoise" />
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
          {unsavedDot && (
            <span
              className="lite-all-controls-dot"
              aria-label="Unsaved changes"
            />
          )}
          <span className="lite-all-controls-arrow" aria-hidden="true">
            →
          </span>
        </button>
      </div>
    </div>
  );
}
