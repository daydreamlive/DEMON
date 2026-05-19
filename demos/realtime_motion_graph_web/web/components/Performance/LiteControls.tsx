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

// Wave 13.3: consolidated mobile mixer. All four primary faders
// (DENOISE / STRUCTURE / FEEDBACK / LORA BLEND) live in one bay; the
// utility cluster (track carousel, REC, "All controls") sits next to
// them. The edge stepper rails are retired — having faders on the
// screen edges duplicated controls and made the canvas feel cramped.
//
// Renders as fixed bottom chrome on mobile. "All controls" opens the
// full 7-tab <MobileFullSheet/>.
export function LiteControls({ onOpenAllControls, unsavedDot }: Props) {
  return (
    <div className="lite-controls">
      <div className="lite-row lite-row--main">
        <SliderGroup param="denoise" label="denoise" />
        <SliderGroup param="hint_strength" label="structure" />
        <SliderGroup param="feedback" label="feedback" />
        <SliderGroup param="lora_blend" label="lora blend" />
      </div>
      <div className="lite-utility">
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
    </div>
  );
}
