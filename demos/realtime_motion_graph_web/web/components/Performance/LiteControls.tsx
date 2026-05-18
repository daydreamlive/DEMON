"use client";

import { LiteTrackCarousel } from "./LiteTrackCarousel";
import { RecordToggle } from "./RecordToggle";

interface Props {
  onOpenAllControls: () => void;
  /** Render a small pulsing dot next to "All controls" — typically
   *  toggled by the host when there are unsaved session tweaks. DEMON
   *  doesn't ship session save (auth + /api/sessions live in
   *  demon-public-demo), so the prop stays optional. */
  unsavedDot?: boolean;
}

// Mobile-first compact action bar. Tracks + record + "All controls"
// gateway. The 3-slider row (denoise/structure/feedback) lived here
// pre-Wave-13 but was redundant: the left rail already handles denoise
// at thumb-reach, and structure/feedback are one tap away inside
// <MobileFullSheet/> → CORE tab. Trimming the bar gives the graph back
// the vertical real estate.
//
// Renders as fixed bottom chrome on mobile (the AdvancedDrawer mobile
// branch mounts this directly, not inside a slide-up drawer). Click
// "All controls" → opens <MobileFullSheet/>, the full 7-tab equivalent.
export function LiteControls({ onOpenAllControls, unsavedDot }: Props) {
  return (
    <div className="lite-controls">
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
