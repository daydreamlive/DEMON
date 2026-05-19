"use client";

import { useRef } from "react";

import { useScrollSyncedTabs } from "@/hooks/useScrollSyncedTabs";

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

type LiteTab = "mix" | "track";

const TABS: LiteTab[] = ["mix", "track"];
const TAB_LABELS: Record<LiteTab, string> = {
  mix: "Mix",
  track: "Track",
};

// Wave 13.5: tabbed + swipeable. Two sections live side-by-side in a
// scroll-snap track; tap a tab pill or swipe horizontally to switch.
// IntersectionObserver mirrors the scroll-position back into `tab` so
// the active pill stays in sync regardless of input method.
//
//   MIX   — 4 faders (denoise / structure / feedback / lora_blend) +
//           "All controls" gateway to the full 7-tab sheet.
//   TRACK — track carousel (upload + mic inside the picker) + REC.
export function LiteControls({ onOpenAllControls, unsavedDot }: Props) {
  const trackRef = useRef<HTMLDivElement | null>(null);
  const { activeTab: tab, gotoTab } = useScrollSyncedTabs<LiteTab>(
    trackRef,
    TABS,
    { attribute: "tab", initial: "mix" },
  );

  return (
    <div className="lite-controls">
      <div
        className="lite-tabs"
        role="tablist"
        aria-label="Mobile mixer"
      >
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            aria-selected={tab === t}
            className={`lite-tab${tab === t ? " lite-tab--active" : ""}`}
            onClick={() => gotoTab(t)}
          >
            {TAB_LABELS[t]}
            {/* Unsaved dot surfaces on whichever pill isn't currently
                active — so the cue is visible regardless of context. */}
            {unsavedDot && tab !== t && t === "mix" && (
              <span
                className="lite-tab-dot"
                aria-label="Unsaved changes"
              />
            )}
          </button>
        ))}
      </div>
      <div ref={trackRef} className="lite-tab-body">
        <section data-tab="mix" className="lite-tab-section">
          <div className="lite-row lite-row--main">
            <SliderGroup param="denoise" label="denoise" />
            <SliderGroup param="hint_strength" label="structure" />
            <SliderGroup param="feedback" label="feedback" />
            <SliderGroup param="lora_blend" label="lora blend" />
          </div>
          <button
            type="button"
            className="lite-all-controls"
            onClick={onOpenAllControls}
            aria-label="All controls"
            data-dd-tooltip="All controls"
            data-dd-tooltip-pos="below"
          >
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
        </section>
        <section data-tab="track" className="lite-tab-section">
          <LiteTrackCarousel />
          <RecordToggle />
        </section>
      </div>
    </div>
  );
}
