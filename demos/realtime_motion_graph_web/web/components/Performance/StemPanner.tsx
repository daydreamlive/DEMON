"use client";

import { useRef } from "react";

import type { StemOverlayKind } from "@/engine/audio/loadFixture";
import { useStemPannerDrag } from "@/hooks/useStemPannerDrag";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { STEM_OVERLAY_MAX, useStemOverlayStore } from "@/store/useStemOverlayStore";

// Horizontal stem-overlay panner. Drag right to mix more of the
// vocal/instrument stem into the model output; click the label to
// mute/unmute without losing the level. Shared by the HeroMacros bay
// and the drawer's CORE tab — the bay version hides when the drawer
// opens, so CORE carries the same control for in-drawer work.
//
// `data-param="stem_<kind>"` arms MIDI-learn via the document-level
// contextmenu handler in useMidi.ts (which routes the bound CC back
// into useStemOverlayStore). Empty slots omit it so right-clicking an
// unloaded panner is a no-op.

export const STEM_LABELS: Record<StemOverlayKind, string> = {
  vocals: "Vocals",
  instruments: "Instr",
};

interface Props {
  kind: StemOverlayKind;
}

export function StemPanner({ kind }: Props) {
  const fixture = usePerformanceStore((s) => s.fixture);
  const stems = useCustomTracksStore((s) =>
    fixture ? s.tracks.get(fixture)?.stems : undefined,
  );
  const enabled = useStemOverlayStore((s) => s.enabled[kind]);
  const volume = useStemOverlayStore((s) => s.volumes[kind]);
  const toggle = useStemOverlayStore((s) => s.toggle);
  const stemsReady = Boolean(stems);
  const trackRef = useRef<HTMLDivElement | null>(null);
  useStemPannerDrag(trackRef, kind, stemsReady);
  const displayValue = enabled ? volume : 0;
  const fraction =
    STEM_OVERLAY_MAX > 0
      ? Math.max(0, Math.min(1, displayValue / STEM_OVERLAY_MAX))
      : 0;
  const label = STEM_LABELS[kind];
  return (
    <div
      className={`hero-stem-panner${stemsReady ? "" : " hero-stem-panner--empty"}`}
      data-param={stemsReady ? `stem_${kind}` : undefined}
    >
      <button
        type="button"
        className="hero-stem-panner-label"
        onClick={() => {
          if (stemsReady) toggle(kind);
        }}
        disabled={!stemsReady}
        aria-pressed={enabled}
        aria-label={`${label} mute`}
        data-midi-learn={stemsReady ? `stem_${kind}_toggle` : undefined}
        data-dd-tooltip={enabled ? "Click to mute layer" : "Click to unmute layer"}
      >
        {label}
      </button>
      <div
        ref={trackRef}
        className="hero-stem-panner-track"
        role="slider"
        aria-label={`${label} overlay volume`}
        aria-orientation="horizontal"
        aria-valuemin={0}
        aria-valuemax={STEM_OVERLAY_MAX}
        aria-valuenow={displayValue}
      >
        <div
          className="hero-stem-panner-fill"
          style={{ width: `${fraction * 100}%` }}
        />
        <div
          className="hero-stem-panner-cap"
          style={{ left: `${fraction * 100}%` }}
        />
      </div>
      <span className="hero-stem-panner-value">{displayValue.toFixed(2)}</span>
    </div>
  );
}
