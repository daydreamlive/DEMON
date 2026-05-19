"use client";

import { useRef } from "react";

import { useLoraFaderDrag } from "@/hooks/useLoraFaderDrag";
import { useLoraStore } from "@/store/useLoraStore";
import { useSessionStore } from "@/store/useSessionStore";
import { LORA_SLIDER_MAX } from "@/types/engine";

// Left-edge "Style" panel — two faders representing the current LoRA
// strengths. Pulled to the LEFT edge so the right edge stays clear for
// the Full Controls sidebar (which slides in from the right on
// desktop). Label says "Style" — these aren't output volumes, they're
// LoRA style strengths, and the prior "Master" wording set the wrong
// expectation. Reads like the inShaper / GrainDust master strip just
// mirrored to the opposite edge.
//
// Hidden while idle so the title screen stays clean. Drag interaction
// is bound to each fader cap (pointer events on the track + cap).

interface FaderProps {
  loraId: string | null;
  label: string;
  /** Slot index — `0` for LoRA-1, `1` for LoRA-2. Lets us look up the
   *  enabled LoRA id from the store at drag time. */
  slotIndex: number;
}

function StyleFader({ loraId, label, slotIndex }: FaderProps) {
  const strengths = useLoraStore((s) => s.strengths);
  const value = loraId ? strengths[loraId] ?? 0 : 0;
  const fraction = LORA_SLIDER_MAX > 0
    ? Math.max(0, Math.min(1, value / LORA_SLIDER_MAX))
    : 0;
  const isEmpty = loraId === null;
  const trackRef = useRef<HTMLDivElement | null>(null);
  useLoraFaderDrag(trackRef, slotIndex, !isEmpty);

  return (
    <div className={`style-fader${isEmpty ? " style-fader--empty" : ""}`}>
      <div className="style-fader-label" title={label}>
        {label}
      </div>
      <div
        ref={trackRef}
        className="style-fader-track"
        data-style-fader-slot={slotIndex}
        role="slider"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={LORA_SLIDER_MAX}
        aria-valuenow={value}
      >
        <div
          className="style-fader-fill"
          style={{ height: `${fraction * 100}%` }}
        />
        <div
          className="style-fader-cap"
          style={{ bottom: `${fraction * 100}%` }}
        />
      </div>
      <div className="style-fader-value">{value.toFixed(2)}</div>
    </div>
  );
}

export function StylePanel() {
  const status = useSessionStore((s) => s.status);
  const enabled = useLoraStore((s) => s.enabled);
  if (status === "idle") return null;

  const enabledIds = Array.from(enabled);
  const lora1 = enabledIds[0] ?? null;
  const lora2 = enabledIds[1] ?? null;

  return (
    <div className="style-panel" aria-label="Style strengths">
      <div className="style-panel-label">Style</div>
      <div className="style-panel-faders">
        <StyleFader
          loraId={lora1}
          label={lora1 ? labelFor(lora1) : "Style 1"}
          slotIndex={0}
        />
        <StyleFader
          loraId={lora2}
          label={lora2 ? labelFor(lora2) : "Style 2"}
          slotIndex={1}
        />
      </div>
    </div>
  );
}

// LoRA id → short human label. The id format is opaque so we just
// truncate; future work can wire this to a LoRA catalog lookup for
// proper display names.
function labelFor(loraId: string): string {
  const short = loraId.replace(/^lora_/, "").slice(0, 8);
  return short.toUpperCase();
}
