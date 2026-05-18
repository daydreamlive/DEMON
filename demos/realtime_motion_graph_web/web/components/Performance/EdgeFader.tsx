"use client";

import { useLoraStore } from "@/store/useLoraStore";
import {
  LORA_DEFAULT_STRENGTH_FRACTION,
  LORA_SIDE_VISIBLE_FLOOR,
  LORA_SLIDER_MAX,
} from "@/types/engine";

// Traditional vertical fader UI rendered on top of each LoRA edge bar.
// Replaces the prior writhing-ribbon visual for the left/right edges
// with the same channel-strip vocabulary used inside VOICE: narrow
// recessed groove + horizontal cap thumb with a center indicator line.
// Drag capture still lives in <DesktopEdgeDrag/>, which sits over this
// fader with pointer-events: auto. The fader itself is decoration —
// pointer-events: none so it never interferes with the drag rect.

interface Props {
  side: "left" | "right";
}

export function EdgeFader({ side }: Props) {
  const enabled = useLoraStore((s) => s.enabled);
  const strengths = useLoraStore((s) => s.strengths);
  const slotIndex = side === "left" ? 0 : 1;
  const loraId = Array.from(enabled)[slotIndex] ?? null;
  const isEmpty = loraId === null;
  const value = isEmpty
    ? LORA_DEFAULT_STRENGTH_FRACTION * LORA_SLIDER_MAX
    : strengths[loraId] ?? 0;
  const rawFraction = LORA_SLIDER_MAX > 0
    ? Math.max(0, Math.min(1, value / LORA_SLIDER_MAX))
    : 0;
  // Match the SIDE_VISIBLE_FLOOR convention so the cap doesn't sink to
  // the bottom edge when strength is zero — the hint+cap stay anchored
  // to the lowest visible position on the rail.
  const fraction = isEmpty
    ? Math.max(rawFraction, LORA_SIDE_VISIBLE_FLOOR)
    : rawFraction;

  return (
    <div
      className={`edge-fader edge-fader--${side}${isEmpty ? " edge-fader--empty" : ""}`}
      aria-hidden="true"
    >
      <div className="edge-fader-track">
        <div
          className="edge-fader-fill"
          style={{ height: `${fraction * 100}%` }}
        />
        <div
          className="edge-fader-cap"
          style={{ bottom: `${fraction * 100}%` }}
        />
      </div>
    </div>
  );
}
