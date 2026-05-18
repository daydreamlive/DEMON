"use client";

import { useEffect } from "react";

import { loraStrengthDispatcher } from "@/engine/lora/dispatcher";
import { useLoraStore } from "@/store/useLoraStore";
import { useSessionStore } from "@/store/useSessionStore";
import { LORA_SLIDER_MAX } from "@/types/engine";

// Right-side master panel — replaces the prior writhing-ribbon LoRA
// rails on both edges with a single consolidated panel containing
// both LoRA faders side-by-side. Reads like the inShaper / GrainDust
// "master strip" — a recessed bay glued to the right edge of the
// canvas holding the channel-out controls.
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

function MasterFader({ loraId, label, slotIndex }: FaderProps) {
  const strengths = useLoraStore((s) => s.strengths);
  const value = loraId ? strengths[loraId] ?? 0 : 0;
  const fraction = LORA_SLIDER_MAX > 0
    ? Math.max(0, Math.min(1, value / LORA_SLIDER_MAX))
    : 0;
  const isEmpty = loraId === null;

  useEffect(() => {
    if (isEmpty) return;
    const trackEl = document.querySelector<HTMLDivElement>(
      `[data-master-fader-slot="${slotIndex}"]`,
    );
    if (!trackEl) return;
    let dragging = false;
    let cachedRect: DOMRect | null = null;
    const commit = (clientY: number) => {
      if (!cachedRect) return;
      const t = 1 - (clientY - cachedRect.top) / cachedRect.height;
      const ids = Array.from(useLoraStore.getState().enabled);
      const id = ids[slotIndex];
      if (!id) return;
      const v = Math.max(0, Math.min(1, t)) * LORA_SLIDER_MAX;
      loraStrengthDispatcher.set(id, v);
    };
    const onDown = (e: PointerEvent) => {
      if (e.button !== 0 && e.pointerType === "mouse") return;
      dragging = true;
      cachedRect = trackEl.getBoundingClientRect();
      trackEl.setPointerCapture(e.pointerId);
      commit(e.clientY);
    };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      commit(e.clientY);
    };
    const onUp = (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      trackEl.releasePointerCapture(e.pointerId);
      cachedRect = null;
    };
    trackEl.addEventListener("pointerdown", onDown);
    trackEl.addEventListener("pointermove", onMove);
    trackEl.addEventListener("pointerup", onUp);
    trackEl.addEventListener("pointercancel", onUp);
    return () => {
      trackEl.removeEventListener("pointerdown", onDown);
      trackEl.removeEventListener("pointermove", onMove);
      trackEl.removeEventListener("pointerup", onUp);
      trackEl.removeEventListener("pointercancel", onUp);
    };
  }, [slotIndex, isEmpty]);

  return (
    <div className={`master-fader${isEmpty ? " master-fader--empty" : ""}`}>
      <div className="master-fader-label" title={label}>
        {label}
      </div>
      <div
        className="master-fader-track"
        data-master-fader-slot={slotIndex}
        role="slider"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={LORA_SLIDER_MAX}
        aria-valuenow={value}
      >
        <div
          className="master-fader-fill"
          style={{ height: `${fraction * 100}%` }}
        />
        <div
          className="master-fader-cap"
          style={{ bottom: `${fraction * 100}%` }}
        />
      </div>
      <div className="master-fader-value">{value.toFixed(2)}</div>
    </div>
  );
}

export function MasterPanel() {
  const status = useSessionStore((s) => s.status);
  const enabled = useLoraStore((s) => s.enabled);
  if (status === "idle") return null;

  const enabledIds = Array.from(enabled);
  const lora1 = enabledIds[0] ?? null;
  const lora2 = enabledIds[1] ?? null;

  return (
    <div className="master-panel" aria-label="Master output">
      <div className="master-panel-label">Master</div>
      <div className="master-panel-faders">
        <MasterFader
          loraId={lora1}
          label={lora1 ? labelFor(lora1) : "LoRA 1"}
          slotIndex={0}
        />
        <MasterFader
          loraId={lora2}
          label={lora2 ? labelFor(lora2) : "LoRA 2"}
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
