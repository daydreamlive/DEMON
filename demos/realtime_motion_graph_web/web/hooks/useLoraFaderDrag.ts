import { useEffect, type RefObject } from "react";

import { loraStrengthDispatcher } from "@/engine/lora/dispatcher";
import { useLoraStore } from "@/store/useLoraStore";
import { LORA_SLIDER_MAX } from "@/types/engine";

// Shared pointer-drag binding for the vertical LoRA strength faders
// (StylePanel's left-edge strip and HeroMacros' hero-bay inline pair).
// Both surfaces share the same gesture: pointer down → cache rect →
// commit on every move → release on up/cancel. Reads the enabled LoRA
// id from the store at commit time so a slot swap mid-drag still lands
// on the currently-enabled lora.
//
// `enabled` mirrors each caller's "isEmpty" guard — when no lora is
// bound to the slot, the listeners stay detached so the empty track
// doesn't swallow pointer events.

export function useLoraFaderDrag(
  trackRef: RefObject<HTMLElement | null>,
  slotIndex: number,
  enabled: boolean,
) {
  useEffect(() => {
    if (!enabled) return;
    const trackEl = trackRef.current;
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
  }, [trackRef, slotIndex, enabled]);
}
