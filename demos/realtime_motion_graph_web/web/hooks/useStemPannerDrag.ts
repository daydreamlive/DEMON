import { useEffect, type RefObject } from "react";

import type { StemOverlayKind } from "@/engine/audio/loadFixture";
import { useStemOverlayStore } from "@/store/useStemOverlayStore";

// Pointer-drag binding for the horizontal stem overlay panners in
// HeroMacros. Same pointer-down → cache rect → commit-on-move → release
// loop as useLoraFaderDrag, but reads `clientX` / `width` instead of
// clientY / height — the panner runs left (zero) to right (max).
// Sliding to zero also clears `enabled` so the underlying overlay
// buffer is audibly silent.
//
// `enabled` mirrors the caller's "isEmpty" guard — when stems aren't
// ready yet, the listeners stay detached so the dimmed track doesn't
// swallow pointer events.

const STEM_OVERLAY_MAX = 1.5;

export function useStemPannerDrag(
  trackRef: RefObject<HTMLElement | null>,
  kind: StemOverlayKind,
  enabled: boolean,
) {
  useEffect(() => {
    if (!enabled) return;
    const trackEl = trackRef.current;
    if (!trackEl) return;

    let dragging = false;
    let cachedRect: DOMRect | null = null;

    const commit = (clientX: number) => {
      if (!cachedRect) return;
      const t = (clientX - cachedRect.left) / cachedRect.width;
      const v = Math.max(0, Math.min(1, t)) * STEM_OVERLAY_MAX;
      const store = useStemOverlayStore.getState();
      store.setVolume(kind, v);
      store.setEnabled(kind, v > 0);
    };

    const onDown = (e: PointerEvent) => {
      if (e.button !== 0 && e.pointerType === "mouse") return;
      dragging = true;
      cachedRect = trackEl.getBoundingClientRect();
      trackEl.setPointerCapture(e.pointerId);
      commit(e.clientX);
    };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      commit(e.clientX);
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
  }, [trackRef, kind, enabled]);
}
