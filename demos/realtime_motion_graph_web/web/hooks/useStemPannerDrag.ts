import { useEffect, type RefObject } from "react";

import type { StemOverlayKind } from "@/engine/audio/loadFixture";
import {
  STEM_OVERLAY_MAX,
  useStemOverlayStore,
} from "@/store/useStemOverlayStore";

// Pointer-drag binding for the horizontal stem overlay panners in
// HeroMacros. Same pointer-down → cache rect → commit-on-move → release
// loop as useLoraFaderDrag, but reads `clientX` / `width` instead of
// clientY / height — the panner runs left (zero) to right (max).
// Sliding to zero also clears `enabled` so the underlying overlay
// buffer is audibly silent.
//
// Beyond drag, the track carries the rest of the fader vocabulary so it
// matches SliderGroup muscle memory: mouse wheel nudges the level (Shift
// = fine), and a double-click resets to the default level. Wheel/dblclick
// work in fraction space (t ∈ [0, 1] of STEM_OVERLAY_MAX) so the gesture
// reads identically to the vertical faders.
//
// `enabled` mirrors the caller's "isEmpty" guard — when stems aren't
// ready yet, the listeners stay detached so the dimmed track doesn't
// swallow pointer events.

// Wheel step as a fraction of the rail — 1% per notch, ÷5 with Shift,
// matching Knob's SCROLL_STEP / FINE_DIVISOR.
const WHEEL_STEP = 0.01;
const WHEEL_STEP_FINE = WHEEL_STEP / 5;
const DBLCLICK_MS = 350;

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
    // DAW-style double-click-to-reset. Track pointerdown timestamps
    // ourselves (same as SliderGroup) so it's reliable under pointer
    // capture rather than leaning on the synthetic dblclick event.
    let lastDownAt = 0;

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
      const now = e.timeStamp;
      if (now - lastDownAt < DBLCLICK_MS) {
        // Second click in the window: reset to default and bail before
        // any drag state is set so the reset isn't clobbered by a commit.
        useStemOverlayStore.getState().reset(kind);
        lastDownAt = 0;
        return;
      }
      lastDownAt = now;
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

    // Mouse wheel over the track nudges the level in fraction space.
    // Reads the live volume (or 0 when muted) so scrolling up off a
    // muted layer ramps from zero and re-enables it, mirroring the drag
    // commit's `enabled = v > 0` rule. Non-passive so the page doesn't
    // scroll while the cursor sits on the rail.
    const onWheel = (e: WheelEvent) => {
      const dir = -Math.sign(e.deltaY);
      if (dir === 0) return;
      e.preventDefault();
      const step = e.shiftKey ? WHEEL_STEP_FINE : WHEEL_STEP;
      const store = useStemOverlayStore.getState();
      const current = store.enabled[kind] ? store.volumes[kind] : 0;
      const t = STEM_OVERLAY_MAX > 0 ? current / STEM_OVERLAY_MAX : 0;
      const v = Math.max(0, Math.min(1, t + dir * step)) * STEM_OVERLAY_MAX;
      store.setVolume(kind, v);
      store.setEnabled(kind, v > 0);
    };

    trackEl.addEventListener("pointerdown", onDown);
    trackEl.addEventListener("pointermove", onMove);
    trackEl.addEventListener("pointerup", onUp);
    trackEl.addEventListener("pointercancel", onUp);
    trackEl.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      trackEl.removeEventListener("pointerdown", onDown);
      trackEl.removeEventListener("pointermove", onMove);
      trackEl.removeEventListener("pointerup", onUp);
      trackEl.removeEventListener("pointercancel", onUp);
      trackEl.removeEventListener("wheel", onWheel);
    };
  }, [trackRef, kind, enabled]);
}
