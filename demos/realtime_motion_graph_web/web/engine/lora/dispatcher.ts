"use client";

import {
  stampManualTouch,
  usePerformanceStore,
} from "@/store/usePerformanceStore";
import { useLoraStore } from "@/store/useLoraStore";
import { LORA_SLIDER_MAX } from "@/types/engine";

// Trailing-edge debounce for LoRA strength updates. LoRA strength
// changes trigger a weight refit on the server, which is far more
// expensive than a decode tick. Streaming the smoothed value at the
// param-sync cadence (8 ms) costs one refit per decode window during
// a drag — typically a stall per refit because the refit blocks the
// next tick. Instead we debounce: only the value the slider settles
// on (or pauses on for DEBOUNCE_MS) reaches the engine, so a long
// drag commits exactly one refit.
//
// We bypass the smoothing tween entirely for LoRA params: smoothing
// would just stretch the refit-storm across the smoothMs window
// instead of preventing it.

const DEBOUNCE_MS = 300;

interface PendingEntry {
  timer: number;
  value: number;
}

const pending = new Map<string, PendingEntry>();

function clamp(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(LORA_SLIDER_MAX, value));
}

export const loraStrengthDispatcher = {
  /** Drag/MIDI/edge-bar entry point. Pushes the value into the UI
   *  stores synchronously so the slider/fill/ribbon tracks the cursor,
   *  then arms (or resets) a trailing-edge timer that finally commits
   *  to sliderValues — where useParamSync picks it up and ships it.
   *
   *  Refit-lock side-effect: when the trailing-edge timer FIRES (the
   *  commit point — the moment the value lands in sliderValues and the
   *  next ~8 ms param tick will actually ship it to the engine), the
   *  dispatcher also marks a pending refit on ``useLoraStore``. The
   *  lock window thus covers the full back half of the latency budget
   *  (server refit, ~270-295 ms observed). A v1 lock that started at
   *  pointerup cleared too early because it didn't account for the
   *  ``DEBOUNCE_MS`` wait before commit. Marking from commit is the
   *  single chokepoint shared by every pointer-driven entry point
   *  (LibraryTile, DesktopEdgeDrag, useLoraFaderDrag, keyboard
   *  shortcuts) — they don't each have to install the lock. */
  set(id: string, rawValue: number): void {
    const value = clamp(rawValue);
    const param = `lora_str_${id}`;

    useLoraStore.getState().setStrength(id, value);
    stampManualTouch(param);
    usePerformanceStore.setState((s) => ({
      sliderTargets: { ...s.sliderTargets, [param]: value },
    }));

    const existing = pending.get(id);
    if (existing) clearTimeout(existing.timer);
    const timer = window.setTimeout(() => {
      pending.delete(id);
      usePerformanceStore.getState().setSliderDirect(param, value);
      // Commit fired — value is in flight to the engine. Start the
      // refit-pending window. Repeated commits within the window
      // re-bump the lock (markPendingRefit resets its internal timer),
      // so a back-to-back commit gets a clean lock for the latest
      // value rather than expiring on the first one's clock.
      useLoraStore.getState().markPendingRefit(id);
    }, DEBOUNCE_MS);
    pending.set(id, { timer, value });
  },
  /** Read-only predicate any pointer-gesture handler can consult on
   *  ``pointerdown`` to decide whether to refuse a new drag. True
   *  while ``useLoraStore.pendingRefit`` holds the id (the window
   *  from commit → refit-clear). MIDI / scheduled curves / MCP
   *  bypass this dispatcher and aren't affected — they aren't
   *  pointer gestures and don't conflict with the user's intent. */
  isLocked(id: string): boolean {
    return useLoraStore.getState().pendingRefit.has(id);
  },
};

/** Seed ``sliderValues["lora_str_<id>"]`` with the current strength.
 *  Closes the first-drag fallback leak: useParamSync prefers
 *  ``sliderValues`` and falls back to ``lora.strengths`` only when
 *  the per-LoRA key is absent. Pre-debounce, the dispatcher writes to
 *  ``lora.strengths`` synchronously on every pointermove but only
 *  commits to ``sliderValues`` after ``DEBOUNCE_MS`` — so the very
 *  first drag of a freshly-enabled LoRA would stream intermediate
 *  values via the fallback path until the first debounce commit.
 *  Seeding here means the fallback is never the source from frame
 *  one onward; the dispatcher's debounce is the only path into the
 *  engine for user-driven changes. Called from ``useLoraToggle`` on
 *  enable and from the store's ``setCatalog``/``reset`` seed paths. */
export function seedLoraSliderValue(id: string, strength: number): void {
  const param = `lora_str_${id}`;
  const value = clamp(strength);
  usePerformanceStore.getState().setSliderDirect(param, value);
}

/** Reconcile the engine-facing slider value of EVERY enabled LoRA with
 *  the store's current ``lora.strengths``. Unlike the per-enable seed
 *  (``seedLoraSliderValue`` fired once when a LoRA flips on), this runs
 *  for ALREADY-enabled LoRAs too.
 *
 *  Why it exists — the session-restore desync: a saved-session resume
 *  writes the persisted strength into ``useLoraStore.strengths`` (which
 *  the fader UI reads, so it shows 1.0) but NOT into
 *  ``perf.sliderValues``, the value ``useParamSync`` actually ships each
 *  tick. ``useParamSync`` prefers ``sliderValues`` and only falls back
 *  to ``lora.strengths`` when the ``lora_str_<id>`` key is ABSENT. The
 *  boot-time catalog seed already wrote a ``sliderValues`` entry for the
 *  default-on LoRAs, so on restore the key is present-but-stale: the
 *  engine stays pinned to the old strength while the slider reads the
 *  restored one. The LoRA is inaudible until a fader drag finally writes
 *  ``sliderValues`` through the dispatcher — exactly the reported bug
 *  ("shows 1.0 but you only hear it once I wiggle the fader").
 *
 *  Idempotent: on a fresh (non-restored) session ``sliderValues`` and
 *  ``lora.strengths`` already agree (both seeded from the same defaults
 *  / config), so every write lands the same value — the server only
 *  refits on a > 0.02 delta, so re-seeding the matching value triggers
 *  no extra refit. Call exactly once per session-ready edge (initial
 *  connect + reconnect), never mid-drag — the dispatcher owns
 *  ``sliderValues`` while a fader is in flight, and ready edges can't
 *  overlap a pointer gesture. */
export function reconcileEnabledLoraStrengths(): void {
  const { enabled, strengths } = useLoraStore.getState();
  for (const id of enabled) {
    const value = strengths[id];
    if (typeof value === "number") seedLoraSliderValue(id, value);
  }
}
