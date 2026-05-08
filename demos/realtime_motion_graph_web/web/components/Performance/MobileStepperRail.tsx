"use client";

import { useCallback, useEffect, useRef } from "react";

import { displayLoraName } from "@/lib/loraLabels";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { SLIDER_META } from "@/types/engine";

// Vertical stepper rail for the screen edges on phones in landscape mode.
// Replaces the drag-rails (MobileRemixRail / MobileLoraBlendRail). Two
// stacked buttons — ▲ at the top end, ▼ at the bottom end — with the
// param's label centered between them and the current value as a small
// readout below the label.
//
// Behavior:
//   - Tap a button: increment/decrement by SLIDER_META[param].step.
//   - Press-and-hold a button: after HOLD_DELAY_MS, repeats every
//     HOLD_INTERVAL_MS and accelerates over time so big moves don't
//     require a marathon of taps.
//   - Each step calls navigator.vibrate(8) where supported (Android),
//     no-op on iOS Safari.
//
// Side bias: the buttons stack vertically. `side="left"` mounts on the
// left edge with vertical labels; `side="right"` mirrors. We reuse the
// same install-edge-{side} writhe canvas underneath: writing --fill on
// the edge keeps the visual ribbon in lockstep with the value.

const HOLD_DELAY_MS = 350;
const HOLD_INTERVAL_MS = 90;
const HOLD_ACCEL_AT_MS = 1500; // after this much hold time, double speed

interface Props {
  side: "left" | "right";
  /** sliderTargets key. */
  param: string;
  /** Slider max — used to clamp + compute --fill on the edge writhe. */
  max: number;
  /** Header label (e.g. "Remix Strength"). */
  label: string;
  /** Optional per-chevron labels — rendered just inside the screen edge,
   *  paired with the up/down chevrons so the user can see at a glance
   *  which direction does what. For LoRA blend: top = the LoRA that
   *  pressing up gives MORE of, bottom = the LoRA pressing down
   *  emphasizes. Empty values render nothing. */
  sublabelTop?: string;
  sublabelBottom?: string;
  /** Inverted means top button decreases, bottom increases. Use for the
   *  blend rail where "top of rail = LoRA A" and pressing up should mean
   *  more A (which is value=0). */
  invert?: boolean;
  /** Visual writhe-fill orientation: when true, --fill = 1 - frac (top of
   *  rail bright when value is 0). Match the rail's `invert`. */
  invertFill?: boolean;
  /** When true, marks the up chevron with `data-gate="up"` so CSS can
   *  pulse it as a "do this" affordance. Used by the per-song "drag to
   *  start" gate on the remix rail — the sideways stepper label isn't
   *  visible enough to act as the prompt, so the chevron itself
   *  becomes the cue. */
  pulseUp?: boolean;
}

function vibrate(ms: number): void {
  if (typeof navigator === "undefined") return;
  const v = (navigator as Navigator & { vibrate?: (n: number) => boolean })
    .vibrate;
  if (typeof v === "function") {
    try {
      v.call(navigator, ms);
    } catch {}
  }
}

export function MobileStepperRail({
  side,
  param,
  max,
  label,
  sublabelTop,
  sublabelBottom,
  invert = false,
  invertFill = false,
  pulseUp = false,
}: Props) {
  // Override takes precedence so the rail's visual demo (the per-song
  // "hear source first" glide) is reflected here too. The increment
  // logic in `apply` below still reads sliderTargets directly via
  // getState(), so taps compute deltas off the engine value, not the
  // demo's transient visual.
  const value = usePerformanceStore(
    (s) => s.sliderDisplayOverride[param] ?? s.sliderTargets[param] ?? 0,
  );
  const setSlider = usePerformanceStore((s) => s.setSlider);
  const meta = SLIDER_META[param];
  const step = meta?.step ?? 0.1;

  // Mirror value into the edge's --fill so the ribbon keeps responding.
  useEffect(() => {
    const edge = document.querySelector<HTMLElement>(`.install-edge-${side}`);
    if (!edge) return;
    const frac = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
    edge.style.setProperty("--fill", (invertFill ? 1 - frac : frac).toString());
  }, [value, max, side, invertFill]);

  const apply = useCallback(
    (delta: number) => {
      const current =
        usePerformanceStore.getState().sliderTargets[param] ?? 0;
      const next = Math.max(0, Math.min(max, current + delta));
      if (next === current) return;
      setSlider(param, next);
      vibrate(8);
    },
    [param, max, setSlider],
  );

  // Press-and-hold orchestration — shared by both buttons via direction.
  const holdStateRef = useRef<{
    startedAt: number;
    interval: ReturnType<typeof setInterval> | null;
    delay: ReturnType<typeof setTimeout> | null;
  }>({ startedAt: 0, interval: null, delay: null });

  const stopHold = useCallback(() => {
    const s = holdStateRef.current;
    if (s.delay) {
      clearTimeout(s.delay);
      s.delay = null;
    }
    if (s.interval) {
      clearInterval(s.interval);
      s.interval = null;
    }
  }, []);

  const startHold = useCallback(
    (delta: number) => {
      stopHold();
      const s = holdStateRef.current;
      s.startedAt = Date.now();
      s.delay = setTimeout(() => {
        s.delay = null;
        s.interval = setInterval(() => {
          const elapsed = Date.now() - s.startedAt;
          // Apply two steps per tick after the accel threshold.
          const reps = elapsed > HOLD_ACCEL_AT_MS ? 2 : 1;
          for (let i = 0; i < reps; i++) apply(delta);
        }, HOLD_INTERVAL_MS);
      }, HOLD_DELAY_MS);
    },
    [apply, stopHold],
  );

  useEffect(() => () => stopHold(), [stopHold]);

  // Top button increases by default (`invert=false`); bottom decreases.
  // For the blend rail (`invert=true`), top decreases and bottom increases
  // because top of rail = LoRA A wins = blend value 0.
  const topDelta = invert ? -step : step;
  const bottomDelta = invert ? step : -step;

  function makeHandlers(delta: number) {
    return {
      onPointerDown: (e: React.PointerEvent) => {
        if (e.button !== 0 && e.pointerType === "mouse") return;
        e.currentTarget.setPointerCapture?.(e.pointerId);
        apply(delta);
        startHold(delta);
      },
      onPointerUp: (e: React.PointerEvent) => {
        e.currentTarget.releasePointerCapture?.(e.pointerId);
        stopHold();
      },
      onPointerCancel: stopHold,
      onPointerLeave: stopHold,
    };
  }

  // The ribbon visual under .install-edge-{side} provides the surface we
  // tap on; we just overlay two large invisible zones (top half = up,
  // bottom half = down) with a small chevron at each rail end as the
  // affordance hint. Reads value-less so the rail stays an instrument,
  // not a metric. Aria-valuenow exposes the value to AT.

  return (
    <div
      className={`stepper-rail stepper-rail--${side}`}
      data-param={param}
      role="slider"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-valuenow={value}
      aria-valuetext={`${label} ${value.toFixed(2)}`}
    >
      <button
        type="button"
        className="stepper-rail-zone stepper-rail-zone--up"
        data-gate={pulseUp ? "up" : undefined}
        aria-label={
          sublabelTop ? `${label} more ${sublabelTop}` : `Increase ${label}`
        }
        {...makeHandlers(topDelta)}
      >
        {sublabelTop && (
          <span className="stepper-rail-zone-label">{sublabelTop}</span>
        )}
        <svg
          className="stepper-rail-chevron stepper-rail-chevron--up"
          viewBox="0 0 24 24"
          width="28"
          height="28"
          aria-hidden="true"
        >
          <path
            d="M5 15 L12 8 L19 15"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      <div className="stepper-rail-readout">
        <div className="stepper-rail-label">{label}</div>
      </div>

      <button
        type="button"
        className="stepper-rail-zone stepper-rail-zone--down"
        aria-label={
          sublabelBottom
            ? `${label} more ${sublabelBottom}`
            : `Decrease ${label}`
        }
        {...makeHandlers(bottomDelta)}
      >
        {sublabelBottom && (
          <span className="stepper-rail-zone-label">{sublabelBottom}</span>
        )}
        <svg
          className="stepper-rail-chevron stepper-rail-chevron--down"
          viewBox="0 0 24 24"
          width="28"
          height="28"
          aria-hidden="true"
        >
          <path
            d="M5 9 L12 16 L19 9"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </div>
  );
}

// Thin wrappers — the left rail drives Remix Strength (denoise); the right
// rail drives the LoRA blend and resolves the paired LoRA names as a
// sublabel. The blend pair is auto-paired by useEdgeLoraBinding; this just
// reads that out for display.

export function MobileRemixStepper() {
  // Mobile equivalent of the desktop top-edge "drag to start" gate.
  // The vertical sideways stepper label isn't visible enough to act as
  // the prompt, so we pulse the up chevron itself as the "do this"
  // affordance while remix hasn't started. Crossing denoise > 0 (via
  // the up chevron, MIDI, or any other path) flips the gate true.
  // Held off until the session is actually ready so the pulse doesn't
  // flash during the loading / connecting phase.
  const remixStarted = usePerformanceStore((s) => s.remixStarted);
  const denoise = usePerformanceStore((s) => s.sliderTargets["denoise"] ?? 0);
  const sessionReady = useSessionStore((s) => s.status === "ready");
  useEffect(() => {
    if (!remixStarted && denoise > 0) {
      usePerformanceStore.getState().setRemixStarted(true);
    }
  }, [remixStarted, denoise]);
  return (
    <MobileStepperRail
      side="left"
      param="denoise"
      max={1.0}
      label="Remix Strength"
      pulseUp={sessionReady && !remixStarted}
    />
  );
}

export function MobileLoraBlendStepper() {
  const enabled = useLoraStore((s) => s.enabled);
  const catalog = useLoraStore((s) => s.catalog);
  const ids = Array.from(enabled);
  while (ids.length < 2) {
    const next = catalog.find((c) => !ids.includes(c.id));
    if (!next) break;
    ids.push(next.id);
  }
  const nameOf = (id: string | undefined) =>
    id ? displayLoraName(id, catalog.find((c) => c.id === id)?.name) : null;
  // With invert=true: top button decreases blend → more LoRA A;
  // bottom button increases blend → more LoRA B. Pair the per-chevron
  // labels accordingly so the user reads "tap up to get more <A>".
  const a = nameOf(ids[0]);
  const b = nameOf(ids[1]);

  return (
    <MobileStepperRail
      side="right"
      param="lora_blend"
      max={1.0}
      label="LoRA Blend"
      sublabelTop={a ?? undefined}
      sublabelBottom={b ?? undefined}
      invert
      invertFill
    />
  );
}
