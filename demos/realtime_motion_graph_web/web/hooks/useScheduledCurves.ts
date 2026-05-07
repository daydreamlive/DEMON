"use client";

import { useEffect } from "react";

import { evaluateCurve } from "@/engine/curves/interp";
import {
  isManualOverrideActive,
  usePerformanceStore,
} from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { useCurveStore } from "@/store/useCurveStore";
import { LORA_SLIDER_MAX, SLIDER_META } from "@/types/engine";

// Per-frame application of curve-scheduled param values.
//
// Each rAF tick:
//   1. Read the active session's playhead in seconds.
//   2. For every curve where `enabled === true`:
//      - Map playhead seconds → t ∈ [0,1] over the track duration.
//      - Evaluate the curve at t (Catmull-Rom / linear / step).
//      - Map y ∈ [0,1] → param's [0, max] via SLIDER_META.
//      - Write directly into sliderValues + sliderTargets via
//        usePerformanceStore.setSliderDirect.
//   3. If the user manually touched a slider in the last 500 ms, the
//      curve yields for that param — manual override briefly wins,
//      then the curve resumes on the next tick past the window.
//
// Subscribes to neither store via reactive selectors — reads via
// .getState() every frame. This is intentional: curves change at
// rAF cadence, not at React-render cadence; reactive subscriptions
// would just churn renders without any visible effect.
export function useScheduledCurves(): void {
  useEffect(() => {
    let raf = 0;
    let cancelled = false;

    const tick = () => {
      if (cancelled) return;
      const curveState = useCurveStore.getState();
      // Master kill-switch: when disabled, no curve drives anything,
      // regardless of per-curve enabled flags. The user's drawings are
      // preserved; they just don't apply.
      if (!curveState.scheduleEnabled) {
        raf = requestAnimationFrame(tick);
        return;
      }
      const session = useSessionStore.getState();
      const player = session.player;
      const remote = session.remote;
      // Need both: player gives us positionSec, remote gives duration
      // (audio length the engine is generating against). Skip cleanly
      // when not playing.
      if (player && remote && remote.duration > 0) {
        const t = Math.min(
          1,
          Math.max(0, player.positionSec / remote.duration),
        );
        const curves = curveState.curves;
        const setSliderDirect =
          usePerformanceStore.getState().setSliderDirect;
        for (const param of Object.keys(curves)) {
          const c = curves[param];
          if (!c.enabled) continue;
          if (isManualOverrideActive(param)) continue;
          const yNorm = evaluateCurve(c.points, t);
          // LoRA strength params (lora_str_<id>) aren't in SLIDER_META;
          // their range is fixed by LORA_SLIDER_MAX. Everything else
          // looks up its own max from SLIDER_META.
          const max = param.startsWith("lora_str_")
            ? LORA_SLIDER_MAX
            : (SLIDER_META[param]?.max ?? 1.0);
          setSliderDirect(param, yNorm * max);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, []);
}
