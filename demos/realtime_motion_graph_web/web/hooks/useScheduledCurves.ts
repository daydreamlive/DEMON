"use client";

import { useEffect } from "react";

import { evaluateCurve } from "@/engine/curves/interp";
import { frameScheduler } from "@/engine/scheduler/FrameScheduler";
import {
  isManualOverrideActive,
  usePerformanceStore,
} from "@/store/usePerformanceStore";
import { useLoraStore } from "@/store/useLoraStore";
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
    // Compute-phase tick: writes sliderValues before useRenderLoop's
    // render phase reads them. Bails cheaply when scheduleEnabled is
    // false — it stays registered but does ~zero work each frame.
    // Pre-2026 this was a self-scheduling rAF that woke up every vsync
    // even when the master kill-switch was off; now it shares a single
    // rAF with the rest of the render path.
    const unregister = frameScheduler.register(
      "scheduled-curves",
      () => {
        const curveState = useCurveStore.getState();
        if (!curveState.scheduleEnabled) return;
        const session = useSessionStore.getState();
        const player = session.player;
        const remote = session.remote;
        if (!player || !remote || remote.duration <= 0) return;
        const t = Math.min(
          1,
          Math.max(0, player.positionSec / remote.duration),
        );
        const curves = curveState.curves;
        const setSliderDirect = usePerformanceStore.getState().setSliderDirect;
        const setLoraStrength = useLoraStore.getState().setStrength;
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
          const value = yNorm * max;
          setSliderDirect(param, value);
          // LoRA strength sliders rendered on the desktop edge rails
          // (DesktopEdgeDrag) read from useLoraStore.strengths — the
          // canonical "current strength" store — not from
          // sliderTargets. Mirror the curve write there so the
          // rails track curves the same way LibraryTile does. We
          // deliberately do NOT route through
          // loraStrengthDispatcher.set() because that stamps a
          // manualTouch on the param, which would suppress the curve
          // on the next tick (isManualOverrideActive == true for
          // 500 ms after every stamp).
          if (param.startsWith("lora_str_")) {
            const id = param.slice("lora_str_".length);
            setLoraStrength(id, value);
          }
        }
      },
      { phase: "compute", budgetMs: 1 },
    );
    return () => unregister();
  }, []);
}
