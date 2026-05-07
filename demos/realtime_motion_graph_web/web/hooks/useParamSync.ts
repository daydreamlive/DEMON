"use client";

import { useEffect } from "react";

import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Continuous param-sync. Mirrors DEMON's app.js Session._sendTick which runs
// on a setInterval (8 ms) and pushes the full slider/seed/dcw raw dict +
// the current playback position EVERY tick — not just on change.
//
// The original cadence matters: the streaming pipeline samples params at
// the start of each generation window. If we only sent on store mutation,
// the engine would drift back to its prior state between user gestures
// (the symptom: "param updates aren't affecting generation"). Continuous
// flow also lets time-keyed curves (e.g. SDE denoise schedules) line up
// against the real playhead in seconds.

const TICK_MS = 33;

export function useParamSync() {
  useEffect(() => {
    let cancelled = false;

    const id = window.setInterval(() => {
      if (cancelled) return;
      const session = useSessionStore.getState();
      if (session.status !== "ready" || !session.remote || !session.player) {
        return;
      }
      if (session.player.ctx?.state === "suspended") return; // paused

      const perf = usePerformanceStore.getState();
      const lora = useLoraStore.getState();

      const raw: Record<string, number | string | boolean> = {
        // seed first so it can never be overwritten by a slider name collision.
        seed: perf.seed,
        ...perf.sliderValues,
      };
      // Per-LoRA strength sliders ride along under lora_str_<id> keys.
      // We prefer perf.sliderValues (smoothed via the tween) and only
      // fall back to lora.strengths when the perf store hasn't seen
      // the LoRA yet (e.g. the user just enabled it but hasn't moved
      // the slider). Without this gate, the LoRA store's instant
      // value used to clobber the smoothed one and smoothing felt
      // broken on LoRA knobs.
      for (const id of lora.enabled) {
        const k = `lora_str_${id}`;
        if (k in raw) continue;
        const v = lora.strengths[id];
        if (typeof v === "number") raw[k] = v;
      }
      // DCW non-numeric controls — server reads raw.get("dcw_enabled") etc.
      raw.dcw_enabled = perf.dcwEnabled;
      raw.dcw_mode = perf.dcwMode;
      raw.dcw_wavelet = perf.dcwWavelet;

      // Playback position is *seconds* (raw audio.positionSec), not a 0..1
      // ratio. The server uses absolute time for curve sampling.
      const playbackSec = session.player.positionSec;
      session.remote.sendParams(raw, playbackSec);
    }, TICK_MS);

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);
}
