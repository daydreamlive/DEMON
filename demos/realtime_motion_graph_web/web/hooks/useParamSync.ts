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

const TICK_MS = 8;

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
      // lora_blend is a UI-only knob (useEdgeLoraBinding fans it out into
      // the paired lora_str_<id> values). The engine doesn't know it.
      delete raw.lora_blend;
      // timbre_strength rides its own dedicated WS message
      // (useTimbreSync → sendSetTimbreStrength) so the server can blend
      // the cached cond_silence/cond_self pair on the spot. The params
      // path doesn't recognize it.
      delete raw.timbre_strength;
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
      // The numeric fader values (dcw_scaler / dcw_high_scaler /
      // dcw_mult_blend / dcw_mag_phase / dcw_soft_thresh) ride through
      // the sliderValues spread above; pipeline.py's
      // ``_build_dcw_advanced`` reads the same keys verbatim.
      raw.dcw_enabled = perf.dcwEnabled;
      raw.dcw_mode = perf.dcwMode;
      raw.dcw_wavelet = perf.dcwWavelet;
      // RCFG mode (off/full/initialize/self). pipeline.py reads
      // raw.rcfg_mode and pairs it with raw.guidance_scale / raw.cfg_rescale
      // (already in the sliderValues spread above) to build the engine's
      // guidance_curve / cfg_rescale_curve.
      raw.rcfg_mode = perf.rcfgMode;

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
