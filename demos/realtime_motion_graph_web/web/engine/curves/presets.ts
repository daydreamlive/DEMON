// Musical preset curves — apply via the tab right-click menu.
// Each preset is a *function* of the current track context (BPM,
// duration), so "sine 1/bar" actually means one full cycle per bar at
// the detected tempo, rather than a fixed-shape curve that doesn't
// know about the music.
//
// Returns a fresh CurvePoint[] anchored to x=0..1. The store's
// setCurvePoints will sort and pin endpoints defensively.

import type { CurvePoint } from "@/types/curves";

export type CurvePresetId =
  | "ramp_up"
  | "ramp_down"
  | "sine_1_bar"
  | "sine_4_bar"
  | "pulse_downbeat"
  | "flat_low"
  | "flat_mid"
  | "flat_high";

export const PRESET_LABEL: Record<CurvePresetId, string> = {
  ramp_up: "Ramp up",
  ramp_down: "Ramp down",
  sine_1_bar: "Sine 1/bar",
  sine_4_bar: "Sine 4/bar",
  pulse_downbeat: "Pulse on downbeats",
  flat_low: "Flat low",
  flat_mid: "Flat mid",
  flat_high: "Flat high",
};

export interface PresetContext {
  /** Track duration in seconds; from RemoteBackend.duration. */
  durationSec: number;
  /** Server-detected BPM; null if unknown — preset falls back to a
   *  fixed musically-vague rate. */
  bpm: number | null;
  /** Beats per bar. Default 4. */
  beatsPerBar?: number;
}

export function buildPreset(
  preset: CurvePresetId,
  ctx: PresetContext,
): CurvePoint[] {
  const beatsPerBar = ctx.beatsPerBar ?? 4;
  const bpm = ctx.bpm ?? 120;
  const beatSec = 60 / bpm;
  const barSec = beatSec * beatsPerBar;
  const totalBars = Math.max(1, ctx.durationSec / barSec);

  switch (preset) {
    case "ramp_up":
      return [
        { x: 0, y: 0, mode: "smooth" },
        { x: 1, y: 1, mode: "smooth" },
      ];

    case "ramp_down":
      return [
        { x: 0, y: 1, mode: "smooth" },
        { x: 1, y: 0, mode: "smooth" },
      ];

    case "flat_low":
      return [
        { x: 0, y: 0.15, mode: "smooth" },
        { x: 1, y: 0.15, mode: "smooth" },
      ];

    case "flat_mid":
      return [
        { x: 0, y: 0.5, mode: "smooth" },
        { x: 1, y: 0.5, mode: "smooth" },
      ];

    case "flat_high":
      return [
        { x: 0, y: 0.85, mode: "smooth" },
        { x: 1, y: 0.85, mode: "smooth" },
      ];

    case "sine_1_bar":
    case "sine_4_bar": {
      const cycles = preset === "sine_1_bar" ? totalBars : totalBars / 4;
      // Sample two control points per cycle (peak + trough) so the
      // Catmull-Rom render gets a recognisable wave without exploding
      // the point count for long tracks.
      const pointsPerCycle = 4;
      const totalPoints = Math.max(2, Math.round(cycles * pointsPerCycle));
      const out: CurvePoint[] = [];
      for (let i = 0; i <= totalPoints; i++) {
        const x = i / totalPoints;
        const phase = x * cycles * 2 * Math.PI;
        const y = 0.5 + 0.45 * Math.sin(phase);
        out.push({ x, y, mode: "smooth" });
      }
      return out;
    }

    case "pulse_downbeat": {
      // Step-up at every downbeat (start of each bar), holds for
      // ~25% of the bar, then drops back. Useful for accent automation.
      const pulses = Math.max(1, Math.floor(totalBars));
      const out: CurvePoint[] = [{ x: 0, y: 0.1, mode: "step" }];
      for (let i = 0; i < pulses; i++) {
        const downbeatX = i / pulses;
        const offX = Math.min(1, downbeatX + 0.25 / pulses);
        out.push({ x: downbeatX, y: 0.9, mode: "step" });
        if (offX < 1) out.push({ x: offX, y: 0.1, mode: "step" });
      }
      out.push({ x: 1, y: 0.1, mode: "step" });
      return out;
    }
  }
}
