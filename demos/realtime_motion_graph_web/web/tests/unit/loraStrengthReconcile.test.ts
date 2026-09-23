// Regression: "LoRAs not loading upon refresh / session reload."
//
// A saved-session resume restores the persisted LoRA strength into
// useLoraStore.strengths (which the fader UI reads, so it shows 1.0) but
// NOT into perf.sliderValues — the value useParamSync ships to the
// engine each tick. useParamSync prefers sliderValues and only falls
// back to lora.strengths when the lora_str_<id> key is ABSENT, so a
// stale entry seeded for a default-on LoRA at boot pins the engine to
// the old strength: the LoRA is inaudible until a fader drag finally
// writes sliderValues. reconcileEnabledLoraStrengths() closes that gap
// on every session-ready edge.

import { beforeEach, describe, expect, it } from "vitest";

import { reconcileEnabledLoraStrengths } from "@/engine/lora/dispatcher";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";

/**
 * Mirror of the LoRA-strength selection useParamSync performs each tick
 * (hooks/useParamSync.ts): start from the sliderValues snapshot, then
 * fall back to lora.strengths only for enabled keys that are absent.
 * Returns the value the engine would actually receive for `id`.
 */
function engineShippedStrength(id: string): number | undefined {
  const perf = usePerformanceStore.getState();
  const lora = useLoraStore.getState();
  const raw: Record<string, number> = { ...perf.sliderValues };
  for (const lid of lora.enabled) {
    const k = `lora_str_${lid}`;
    if (k in raw) continue;
    const v = lora.strengths[lid];
    if (typeof v === "number") raw[k] = v;
  }
  return raw[`lora_str_${id}`];
}

beforeEach(() => {
  useLoraStore.setState({ enabled: new Set<string>(), strengths: {} });
  usePerformanceStore.setState({ sliderValues: {}, sliderTargets: {} });
});

describe("reconcileEnabledLoraStrengths", () => {
  it("pushes a restored strength to the engine when sliderValues is stale", () => {
    // Boot seed left the default-on LoRA at 0.8 in the engine-facing
    // sliderValues; the restore then wrote 1.0 into lora.strengths only.
    usePerformanceStore.getState().setSliderDirect("lora_str_deathstep", 0.8);
    useLoraStore.setState({
      enabled: new Set(["deathstep"]),
      strengths: { deathstep: 1.0 },
    });

    // Before the fix: the engine keeps streaming the stale 0.8.
    expect(engineShippedStrength("deathstep")).toBe(0.8);

    // On session-ready, the reconcile runs.
    reconcileEnabledLoraStrengths();

    // The engine now receives the restored 1.0 — no fader drag needed.
    expect(usePerformanceStore.getState().sliderValues["lora_str_deathstep"]).toBe(
      1.0,
    );
    expect(engineShippedStrength("deathstep")).toBe(1.0);
  });

  it("is a no-op on a fresh session where the values already agree", () => {
    usePerformanceStore.getState().setSliderDirect("lora_str_synthpop", 0.8);
    useLoraStore.setState({
      enabled: new Set(["synthpop"]),
      strengths: { synthpop: 0.8 },
    });

    reconcileEnabledLoraStrengths();

    expect(usePerformanceStore.getState().sliderValues["lora_str_synthpop"]).toBe(
      0.8,
    );
    expect(engineShippedStrength("synthpop")).toBe(0.8);
  });

  it("only reconciles enabled LoRAs, leaving disabled ones untouched", () => {
    usePerformanceStore.getState().setSliderDirect("lora_str_ambient", 0.5);
    useLoraStore.setState({
      // ambient is NOT enabled — its persisted strength must not be
      // pushed to the engine.
      enabled: new Set(["deathstep"]),
      strengths: { deathstep: 1.0, ambient: 0.9 },
    });

    reconcileEnabledLoraStrengths();

    expect(usePerformanceStore.getState().sliderValues["lora_str_deathstep"]).toBe(
      1.0,
    );
    expect(usePerformanceStore.getState().sliderValues["lora_str_ambient"]).toBe(
      0.5,
    );
  });
});
