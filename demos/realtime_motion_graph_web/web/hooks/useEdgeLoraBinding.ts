"use client";

import { useEffect } from "react";

import { useIsMobile } from "@/hooks/useIsMobile";
import { loraDisplayName } from "@/lib/loraLabels";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import {
  LORA_DEFAULT_STRENGTH_FRACTION,
  LORA_SLIDER_MAX,
} from "@/types/engine";

// Subscribes to useLoraStore (and on mobile, usePerformanceStore) to keep the
// .install-edge-left / .install-edge-right HUD bars wired up correctly.
//
// Desktop ─────────────────────────────────────────────────────────────────
// Mirrors the first two enabled LoRAs onto the left / right bars.
// Sets data-bar, label text, --fill, and toggles .install-edge-empty so the
// perimeter ribbons reflect LoRA strength visually. DesktopEdgeDrag reads
// data-bar to know which LoRA each side controls and writes back through
// useLoraStore.setStrength.
//
// Mobile ──────────────────────────────────────────────────────────────────
// The right edge becomes a single "LoRA blend" knob: the rail (and the
// edge itself, via --fill) reflect the lora_blend value, and we fan that
// value out into the paired LoRAs' lora_str_<id> via setStrength using a
// linear crossfade. The left edge remains driven by MobileRemixRail
// (denoise) — this hook never touches it on mobile.
//
// Lives outside useRenderLoop because LoRA / blend state moves much less
// often than audio-driven values; subscribing only writes when state
// actually changes and keeps the per-frame loop focused.

const SIDES = ["left", "right"] as const;

// Module-level latch: ensures the auto-enable safety net for mobile blend
// only fires once per page load. Resets naturally on full reload.
let didInitialAutoPair = false;

function applyDesktopBindings(): void {
  const { enabled, strengths, catalog } = useLoraStore.getState();
  const ids = Array.from(enabled);

  for (let i = 0; i < SIDES.length; i++) {
    const side = SIDES[i];
    const id = ids[i] ?? null;
    const edge = document.querySelector<HTMLElement>(`.install-edge-${side}`);
    if (!edge) continue;

    const labelEl = edge.querySelector<HTMLElement>(".install-edge-label");

    if (id === null) {
      edge.classList.add("install-edge-empty");
      delete edge.dataset.bar;
      if (labelEl) labelEl.textContent = "";
      edge.style.setProperty(
        "--fill",
        LORA_DEFAULT_STRENGTH_FRACTION.toString(),
      );
      continue;
    }

    edge.classList.remove("install-edge-empty");
    edge.dataset.bar = `lora_str_${id}`;

    if (labelEl) {
      const entry = catalog.find((e) => e.id === id);
      labelEl.textContent = loraDisplayName(entry ?? { id });
    }

    const strength = strengths[id] ?? 0;
    const frac = LORA_SLIDER_MAX > 0 ? strength / LORA_SLIDER_MAX : 0;
    edge.style.setProperty(
      "--fill",
      Math.max(0, Math.min(1, frac)).toString(),
    );
  }
}

function pickPairAndAutoEnable(): [string | null, string | null] {
  const { enabled, catalog, enable } = useLoraStore.getState();
  const ids = Array.from(enabled);

  if (!didInitialAutoPair && catalog.length > 0) {
    for (const entry of catalog) {
      if (ids.length >= 2) break;
      if (!ids.includes(entry.id)) {
        enable(entry.id);
        ids.push(entry.id);
      }
    }
    didInitialAutoPair = true;
  }

  return [ids[0] ?? null, ids[1] ?? null];
}

function fanOutBlend(
  blend: number,
  idA: string | null,
  idB: string | null,
): void {
  const { setStrength } = useLoraStore.getState();
  const clamped = Math.max(0, Math.min(1, blend));
  if (idA) setStrength(idA, (1 - clamped) * LORA_SLIDER_MAX);
  if (idB) setStrength(idB, clamped * LORA_SLIDER_MAX);
}

function applyMobileRightEdge(
  blend: number,
  idA: string | null,
  idB: string | null,
): void {
  const edge = document.querySelector<HTMLElement>(".install-edge-right");
  if (!edge) return;
  const labelEl = edge.querySelector<HTMLElement>(".install-edge-label");
  const { catalog } = useLoraStore.getState();
  const nameOf = (id: string | null) =>
    id ? loraDisplayName(catalog.find((e) => e.id === id) ?? { id }) : "—";

  edge.dataset.bar = "lora_blend";
  if (idA && idB) {
    edge.classList.remove("install-edge-empty");
    if (labelEl) labelEl.textContent = `${nameOf(idA)} ↔ ${nameOf(idB)}`;
  } else {
    edge.classList.add("install-edge-empty");
    if (labelEl) labelEl.textContent = "LoRA Blend";
  }

  // Top of rail = LoRA A wins → fill 1.0 ; bottom = B wins → fill 0.0 .
  // Same orientation MobileLoraBlendRail uses.
  edge.style.setProperty("--fill", (1 - blend).toString());
}

export function useEdgeLoraBinding(): void {
  const isMobile = useIsMobile();

  useEffect(() => {
    if (!isMobile) {
      // Desktop: original three-edge LoRA binding. left + right each track
      // the first/second enabled LoRA's strength.
      applyDesktopBindings();
      const unsub = useLoraStore.subscribe(applyDesktopBindings);
      return () => unsub();
    }

    // Mobile: right edge is the blend knob; left edge stays owned by
    // MobileRemixRail. Re-applies on either lora-store or perf-store
    // changes (auto-pair + fan-out).
    let lastBlend = -1;
    let lastA: string | null = null;
    let lastB: string | null = null;

    const apply = () => {
      const blend =
        usePerformanceStore.getState().sliderTargets["lora_blend"] ?? 0.5;
      const [a, b] = pickPairAndAutoEnable();

      const pairChanged = a !== lastA || b !== lastB;
      const blendChanged = blend !== lastBlend;
      if (!pairChanged && !blendChanged) return;

      applyMobileRightEdge(blend, a, b);
      fanOutBlend(blend, a, b);
      lastBlend = blend;
      lastA = a;
      lastB = b;
    };

    apply();
    const unsubLora = useLoraStore.subscribe(apply);
    const unsubPerf = usePerformanceStore.subscribe(apply);
    return () => {
      unsubLora();
      unsubPerf();
    };
  }, [isMobile]);
}
