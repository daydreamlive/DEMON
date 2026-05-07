"use client";

import { useEffect } from "react";

import { useLoraStore } from "@/store/useLoraStore";
import {
  LORA_DEFAULT_STRENGTH_FRACTION,
  LORA_SLIDER_MAX,
} from "@/types/engine";

// Subscribes to useLoraStore and mirrors the first two enabled LoRAs onto
// the .install-edge-left / .install-edge-right HUD bars. Sets data-bar,
// label text, --fill, and toggles .install-edge-empty so the perimeter
// ribbons reflect LoRA strength visually. DesktopEdgeDrag reads data-bar
// to know which LoRA each side controls.
//
// Lives outside useRenderLoop because LoRA state moves much less often
// than audio-driven values; subscribing only writes when state actually
// changes and keeps the per-frame loop focused.

const SIDES = ["left", "right"] as const;

function applyBindings() {
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
      // Show the empty-state slider at the shared default fill so the
      // ribbon canvas (which reads --fill directly) renders at a useful
      // length and the hint's value-driven head position lands at the
      // matching point on the bar.
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
      labelEl.textContent = entry?.name ?? id;
    }

    const strength = strengths[id] ?? 0;
    const frac =
      LORA_SLIDER_MAX > 0 ? strength / LORA_SLIDER_MAX : 0;
    edge.style.setProperty(
      "--fill",
      Math.max(0, Math.min(1, frac)).toString(),
    );
  }
}

export function useEdgeLoraBinding(): void {
  useEffect(() => {
    // Apply once on mount in case the store already has state.
    applyBindings();
    // Re-apply on every store change. The selector returns the same
    // object reference each time so we just use the global subscribe API.
    const unsub = useLoraStore.subscribe(() => {
      applyBindings();
    });
    return () => unsub();
  }, []);
}
