"use client";

import { useCallback, useEffect, useRef } from "react";

import { listLoras } from "@/engine/lora/listLoras";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { LORA_SLIDER_MAX } from "@/types/engine";

// LoRA library tile. Each row: pill-style enable switch, name, full-width
// strength slider with a colored fill.
//
// Right-click anywhere on the row triggers MIDI learn for the
// `lora_str_<id>` param — useMidi.ts looks for `.lora-row[data-param=…]`.
// The strength is stored in BOTH usePerformanceStore.sliderTargets (so
// the smooth tween sees it + useParamSync sends it) and useLoraStore
// (so subscribed components / RemoteBackend.sendEnableLora pick it up).

interface RowProps {
  id: string;
  name: string;
}

function LoraRow({ id, name }: RowProps) {
  const enabled = useLoraStore((s) => s.enabled.has(id));
  // Read the user's intent from the perf store (instant; doesn't lag
  // behind the smoothing tween) like SliderGroup does, so dragging
  // tracks the cursor without waiting on smoothMs. Falls back to the
  // LoRA store for never-touched LoRAs.
  const strength = usePerformanceStore(
    (s) => s.sliderTargets[`lora_str_${id}`],
  );
  const fallbackStrength = useLoraStore((s) => s.strengths[id] ?? 0);
  const value = typeof strength === "number" ? strength : fallbackStrength;
  const setStrength = useLoraStore((s) => s.setStrength);
  const enable = useLoraStore((s) => s.enable);
  const disable = useLoraStore((s) => s.disable);

  const trackRef = useRef<HTMLDivElement | null>(null);

  function toggle() {
    const remote = useSessionStore.getState().remote;
    if (enabled) {
      disable(id);
      remote?.sendDisableLora(id);
    } else {
      enable(id);
      const s = useLoraStore.getState().strengths[id] ?? 0;
      remote?.sendEnableLora(id, s);
    }
  }

  const setFromClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const t = (clientX - rect.left) / rect.width;
      const v = Math.max(0, Math.min(1, t)) * LORA_SLIDER_MAX;
      // Mirror into both stores so the engine sees it (perf → param-sync)
      // and any non-React subscriber (RemoteBackend.sendEnableLora) can
      // read it via useLoraStore.
      usePerformanceStore.getState().setSlider(`lora_str_${id}`, v);
      setStrength(id, v);
    },
    [id, setStrength],
  );

  useEffect(() => {
    const el = trackRef.current;
    if (!el || !enabled) return;
    let dragging = false;
    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return; // right-click → MIDI learn
      dragging = true;
      el.setPointerCapture(e.pointerId);
      setFromClientX(e.clientX);
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!dragging) return;
      setFromClientX(e.clientX);
    };
    const onPointerUp = (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      el.releasePointerCapture(e.pointerId);
    };
    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", onPointerUp);
    el.addEventListener("pointercancel", onPointerUp);
    return () => {
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", onPointerUp);
      el.removeEventListener("pointercancel", onPointerUp);
    };
  }, [enabled, setFromClientX]);

  const pct = Math.max(0, Math.min(1, value / LORA_SLIDER_MAX)) * 100;

  return (
    <div
      className={`lora-row${enabled ? " enabled" : ""}`}
      data-param={`lora_str_${id}`}
      data-state={enabled ? "enabled" : "disabled"}
    >
      <button
        type="button"
        className="lora-switch"
        role="switch"
        aria-checked={enabled}
        onClick={toggle}
        title={enabled ? "Disable" : "Enable"}
      >
        <span className="lora-switch-thumb" aria-hidden="true" />
      </button>
      <span className="lora-row-name" title={id} onClick={toggle}>
        {name}
      </span>
      <div className="lora-strength">
        <div className="lora-strength-track" ref={trackRef}>
          <div className="lora-strength-fill" style={{ width: `${pct}%` }} />
        </div>
        <span className="lora-strength-value">{value.toFixed(2)}</span>
      </div>
    </div>
  );
}

export function LibraryTile() {
  const catalog = useLoraStore((s) => s.catalog);
  const setCatalog = useLoraStore((s) => s.setCatalog);
  // /api/pod/* requires ?session=<id>; without it the proxy 401s. Wait
  // for the queue to admit us before fetching so we don't spam 401s
  // pre-admission. The fetch fires the moment sessionWsUrl flips truthy
  // and re-runs whenever it changes.
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);

  useEffect(() => {
    if (!sessionWsUrl) return;
    void listLoras().then(setCatalog).catch(() => {});
  }, [setCatalog, sessionWsUrl]);

  if (catalog.length === 0) {
    return (
      <div className="mixer-tile" data-tile="library">
        <div className="mixer-tile-label">Library</div>
        <div className="lora-empty">no LoRAs found</div>
      </div>
    );
  }

  return (
    <div className="mixer-tile" data-tile="library">
      <div className="mixer-tile-label">Library</div>
      <div className="lora-list">
        {catalog.map((entry) => (
          <LoraRow
            key={entry.id}
            id={entry.id}
            name={entry.name ?? entry.id}
          />
        ))}
      </div>
    </div>
  );
}
