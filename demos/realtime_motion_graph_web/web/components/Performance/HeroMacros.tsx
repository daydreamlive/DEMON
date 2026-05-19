"use client";

import { useEffect, useRef, useState } from "react";

import { useLoraFaderDrag } from "@/hooks/useLoraFaderDrag";
import { useCurveStore } from "@/store/useCurveStore";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import {
  elapsedMs,
  isActive,
  useRecordingStore,
} from "@/store/useRecordingStore";
import { useSessionStore } from "@/store/useSessionStore";
import { LORA_SLIDER_MAX } from "@/types/engine";

import { Knob } from "./Knob";
import { SeedKnob } from "./SeedKnob";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";

// Bottom-center bay. Three zones, left to right:
//   1. Macros — denoise / structure / feedback knobs + seed randomizer.
//   2. Style faders — the two LoRA strengths inline (was the left-edge
//      StylePanel; consolidated here so the canvas reads as one unit).
//   3. Tools — Record / Curve Editor / Full Controls stack.
//
// Visibility:
//   - Hidden when the session is idle.
//   - Knobs + style faders hide when the drawer is open (CORE + LIB
//     tabs cover the same params). Tools stay reachable.
//   - Hidden below 768 px (mobile gets LiteControls).

const HERO_PARAMS = ["denoise", "hint_strength", "timbre_strength"] as const;

function CurveIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      width={12}
      height={12}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.4}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M2 12 C 5 12, 5 4, 8 4 S 11 12, 14 12" />
    </svg>
  );
}

function fmtTime(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${mm}:${ss.toString().padStart(2, "0")}`;
}

function RecordPill() {
  const state = useRecordingStore((s) => s.state);
  const active = isActive(state);
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (state.kind !== "recording") return;
    const id = window.setInterval(() => setNow(performance.now()), 250);
    return () => window.clearInterval(id);
  }, [state.kind]);
  const elapsed = state.kind === "recording" ? elapsedMs(state, now) : 0;
  const label =
    state.kind === "recording"
      ? fmtTime(elapsed)
      : state.kind === "arming"
        ? "..."
        : state.kind === "finalizing"
          ? "Saving"
          : "Record";
  const onClick = () => {
    if (state.kind === "arming" || state.kind === "finalizing") return;
    document.dispatchEvent(new CustomEvent("dd:toggle-record"));
  };
  return (
    <button
      type="button"
      className={`hero-macros-tool hero-macros-rec${active ? " hero-macros-rec--active" : ""}`}
      onClick={onClick}
      aria-pressed={active}
      aria-label={active ? "Stop recording" : "Start recording"}
    >
      <span className="hero-macros-rec-dot" aria-hidden="true" />
      <span className="hero-macros-tool-label">{label}</span>
    </button>
  );
}

interface HeroStyleFaderProps {
  slotIndex: 0 | 1;
}
function HeroStyleFader({ slotIndex }: HeroStyleFaderProps) {
  const strengths = useLoraStore((s) => s.strengths);
  const enabled = useLoraStore((s) => s.enabled);
  const enabledIds = Array.from(enabled);
  const loraId = enabledIds[slotIndex] ?? null;
  const value = loraId ? strengths[loraId] ?? 0 : 0;
  const fraction = LORA_SLIDER_MAX > 0
    ? Math.max(0, Math.min(1, value / LORA_SLIDER_MAX))
    : 0;
  const isEmpty = loraId === null;
  const trackRef = useRef<HTMLDivElement | null>(null);
  useLoraFaderDrag(trackRef, slotIndex, !isEmpty);

  const displayLabel = loraId ? labelFor(loraId) : `Style ${slotIndex + 1}`;
  return (
    <div className={`hero-style-fader${isEmpty ? " hero-style-fader--empty" : ""}`}>
      <div className="hero-style-fader-label" title={displayLabel}>
        {displayLabel}
      </div>
      <div
        ref={trackRef}
        className="hero-style-fader-track"
        role="slider"
        aria-label={displayLabel}
        aria-valuemin={0}
        aria-valuemax={LORA_SLIDER_MAX}
        aria-valuenow={value}
      >
        <div
          className="hero-style-fader-fill"
          style={{ height: `${fraction * 100}%` }}
        />
        <div
          className="hero-style-fader-cap"
          style={{ bottom: `${fraction * 100}%` }}
        />
      </div>
      <div className="hero-style-fader-value">{value.toFixed(2)}</div>
    </div>
  );
}

// LoRA id → short human label; mirrors the helper in StylePanel.
function labelFor(loraId: string): string {
  return loraId.replace(/^lora_/, "").slice(0, 8).toUpperCase();
}

export function HeroMacros() {
  const status = useSessionStore((s) => s.status);
  const started = status !== "idle";
  const curveOpen = useCurveStore((s) => s.overlayOpen);
  const toggleCurve = useCurveStore((s) => s.toggleOverlay);
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Clear the per-song remix gate when the user moves the bay's
  // DENOISE knob above zero. Mirrors MobileRemixStepper's behavior so
  // touching the bottom bay counts as "engaging the remix" — the
  // top-edge RemixHint is no longer rendered, so the bay is the user-
  // facing gate-clearing affordance on desktop.
  const remixStarted = usePerformanceStore((s) => s.remixStarted);
  const denoise = usePerformanceStore((s) => s.sliderTargets["denoise"] ?? 0);
  useEffect(() => {
    if (!remixStarted && denoise > 0) {
      usePerformanceStore.getState().setRemixStarted(true);
    }
  }, [remixStarted, denoise]);

  // Mirror body.drawer-open so the toggle label/caret flip with the
  // drawer state. The drawer is the source of truth.
  useEffect(() => {
    const sync = () => {
      setDrawerOpen(document.body.classList.contains("drawer-open"));
    };
    const obs = new MutationObserver(sync);
    obs.observe(document.body, { attributes: true, attributeFilter: ["class"] });
    sync();
    return () => obs.disconnect();
  }, []);

  if (!started) return null;
  return (
    <div
      className={`hero-macros${drawerOpen ? " hero-macros--drawer-open" : ""}${curveOpen ? " hero-macros--curve-open" : ""}`}
      data-hero-macros
    >
      <div className="hero-macros-knobs">
        {HERO_PARAMS.map((p) => (
          <Knob
            key={p}
            param={p}
            label={defaultLabelFor(p)}
            kbd={kbdHintFor(p)}
          />
        ))}
        <SeedKnob />
      </div>
      <div className="hero-macros-divider" aria-hidden="true" />
      <div className="hero-macros-styles">
        <HeroStyleFader slotIndex={0} />
        <HeroStyleFader slotIndex={1} />
      </div>
      <div className="hero-macros-divider" aria-hidden="true" />
      <div className="hero-macros-tools">
        <RecordPill />
        <button
          type="button"
          className={`hero-macros-tool${curveOpen ? " hero-macros-tool--active" : ""}`}
          onClick={() => toggleCurve()}
          aria-pressed={curveOpen}
          aria-label="Toggle curve editor"
          data-midi-learn="schedule_curves_toggle"
          data-dd-tooltip="Open the curve scheduler — draw param automation against the track (right-click to MIDI-learn)"
        >
          <CurveIcon />
          <span className="hero-macros-tool-label">Curve Editor</span>
        </button>
        <button
          type="button"
          className="hero-macros-toggle"
          onClick={() => document.dispatchEvent(new Event("dd:toggle-drawer"))}
          aria-label={drawerOpen ? "Close Full Controls" : "Open Full Controls"}
          aria-expanded={drawerOpen}
        >
          <span className="hero-macros-toggle-label">
            {drawerOpen ? "Simple Controls" : "Full Controls"}
          </span>
          <span className="hero-macros-toggle-caret" aria-hidden="true">
            {drawerOpen ? "◂" : "▸"}
          </span>
        </button>
      </div>
    </div>
  );
}
