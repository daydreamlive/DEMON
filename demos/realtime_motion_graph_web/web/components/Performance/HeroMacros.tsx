"use client";

import { useEffect, useRef, useState } from "react";

import type { StemOverlayKind } from "@/engine/audio/loadFixture";
import { LORA_SLOT_MARKER } from "@/engine/midi/types";
import { useLoraFaderDrag } from "@/hooks/useLoraFaderDrag";
import { useStemPannerDrag } from "@/hooks/useStemPannerDrag";
import { loraDisplayName } from "@/lib/loraLabels";
import { useCurveStore } from "@/store/useCurveStore";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import {
  elapsedMs,
  isActive,
  useRecordingStore,
} from "@/store/useRecordingStore";
import { useSessionStore } from "@/store/useSessionStore";
import { useStemOverlayStore } from "@/store/useStemOverlayStore";
import { LORA_SLIDER_MAX } from "@/types/engine";

import { Knob } from "./Knob";
import { MidiInToggle } from "./MidiInToggle";
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
  const catalog = useLoraStore((s) => s.catalog);
  const enabledIds = Array.from(enabled);
  const loraId = enabledIds[slotIndex] ?? null;
  const value = loraId ? strengths[loraId] ?? 0 : 0;
  const fraction = LORA_SLIDER_MAX > 0
    ? Math.max(0, Math.min(1, value / LORA_SLIDER_MAX))
    : 0;
  const isEmpty = loraId === null;
  const trackRef = useRef<HTMLDivElement | null>(null);
  useLoraFaderDrag(trackRef, slotIndex, !isEmpty);

  const displayLabel = loraId
    ? loraDisplayName(catalog.find((c) => c.id === loraId) ?? { id: loraId })
    : `Style ${slotIndex + 1}`;
  const kbdHint = loraId
    ? `${slotIndex === 0 ? "Z" : "X"} + ▲▼`
    : null;
  return (
    <div
      className={`hero-style-fader${isEmpty ? " hero-style-fader--empty" : ""}`}
      // Right-click → MIDI-learn. Use the slot marker (lora_slot_0 /
      // lora_slot_1) rather than the concrete `lora_str_<id>` so a CC
      // binding survives swapping the LoRA in this slot — matches the
      // default keymap (CC71→lora_slot_0, CC72→lora_slot_1) and the
      // resolveCcParam slot-marker branch in useMidi.ts. Empty slots
      // omit the attribute so right-clicking an unloaded fader is a
      // no-op rather than arming a binding that resolves to null.
      data-param={isEmpty ? undefined : LORA_SLOT_MARKER[slotIndex]}
    >
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
      {kbdHint && <kbd className="hero-style-fader-kbd">{kbdHint}</kbd>}
    </div>
  );
}

const STEM_OVERLAY_MAX = 1.5;
const STEM_LABELS: Record<StemOverlayKind, string> = {
  vocals: "Vocals",
  instruments: "Instr",
};
// Keyboard hold-chord shortcuts (V/I + ▲▼) still work via
// useKeyboardShortcuts.ts — they're documented in the section-header
// tooltip below rather than under each panner.
const STEM_SECTION_TOOLTIP =
  "Vocal and instrumental stems extracted from the source track. Drag a panner right to mix that layer into the model output. Click the layer name to mute or unmute without losing the level. Hold V (vocals) or I (instruments) + ▲▼ to nudge from the keyboard.";

interface HeroStemPannerProps {
  kind: StemOverlayKind;
}
function HeroStemPanner({ kind }: HeroStemPannerProps) {
  const fixture = usePerformanceStore((s) => s.fixture);
  const stems = useCustomTracksStore((s) =>
    fixture ? s.tracks.get(fixture)?.stems : undefined,
  );
  const enabled = useStemOverlayStore((s) => s.enabled[kind]);
  const volume = useStemOverlayStore((s) => s.volumes[kind]);
  const toggle = useStemOverlayStore((s) => s.toggle);
  const stemsReady = Boolean(stems);
  const trackRef = useRef<HTMLDivElement | null>(null);
  useStemPannerDrag(trackRef, kind, stemsReady);
  const displayValue = enabled ? volume : 0;
  const fraction =
    STEM_OVERLAY_MAX > 0
      ? Math.max(0, Math.min(1, displayValue / STEM_OVERLAY_MAX))
      : 0;
  const label = STEM_LABELS[kind];
  return (
    <div
      className={`hero-stem-panner${stemsReady ? "" : " hero-stem-panner--empty"}`}
      data-param={stemsReady ? `stem_${kind}` : undefined}
    >
      {/* Click the label to mute / unmute without losing the level —
          mirrors the original StemOverlayPanel toggle. The drag still
          sets enabled = volume > 0, so the two interactions cooperate.
          Tiny non-wide tooltip — uses the CSS pseudo since the rich
          two-tone HeroMacrosTooltip is reserved for the section
          header above. */}
      <button
        type="button"
        className="hero-stem-panner-label"
        onClick={() => {
          if (stemsReady) toggle(kind);
        }}
        disabled={!stemsReady}
        aria-pressed={enabled}
        data-dd-tooltip={enabled ? "Click to mute layer" : "Click to unmute layer"}
      >
        {label}
      </button>
      <div
        ref={trackRef}
        className="hero-stem-panner-track"
        role="slider"
        aria-label={`${label} overlay volume`}
        aria-orientation="horizontal"
        aria-valuemin={0}
        aria-valuemax={STEM_OVERLAY_MAX}
        aria-valuenow={displayValue}
      >
        <div
          className="hero-stem-panner-fill"
          style={{ width: `${fraction * 100}%` }}
        />
        <div
          className="hero-stem-panner-cap"
          style={{ left: `${fraction * 100}%` }}
        />
      </div>
      <span className="hero-stem-panner-value">{displayValue.toFixed(2)}</span>
    </div>
  );
}

export function HeroMacros() {
  const status = useSessionStore((s) => s.status);
  const started = status !== "idle";
  const curveOpen = useCurveStore((s) => s.overlayOpen);
  const toggleCurve = useCurveStore((s) => s.toggleOverlay);
  const [drawerOpen, setDrawerOpen] = useState(false);
  // Stem section is only visible when the active track was uploaded
  // with a non-"full" sourceMode — i.e. the user opted into stem
  // separation. For "full" mode the backend produces no stems, so
  // the faders would never become live.
  const fixture = usePerformanceStore((s) => s.fixture);
  const sourceMode = useCustomTracksStore((s) =>
    fixture ? s.tracks.get(fixture)?.sourceMode : undefined,
  );
  const stemStatus = useCustomTracksStore((s) =>
    fixture ? s.tracks.get(fixture)?.stemStatus : undefined,
  );
  const stemError = useCustomTracksStore((s) =>
    fixture ? s.tracks.get(fixture)?.stemError : undefined,
  );
  const stemsReady = useCustomTracksStore((s) =>
    Boolean(fixture && s.tracks.get(fixture)?.stems),
  );
  const showStems = !!sourceMode && sourceMode !== "full";
  // Mirrors the status copy from the original StemOverlayPanel — sits
  // italicised just under the section header so the operator sees what
  // the pipeline is doing while the panners are still inert.
  const stemSummary = !showStems
    ? null
    : stemStatus === "processing"
      ? "Ripping stems…"
      : stemStatus === "failed"
        ? stemError || "Stem rip failed"
        : stemsReady
          ? `Inference source: ${sourceMode}`
          : "Stems will load on play";

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
        <div className="hero-macros-group-row">
          <HeroStyleFader slotIndex={0} />
          <HeroStyleFader slotIndex={1} />
        </div>
      </div>
      {showStems && (
        <>
          <div className="hero-macros-divider" aria-hidden="true" />
          <div className="hero-macros-stems">
            <div
              className="hero-macros-group-label"
              data-dd-tooltip={STEM_SECTION_TOOLTIP}
              data-dd-tooltip-wide=""
              data-dd-tooltip-title="Stem Layers"
            >
              Stem Layers
            </div>
            {stemSummary && (
              <div
                className="hero-macros-group-status"
                title={stemError || undefined}
              >
                {stemSummary}
              </div>
            )}
            <div className="hero-stem-panners">
              <HeroStemPanner kind="vocals" />
              <HeroStemPanner kind="instruments" />
            </div>
          </div>
        </>
      )}
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
          data-dd-tooltip="Draw param automation curves against the track timeline. Curves drive denoise / structure / timbre / LoRA strengths / etc. over time so the model performs an arrangement instead of a static patch. Right-click to MIDI-learn."
          data-dd-tooltip-wide=""
          data-dd-tooltip-title="Curve Editor"
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
        <MidiInToggle />
      </div>
    </div>
  );
}
