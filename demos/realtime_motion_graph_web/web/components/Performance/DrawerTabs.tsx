"use client";

import { useEffect, useState, type ReactElement } from "react";

import { useCurveStore } from "@/store/useCurveStore";
import { useRecordingStore } from "@/store/useRecordingStore";

// Tab strip for the Full Controls panel. Combo segmented-control +
// icon-led tabs — a single bordered hardware shell containing six
// primary cells (top row) + two tool-trigger cells (bottom row),
// each with a small monoline icon above its label. Active body-tab
// cells sit recessed (inset shadow); trigger cells fire side effects
// (curve editor overlay, record start/stop) and badge themselves
// active while their underlying surface is engaged.
//
// Body tabs (top row):
//   CORE   — MIX / TRACK / TIMBRE / FEEDBACK / BASS / TREBLE (knobs)
//   MOD    — SHIFT / N.SHARE / JITTER (knobs) + DCW config
//   VOICE  — the 14 latent channels (V1–V8 + M1–M6, faders)
//   PROMPT — prompts, key, time signature, seed
//   LIB    — LoRAs
//   CONFIG — session controls: track/key/sig, transport, MIDI, prefs
//
// Tool-trigger tabs (bottom row):
//   CURVE EDITOR — toggles the existing ScheduleCurvesOverlay
//   REC          — dispatches dd:toggle-record (same event the
//                  floating turntable uses)

export const DRAWER_TABS = ["core", "mod", "voice", "prompt", "lib", "config"] as const;
export type DrawerTab = (typeof DRAWER_TABS)[number];

const TOOL_TABS = ["curve-editor", "rec"] as const;
type ToolTab = (typeof TOOL_TABS)[number];

const TAB_LABELS: Record<DrawerTab, string> = {
  core: "Core",
  mod: "Mod",
  voice: "Voice",
  prompt: "Prompt",
  lib: "Lib",
  config: "Config",
};

const TOOL_LABELS: Record<ToolTab, string> = {
  "curve-editor": "Curve Editor",
  rec: "Rec",
};

// Monoline 16x16 icons — same vocabulary as the halo menu (1.4px
// stroke, round caps/joins, no fill).
const TAB_ICONS: Record<DrawerTab, ReactElement> = {
  core: (
    <>
      <circle cx="8" cy="8" r="5.2" />
      <line x1="8" y1="3.2" x2="8" y2="5.6" />
    </>
  ),
  mod: <path d="M2 8 Q 4.5 3.5 7 8 T 12 8 T 14 8" />,
  voice: (
    <>
      <line x1="4" y1="2.5" x2="4" y2="13.5" />
      <line x1="8" y1="2.5" x2="8" y2="13.5" />
      <line x1="12" y1="2.5" x2="12" y2="13.5" />
      <rect x="2.5" y="6" width="3" height="2" rx="0.4" />
      <rect x="6.5" y="9.5" width="3" height="2" rx="0.4" />
      <rect x="10.5" y="4.5" width="3" height="2" rx="0.4" />
    </>
  ),
  prompt: (
    <>
      <path d="M2.5 3.5h11a1 1 0 0 1 1 1v6a1 1 0 0 1-1 1H9.5l-3 2.5v-2.5H2.5a1 1 0 0 1-1-1v-6a1 1 0 0 1 1-1z" />
      <line x1="5" y1="7" x2="11" y2="7" />
      <line x1="5" y1="9.5" x2="9" y2="9.5" />
    </>
  ),
  lib: (
    <>
      <rect x="2" y="4" width="12" height="8" rx="1.2" />
      <circle cx="6" cy="9" r="1.4" />
      <circle cx="10" cy="9" r="1.4" />
    </>
  ),
  config: (
    <>
      <circle cx="8" cy="8" r="2.2" />
      <path d="M8 1.8v1.6 M8 12.6v1.6 M14.2 8h-1.6 M3.4 8H1.8 M12.4 3.6l-1.1 1.1 M4.7 11.3l-1.1 1.1 M12.4 12.4l-1.1-1.1 M4.7 4.7L3.6 3.6" />
    </>
  ),
};

const TOOL_ICONS: Record<ToolTab, ReactElement> = {
  // Curve editor — a sinuous bezier with two control points
  "curve-editor": (
    <>
      <path d="M2 12 C 5 12, 5 4, 8 4 S 11 12, 14 12" />
      <circle cx="2" cy="12" r="1.1" fill="currentColor" stroke="none" />
      <circle cx="14" cy="12" r="1.1" fill="currentColor" stroke="none" />
    </>
  ),
  // REC — filled circle (universal record glyph)
  rec: <circle cx="8" cy="8" r="3.6" fill="currentColor" stroke="none" />,
};

interface Props {
  active: DrawerTab;
  onChange: (tab: DrawerTab) => void;
}

export function DrawerTabs({ active, onChange }: Props) {
  // Subscribe to overlay + recording state so the tool tabs can
  // active-style themselves while their underlying surface is engaged.
  const curveOpen = useCurveStore((s) => s.overlayOpen);
  const toggleCurve = useCurveStore((s) => s.toggleOverlay);
  const [recState, setRecState] = useState<string>("idle");
  useEffect(() => {
    const update = () => {
      const s = useRecordingStore.getState().state;
      setRecState(s.kind);
    };
    update();
    const unsub = useRecordingStore.subscribe(update);
    return unsub;
  }, []);
  const isRecording = recState === "recording" || recState === "arming";

  return (
    <div className="drawer-tabs" role="tablist" aria-label="Full controls">
      <div className="drawer-tabs-row drawer-tabs-row--primary">
        {DRAWER_TABS.map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            aria-selected={active === t}
            className={`drawer-tab${active === t ? " drawer-tab--active" : ""}`}
            onClick={() => onChange(t)}
          >
            <svg
              className="drawer-tab-icon"
              viewBox="0 0 16 16"
              width={16}
              height={16}
              fill="none"
              stroke="currentColor"
              strokeWidth={1.4}
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              {TAB_ICONS[t]}
            </svg>
            <span className="drawer-tab-label">{TAB_LABELS[t]}</span>
          </button>
        ))}
      </div>
      <div className="drawer-tabs-row drawer-tabs-row--tools">
        <button
          type="button"
          className={`drawer-tab drawer-tab--tool${curveOpen ? " drawer-tab--active" : ""}`}
          onClick={() => toggleCurve()}
          aria-pressed={curveOpen}
        >
          <svg
            className="drawer-tab-icon"
            viewBox="0 0 16 16"
            width={16}
            height={16}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.4}
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            {TOOL_ICONS["curve-editor"]}
          </svg>
          <span className="drawer-tab-label">{TOOL_LABELS["curve-editor"]}</span>
        </button>
        <button
          type="button"
          className={`drawer-tab drawer-tab--tool drawer-tab--rec${isRecording ? " drawer-tab--active drawer-tab--rec-active" : ""}`}
          onClick={() => document.dispatchEvent(new Event("dd:toggle-record"))}
          aria-pressed={isRecording}
        >
          <svg
            className="drawer-tab-icon drawer-tab-icon--rec"
            viewBox="0 0 16 16"
            width={16}
            height={16}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.4}
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            {TOOL_ICONS.rec}
          </svg>
          <span className="drawer-tab-label">
            {isRecording ? "Stop" : TOOL_LABELS.rec}
          </span>
        </button>
      </div>
    </div>
  );
}

export function useDrawerTab(initial: DrawerTab = "core") {
  return useState<DrawerTab>(initial);
}
