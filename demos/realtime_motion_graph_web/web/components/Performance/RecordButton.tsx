"use client";

import { useEffect, useRef, useState } from "react";

import { useOneShotTooltip } from "@/hooks/useOneShotTooltip";
import { isRecordingSupported } from "@/hooks/useRecording";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import {
  elapsedMs,
  isActive,
  useRecordingStore,
} from "@/store/useRecordingStore";
import { useSessionStore } from "@/store/useSessionStore";

function fmtTime(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${mm}:${ss.toString().padStart(2, "0")}`;
}

export function RecordButton() {
  const state = useRecordingStore((s) => s.state);
  const warning = useRecordingStore((s) => s.warning);
  const status = useSessionStore((s) => s.status);
  const kiosk = usePerformanceStore((s) => s.kiosk);

  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (state.kind !== "recording") return;
    const id = window.setInterval(() => setNow(performance.now()), 250);
    return () => window.clearInterval(id);
  }, [state.kind]);

  const [burst, setBurst] = useState(0);
  const prevKind = useRef(state.kind);
  useEffect(() => {
    if (prevKind.current === "finalizing" && state.kind === "preview") {
      setBurst((n) => n + 1);
    }
    prevKind.current = state.kind;
  }, [state.kind]);

  // Derive everything we need from state UP FRONT so the one-shot
  // tooltip hook below stays above the early returns (Rules of Hooks
  // — hooks must run in the same order on every render).
  const active = isActive(state);
  const busy = state.kind === "arming" || state.kind === "finalizing";
  const elapsed = state.kind === "recording" ? elapsedMs(state, now) : 0;
  const label =
    state.kind === "recording"
      ? `Stop recording (${fmtTime(elapsed)})`
      : state.kind === "paused"
        ? "Resume recording"
        : state.kind === "arming"
          ? "Starting…"
          : state.kind === "finalizing"
            ? "Saving…"
            : "Record (R)";
  // One-shot tooltip — shows the first time, never again. The
  // permanent "REC" caption beneath the disc carries the affordance
  // once the user has seen the tooltip; repeated tooltips become
  // noise during active performance.
  const tipProps = useOneShotTooltip("record", label);

  const supported = isRecordingSupported();
  if (!supported) return null;
  if (status === "idle" || kiosk) return null;

  const onClick = () => {
    if (busy) return;
    document.dispatchEvent(new CustomEvent("dd:toggle-record"));
  };

  const cls = [
    "turntable",
    active ? "turntable--active" : "",
    state.kind === "recording" ? "turntable--recording" : "",
    state.kind === "paused" ? "turntable--paused" : "",
    busy ? "turntable--busy" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="turntable-wrap">
      {warning && (
        <div className="rec-warning" role="status" aria-live="polite">
          {warning}
        </div>
      )}
      <button
      type="button"
      className={cls}
      onClick={onClick}
      disabled={busy}
      aria-label={label}
      // Custom tooltip pattern — see [data-dd-tooltip] in globals.css.
      // One-shot via useOneShotTooltip: tipProps supplies the tooltip
      // attr only the first time.
      {...tipProps}
    >
      {/* Disc — platter, audio-reactive grooves, label, spindle. The whole
          disc rotates while recording (CSS animation on .turntable-disc). */}
      <span className="turntable-disc">
        <span className="turntable-platter" />
        <canvas className="turntable-grooves" aria-hidden="true" />
        <span className="turntable-label" />
        <span className="turntable-spindle" />
      </span>
      {/* Small caption beneath the disc — tactful word label so first-time
          visitors don't have to hover to learn what the turntable does.
          Same mono+caps+wide-tracking treatment used everywhere else. */}
      <span className="turntable-caption" aria-hidden="true">REC</span>

      {/* Tonearm — sits to the upper-right of the disc. Idle: parked,
          rotated up-right; recording: dropped onto the outer groove. The
          rotation is purely CSS, transitioning over 700ms with a weighted
          ease so it reads mechanical. */}
      <svg
        className="turntable-tonearm"
        viewBox="0 0 110 84"
        preserveAspectRatio="xMidYMid meet"
        aria-hidden="true"
      >
        <g className="turntable-tonearm-arm">
          <line
            x1="93"
            y1="14"
            x2="50"
            y2="22"
            stroke="#9ba0a6"
            strokeWidth="2.4"
            strokeLinecap="round"
          />
          <rect
            x="46"
            y="19"
            width="10"
            height="6"
            rx="1.2"
            fill="#23232c"
            stroke="#5a5d63"
            strokeWidth="0.8"
          />
          <circle cx="48" cy="24" r="1.6" fill="#e84f3d" />
          <circle
            cx="95"
            cy="12"
            r="6"
            fill="#1a1a22"
            stroke="#5a5d63"
            strokeWidth="1"
          />
          <circle cx="95" cy="12" r="2" fill="#3a3d42" />
        </g>
      </svg>

      {state.kind === "recording" && (
        <span className="turntable-time" aria-hidden="true">
          {fmtTime(elapsed)}
        </span>
      )}

      {burst > 0 && (
        <span
          key={burst}
          className="turntable-confetti"
          aria-hidden="true"
        >
          {Array.from({ length: 12 }).map((_, i) => (
            <span
              key={i}
              className={`turntable-confetti-dot turntable-confetti-dot--${i}`}
            />
          ))}
        </span>
      )}
    </button>
    </div>
  );
}
