"use client";

import { useEffect, useState } from "react";

import { isRecordingSupported } from "@/hooks/useRecording";
import {
  elapsedMs,
  isActive,
  useRecordingStore,
} from "@/store/useRecordingStore";

// Inline record toggle — uses the existing .pause-btn shell so it sits
// next to the play/pause / KBD / KIOSK buttons and reads as part of the
// same control row. Dispatches the same dd:toggle-record event as the
// floating Turntable, so both surfaces stay in sync.

function fmtTime(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${mm}:${ss.toString().padStart(2, "0")}`;
}

export function RecordToggle() {
  const state = useRecordingStore((s) => s.state);

  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (state.kind !== "recording") return;
    const id = window.setInterval(() => setNow(performance.now()), 250);
    return () => window.clearInterval(id);
  }, [state.kind]);

  // Defer render until after mount so SSR + first client render agree.
  // `isRecordingSupported()` checks `window.MediaRecorder`, which is
  // missing on the server — without this gate, the server returns null
  // here and the client returns a button, shifting siblings and
  // tripping React's hydration mismatch detector.
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);
  if (!mounted) return null;
  if (!isRecordingSupported()) return null;

  const active = isActive(state);
  const busy = state.kind === "arming" || state.kind === "finalizing";
  const elapsed = state.kind === "recording" ? elapsedMs(state, now) : 0;

  const labelText =
    state.kind === "recording"
      ? `REC ${fmtTime(elapsed)}`
      : state.kind === "paused"
        ? "PAUSED"
        : state.kind === "arming"
          ? "…"
          : state.kind === "finalizing"
            ? "SAVING"
            : "REC";

  const cls = [
    "pause-btn",
    "rec-btn",
    active ? "active" : "",
    state.kind === "recording" ? "rec-btn--recording" : "",
    busy ? "rec-btn--busy" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <button
      type="button"
      className={cls}
      onClick={(e) => {
        // Prevent bubbling to drawer-handle parent which would close the drawer.
        e.stopPropagation();
        if (busy) return;
        document.dispatchEvent(new CustomEvent("dd:toggle-record"));
      }}
      disabled={busy}
      aria-label={active ? "Stop recording" : "Record audio"}
      title={active ? "Stop recording (R)" : "Record audio (R)"}
    >
      {state.kind === "recording" && (
        <span className="rec-btn-dot" aria-hidden="true" />
      )}
      <span className="rec-btn-label">{labelText}</span>
    </button>
  );
}
