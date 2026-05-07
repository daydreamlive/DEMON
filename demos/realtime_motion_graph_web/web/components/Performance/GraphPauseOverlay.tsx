"use client";

import { useEffect, useRef, useState } from "react";

import { togglePauseAndAudio } from "@/engine/audio/togglePauseAndAudio";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

const FLASH_MS = 700;

// Click-to-pause overlay layered on top of #graph-wrap. The whole canvas
// area is the click target; the centered glyph fades in on each toggle and
// fades out, YouTube-style. A one-shot discovery flash fires the first
// time the session reaches "ready" so first-time users see the affordance
// without having to click first.
export function GraphPauseOverlay() {
  const paused = usePerformanceStore((s) => s.paused);
  const status = useSessionStore((s) => s.status);

  const [flashing, setFlashing] = useState(false);
  // Glyph reflects the state being shown — for clicks that's the
  // post-toggle state, for the discovery flash it's the current (playing)
  // state. Tracked separately from `paused` so the visual doesn't get
  // overwritten by an unrelated state change mid-flash.
  const [glyph, setGlyph] = useState<"play" | "pause">("pause");
  const flashTimerRef = useRef<number | null>(null);
  const discoveryFiredRef = useRef(false);

  function flash(showGlyph: "play" | "pause") {
    setGlyph(showGlyph);
    setFlashing(true);
    if (flashTimerRef.current !== null) {
      window.clearTimeout(flashTimerRef.current);
    }
    flashTimerRef.current = window.setTimeout(() => {
      setFlashing(false);
      flashTimerRef.current = null;
    }, FLASH_MS);
  }

  useEffect(() => {
    return () => {
      if (flashTimerRef.current !== null) {
        window.clearTimeout(flashTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (status !== "ready" || discoveryFiredRef.current) return;
    discoveryFiredRef.current = true;
    // Small delay so the discovery flash lands after the launch transition
    // settles, not during it.
    const t = window.setTimeout(() => flash("pause"), 600);
    return () => window.clearTimeout(t);
  }, [status]);

  function onClick() {
    // YouTube-style: flash the icon for the action the user just took.
    // `paused` here is still the pre-toggle value, so a click while
    // playing (paused=false) flashes ⏸, and a click while paused
    // (paused=true) flashes ▶.
    flash(paused ? "play" : "pause");
    togglePauseAndAudio();
  }

  return (
    <button
      type="button"
      className={`graph-pause-overlay${flashing ? " is-flashing" : ""}`}
      onClick={onClick}
      aria-label={paused ? "Resume" : "Pause"}
    >
      <span className="graph-pause-overlay__glyph" aria-hidden="true">
        {glyph === "pause" ? "⏸" : "▶"}
      </span>
    </button>
  );
}
