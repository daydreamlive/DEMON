"use client";

import { useEffect, useRef, useState } from "react";

import { computePeaks, drawPeaks } from "@/engine/curves/waveformPeaks";
import { frameScheduler } from "@/engine/scheduler/FrameScheduler";
import { useCurveStore } from "@/store/useCurveStore";
import { useSessionStore } from "@/store/useSessionStore";

// Bottom-center scrub strip. Shows the source-track waveform, overlays
// a playhead at the current AudioPlayer.positionSec, lets the operator
// click/drag anywhere to call player.seek(t).
//
// The audio buffer here IS the looped source track — the engine streams
// generated `audio_slice` messages that `patch` into this same buffer.
// Seek is therefore client-only and instant: jumping to t=X plays
// whichever lives there now (generated audio if that region has been
// touched in a prior lap; otherwise the original source). No engine
// protocol message is involved — see audio-worklet.js:106 and
// AudioPlayer.seek():339.
//
// Loop bands (v1):
//   • Shift + drag      → draw a new band (start at down, end at up)
//   • Drag band body    → move the whole band
//   • Drag band edge    → resize that edge
//   • Right-click band  → clear
// All client-side via AudioPlayer.setLoopBand/clearLoopBand, which the
// AudioWorklet honours by wrapping end→start on each pass.

const WAVEFORM_BUCKETS = 640;
const EDGE_PADDING_PX = 4;
const BAND_EDGE_HIT_PX = 7; // hit-zone radius around each band edge
const MIN_BAND_SEC = 0.05;  // 50 ms — below this the band is meaningless

function ensureCanvasSize(canvas: HTMLCanvasElement): {
  w: number;
  h: number;
  dpr: number;
} {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  const targetW = Math.floor(w * dpr);
  const targetH = Math.floor(h * dpr);
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  return { w, h, dpr };
}

type Band = { start: number; end: number };
type DragMode =
  | "seek"
  | "draw-band"
  | "move-band"
  | "resize-band-start"
  | "resize-band-end";

interface DragState {
  mode: DragMode;
  /** Anchor time at the moment of pointerdown — interpretation depends
   *  on mode. seek/draw-band: t at pointerdown. move-band: original
   *  pointerdown t. resize-band-*: original t of the edge being grabbed. */
  anchorT: number;
  /** For move-band: original band at pointerdown so we can recompute
   *  start/end from delta on every move without drift. */
  startBand?: Band;
}

export function WaveformScrubBox() {
  const player = useSessionStore((s) => s.player);
  const curvesOpen = useCurveStore((s) => s.overlayOpen);
  const boxRef = useRef<HTMLDivElement>(null);
  const bgCanvasRef = useRef<HTMLCanvasElement>(null);
  const fgCanvasRef = useRef<HTMLCanvasElement>(null);

  // Active band (seconds). Mirrored to the worklet via player.setLoopBand
  // any time it changes to a complete band. Kept in a ref so the rAF
  // foreground tick can read it without making React state changes
  // invalidate the tick's frameScheduler subscription.
  const [bandState, setBandState] = useState<Band | null>(null);
  const bandRef = useRef<Band | null>(null);
  bandRef.current = bandState;

  const [hasPeaks, setHasPeaks] = useState(false);
  const hasPlayer = player !== null;

  // ── Background canvas: waveform ───────────────────────────────────
  useEffect(() => {
    if (!hasPlayer) {
      setHasPeaks(false);
      return;
    }
    const canvas = bgCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let peaks: Float32Array | null = null;

    const redraw = () => {
      const { w, h, dpr } = ensureCanvasSize(canvas);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (!peaks) return;
      ctx.fillStyle = "rgba(240, 138, 72, 0.28)";
      drawPeaks(ctx, peaks, w, h);
      ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
      drawPeaks(ctx, peaks, w, h);
    };

    const recompute = () => {
      const p = useSessionStore.getState().player;
      if (!p) {
        peaks = null;
        setHasPeaks(false);
      } else {
        const mirror = p.getMirror();
        peaks = computePeaks(mirror, p.channels, WAVEFORM_BUCKETS);
        setHasPeaks(true);
      }
      redraw();
    };

    recompute();

    const ro = new ResizeObserver(redraw);
    ro.observe(canvas);

    const unsubMirror = player.onMirrorChange?.(() => recompute()) ?? (() => {});
    const unsubSession = useSessionStore.subscribe((s, prev) => {
      if (s.player !== prev.player) recompute();
    });

    return () => {
      ro.disconnect();
      unsubMirror();
      unsubSession();
    };
  }, [hasPlayer, player]);

  // ── Foreground canvas: playhead + active band ─────────────────────
  useEffect(() => {
    if (!hasPlayer) return;
    const canvas = fgCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const tick = () => {
      const p = useSessionStore.getState().player;
      if (!p) return;
      const duration = p.duration;
      if (duration <= 0) return;
      const { w, h, dpr } = ensureCanvasSize(canvas);
      const innerW = Math.max(1, w - 2 * EDGE_PADDING_PX);
      const tToX = (t: number) =>
        EDGE_PADDING_PX + (Math.min(1, Math.max(0, t / duration))) * innerW;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // Active band — translucent orange rect.
      const band = bandRef.current;
      if (band) {
        const x0 = tToX(band.start);
        const x1 = tToX(band.end);
        ctx.fillStyle = "rgba(240, 138, 72, 0.18)";
        ctx.fillRect(Math.min(x0, x1), 0, Math.abs(x1 - x0), h);
        // Edge markers — slightly brighter so the resize hit-zones are
        // findable by eye.
        ctx.fillStyle = "rgba(255, 222, 196, 0.55)";
        ctx.fillRect(x0 - 0.5, 0, 1.5, h);
        ctx.fillRect(x1 - 0.5, 0, 1.5, h);
      }

      // Playhead — orange line with halo.
      const x = tToX(p.positionSec);
      ctx.strokeStyle = "rgba(240, 138, 72, 0.28)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x, 2);
      ctx.lineTo(x, h - 2);
      ctx.stroke();
      ctx.strokeStyle = "rgba(255, 222, 196, 0.95)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 2);
      ctx.lineTo(x, h - 2);
      ctx.stroke();
    };

    const unregister = frameScheduler.register("waveform-scrub", tick, {
      phase: "compute",
      budgetMs: 0.2,
    });
    return () => unregister();
  }, [hasPlayer]);

  // Apply / clear band on the worklet whenever the React band state
  // settles to a valid range (or null). Guarded with `typeof === "function"`
  // because a session that started before this code shipped has an
  // AudioPlayer instance whose prototype predates these methods —
  // calling them blind would crash the render and tear the session.
  useEffect(() => {
    if (!hasPlayer || !player) return;
    const setBand = (player as unknown as {
      setLoopBand?: (s: number, e: number) => void;
    }).setLoopBand;
    const clearBand = (player as unknown as {
      clearLoopBand?: () => void;
    }).clearLoopBand;
    if (
      bandState &&
      bandState.end - bandState.start >= MIN_BAND_SEC &&
      bandState.start >= 0
    ) {
      if (typeof setBand === "function") {
        setBand.call(player, bandState.start, bandState.end);
      }
    } else if (bandState === null) {
      if (typeof clearBand === "function") clearBand.call(player);
    }
  }, [bandState, hasPlayer, player]);

  // ── Pointer state machine ─────────────────────────────────────────
  useEffect(() => {
    if (!hasPlayer) return;
    const box = boxRef.current;
    if (!box) return;

    let drag: DragState | null = null;

    const tFromEvent = (e: PointerEvent): number => {
      const p = useSessionStore.getState().player;
      if (!p) return 0;
      const duration = p.duration;
      if (duration <= 0) return 0;
      const rect = box.getBoundingClientRect();
      const innerW = Math.max(1, rect.width - 2 * EDGE_PADDING_PX);
      const x = e.clientX - rect.left - EDGE_PADDING_PX;
      return Math.min(duration, Math.max(0, (x / innerW) * duration));
    };

    /** Pixels per second at the current canvas width. Used to convert
     *  the band-edge hit-zone (defined in pixels) to a tolerance in
     *  seconds at pointerdown. */
    const secPerPx = (): number => {
      const p = useSessionStore.getState().player;
      if (!p || p.duration <= 0) return 0;
      const rect = box.getBoundingClientRect();
      const innerW = Math.max(1, rect.width - 2 * EDGE_PADDING_PX);
      return p.duration / innerW;
    };

    const onDown = (e: PointerEvent) => {
      // Right-click → clear band (if any). Don't preventDefault here —
      // contextmenu handler below also fires and is the place to
      // suppress the browser's native menu.
      if (e.button === 2) return;
      if (e.button !== 0) return;

      const t = tFromEvent(e);
      const band = bandRef.current;
      const tol = secPerPx() * BAND_EDGE_HIT_PX;

      let mode: DragMode = "seek";
      let anchorT = t;
      let startBand: Band | undefined;

      if (e.shiftKey) {
        // Draw a brand-new band from this point. Clears any existing one
        // so the user always sees their freshest gesture.
        mode = "draw-band";
        setBandState({ start: t, end: t });
      } else if (band) {
        // Existing band — check for edge / body hits.
        if (Math.abs(t - band.start) <= tol) {
          mode = "resize-band-start";
        } else if (Math.abs(t - band.end) <= tol) {
          mode = "resize-band-end";
        } else if (t >= band.start && t <= band.end) {
          mode = "move-band";
          anchorT = t;
          startBand = { ...band };
        } else {
          // Plain click outside the band — regular seek. Leave the band
          // in place; the worklet keeps looping it. (Seeks outside the
          // band fall back into it on the next wrap.)
          mode = "seek";
        }
      }

      drag = { mode, anchorT, startBand };
      box.setPointerCapture(e.pointerId);

      if (mode === "seek") {
        const p = useSessionStore.getState().player;
        p?.seek(t);
      }
    };

    const onMove = (e: PointerEvent) => {
      if (!drag) return;
      const t = tFromEvent(e);
      const p = useSessionStore.getState().player;
      const duration = p?.duration ?? 0;

      switch (drag.mode) {
        case "seek":
          p?.seek(t);
          return;
        case "draw-band": {
          const a = drag.anchorT;
          const start = Math.max(0, Math.min(a, t));
          const end = Math.min(duration, Math.max(a, t));
          setBandState({ start, end });
          return;
        }
        case "move-band": {
          const sb = drag.startBand;
          if (!sb) return;
          const len = sb.end - sb.start;
          const delta = t - drag.anchorT;
          let start = sb.start + delta;
          let end = sb.end + delta;
          // Clamp to buffer ends without resizing.
          if (start < 0) {
            end -= start;
            start = 0;
          }
          if (end > duration) {
            start -= end - duration;
            end = duration;
          }
          setBandState({ start, end: start + len });
          return;
        }
        case "resize-band-start": {
          const b = bandRef.current;
          if (!b) return;
          const newStart = Math.min(b.end - MIN_BAND_SEC, Math.max(0, t));
          setBandState({ start: newStart, end: b.end });
          return;
        }
        case "resize-band-end": {
          const b = bandRef.current;
          if (!b) return;
          const newEnd = Math.max(
            b.start + MIN_BAND_SEC,
            Math.min(duration, t),
          );
          setBandState({ start: b.start, end: newEnd });
          return;
        }
      }
    };

    const onUp = (e: PointerEvent) => {
      if (!drag) return;
      // Draw mode finalises on release: if the user just tap-clicked
      // with shift (no drag), kill the band so we don't lock playback
      // to a zero-width sliver.
      if (drag.mode === "draw-band") {
        const b = bandRef.current;
        if (!b || b.end - b.start < MIN_BAND_SEC) {
          setBandState(null);
        }
      }
      drag = null;
      try {
        box.releasePointerCapture(e.pointerId);
      } catch {}
    };

    const onContextMenu = (e: MouseEvent) => {
      // Right-click clears the band entirely. Operators familiar with
      // DAWs expect "right-click loop marker = remove" so we mirror that
      // without spinning up a real context menu in v1.
      if (bandRef.current) {
        e.preventDefault();
        setBandState(null);
      }
    };

    box.addEventListener("pointerdown", onDown);
    box.addEventListener("pointermove", onMove);
    box.addEventListener("pointerup", onUp);
    box.addEventListener("pointercancel", onUp);
    box.addEventListener("contextmenu", onContextMenu);
    return () => {
      box.removeEventListener("pointerdown", onDown);
      box.removeEventListener("pointermove", onMove);
      box.removeEventListener("pointerup", onUp);
      box.removeEventListener("pointercancel", onUp);
      box.removeEventListener("contextmenu", onContextMenu);
    };
  }, [hasPlayer]);

  // Render the DOM as soon as we have a player so the peak-compute
  // effect can find the canvases in the DOM. The strip stays visually
  // hidden (opacity 0) until the first peak-pass lands.
  if (!hasPlayer) return null;

  return (
    <div
      ref={boxRef}
      className="waveform-scrub-box"
      data-curves-open={curvesOpen ? "true" : undefined}
      data-ready={hasPeaks ? "true" : undefined}
      data-has-band={bandState ? "true" : undefined}
      role="slider"
      aria-label="Scrub playhead"
    >
      <canvas ref={bgCanvasRef} className="waveform-scrub-bg" aria-hidden="true" />
      <canvas ref={fgCanvasRef} className="waveform-scrub-fg" aria-hidden="true" />
    </div>
  );
}
