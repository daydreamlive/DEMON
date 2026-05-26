"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { computePeaks } from "@/engine/curves/waveformPeaks";
import type { DecodedFixture } from "@/engine/audio/loadFixture";

// Interactive trim step that runs between decodeAudioFile and the
// AlmostReadyDialog for every upload. The user picks a window of up
// to ``capS`` seconds from the decoded waveform; only the slice is
// passed downstream. Replaces the prior silent head-trim behaviour
// (decodeAudioFile used to clip anything over 240 s and just tell the
// user "we trimmed it" after the fact).
//
// Why always-on (not just for over-cap uploads):
//   - A 5-minute upload obviously needs trimming.
//   - A 30-second upload often ALSO benefits from trimming — pick
//     the chorus / drop instead of feeding the intro.
//   - Consistent UX: there's one trim step, not two flows the user
//     has to memorise.
//
// The cap (``capS``) comes from config — see
// ``engine.max_source_duration_s`` in lib/config.ts.

export interface WaveformTrimDialogProps {
  decoded: DecodedFixture;
  fileName: string;
  /** Max length of the selected window, in seconds. The dialog
   *  refuses to let the user expand the window past this. */
  capS: number;
  /** Minimum selectable window. Set by the smallest TRT profile +
   *  pool alignment; well below this and the slice can't even fill
   *  one engine forward pass. */
  minS?: number;
  onConfirm: (startS: number, endS: number) => void;
  onCancel: () => void;
}

// Match AlmostReadyDialog's portal-mount tick so the overlay's
// transition (opacity, transform) gets a chance to animate in.
const MIN_S_DEFAULT = 3;

function fmtMMSS(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function WaveformTrimDialog({
  decoded,
  fileName,
  capS,
  minS = MIN_S_DEFAULT,
  onConfirm,
  onCancel,
}: WaveformTrimDialogProps) {
  const durationS = decoded.frames / decoded.sampleRate;
  const initialEnd = Math.min(capS, durationS);
  const [startS, setStartS] = useState(0);
  const [endS, setEndS] = useState(initialEnd);
  const [mounted, setMounted] = useState(false);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const peaksRef = useRef<Float32Array | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Measure container width — drives both canvas pixel sizing and
  // the px↔seconds mapping. ResizeObserver covers viewport changes
  // (mobile rotate, drawer resize) without redrawing the waveform
  // unless the width actually changes.
  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => setContainerWidth(el.clientWidth);
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Compute peaks once per (decoded, width) — the file doesn't change
  // mid-dialog but width can on resize. ``computePeaks`` returns
  // alternating min/max pairs; the canvas draw below treats them as
  // vertical bars per pixel column.
  useEffect(() => {
    if (!containerWidth || !canvasRef.current) return;
    const dpr = window.devicePixelRatio || 1;
    const cssW = containerWidth;
    const cssH = 120;
    const canvas = canvasRef.current;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    const peaks = computePeaks(
      decoded.interleaved,
      decoded.channels,
      cssW,
    );
    peaksRef.current = peaks;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    // Background bars use a muted tone; the selection overlay above
    // re-paints the in-window region in the accent colour.
    ctx.fillStyle = "rgba(255,255,255,0.18)";
    const mid = cssH / 2;
    const amp = mid - 4;
    for (let i = 0; i < cssW; i++) {
      const mn = peaks[i * 2] ?? 0;
      const mx = peaks[i * 2 + 1] ?? 0;
      const y0 = mid - mx * amp;
      const y1 = mid - mn * amp;
      const h = Math.max(1, y1 - y0);
      ctx.fillRect(i, y0, 1, h);
    }
  }, [containerWidth, decoded]);

  const pxPerSecond = containerWidth > 0 && durationS > 0
    ? containerWidth / durationS
    : 0;
  const startPx = startS * pxPerSecond;
  const endPx = endS * pxPerSecond;
  const widthPx = Math.max(0, endPx - startPx);

  const clampStart = (next: number, currentEnd: number): number => {
    const upper = Math.min(currentEnd - minS, durationS - minS);
    return Math.max(0, Math.min(upper, next));
  };
  const clampEnd = (next: number, currentStart: number): number => {
    const lower = Math.max(currentStart + minS, minS);
    const upper = Math.min(durationS, currentStart + capS);
    return Math.max(lower, Math.min(upper, next));
  };

  type DragKind = "start" | "end" | "body" | null;
  const dragRef = useRef<{
    kind: DragKind;
    pointerId: number;
    grabPx: number;
    grabStartS: number;
    grabEndS: number;
  }>({ kind: null, pointerId: -1, grabPx: 0, grabStartS: 0, grabEndS: 0 });

  function onPointerDown(kind: Exclude<DragKind, null>) {
    return (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      dragRef.current = {
        kind,
        pointerId: e.pointerId,
        grabPx: e.clientX,
        grabStartS: startS,
        grabEndS: endS,
      };
    };
  }

  function onPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    const d = dragRef.current;
    if (d.kind === null || e.pointerId !== d.pointerId || pxPerSecond === 0) {
      return;
    }
    const deltaPx = e.clientX - d.grabPx;
    const deltaS = deltaPx / pxPerSecond;
    if (d.kind === "start") {
      setStartS(clampStart(d.grabStartS + deltaS, d.grabEndS));
    } else if (d.kind === "end") {
      setEndS(clampEnd(d.grabEndS + deltaS, d.grabStartS));
    } else {
      // Body drag: translate both, clamping so neither edge escapes.
      const windowS = d.grabEndS - d.grabStartS;
      let nextStart = d.grabStartS + deltaS;
      if (nextStart < 0) nextStart = 0;
      if (nextStart + windowS > durationS) nextStart = durationS - windowS;
      setStartS(nextStart);
      setEndS(nextStart + windowS);
    }
  }

  function onPointerUp(e: React.PointerEvent<HTMLDivElement>) {
    const d = dragRef.current;
    if (d.kind === null || e.pointerId !== d.pointerId) return;
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      // Ignore — releasePointerCapture throws if the capture was lost
      // (e.g. drag dragged off-screen).
    }
    dragRef.current = { kind: null, pointerId: -1, grabPx: 0, grabStartS: 0, grabEndS: 0 };
  }

  const selectedS = endS - startS;
  // The cap and the file length are independent constraints. Surface
  // whichever ceiling the user is actually pressing against so the
  // hint reads naturally ("max 2:00" for cap, "end of track" for short
  // file).
  const atCap = Math.abs(selectedS - Math.min(capS, durationS)) < 0.5;
  const fileShorterThanCap = durationS < capS;

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") onCancel();
      if (ev.key === "Enter") onConfirm(startS, endS);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onCancel, onConfirm, startS, endS]);

  if (typeof document === "undefined") return null;
  return createPortal(
    <div
      className={`almost-ready-overlay${mounted ? " mounted" : ""}`}
      onClick={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
      role="dialog"
      aria-modal="true"
      aria-label="Trim upload"
    >
      <div className="almost-ready-card waveform-trim-card">
        <header className="almost-ready-header">
          <h2 className="almost-ready-title">Trim upload</h2>
          <p className="almost-ready-sub">
            <strong>{fileName}</strong> — {fmtMMSS(durationS)} total.
            {" "}
            {fileShorterThanCap
              ? "Pick any section."
              : `Max ${fmtMMSS(capS)} of audio per session.`}
          </p>
        </header>

        <div
          ref={containerRef}
          className="waveform-trim-container"
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        >
          <canvas ref={canvasRef} className="waveform-trim-canvas" />
          {containerWidth > 0 && (
            <>
              {/* Dimmed regions outside the selection. Pointer events
                  pass through to the container so a stray pointerdown
                  on a dimmed region doesn't break the drag. */}
              <div
                className="waveform-trim-dim"
                style={{ left: 0, width: startPx }}
              />
              <div
                className="waveform-trim-dim"
                style={{ left: endPx, right: 0 }}
              />
              {/* Selection box — draggable body, with two grab handles
                  poking out at the edges. */}
              <div
                className="waveform-trim-selection"
                style={{ left: startPx, width: widthPx }}
                onPointerDown={onPointerDown("body")}
              />
              <div
                className="waveform-trim-handle waveform-trim-handle-start"
                style={{ left: startPx }}
                onPointerDown={onPointerDown("start")}
                aria-label="Trim start"
                role="slider"
                aria-valuemin={0}
                aria-valuemax={Math.floor(durationS)}
                aria-valuenow={Math.floor(startS)}
              />
              <div
                className="waveform-trim-handle waveform-trim-handle-end"
                style={{ left: endPx }}
                onPointerDown={onPointerDown("end")}
                aria-label="Trim end"
                role="slider"
                aria-valuemin={0}
                aria-valuemax={Math.floor(durationS)}
                aria-valuenow={Math.floor(endS)}
              />
            </>
          )}
        </div>

        <div className="waveform-trim-readout">
          <span>
            <span className="waveform-trim-readout-label">Start</span>{" "}
            {fmtMMSS(startS)}
          </span>
          <span>
            <span className="waveform-trim-readout-label">End</span>{" "}
            {fmtMMSS(endS)}
          </span>
          <span
            className={`waveform-trim-readout-selected${atCap ? " at-cap" : ""}`}
          >
            <span className="waveform-trim-readout-label">Selected</span>{" "}
            {fmtMMSS(selectedS)}
            {atCap && !fileShorterThanCap && (
              <span className="waveform-trim-readout-cap"> (max)</span>
            )}
          </span>
        </div>

        <footer className="almost-ready-footer">
          <button
            type="button"
            className="almost-ready-btn almost-ready-btn--secondary"
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            type="button"
            className="almost-ready-btn almost-ready-btn--primary"
            onClick={() => onConfirm(startS, endS)}
          >
            Use this section
          </button>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
