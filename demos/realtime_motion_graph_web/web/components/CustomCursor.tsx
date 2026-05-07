"use client";

// Mount point for the canvas-rendered cursor. cursor.ts grabs this
// element on init and drives it from the shared render loop. Single
// composited layer so the orbital cluster + sparks + confetti all
// stay on the GPU's fast path in every browser.

export function CustomCursor() {
  return <canvas className="cursor-canvas" aria-hidden="true" />;
}
