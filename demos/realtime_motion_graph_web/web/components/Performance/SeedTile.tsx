"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";

export function SeedTile() {
  const seed = usePerformanceStore((s) => s.seed);
  const randomize = usePerformanceStore((s) => s.randomizeSeed);
  return (
    <div className="mixer-tile mixer-tile-seed" data-tile="seed">
      <div className="mixer-tile-label">Seed</div>
      <div className="seed-content">
        <button
          id="seed-btn"
          className="seed-btn"
          data-midi-learn="seed"
          title="Randomize seed (right-click to MIDI-learn)"
          type="button"
          onClick={randomize}
          aria-label="Randomize seed"
        >
          {/* Inline SVG dice — monoline stroke, no fill, matches the
              custom-cursor / ribbon vocabulary. Two pips visible
              (face-2) suggest "the dice is mid-roll" without literally
              animating. */}
          <svg
            className="seed-dice"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <rect x="3" y="3" width="18" height="18" rx="3" />
            <circle cx="8.5" cy="8.5" r="1.1" fill="currentColor" stroke="none" />
            <circle cx="15.5" cy="15.5" r="1.1" fill="currentColor" stroke="none" />
          </svg>
        </button>
        <div className="slider-value" id="seed-value">
          {seed.toFixed(2)}
        </div>
        <kbd className="desktop-only">F</kbd>
      </div>
    </div>
  );
}
