"use client";

import { useState } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";

import { KnobCap } from "./KnobCap";

// SeedKnob — knob-shaped randomize button, lives at the end of the CORE
// row. Shares the skeumorphic cap chrome with <Knob/> via <KnobCap/>
// so the row reads as one continuous hardware unit. Difference: no
// arc, no value indicator — the cap surface holds a dice glyph that
// snaps to a new face on every click.
//
// Interactions:
//   - Click            → randomize seed (calls store.randomizeSeed)
//   - Double-click cap → manual numeric input
//   - Right-click cap  → MIDI-learn (data-midi-learn attr handled by
//                        useMidi)
//   - "F" keyboard shortcut (handled in useKeyboardShortcuts)

// Dice face configurations. Each face is a list of pip [x, y] coords
// in the local 24x24 viewBox.
const PIPS_BY_FACE: ReadonlyArray<ReadonlyArray<readonly [number, number]>> = [
  // face 1
  [[12, 12]],
  // face 2
  [[8.5, 8.5], [15.5, 15.5]],
  // face 3
  [[8, 8], [12, 12], [16, 16]],
  // face 4
  [[8, 8], [16, 8], [8, 16], [16, 16]],
  // face 5
  [[8, 8], [16, 8], [12, 12], [8, 16], [16, 16]],
  // face 6
  [[8, 8.5], [16, 8.5], [8, 12], [16, 12], [8, 15.5], [16, 15.5]],
];

export function SeedKnob() {
  const seed = usePerformanceStore((s) => s.seed);
  const randomize = usePerformanceStore((s) => s.randomizeSeed);
  const setSeed = usePerformanceStore((s) => s.setSeed);

  // Dice face cycles on each click so the cap visibly reacts to the
  // randomize action. The face index doesn't carry information — it's
  // a visual "the dice rolled" cue. Initial face derived from the
  // current seed so a fresh page-load shows a stable face per session.
  const [face, setFace] = useState(() => seed % 6);

  // Double-click on the value cell swaps it for a text input.
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState("");
  const startEdit = () => {
    setEditText(String(seed));
    setEditing(true);
  };
  const commitEdit = () => {
    // parseFloat handles plain ints, decimals, AND exponential notation
    // (the compactSeed display uses "1.2e7" for big seeds, so users
    // re-editing a compact value need the parser to accept it). setSeed
    // floors internally — fractional inputs round to integer.
    const parsed = parseFloat(editText);
    if (Number.isFinite(parsed)) setSeed(parsed);
    setEditing(false);
  };
  const cancelEdit = () => setEditing(false);

  const onClick = () => {
    randomize();
    setFace((f) => (f + 1) % 6); // cycle to a different face
  };

  const pips = PIPS_BY_FACE[face];

  return (
    <div
      className="knob-group seed-knob-group"
      // Tooltip trio on the wrapper — matches Knob.tsx so the
      // HeroMacrosTooltip portal picks up the whole knob-group
      // (label + body + value + kbd) instead of just the SVG button.
      data-dd-tooltip="Randomize the generation seed for a fresh take on the same source. Same seed + same source → reproducible output. Double-click the value to type a specific seed. Right-click to MIDI-learn."
      data-dd-tooltip-wide=""
      data-dd-tooltip-title="Seed"
    >
      <div className="knob-label">Seed</div>
      <button
        type="button"
        className="knob-body seed-knob-body"
        onClick={onClick}
        data-midi-learn="seed"
        aria-label="Randomize seed"
      >
        <svg
          className="knob-svg"
          viewBox="0 0 48 48"
          width="48"
          height="48"
          aria-hidden="true"
        >
          {/* Shared 3D cap chrome: shadow + body + beveled rim. */}
          <KnobCap />
          {/* Dice glyph centered on the cap. Scaled + translated from
              the dice's local 24x24 viewBox into the cap's center. The
              dice face cycles on each click so the cap visibly reacts. */}
          <g transform="translate(15, 15) scale(0.75)" className="seed-dice-glyph">
            <rect
              x="3"
              y="3"
              width="18"
              height="18"
              rx="3"
              fill="none"
              stroke="rgba(220, 224, 232, 0.9)"
              strokeWidth="1.4"
              strokeLinejoin="round"
            />
            {pips.map(([px, py], i) => (
              <circle
                key={i}
                cx={px}
                cy={py}
                r={1.1}
                fill="rgba(220, 224, 232, 0.95)"
              />
            ))}
          </g>
        </svg>
      </button>
      <div
        className="knob-value seed-knob-value"
        onDoubleClick={startEdit}
        title="Double-click to edit"
      >
        {editing ? (
          <input
            type="text"
            className="knob-value-input"
            inputMode="numeric"
            value={editText}
            autoFocus
            onChange={(e) => setEditText(e.target.value)}
            onFocus={(e) => e.currentTarget.select()}
            onBlur={commitEdit}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                e.preventDefault();
                cancelEdit();
              }
            }}
          />
        ) : (
          // uint32 seeds get wide — truncate the display so the cell
          // doesn't blow up the row width. Hover-title shows the full
          // value; double-click reveals it for editing.
          <span title={String(seed)}>{compactSeed(seed)}</span>
        )}
      </div>
      <kbd className="knob-kbd desktop-only">F</kbd>
    </div>
  );
}

// uint32 → at-most-7-char display ("1234567" or "1.2e7"). Keeps the
// value cell the same width as a regular knob's "0.50" / "-26.1 dB".
function compactSeed(v: number): string {
  if (v < 100000) return String(v);
  if (v < 10_000_000) return v.toFixed(0);
  return v.toExponential(1).replace("e+", "e");
}
