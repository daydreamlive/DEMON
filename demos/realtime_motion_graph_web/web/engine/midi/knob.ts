// Per-CC knob mode auto-detect (absolute vs relative 2's complement). Some
// MIDI controllers (MPK Mini in increment/decrement mode, many endless
// encoders) send relative deltas instead of absolute positions. Direct port
// of the decodeKnob() helper in static/app.js.

interface KnobState {
  history: number[];
  mode: "unknown" | "absolute" | "relative";
}

export type KnobDecoded =
  | { mode: "absolute"; absolute: number }
  | { mode: "relative"; delta: number };

const knobState = new Map<number, KnobState>();

export function decodeKnob(cc: number, value: number): KnobDecoded {
  let s = knobState.get(cc);
  if (!s) {
    s = { history: [], mode: "unknown" };
    knobState.set(cc, s);
  }
  s.history.push(value);
  if (s.history.length > 6) s.history.shift();

  if (s.mode === "unknown" && s.history.length >= 3) {
    const allExtreme = s.history.every((v) => v <= 4 || v >= 123);
    s.mode = allExtreme ? "relative" : "absolute";
  }

  // Demote a misclassified "relative" knob back to absolute the moment
  // it sends a value clearly inconsistent with 2's-complement deltas.
  // Real relative encoders only emit small CW values (1-15) or values
  // close to 128 (113-127, i.e. -1..-15). A value in 5-122 means we're
  // looking at an absolute knob that happened to be parked near 0 or
  // 127 when the user first nudged it through the auto-detect window.
  // Without this, the relative branch interprets value=30 as "+30 ticks"
  // and slams the slider against its bound.
  if (s.mode === "relative" && value >= 5 && value <= 122) {
    s.mode = "absolute";
  }

  if (s.mode === "absolute") {
    return { mode: "absolute", absolute: value / 127 };
  }
  if (s.mode === "relative") {
    let delta = 0;
    if (value > 0 && value < 64) delta = value;
    else if (value > 64 && value <= 127) delta = value - 128;
    return { mode: "relative", delta };
  }
  // Unknown: extreme → relative tick; mid-range → commits to absolute.
  if (value <= 4 || value >= 123) {
    let delta = 0;
    if (value > 0 && value < 64) delta = value;
    else if (value > 64 && value <= 127) delta = value - 128;
    return { mode: "relative", delta };
  }
  s.mode = "absolute";
  return { mode: "absolute", absolute: value / 127 };
}
