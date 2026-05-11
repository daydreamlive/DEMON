// Client-side display overrides for LoRA names. The underlying lora id
// (e.g. `"deathstep"`) stays untouched everywhere it crosses an engine
// boundary — prompts, MIDI keys, curve storage, the catalog wire format —
// only the user-visible label is rewritten. Looked up at every render
// site that shows a LoRA name (LibraryTile, edge bars, mobile stepper,
// curve scheduler tabs).

import { getLoraTrigger } from "./loraTriggers";

const LORA_DISPLAY_OVERRIDES: Record<string, string> = {
  deathstep: "DUBSTEP",
};

export function displayLoraName(id: string, fallback?: string): string {
  // Exact-match override wins (e.g. `deathstep` → `DUBSTEP`). Then the
  // prefix-matched trigger map's `label` (e.g. any `bptkno-*` →
  // `TECHNO`) — same source-of-truth as the auto-prepend logic, so a
  // new trained LoRA only needs an entry in `loraTriggers.ts` and the
  // library tile picks up the right name automatically.
  const exact = LORA_DISPLAY_OVERRIDES[id];
  if (exact) return exact;
  const triggered = getLoraTrigger(id);
  if (triggered?.label) return triggered.label;
  return fallback ?? id;
}
