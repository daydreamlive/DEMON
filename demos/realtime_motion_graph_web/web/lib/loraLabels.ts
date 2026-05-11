// Client-side display overrides for LoRA names. The underlying lora id
// (e.g. `"deathstep"`) stays untouched everywhere it crosses an engine
// boundary — prompts, MIDI keys, curve storage, the catalog wire format —
// only the user-visible label is rewritten. Looked up at every render
// site that shows a LoRA name (LibraryTile, edge bars, mobile stepper,
// curve scheduler tabs).
const LORA_DISPLAY_OVERRIDES: Record<string, string> = {
  deathstep: "DUBSTEP",
};

export function displayLoraName(id: string, fallback?: string): string {
  return LORA_DISPLAY_OVERRIDES[id] ?? fallback ?? id;
}
