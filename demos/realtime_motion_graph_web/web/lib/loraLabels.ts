// Hardcoded display labels for the bare "v6" LoRAs.
//
// These LoRAs ship without a metadata sidecar, so their catalog entry
// has no `name` and the UI would otherwise fall back to the raw
// filename stem (`bach`, `deathstep`, …). The team picked these
// caps-styled names (originally PR #84 / #113). LoRAs that DO carry a
// real metadata `name` — e.g. the genre LoRAs — are not in this map
// and keep their own name.

export const LORA_LABELS: Record<string, string> = {
  deathstep: "DUBSTEP",
  bach: "BAROQUE",
  bptkno: "TECHNO",
  hardrock: "HARDROCK",
  discofunk: "DISCOFUNK",
};

/** Display name for a LoRA: hardcoded label → metadata name → id. */
export function loraDisplayName(entry: {
  id: string;
  name?: string | null;
}): string {
  return LORA_LABELS[entry.id] ?? (entry.name || entry.id);
}
