// Per-LoRA trigger word + optional display label.
//
// A LoRA's trigger is the text token its training captions all shared
// (the v5 "pruned captions" rule — the trigger is the only token that
// appears in every clip, so the entire learned style concentrates on
// its embedding). At inference, including the trigger in the prompt
// activates the style; omitting it leaves the LoRA dormant.
//
// This module is the UI-side source of truth for trigger words. When
// the user toggles a LoRA on, the trigger is auto-prepended to the
// active prompts so the style actually fires; when they toggle off,
// the auto-injected trigger is stripped. Without this glue, toggling
// a LoRA in the UI only registers it with the engine — it does NOT
// activate the style unless the user happens to type the trigger
// themselves, which is the failure mode of every non-deathstep LoRA
// today (deathstep accidentally activates because its trigger
// `afxdump` is hard-coded into the default prompt).
//
// Prefix-matched: checkpoint variants of the same training run
// (`bptkno-100`, `bptkno-200`, `bptkno-300`, ...) all map to the same
// trigger + label. Order matters — the first matching prefix wins.
//
// Long-term, this lookup should be replaced by trigger metadata
// embedded in each LoRA's `.safetensors.json` sidecar (written by the
// training pipeline at convert time, surfaced through the pod's
// /api/loras response). The map below stays as the fallback for any
// LoRA whose sidecar doesn't declare a trigger.

interface LoraTriggerEntry {
  /** Word to prepend to the active prompt when the LoRA toggles on.
   *  Lower-case, single token, no surrounding whitespace — the
   *  prepend logic adds the leading comma+space and dedupes. */
  trigger: string;
  /** Display label shown in the LoRA library tile / edge bars / mobile
   *  stepper. Optional — falls back to the bare LoRA id. Capitalised
   *  by convention to match the existing "DUBSTEP" override style. */
  label?: string;
}

/** Prefix-keyed map. The first entry whose `prefix` is a (lowercase)
 *  start-substring of the LoRA id wins. Bare ids (no checkpoint
 *  suffix) match too because `"bptkno".startsWith("bptkno") === true`. */
const PREFIX_TRIGGERS: ReadonlyArray<{
  prefix: string;
  entry: LoraTriggerEntry;
}> = [
  // v6 — Beatport Sept 2014 techno LoRA. Variants `bptkno-100`,
  // `bptkno-200`, ..., `bptkno-500` all share the trigger; the
  // display label is "TECHNO" because the user-facing genre is the
  // useful name, not the internal training-run shortcode.
  { prefix: "bptkno", entry: { trigger: "bptkno", label: "TECHNO" } },
  // v5 — RAM-era Daft Punk pruned-captions LoRA. Trigger collides
  // with the genre word "discofunk" by design (the v5 retrospective
  // noted this softens activation vs a nonsense trigger; we keep it
  // for back-compat with the trained adapter).
  { prefix: "discofunk", entry: { trigger: "discofunk", label: "DISCOFUNK" } },
  // hardrock — AC/DC pruned-captions LoRA. Same real-vocab caveat.
  { prefix: "hardrock", entry: { trigger: "hardrock", label: "HARDROCK" } },
  // deathstep — Ryan's canonical dubstep LoRA. Trigger is `afxdump`
  // (per the deathstep-good README in his training-state archive).
  // The default promptA happens to already include both `deathstep`
  // and `afxdump` for legacy back-compat — `withTrigger` is
  // idempotent so the initial-seed prepend is a no-op there; toggling
  // off cleanly strips `afxdump` only (the literal `deathstep` word
  // in the default prompt stays — it's a stylistic descriptor, not
  // the LoRA's training trigger). Label override "DUBSTEP" lives in
  // loraLabels.ts's exact-match map and takes precedence over the
  // optional label here.
  { prefix: "deathstep", entry: { trigger: "afxdump", label: "DUBSTEP" } },
  // synthpop — intentionally NOT in this list. The pre-existing LoRA
  // didn't ship with a documented training trigger; treating it as a
  // no-trigger LoRA means toggling it does nothing to the prompt,
  // which matches the engine-side behaviour of registering it without
  // an associated text-side activation.
];

export function getLoraTrigger(loraId: string): LoraTriggerEntry | null {
  if (!loraId) return null;
  const lower = loraId.toLowerCase();
  for (const { prefix, entry } of PREFIX_TRIGGERS) {
    if (lower.startsWith(prefix)) return entry;
  }
  return null;
}

/** Check whether `prompt` already contains `trigger` as a standalone,
 *  comma-separated token (case-insensitive). Used to make the prepend
 *  idempotent and to detect user-typed triggers we shouldn't strip. */
export function containsTrigger(prompt: string, trigger: string): boolean {
  if (!trigger) return false;
  const lowerTrig = trigger.toLowerCase();
  return prompt
    .split(",")
    .map((t) => t.trim().toLowerCase())
    .includes(lowerTrig);
}

/** Prepend `trigger` to `prompt` as the first comma-separated token.
 *  Idempotent: a no-op when the trigger is already present. The
 *  original prompt is preserved verbatim (whitespace untrimmed apart
 *  from the inserted "<trigger>, " prefix) so the user's formatting
 *  stays intact. */
export function withTrigger(prompt: string, trigger: string): string {
  if (!trigger) return prompt;
  if (containsTrigger(prompt, trigger)) return prompt;
  const trimmed = prompt.trim();
  return trimmed ? `${trigger}, ${trimmed}` : trigger;
}

/** Remove a single trigger token from a prompt. Strips by matching
 *  whole comma-separated tokens (case-insensitive), regardless of
 *  position. The remaining tokens are re-joined with ", "; if the
 *  caller cares about preserving the user's exact whitespace they
 *  should call `containsTrigger` first and skip the strip when the
 *  trigger isn't auto-owned. */
export function withoutTrigger(prompt: string, trigger: string): string {
  if (!trigger) return prompt;
  const lowerTrig = trigger.toLowerCase();
  const tokens = prompt
    .split(",")
    .map((t) => t.trim())
    .filter((t) => t.length > 0 && t.toLowerCase() !== lowerTrig);
  return tokens.join(", ");
}
