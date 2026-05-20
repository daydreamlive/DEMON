"use client";

// Shared helpers for the "visible LoRA trigger prepend" feature.
//
// The contract (mirroring the LibraryTile header comment): when a LoRA
// with a primary_trigger_word is enabled and engine.auto_prepend_lora_triggers
// is on, its trigger is prepended to promptA + promptB as a leading
// comma-delimited token so the encoder sees what the operator sees. On
// disable, the trigger is stripped wherever it appears as a standalone
// token (substrings inside larger user-typed phrases are left alone).
//
// These helpers used to live inline in components/Performance/LibraryTile.tsx
// where the click-toggle was the only caller. Lifting them here lets the
// LoRA store's enable()/disable() actions also run them, so EVERY enable
// path (manual click, boot auto-enable seed, MIDI, future programmatic
// callers) gets the prepend without the call site needing to remember to
// fire it.

import { usePerformanceStore } from "@/store/usePerformanceStore";

function containsTrigger(prompt: string, trigger: string): boolean {
  return prompt.toLowerCase().includes(trigger.toLowerCase());
}

function prependTrigger(prompt: string, trigger: string): string {
  const trimmed = prompt.trim();
  if (trimmed.length === 0) return trigger;
  return `${trigger}, ${prompt}`;
}

/** Remove every occurrence of ``trigger`` from ``prompt`` where it
 *  appears as a standalone comma-delimited token (head, middle, tail,
 *  or sole token). Substrings of larger phrases are left alone — if
 *  the user typed "punk rock" and the trigger is "rock", we don't
 *  mangle their prompt. Returns the rewritten prompt, or null when
 *  the trigger isn't present as a token (so callers can skip the
 *  write). Case-insensitive on the trigger comparison; preserves the
 *  case of surrounding tokens. */
function stripTrigger(prompt: string, trigger: string): string | null {
  const needle = trigger.trim().toLowerCase();
  if (!needle) return null;
  const tokens = prompt.split(", ");
  const kept = tokens.filter((tok) => tok.trim().toLowerCase() !== needle);
  if (kept.length === tokens.length) return null;
  return kept.join(", ");
}

// Both prompts get the same treatment when a LoRA is toggled. Iterate
// over [current, setter] pairs so the per-side logic lives once.
function promptSides(): ReadonlyArray<readonly [string, (v: string) => void]> {
  const perf = usePerformanceStore.getState();
  return [
    [perf.promptA, perf.setPromptA],
    [perf.promptB, perf.setPromptB],
  ] as const;
}

export function prependTriggerToPrompts(trigger: string): void {
  for (const [cur, setter] of promptSides()) {
    if (!containsTrigger(cur, trigger)) {
      setter(prependTrigger(cur, trigger));
    }
  }
}

export function removeTriggerFromPrompts(trigger: string): void {
  for (const [cur, setter] of promptSides()) {
    const next = stripTrigger(cur, trigger);
    if (next !== null) setter(next);
  }
}
