"use client";

// Wire-side LoRA trigger injection.
//
// Each LoRA in the catalog can carry a `metadata.primary_trigger_word`
// — the activation word the LoRA was trained against. For the LoRA's
// style to actually fire, that word has to reach the engine's text
// encoder. We do NOT store it in promptA/promptB (the Tags A/B
// textareas stay the operator's clean prompt text); instead we inject
// the triggers onto the WIRE at send-time.
//
// `enabledLoraTriggerPrefix()` builds the comma-joined prefix for the
// currently-enabled LoRAs. `wirePromptTransform` (injected into
// RemoteBackend via RemoteBackendOptions.promptTransform at session
// start) applies it to both `tags` and `tags_b` right before the WS
// `prompt` message goes out. Callers always pass the clean prompt text;
// the transform adds the triggers. The prefix is computed fresh on
// every send, so there is no double-prepend and toggling a LoRA
// immediately changes what the encoder sees on the next send.
//
// Gated on `engine.auto_prepend_lora_triggers` (default true): with it
// off, the operator owns the trigger workflow manually and the prefix
// is empty.

import { getConfig } from "@/lib/config";
import { useLoraStore } from "@/store/useLoraStore";

/** Comma-joined trigger prefix for the currently-enabled LoRAs, with a
 *  trailing ", " so it can be cheaply concatenated ahead of a prompt.
 *
 *  Reads the live `useLoraStore` state (the `enabled` Set + `catalog`),
 *  collects each enabled LoRA's `metadata.primary_trigger_word`,
 *  skipping null/empty values, de-duping while preserving insertion
 *  order. Returns "" when no enabled LoRA has a trigger, or when
 *  `engine.auto_prepend_lora_triggers` is false (manual workflow). */
export function enabledLoraTriggerPrefix(): string {
  if ((getConfig().engine.auto_prepend_lora_triggers ?? true) === false) {
    return "";
  }
  const { enabled, catalog } = useLoraStore.getState();
  if (enabled.size === 0) return "";
  const seen = new Set<string>();
  const triggers: string[] = [];
  for (const entry of catalog) {
    if (!enabled.has(entry.id)) continue;
    const trigger = entry.metadata?.primary_trigger_word;
    if (!trigger) continue;
    const trimmed = trigger.trim();
    if (!trimmed) continue;
    const key = trimmed.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    triggers.push(trimmed);
  }
  if (triggers.length === 0) return "";
  return `${triggers.join(", ")}, `;
}

/** Every known LoRA trigger word in the catalog (lowercased), enabled
 *  or not. The basis for stripping a trigger prefix off a prompt. */
function catalogTriggerWords(): Set<string> {
  const out = new Set<string>();
  for (const entry of useLoraStore.getState().catalog) {
    const trimmed = entry.metadata?.primary_trigger_word?.trim().toLowerCase();
    if (trimmed) out.add(trimmed);
  }
  return out;
}

/** Strip a leading LoRA-trigger prefix off a prompt, returning the
 *  operator's clean text.
 *
 *  Drops leading comma-separated tokens that match a known catalog
 *  trigger word — ANY trigger, enabled or not, however many times it
 *  repeats. It's the inverse of the prefix `enabledLoraTriggerPrefix`
 *  builds, but resilient: it recovers the clean prompt from a stale
 *  prefix, a prefix for a since-disabled LoRA, or a prefix accidentally
 *  stacked N times. Matching is case-insensitive; the first non-trigger
 *  token ends the strip.
 *
 *  This is the guarantee behind "a disabled LoRA's trigger is never on
 *  the wire, an enabled LoRA's trigger is on it exactly once": sendPrompt
 *  runs the incoming text through here before prepending the current
 *  prefix, so whatever prefix drift happened upstream is erased and
 *  rebuilt cleanly. Trigger words are deliberately distinctive
 *  activation tokens, so a clean prompt legitimately leading with one
 *  (then a comma) is vanishingly unlikely. */
export function stripLeadingTriggers(text: string): string {
  if (!text) return text;
  const triggers = catalogTriggerWords();
  if (triggers.size === 0) return text;
  const parts = text.split(",");
  let i = 0;
  while (i < parts.length && triggers.has(parts[i].trim().toLowerCase())) {
    i += 1;
  }
  if (i === 0) return text;
  return parts.slice(i).join(",").replace(/^\s+/, "");
}

/** The app's wire-prompt transform, passed to RemoteBackend as
 *  `RemoteBackendOptions.promptTransform`.
 *
 *  Hard guarantee, independent of upstream state: stripLeadingTriggers
 *  removes ANY trigger prefix already on the text (stale, stacked, or
 *  belonging to a since-disabled LoRA), then we prepend exactly the
 *  de-duped enabled-set prefix. So the wire prompt always carries a
 *  disabled LoRA's trigger zero times and an enabled LoRA's trigger
 *  exactly once — per tag (A and B alike). */
export function wirePromptTransform(tags: string): string {
  return enabledLoraTriggerPrefix() + stripLeadingTriggers(tags);
}
