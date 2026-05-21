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
// currently-enabled LoRAs. `RemoteBackend.sendPrompt` prepends it to
// both `tags` and `tags_b` right before the WS `prompt` message goes
// out. Callers always pass the clean prompt text; sendPrompt adds the
// triggers. The prefix is computed fresh on every send, so there is no
// double-prepend and toggling a LoRA immediately changes what the
// encoder sees on the next send.
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
