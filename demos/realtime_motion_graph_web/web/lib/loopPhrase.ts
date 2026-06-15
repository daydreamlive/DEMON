"use client";

// Wire-side loop-phrase injection.
//
// The loop-focused workflow loops a single generated section. Telling
// the text encoder up front that the section is a *loop* measurably
// tightens the loop seam (offline A/B study in
// scripts/experiments/loop_prompting/: ~12–14% lower seam discontinuity
// on average, concentrated on rhythmic material, no downside).
//
// Like LoRA triggers, we do NOT store the phrase in promptA/promptB (the
// Tags A/B textareas stay the operator's clean text); we prepend it onto
// the WIRE at send-time in `wirePromptTransform` (lib/loraTriggers.ts).
// The prefix is recomputed on every send and stripped from the incoming
// text first, so there is no double-prepend and toggling the flag
// (engine.auto_prepend_loop_phrase via config.json + refresh) immediately
// changes what the encoder sees on the next send.
//
// Gated on `engine.auto_prepend_loop_phrase` (default false, opt-in). The
// phrase itself is `engine.loop_phrase` (default below).

import { getConfig } from "@/lib/config";

/** Fallback when `engine.loop_phrase` is absent. The study's best
 *  performer (it edged "seamless repeating loop of"). */
export const DEFAULT_LOOP_PHRASE = "a short perfect loop of";

function configuredPhrase(): string {
  return (getConfig().engine.loop_phrase ?? DEFAULT_LOOP_PHRASE).trim();
}

/** The loop phrase with a trailing space, ready to concatenate ahead of
 *  a prompt — or "" when the flag is off or the phrase is empty. */
export function loopPhrasePrefix(): string {
  if ((getConfig().engine.auto_prepend_loop_phrase ?? false) !== true) {
    return "";
  }
  const phrase = configuredPhrase();
  return phrase ? `${phrase} ` : "";
}

/** Strip a leading copy (or accidental stack) of the configured loop
 *  phrase off a prompt, returning the operator's clean text. Inverse of
 *  `loopPhrasePrefix`; case-insensitive; best-effort (only the currently
 *  configured phrase is recognised, mirroring the LoRA-trigger strip's
 *  best-effort contract). Requires a trailing space after the phrase so a
 *  bare prompt equal to the phrase is left untouched. */
export function stripLeadingLoopPhrase(text: string): string {
  if (!text) return text;
  const phrase = configuredPhrase();
  if (!phrase) return text;
  const needle = `${phrase.toLowerCase()} `;
  let out = text;
  // Bounded loop: drop repeated leading occurrences (prefix drift / a
  // phrase change that stacked), capped so a pathological input can't spin.
  for (let guard = 0; guard < 8; guard += 1) {
    const lead = out.replace(/^\s+/, "");
    if (!lead.toLowerCase().startsWith(needle)) {
      out = lead;
      break;
    }
    out = lead.slice(phrase.length).replace(/^\s+/, "");
  }
  return out;
}
