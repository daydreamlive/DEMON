"use client";

import { useEffect } from "react";

import { getConfig } from "@/lib/config";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Re-send the prompt whenever the enabled-LoRA set changes.
//
// A LoRA's trigger word reaches the engine only on a `prompt` WS
// message, and it's injected onto the wire by RemoteBackend.sendPrompt
// (via enabledLoraTriggerPrefix) — the Tags A/B textareas stay the
// operator's clean prompt text. But most LoRA-enable paths don't
// themselves send a `prompt` (Send Tags / Enter / key change are the
// usual senders). LibraryTile's click-toggle now sends one directly;
// this hook is the BACKSTOP for the other enable paths — MIDI,
// edge-binding, reset — so they too commit the new trigger set to the
// encoder.
//
// On any change to `useLoraStore.enabled` it debounce-sends the
// current clean promptA/promptB; sendPrompt rebuilds the trigger
// prefix fresh. Debounce — not throttle — because auditioning LoRAs
// produces rapid enable/disable bursts; we want a single send once the
// burst settles, not one per toggle.
//
// Gated on `engine.auto_prepend_lora_triggers`: when an operator turns
// auto-prepend off (a fully manual trigger workflow) they also own
// prompt sends, so the auto-send stays out of their way.
//
// `enabled` is replaced with a fresh Set on every real membership
// change (enable/disable build a new Set; the no-op guards return the
// old one), so a reference check is a reliable change signal.

const DEBOUNCE_MS = 250;

export function useLoraTriggerSync() {
  useEffect(() => {
    let timerId = 0;

    const flush = () => {
      timerId = 0;
      const session = useSessionStore.getState();
      // Match PromptsTile.sendPrompt — the proven send path — which
      // only checks `remote`. The WS readyState guard inside
      // RemoteBackend.sendPrompt is the real gate; an extra
      // status !== "ready" check here only drops legitimate sends.
      if (!session.remote) return;
      const perf = usePerformanceStore.getState();
      session.remote.sendPrompt(
        perf.promptA,
        perf.activeKey,
        perf.activeTimeSignature,
        perf.promptB,
      );
    };

    const unsub = useLoraStore.subscribe((s, prev) => {
      if (s.enabled === prev.enabled) return;
      if ((getConfig().engine.auto_prepend_lora_triggers ?? true) === false) {
        return;
      }
      if (timerId !== 0) window.clearTimeout(timerId);
      timerId = window.setTimeout(flush, DEBOUNCE_MS);
    });

    return () => {
      if (timerId !== 0) window.clearTimeout(timerId);
      unsub();
    };
  }, []);
}
