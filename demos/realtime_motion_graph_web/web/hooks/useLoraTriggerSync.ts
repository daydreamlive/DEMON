"use client";

import { useEffect } from "react";

import { getConfig } from "@/lib/config";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Re-send the prompt whenever the enabled-LoRA set changes.
//
// Enabling/disabling a LoRA prepends/strips its trigger word to/from
// Tags A + B (useLoraStore.enable/disable → loraTriggers), but that
// edit only reaches the engine on a `prompt` WS message — and toggling
// a LoRA does NOT itself send one (Send Tags / Enter / key change are
// the only senders). Without this hook the operator enables a LoRA,
// sees its trigger appear in the Tags box, but the engine keeps
// generating against the OLD prompt with no trigger: the LoRA's
// matrices apply via refit, yet its activation word never lands — the
// style barely fires. This is exactly the "LoRAs apply weird" report.
//
// This hook closes the gap: on any change to `useLoraStore.enabled` it
// debounce-sends the current promptA/promptB so the just-prepended (or
// just-stripped on disable) trigger commits to the encoder. Debounce —
// not throttle — because auditioning LoRAs produces rapid enable/
// disable bursts; we want a single send once the burst settles, not
// one per toggle.
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
      if (session.status !== "ready" || !session.remote) return;
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
