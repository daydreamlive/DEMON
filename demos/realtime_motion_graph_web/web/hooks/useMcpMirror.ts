"use client";

import { useEffect } from "react";

import { stripLeadingTriggers } from "@/lib/loraTriggers";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { isRcfgMode, isTimeSignature } from "@/types/engine";
import type { LoraCatalogEntry } from "@/types/protocol";

// Mirror state changes driven by the onboard MCP control bus back into
// the front-end stores so the user can see Claude's edits land in the
// UI while it's driving the demo.
//
// The MCP injects commands through the server's HTTP control bus, which
// dispatches them through the same handler as the browser's own
// WebSocket frames. For most actions (prompt, LoRA enable/disable,
// timbre, structure, swap) the server already broadcasts an ack that
// existing hooks consume, so the UI updates without any extra wiring.
//
// Knob changes are the one case the UI doesn't already mirror — the
// browser owns sliderValues as the source of truth and pushes them at
// 8 ms ticks, so an MCP-driven write needs to go through the same
// slider path a user drag would. The server doesn't apply MCP knob
// writes server-side; instead it tags them with `params_echo` (or
// `prompt_blend_echo` for the dedicated blend slider). We feed those
// into setSlider, which retargets and — when Smooth is on — kicks off
// the cubic-ease tween. useParamSync / usePromptBlendSync then ship
// the smoothed sequence back to the server as normal WS messages, so
// virtual_knobs sees the same curve a real drag would produce.
//
// With Smooth off, setSlider snaps both sliderTargets and sliderValues
// in one shot, equivalent to the prior setSliderDirect behavior.
//
// `prompt_applied` echoes are also consumed here: the engine echoes the
// WIRE prompt — the operator's clean text with the enabled-LoRA trigger
// prefix that sendPrompt injects. We strip that prefix back off, then,
// when the engine's clean prompt diverges from the typed promptA, adopt
// it so the input matches what's actually being encoded. Adopting the
// raw (prefixed) echo would bake the trigger prefix into promptA and
// every later sendPrompt would re-prepend it — unbounded accumulation.

export function useMcpMirror() {
  useEffect(() => {
    let attached: { remote: EventTarget; off: () => void } | null = null;

    const attach = (remote: EventTarget) => {
      const onParamsEcho = (e: Event) => {
        const raw = (e as CustomEvent<Record<string, number | string | boolean>>)
          .detail;
        if (!raw || typeof raw !== "object") return;
        const perf = usePerformanceStore.getState();
        const lora = useLoraStore.getState();
        for (const [name, value] of Object.entries(raw)) {
          // String-valued knobs that the MCP can drive. Handled before
          // the numeric filter below because the underlying store fields
          // are not in sliderValues — they have dedicated setters and
          // useParamSync ships them every tick from the store. Without
          // mirroring here, the next tick would overwrite the MCP write.
          if (name === "rcfg_mode") {
            if (isRcfgMode(value)) perf.setRcfgMode(value);
            continue;
          }
          if (typeof value !== "number" || !Number.isFinite(value)) continue;
          if (name === "seed") {
            perf.setSeed(value);
            continue;
          }
          if (name.startsWith("lora_str_")) {
            const id = name.slice("lora_str_".length);
            if (id) lora.setStrength(id, value);
            // Also retarget the slider so useParamSync's full-snapshot
            // payload reflects the new value next tick. setSlider runs
            // the Smooth tween when enabled, otherwise snaps.
            perf.setSlider(name, value);
            continue;
          }
          // Skip non-slider keys the engine carries that aren't UI knobs
          // (e.g. dcw_enabled / dcw_mode are string/bool and already
          // filtered by the numeric guard). Everything else maps to
          // sliderValues.
          perf.setSlider(name, value);
        }
      };

      const onPromptBlendEcho = (e: Event) => {
        const v = (e as CustomEvent<number>).detail;
        if (typeof v !== "number" || !Number.isFinite(v)) return;
        // setSlider runs the Smooth tween on sliderValues.prompt_blend;
        // usePromptBlendSync subscribes to that and ships the tweened
        // sequence back to the server via set_prompt_blend WS messages.
        usePerformanceStore.getState().setSlider("prompt_blend", v);
      };

      const onPromptApplied = (e: Event) => {
        const raw = (e as CustomEvent<string>).detail;
        if (typeof raw !== "string") return;
        const perf = usePerformanceStore.getState();
        // The echo is the wire prompt: clean text + the LoRA trigger
        // prefix sendPrompt prepended. stripLeadingTriggers removes any
        // trigger prefix (current, stale, or stacked) so we only ever
        // adopt the clean prompt into promptA — otherwise the prefix is
        // baked into promptA and every later sendPrompt re-prepends it
        // (and a disabled LoRA's trigger lingers as literal text).
        const tags = stripLeadingTriggers(raw);
        // Only adopt when the engine's clean prompt diverges from what's
        // in the input box — protects the user's mid-typing state when
        // the echo is just confirming their own send.
        if (perf.promptA !== tags) {
          perf.setPromptA(tags);
        }
      };

      const onLoraCatalog = (e: Event) => {
        // Server-broadcast catalog after the runner thread applies a
        // pending enable/disable. Reconcile the UI's enabled set +
        // strengths against the catalog so MCP-driven LoRA toggles
        // surface in the Library panel and useParamSync starts sending
        // the right per-LoRA strengths next tick. Without this, the
        // front-end's `enabled` set keeps its own user-click history
        // regardless of what's actually loaded in the engine.
        const catalog = (e as CustomEvent<LoraCatalogEntry[]>).detail;
        if (!Array.isArray(catalog)) return;
        const lora = useLoraStore.getState();
        lora.setCatalog(catalog);
        for (const entry of catalog) {
          if (!entry || typeof entry.id !== "string") continue;
          if (entry.state === "enabled") {
            lora.enable(entry.id);
            if (typeof entry.strength === "number" && entry.strength > 0) {
              lora.setStrength(entry.id, entry.strength);
            }
          } else {
            lora.disable(entry.id);
          }
        }
      };

      const onSwapReady = (e: Event) => {
        // Mirror MCP-driven source swaps into the UI: the audio buffer
        // already swapped via protocol.ts's own swap_ready handler; we
        // just sync the fixture dropdown + detected key/time signature
        // and register the buffer in customTracks so later
        // loadFixtureAudio calls (e.g. re-Play) resolve the name.
        //
        // For user-initiated swaps (useFixtureSwap drove the
        // sendSwapSource), perf.fixture already equals the new name —
        // no-op so useFixtureSwap's own in-flight listener owns the
        // adoption (status messages, denoise gate, pendingKeyOverride,
        // prompt re-send all live there).
        const detail = (e as CustomEvent<{
          fixture_name?: string;
          interleaved: Float32Array;
          channels: number;
          key?: string;
          time_signature?: string;
        }>).detail;
        if (!detail || !detail.fixture_name) return;
        const perf = usePerformanceStore.getState();
        if (perf.fixture === detail.fixture_name) return;

        // Register the resulting PCM in the custom-tracks store under
        // its name. Loads checked customTracks first, so this lets a
        // future swap back to the same name (e.g. Library carousel) hit
        // the in-memory buffer instead of attempting a /fixtures/ fetch
        // for a name the pod never had on disk.
        const channels = Math.max(1, detail.channels);
        useCustomTracksStore.getState().add(detail.fixture_name, {
          interleaved: detail.interleaved,
          channels,
          sampleRate: 48000,
          frames: detail.interleaved.length / channels,
        });

        const rawTs = detail.time_signature;
        const detectedTs = rawTs != null && isTimeSignature(rawTs)
          ? rawTs
          : null;
        if (detail.key || detectedTs) {
          perf.setDetected(
            perf.detectedBpm,
            detail.key ?? perf.detectedKey,
            detectedTs ?? perf.detectedTimeSignature,
          );
        }
        if (detail.key) perf.setKey(detail.key);
        if (detectedTs) perf.setTimeSignature(detectedTs);

        // Suppress useFixtureSwap's reaction to the perf.fixture write
        // below — the server already has the new source loaded, the
        // user-swap pipeline would just re-upload + re-swap to the same
        // thing.
        perf.setSkipNextFixtureSwap(true);
        perf.setFixture(detail.fixture_name);
      };

      remote.addEventListener("params_echo", onParamsEcho);
      remote.addEventListener("prompt_blend_echo", onPromptBlendEcho);
      remote.addEventListener("prompt_applied", onPromptApplied);
      remote.addEventListener("lora_catalog", onLoraCatalog);
      remote.addEventListener("swap_ready", onSwapReady);
      attached = {
        remote,
        off: () => {
          remote.removeEventListener("params_echo", onParamsEcho);
          remote.removeEventListener("prompt_blend_echo", onPromptBlendEcho);
          remote.removeEventListener("prompt_applied", onPromptApplied);
          remote.removeEventListener("lora_catalog", onLoraCatalog);
          remote.removeEventListener("swap_ready", onSwapReady);
        },
      };
    };

    const detach = () => {
      attached?.off();
      attached = null;
    };

    const apply = (remote: EventTarget | null) => {
      if (attached?.remote === remote) return;
      detach();
      if (remote) attach(remote);
    };

    apply(useSessionStore.getState().remote ?? null);

    const unsub = useSessionStore.subscribe((s, prev) => {
      if (s.remote !== prev.remote) {
        apply(s.remote ?? null);
      }
    });

    return () => {
      detach();
      unsub();
    };
  }, []);
}
