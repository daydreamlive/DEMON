"use client";

import { useEffect, useRef } from "react";

import {
  loadFixtureAudio,
  type StemSourceMode,
} from "@/engine/audio/loadFixture";
import {
  applyLoraCapWithServerSync,
  getConfig,
  resolveLoraCapForSource,
} from "@/lib/config";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { isTimeSignature } from "@/types/engine";

// In-place fixture swap. Mirrors swapToFixture() in DEMON's app.js: when the
// user picks a different fixture mid-session, the server keeps the model
// loaded and re-encodes the new source; the worklet crossfades the new
// buffer in over 50 ms. We surface "Decoding ..." then "Swapping to ..."
// in the status bar so the user knows the swap is in flight.
//
// Falls back to a full session restart on swap_failed (e.g. server in a
// state where it can't accept a new source). The full restart is delegated
// back to useStartSession via the same fixture name.

export function useFixtureSwap() {
  // Skip the very first fixture write (which fires when the catalog populates
  // and writes the default name into the store).
  const lastSwappedTo = useRef<string | null>(null);
  // The stem_source_mode last actually sent to the server. Used to (a)
  // dedupe the server's `stem_assets` source_mode echo so it doesn't
  // bounce back as a fresh swap, and (b) let the source-mode hotswap
  // subscription re-run a swap for the *same* fixture when only the mode
  // changed (the name-based `lastSwappedTo` guard would otherwise block it).
  const lastSwappedMode = useRef<StemSourceMode | null>(null);

  useEffect(() => {
    let cancelled = false;

    // `force` re-runs the swap for the currently-loaded fixture (used by
    // the source-mode hotswap). It bypasses the same-name short-circuit
    // and skips the "new track" denoise gate below — toggling vocals ↔
    // instruments is the same song, so it shouldn't yank the performer
    // back to the hear-the-source start gate.
    const run = async (name: string, force = false) => {
      if (cancelled) return;
      const session = useSessionStore.getState();
      if (session.status !== "ready" || !session.remote || !session.player) {
        return; // No live session yet; the next Play will pick the new fixture.
      }
      if (!force && lastSwappedTo.current === name) return;

      const { setStatus } = useSessionStore.getState();

      // Server-resident tracks (built-in fixtures + persisted uploads)
      // swap by name: the pod loads the waveform off its own disk so the
      // sidecar + stem caches hit, instead of the browser decoding the
      // file and re-uploading PCM only for the server to re-encode and
      // re-rip stems. The playback buffer still comes back in the
      // swap_ready echo. Only tracks that live solely in browser memory
      // (no-pod fallback, MCP mirror) take the decode + upload path.
      const serverResident = useCustomTracksStore
        .getState()
        .isServerResident(name);

      let interleaved: Float32Array | null = null;
      let channels = 0;
      if (!serverResident) {
        setStatus("ready", `Loading ${name}…`);
        try {
          const decoded = await loadFixtureAudio(name);
          interleaved = decoded.interleaved;
          channels = decoded.channels;
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          setStatus("ready", `Load failed: ${msg}`);
          return;
        }
        if (cancelled) return;
      }

      setStatus("ready", `Swapping to ${name}…`);

      const remote = session.remote;
      const ok = await new Promise<boolean>((resolve) => {
        const onReady = (e: Event) => {
          remote.removeEventListener("swap_ready", onReady);
          remote.removeEventListener("swap_failed", onFail);
          const detail = (e as CustomEvent<{
            interleaved: Float32Array;
            channels: number;
            bpm?: number | null;
            key?: string;
            time_signature?: string;
          }>).detail;
          // Recompute the duration-aware LoRA cap against the new
          // source. protocol.ts already set ``remote.duration`` from
          // the swap_ready message; reading it here gives the
          // authoritative value. Tiers swap to the right cap (e.g.
          // 60s→3 LoRAs, 240s→1) before the user can interact with
          // the new source.
          //
          // Critical: use the server-syncing helper, not setMaxEnabled
          // alone. A tightening cap (60s→240s source swap) clips the
          // client's enabled set, but if we don't WS-disable the
          // dropped LoRAs the server keeps them materialized — the
          // ghost-LoRA leak. The helper sends disable_lora for each
          // dropped id and re-issues the prompt so the trigger
          // prefix drops their triggers too.
          applyLoraCapWithServerSync(resolveLoraCapForSource(remote.duration));
          session.player?.swap(detail.interleaved, detail.channels);
          // Both backends keep the playhead phase across a swap (the
          // worklet leaves `position` untouched; AudioPlayer.swap clamps
          // `_spPosition` into the new buffer rather than zeroing it), so a
          // swap at 1:30 resumes at 1:30 of the new track unless we seek.
          // This gate is the single source of truth for restart-on-swap on
          // both paths. Operator can disable via config.
          if (getConfig().restart_song_on_swap) {
            session.player?.seek(0);
          }
          // Always update detectedBpm / detectedKey / detectedTimeSignature
          // so the advanced strip's "Detected: …" readout reflects what
          // the server actually resolved — even when the user overrode
          // it below.
          const perf = usePerformanceStore.getState();
          const rawTs = detail.time_signature;
          const detectedTs = rawTs != null && isTimeSignature(rawTs)
            ? rawTs
            : null;
          const detectedBpm =
            typeof detail.bpm === "number" ? detail.bpm : perf.detectedBpm;
          if (detail.bpm != null || detail.key || detectedTs) {
            perf.setDetected(
              detectedBpm,
              detail.key ?? perf.detectedKey,
              detectedTs ?? perf.detectedTimeSignature,
            );
          }
          // One-shot override (set by AlmostReadyDialog's "Set
          // manually" mode) wins over the server's detection and is
          // cleared after use. Tell the server too so the model hint
          // matches what the UI shows. The same prompt re-encode
          // carries any time-signature override; the server's prompt
          // handler reads both fields off the same message.
          const keyOverride = perf.pendingKeyOverride;
          const tsOverride = perf.pendingTimeSignatureOverride;
          if (keyOverride || tsOverride) {
            if (keyOverride) {
              perf.setKey(keyOverride);
              perf.setPendingKeyOverride(null);
            } else if (detail.key) {
              perf.setKey(detail.key);
            }
            if (tsOverride) {
              perf.setTimeSignature(tsOverride);
              perf.setPendingTimeSignatureOverride(null);
            } else if (detectedTs) {
              perf.setTimeSignature(detectedTs);
            }
            const finalKey =
              keyOverride
              ?? detail.key
              ?? usePerformanceStore.getState().activeKey;
            const finalTs =
              tsOverride
              ?? usePerformanceStore.getState().activeTimeSignature;
            remote.sendPrompt(perf.promptA, finalKey, finalTs);
          } else {
            if (detail.key) perf.setKey(detail.key);
            if (detectedTs) perf.setTimeSignature(detectedTs);
          }
          resolve(true);
        };
        const onFail = (e: Event) => {
          remote.removeEventListener("swap_ready", onReady);
          remote.removeEventListener("swap_failed", onFail);
          const error = (e as CustomEvent).detail;
          console.warn("[fixture-swap] server swap_failed:", error);
          if (useCustomTracksStore.getState().resolveSourceMode(name)) {
            useCustomTracksStore
              .getState()
              .setStemStatus(name, "failed", String(error || "Swap failed"));
          }
          resolve(false);
        };
        remote.addEventListener("swap_ready", onReady);
        remote.addEventListener("swap_failed", onFail);

        const perf = usePerformanceStore.getState();
        const sourceMode = useCustomTracksStore
          .getState()
          .resolveSourceMode(name);
        // Record what we're about to send BEFORE the round-trip so the
        // server's `stem_assets` source_mode echo (which calls
        // setSourceMode) is recognised as already-applied and doesn't
        // re-enter the hotswap subscription below.
        lastSwappedMode.current = sourceMode ?? null;
        if (sourceMode) {
          useCustomTracksStore.getState().setStemStatus(name, "processing");
        }
        // Key is intentionally NOT sent: the server resolves via the
        // new fixture's sidecar (or CNN-detects on a miss) and echoes
        // the result in `swap_ready.key`, which we write into the
        // dropdown via setKey(detail.key) above. Sending perf.activeKey
        // here was the regression that made switching between test
        // fixtures stick on the previous fixture's key — the dropdown's
        // stale value was applied as `key_override` and beat the new
        // fixture's sidecar.key on the server side.
        // Operator overrides flow through the OperatorStrip dropdown's
        // onChange handler (sendPrompt), not through swap_source.
        const sent = serverResident
          ? remote.sendSwapSourceByName(
              name,
              perf.promptA,
              undefined,
              undefined,
              sourceMode,
            )
          : remote.sendSwapSource(
              interleaved as Float32Array,
              channels,
              perf.promptA,
              undefined,
              name,
              undefined,
              sourceMode,
            );
        if (!sent) {
          remote.removeEventListener("swap_ready", onReady);
          remote.removeEventListener("swap_failed", onFail);
          resolve(false);
        }
      });

      if (cancelled) return;
      if (!ok) {
        setStatus("ready", "Swap failed — please try again");
        return;
      }
      lastSwappedTo.current = name;
      // A source-mode hotswap (force) is the same song with a different
      // stem feeding inference — skip the new-track gate entirely so the
      // performer's denoise / remix-started state is left untouched.
      if (!force) {
        // Each new track re-enters the "hear source first" gate when
        // enabled in config: snap engine denoise to 0 (user hears the
        // source from frame 1) and play a visual-only glide on the ribbon
        // from its prior position down to 0 as a "this is a slider" hint.
        // remixStarted always clears so the "drag to start" affordance
        // shows again; side-rail hints stay suppressed until the user
        // drags the top ribbon up. Shares config with useStartSession so
        // one knob controls both Play and swap.
        //
        // skipNextDenoiseGate is a one-shot opt-out for saved-session
        // resumes: the demo-side applySessionState sets it before
        // writing perf.fixture so the gate doesn't trample the restored
        // denoise value with a snap-to-zero. Consume-and-clear here so
        // the next legitimate swap reverts to the normal behaviour.
        const perfState = usePerformanceStore.getState();
        const gate = getConfig().denoise_session_gate;
        if (perfState.skipNextDenoiseGate) {
          perfState.setSkipNextDenoiseGate(false);
        } else if (gate.enabled) {
          const prevDenoise = perfState.sliderTargets["denoise"] ?? 0;
          perfState.setSliderDirect("denoise", 0);
          perfState.animateSliderDisplayFrom("denoise", prevDenoise, gate.glide_ms);
        }
        perfState.setRemixStarted(false);
      }
      setStatus("ready", "Playing");
    };

    const unsub = usePerformanceStore.subscribe((s, prev) => {
      if (s.fixture !== prev.fixture && s.fixture) {
        // One-shot opt-out: useMcpMirror sets this when adopting an
        // MCP-driven swap whose audio was already swapped server-side.
        // Skip the load+sendSwapSource round-trip, just record the new
        // fixture as already-swapped so a later real user pick still
        // works. Consumed regardless of source.
        if (s.skipNextFixtureSwap) {
          usePerformanceStore.getState().setSkipNextFixtureSwap(false);
          lastSwappedTo.current = s.fixture;
          return;
        }
        void run(s.fixture);
      }
    });

    // Hotswap the inference source (full ↔ vocals ↔ instruments) for the
    // active upload. The source-mode switch writes setSourceMode(); we
    // re-run the swap for the same fixture so the server re-encodes the
    // chosen stem. Guards: only the *active* fixture, only a real change,
    // and never the server's own source_mode echo (lastSwappedMode).
    const unsubSource = useCustomTracksStore.subscribe((s, prev) => {
      const fixture = usePerformanceStore.getState().fixture;
      if (!fixture) return;
      const mode = s.tracks.get(fixture)?.sourceMode;
      if (!mode) return;
      const prevMode = prev.tracks.get(fixture)?.sourceMode;
      if (mode === prevMode) return; // some other track / field changed
      if (mode === lastSwappedMode.current) return; // server echo, already live
      void run(fixture, true);
    });

    // Seed lastSwappedTo with the current fixture so the initial population
    // (catalog → default fixture write) doesn't trigger a no-op swap, and
    // seed lastSwappedMode so the first stem_assets echo is recognised.
    lastSwappedTo.current = usePerformanceStore.getState().fixture;
    lastSwappedMode.current =
      useCustomTracksStore
        .getState()
        .resolveSourceMode(lastSwappedTo.current) ?? null;

    return () => {
      cancelled = true;
      unsub();
      unsubSource();
    };
  }, []);
}
