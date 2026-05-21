"use client";

import { useEffect, useRef } from "react";

import { loadFixtureAudio } from "@/engine/audio/loadFixture";
import { getConfig } from "@/lib/config";
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

  useEffect(() => {
    let cancelled = false;

    const run = async (name: string) => {
      if (cancelled) return;
      const session = useSessionStore.getState();
      if (session.status !== "ready" || !session.remote || !session.player) {
        return; // No live session yet; the next Play will pick the new fixture.
      }
      if (lastSwappedTo.current === name) return;

      const { setStatus } = useSessionStore.getState();
      setStatus("ready", `Loading ${name}…`);

      let interleaved: Float32Array;
      let channels: number;
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
          session.player?.swap(detail.interleaved, detail.channels);
          // Worklet's swap message keeps `position` untouched, so a swap
          // at 1:30 into the old track would otherwise resume at 1:30 of
          // the new track. The ScriptProcessor fallback already restarts
          // (AudioPlayer.swap sets _spPosition = 0); this aligns the
          // worklet path. Operator can disable via config.
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
        const sent = remote.sendSwapSource(
          interleaved,
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

    // Seed lastSwappedTo with the current fixture so the initial population
    // (catalog → default fixture write) doesn't trigger a no-op swap.
    lastSwappedTo.current = usePerformanceStore.getState().fixture;

    return () => {
      cancelled = true;
      unsub();
    };
  }, []);
}
