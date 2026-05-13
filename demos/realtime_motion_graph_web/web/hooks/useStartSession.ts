"use client";

import { useCallback } from "react";

import { AudioPlayer } from "@/engine/audio/AudioPlayer";
import { listFixtures, loadFixtureAudio, pickDefaultFixture } from "@/engine/audio/loadFixture";
import { resetKnobDelta } from "@/engine/midi/absoluteDelta";
import { createNetworkMonitor } from "@/engine/networkMonitor";
import { defaultWsUrl } from "@/engine/podUrl";
import { RemoteBackend, SLICE_FLAG_DELTA } from "@/engine/protocol";
import { getApiKey } from "@/engine/rtmgConfig";
import { getConfig } from "@/lib/config";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { isTimeSignature } from "@/types/engine";
import type { AudioSlice, SessionConfig } from "@/types/protocol";

/**
 * Resolve the WS URL for this session. Preference order:
 *   1. server-issued wsUrl (from /api/queue/join, signed when RTMG_TOKEN_SECRET set)
 *   2. defaultWsUrl() — `?ws=` URL override or NEXT_PUBLIC_POD_BASE_URL
 *
 * If signed in, the daydream apiKey is appended as `?apiKey=<key>` so the
 * pod can identify the user it's debiting via webhook. Mirrors the rtmg
 * engine's WS handshake — the pod side must accept the parameter (out of
 * scope for this repo).
 */
function resolveWsUrl(serverWsUrl: string | null): string {
  let url = serverWsUrl ?? defaultWsUrl();
  const apiKey = getApiKey();
  if (apiKey) {
    const sep = url.includes("?") ? "&" : "?";
    url = `${url}${sep}apiKey=${encodeURIComponent(apiKey)}`;
  }
  return url;
}

// Drives the whole "click Play" flow:
//   1. resolve fixture (use store, fall back to first listed)
//   2. load + decode audio
//   3. open WS with config built from current store state
//   4. on "ready": init AudioPlayer with initial buffer
//   5. wire slice → patch/addDelta, lora_catalog → useLoraStore, etc.
//   6. resume audio context

function buildConfig(fixtureName: string): SessionConfig {
  const perf = usePerformanceStore.getState();
  const lora = useLoraStore.getState();
  const cfg = getConfig().engine;
  const enabledLoras = Array.from(lora.enabled);
  const loraStrengths: Record<string, number> = {};
  for (const id of enabledLoras) {
    const v = lora.strengths[id];
    if (typeof v === "number") loraStrengths[id] = v;
  }
  // Engine fields come from web/public/config.json (overridable per
  // installation). Default depth=4 over depth=8: ~½ VRAM, ~⅛ param
  // latency, ~11.3/s vs 12.3/s throughput on a 32 GB card. The VRAM
  // headroom is what unlocks longer audio uploads (cap lives in
  // loadFixture.ts; depth=4 makes future bumps VRAM-safe).
  return {
    sde: cfg.sde,
    lora: cfg.lora,
    depth: cfg.depth,
    vae_window: cfg.vae_window,
    crop: cfg.crop,
    steps: cfg.steps,
    fast_vae: cfg.fast_vae,
    walk_window: cfg.walk_window ?? false,
    walk_window_s: cfg.walk_window_s ?? 60,
    enabled_loras: enabledLoras,
    prompt: perf.promptA,
    lora_strengths: loraStrengths,
    // Lets the server look up a precomputed sidecar (BPM, key, source
    // latent, context_latent). Absent / unknown name -> live path.
    // Key is intentionally not sent: the server's session-init resolver
    // ignores config.key anyway and uses sidecar.key for known fixtures
    // (or CNN-detects on a miss). The result echoes back in `ready.key`
    // and `setDetected` writes it into the dropdown. Sending the
    // dropdown's stale value here would only re-introduce the
    // override-wins-over-sidecar regression.
    fixture_name: fixtureName,
  };
}

export function useStartSession() {
  return useCallback(async () => {
    const { setStatus, setSession, reset } = useSessionStore.getState();

    // Tear down any in-flight session.
    const prev = useSessionStore.getState();
    try {
      await prev.player?.close();
    } catch {}
    try {
      prev.remote?.close();
    } catch {}
    reset();
    // Clear the per-CC MIDI delta cache. Stale `lastValue` entries
    // from the previous session would otherwise produce phantom
    // deltas on the first knob wiggle of the new one — a long-
    // reported "knobs go crazy on session start" complaint.
    resetKnobDelta();

    setStatus("loading-fixture", "Loading track…");

    let fixtureName = usePerformanceStore.getState().fixture;
    if (!fixtureName) {
      try {
        const list = await listFixtures();
        fixtureName = pickDefaultFixture(list);
        if (fixtureName) {
          usePerformanceStore.getState().setFixture(fixtureName);
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setStatus("error", `Track list failed: ${msg}`);
        return;
      }
    }
    if (!fixtureName) {
      setStatus("error", "No tracks available. Please refresh.");
      return;
    }

    let interleaved: Float32Array;
    let channels: number;
    try {
      const decoded = await loadFixtureAudio(fixtureName);
      interleaved = decoded.interleaved;
      channels = decoded.channels;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus("error", `Track failed to load: ${msg}`);
      return;
    }

    setStatus("connecting", "Connecting…");
    const config = buildConfig(fixtureName);
    const wsUrl = resolveWsUrl(useSessionStore.getState().wsUrl);
    const remote = new RemoteBackend(
      wsUrl,
      interleaved,
      channels,
      config,
    );

    // Wire engine → store. Bind BEFORE connect() so we never miss the first
    // few slices (server can send within milliseconds of "ready").
    remote.addEventListener("slice", (e) => {
      const detail = (e as CustomEvent<AudioSlice>).detail;
      const player = useSessionStore.getState().player;
      if (!player) return;
      // Drop slices that were generated for a previous source. Without
      // this, slices already in the WS queue (or mid-decode in the
      // worker) at the moment the user swaps tracks would write into
      // the new buffer — audible as chunks of the previous song
      // bleeding through after a swap.
      if (detail.epoch !== player.swapCount) return;
      const startFrame = Math.floor(detail.startSample);
      if (detail.flags === SLICE_FLAG_DELTA) {
        player.addDelta(startFrame, detail.audio);
      } else {
        player.patch(startFrame, detail.audio);
      }
    });

    remote.addEventListener("lora_catalog", (e) => {
      const detail = (e as CustomEvent).detail;
      useLoraStore.getState().setCatalog(detail);
    });

    remote.addEventListener("close", (e) => {
      const detail = (e as CustomEvent<CloseEvent>).detail;
      const reason = detail?.reason || `code ${detail?.code}`;
      useSessionStore.getState().setStatus(
        "closed",
        `Something went wrong and you’ve been disconnected. Disconnect code: (${reason})`,
      );
    });

    remote.addEventListener("error", () => {
      // The connect() promise will reject too; defer messaging to that path.
    });

    try {
      await remote.connect();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus("error", msg);
      return;
    }

    if (!remote.initialBuffer) {
      setStatus("error", "Track failed to load.");
      return;
    }

    setStatus("connecting", "Starting audio…");

    const player = new AudioPlayer();
    try {
      await player.init(remote.initialBuffer, remote.channels);
      await player.resume();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus("error", `Audio failed to start: ${msg}`);
      return;
    }

    // Server-detected metadata flows into the perf store so the key /
    // time-signature selects + HUD reflect it without manual sync.
    const rawTs = remote.detectedTimeSignature;
    const detectedTs = rawTs != null && isTimeSignature(rawTs) ? rawTs : null;
    usePerformanceStore
      .getState()
      .setDetected(remote.detectedBpm, remote.detectedKey, detectedTs);
    if (remote.loraCatalog.length > 0) {
      useLoraStore.getState().setCatalog(remote.loraCatalog);
    }

    // "Hear the source first" gate: when enabled in config.json, every
    // session start snaps engine denoise to 0 and plays a visual-only
    // glide from the slider's prior value down to 0 over glide_ms. The
    // engine value never moves with the glide; it's a hint that the top
    // ribbon is a slider, and the "drag to start" affordance prompts the
    // user to dial it back up. controls.denoise in config.json seeds the
    // initial fresh-load value (only applyConfig() at module load sees
    // it); later sessions need this explicit reset to restore the gate.
    //
    // skipNextDenoiseGate is a one-shot opt-out: the import path
    // (applySessionState in demon-public-demo's lib/sessions/client.ts)
    // sets it after writing the sharer's sliderTargets, so a session
    // started from a shared / saved / pasted state plays from frame 1 at
    // the imported denoise instead of being snapped to 0 by the
    // onboarding cue. Consumed-and-cleared regardless of gate.enabled so
    // the flag never survives past one session start.
    //
    // remixStarted always resets so the affordance shows again.
    const perfState = usePerformanceStore.getState();
    const gate = getConfig().denoise_session_gate;
    const skipGate = perfState.skipNextDenoiseGate;
    if (skipGate) perfState.setSkipNextDenoiseGate(false);
    if (gate.enabled && !skipGate) {
      const prevDenoise = perfState.sliderTargets["denoise"] ?? 0;
      perfState.setSliderDirect("denoise", 0);
      perfState.animateSliderDisplayFrom("denoise", prevDenoise, gate.glide_ms);
    }
    perfState.setRemixStarted(false);

    setSession(remote, player);
    setStatus("ready", "Playing");

    // Start the network-quality monitor now that the WS is "ready".
    // Lives on the session store so reset() (called at next session
    // start) tears it down — no orphan intervals across hot reloads.
    useSessionStore.getState().setMonitor(createNetworkMonitor(remote));
  }, []);
}
