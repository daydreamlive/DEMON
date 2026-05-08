"use client";

import { useCallback } from "react";

import { AudioPlayer } from "@/engine/audio/AudioPlayer";
import { listFixtures, loadFixtureAudio, pickDefaultFixture } from "@/engine/audio/loadFixture";
import { defaultWsUrl } from "@/engine/podUrl";
import { RemoteBackend, SLICE_FLAG_DELTA } from "@/engine/protocol";
import { getApiKey } from "@/engine/rtmgConfig";
import { getConfig } from "@/lib/config";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
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
    key: perf.activeKey,
    enabled_loras: enabledLoras,
    prompt: perf.promptA,
    lora_strengths: loraStrengths,
    // Lets the server look up a precomputed sidecar (BPM, key, source
    // latent, context_latent). Absent / unknown name -> live path.
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

    // Server-detected metadata flows into the perf store so the key select
    // and HUD reflect it without manual sync.
    usePerformanceStore
      .getState()
      .setDetected(remote.detectedBpm, remote.detectedKey);
    if (remote.loraCatalog.length > 0) {
      useLoraStore.getState().setCatalog(remote.loraCatalog);
    }

    // "Hear the source first" gate: every fresh session starts with the
    // engine value at 0 so the user hears the unmodified track from
    // frame 1. The top-edge ribbon then plays a *visual-only* glide
    // from its prior position down to 0 — purely a hint that the
    // ribbon is a slider; the engine value never moves with it. The
    // "drag to start" affordance prompts them to dial it back up; the
    // first value-changing drag flips remixStarted true.
    const perfState = usePerformanceStore.getState();
    const prevDenoise = perfState.sliderTargets["denoise"] ?? 0;
    perfState.setSliderDirect("denoise", 0);
    perfState.animateSliderDisplayFrom("denoise", prevDenoise, 700);
    perfState.setRemixStarted(false);

    setSession(remote, player);
    setStatus("ready", "Playing");
  }, []);
}
