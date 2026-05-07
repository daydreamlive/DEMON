"use client";

import { useCallback } from "react";

import { AudioPlayer } from "@/engine/audio/AudioPlayer";
import { listFixtures, loadFixtureAudio } from "@/engine/audio/loadFixture";
import { defaultWsUrl } from "@/engine/podUrl";
import { RemoteBackend, SLICE_FLAG_DELTA } from "@/engine/protocol";
import { getApiKey } from "@/engine/rtmgConfig";
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

function buildConfig(): SessionConfig {
  const perf = usePerformanceStore.getState();
  const lora = useLoraStore.getState();
  const enabledLoras = Array.from(lora.enabled);
  const loraStrengths: Record<string, number> = {};
  for (const id of enabledLoras) {
    const v = lora.strengths[id];
    if (typeof v === "number") loraStrengths[id] = v;
  }
  return {
    sde: false,
    lora: true,
    depth: 8,
    vae_window: 6,
    crop: 0,
    steps: 8,
    fast_vae: false,
    key: perf.activeKey,
    enabled_loras: enabledLoras,
    prompt: perf.promptA,
    lora_strengths: loraStrengths,
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
        fixtureName = list[0] ?? "";
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
    const config = buildConfig();
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

    setSession(remote, player);
    setStatus("ready", "Playing");
  }, []);
}
