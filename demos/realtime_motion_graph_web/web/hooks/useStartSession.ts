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
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
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

/**
 * Capability probe: ask the target pod (via its HTTP origin, derived
 * from the ws URL) whether it can load known fixtures server-side.
 * Returns the advertised `server_side_fixtures` list, or `[]` on any
 * failure / old backend. Never throws.
 *
 * This is what makes the server-side-fixture path safe across a mixed
 * fleet and ANY deploy/merge order: an old backend doesn't advertise
 * the capability, so the UI falls back to the (unchanged) upload path
 * instead of omitting the audio frame and hanging the old pod's recv.
 */
async function probeServerSideFixtures(wsUrl: string): Promise<string[]> {
  try {
    const u = new URL(wsUrl);
    u.protocol = u.protocol === "wss:" ? "https:" : "http:";
    u.pathname = "/api/server-info";
    u.search = "";
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 4000);
    let res: Response;
    try {
      res = await fetch(u.toString(), { signal: ctrl.signal });
    } finally {
      clearTimeout(timer);
    }
    if (!res.ok) return [];
    const info = (await res.json()) as { server_side_fixtures?: unknown };
    return Array.isArray(info.server_side_fixtures)
      ? (info.server_side_fixtures as string[])
      : [];
  } catch {
    return [];
  }
}

// Drives the whole "click Play" flow:
//   1. resolve fixture (use store, fall back to first listed)
//   2. load + decode audio
//   3. open WS with config built from current store state
//   4. on "ready": init AudioPlayer with initial buffer
//   5. wire slice → patch/addDelta, lora_catalog → useLoraStore, etc.
//   6. resume audio context

function buildConfig(
  fixtureName: string,
  useServerFixture: boolean,
): SessionConfig {
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
  const custom = useCustomTracksStore.getState();
  const sourceMode = custom.resolveSourceMode(fixtureName);
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
    prompt_b: perf.promptB,
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
    ...(sourceMode ? { stem_source_mode: sourceMode } : {}),
    // Only set when the target pod advertised it can load this fixture
    // server-side (capability-gated by the caller via
    // probeServerSideFixtures). When true the pod reads the waveform
    // from its own /fixtures cache and the client sends no audio frame
    // (saves the ~20 MB / ~11 s round-trip). When false we fall back to
    // the unchanged upload path, so this is safe on a mixed fleet in
    // any deploy/merge order.
    use_server_fixture: useServerFixture,
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

    // Resolve the target pod first, then ask it (server-info) whether
    // it can load this fixture server-side. Capability-gated so the UI
    // is safe across a mixed fleet / any deploy order: an old backend
    // doesn't advertise the fixture, so we keep the upload path.
    const wsUrl = resolveWsUrl(useSessionStore.getState().wsUrl);
    const serverSideFixtures = await probeServerSideFixtures(wsUrl);
    const useServerFixture = serverSideFixtures.includes(fixtureName);

    let interleaved: Float32Array;
    let channels: number;
    if (useServerFixture) {
      // Pod loads the waveform from its own /fixtures cache; skip the
      // client download/decode/upload entirely. Playback comes from the
      // server's echoed initial buffer (player.init uses
      // remote.initialBuffer), so the local decode was only ever
      // feeding the now-eliminated upload.
      interleaved = new Float32Array(0);
      channels = 2;
    } else {
      // Old backend or unknown fixture: upload as before.
      try {
        const decoded = await loadFixtureAudio(fixtureName);
        interleaved = decoded.interleaved;
        channels = decoded.channels;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setStatus("error", `Track failed to load: ${msg}`);
        return;
      }
    }

    setStatus("connecting", "Connecting…");
    const sourceMode = useCustomTracksStore
      .getState()
      .resolveSourceMode(fixtureName);
    if (sourceMode) {
      useCustomTracksStore.getState().setStemStatus(fixtureName, "processing");
    }
    const config = buildConfig(fixtureName, useServerFixture);
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

    remote.addEventListener("stem_assets", (e) => {
      const detail = (e as CustomEvent<{
        fixture_name?: string;
        sample_rate: number;
        channels: number;
        frames: number;
        source_mode?: "full" | "vocals" | "instruments";
        buffers: Record<"vocals" | "instruments", Float32Array>;
      }>).detail;
      const name = detail.fixture_name || usePerformanceStore.getState().fixture;
      if (!name) return;
      if (detail.source_mode) {
        useCustomTracksStore.getState().setSourceMode(name, detail.source_mode);
      }
      useCustomTracksStore.getState().setStems(name, {
        vocals: {
          interleaved: detail.buffers.vocals,
          channels: detail.channels,
          frames: detail.frames,
          sampleRate: detail.sample_rate,
        },
        instruments: {
          interleaved: detail.buffers.instruments,
          channels: detail.channels,
          frames: detail.frames,
          sampleRate: detail.sample_rate,
        },
      });
    });

    remote.addEventListener("stem_failed", (e) => {
      const detail = (e as CustomEvent<{
        fixture_name?: string;
        error?: string;
      }>).detail;
      const name = detail.fixture_name || usePerformanceStore.getState().fixture;
      if (!name) return;
      useCustomTracksStore
        .getState()
        .setStemStatus(name, "failed", detail.error || "Stem extraction failed");
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
    // remixStarted always resets so the affordance shows again.
    //
    // skipNextDenoiseGate is a one-shot opt-out for saved-session
    // resumes: applySessionState sets it before writing perf.fixture so
    // the just-restored denoise survives this hook's snap-to-zero. The
    // useFixtureSwap consumer normally clears it, but on a fresh resume
    // its run() bails on session.status !== "ready" without firing, so
    // we consume-and-clear here too. Fresh sessions don't set the flag,
    // so their gate behaviour is unchanged.
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

    setSession(remote, player);
    setStatus("ready", "Playing");

    // Start the network-quality monitor now that the WS is "ready".
    // Lives on the session store so reset() (called at next session
    // start) tears it down — no orphan intervals across hot reloads.
    useSessionStore.getState().setMonitor(createNetworkMonitor(remote));
  }, []);
}
