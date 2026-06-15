"use client";

import { useCallback } from "react";

import { AudioPlayer } from "@demon/client";
import { listFixtures, loadFixtureAudio, pickDefaultFixture } from "@/engine/audio/loadFixture";
import { createNetworkMonitor } from "@/engine/networkMonitor";
import { defaultWsUrl } from "@/engine/podUrl";
import { PREEMPTED_CLOSE_CODE, RemoteBackend, SAMPLE_RATE, SLICE_FLAG_DELTA } from "@demon/client";
import { getApiKey, getClientId } from "@/engine/rtmgConfig";
import { WsReconnector } from "@demon/client";
import {
  applyLoraCapWithServerSync,
  getConfig,
  resolveLoraCapForSource,
} from "@/lib/config";
import { wirePromptTransform } from "@/lib/loraTriggers";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore, type RefSource } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { isTimeSignature } from "@/types/engine";
import type { AudioSlice, SessionConfig, WsTrace } from "@demon/client";

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
    const basePath = u.pathname.replace(/\/+$/, "");
    u.pathname = `${basePath}/api/server-info`;
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
//
// After "ready", a close-event handler watches the live WebSocket. If
// it sees a network-side close (not user-initiated — i.e. the pod
// tunnel dropped, the network blipped, RunPod hard-reset the worker),
// it kicks off WsReconnector to re-run essentially the same flow:
// fresh WS handshake with the same store state, then swap the new
// RemoteBackend into the session store. The AudioPlayer stays alive
// across reconnects so playback keeps looping the source while slices
// pause, then resumes streaming when the new backend reaches "ready".

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
  // Optional opaque per-browser identifier from the host (the demo's
  // standalone shell wires no getter, so this is null and the field is
  // omitted; demon-public-demo wires PostHog's distinct_id).
  const clientId = getClientId();
  return {
    telemetry_version: 1,
    sde: cfg.sde,
    lora: cfg.lora,
    depth: cfg.depth,
    vae_window: cfg.vae_window,
    crop: cfg.crop,
    steps: cfg.steps,
    fast_vae: cfg.fast_vae,
    walk_window: cfg.walk_window ?? false,
    walk_window_s: cfg.walk_window_s ?? 60,
    lead_floor_s: cfg.lead_floor_s,
    lead_ceiling_s: cfg.lead_ceiling_s,
    lead_release_tau_s: cfg.lead_release_tau_s,
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
    ...(clientId ? { client_id: clientId } : {}),
  };
}

/**
 * Attach every per-session listener (slice → AudioPlayer, lora_catalog,
 * stem assets, close) to a RemoteBackend. Extracted from the inline
 * setup so the reconnect path can re-attach the same handlers to a
 * freshly-built backend.
 *
 * The `onUnexpectedClose` callback fires when the WS closes for any
 * reason other than `RemoteBackend.close()` being called from the app
 * (e.g. starting a new session). 1006 (abnormal closure) and 1011
 * (server internal error) both flow through here; the reconnect
 * orchestrator decides whether to retry based on attempt count, not
 * close code. The previous "set status to closed" behaviour was a
 * dead-end — the user had to refresh to recover. Now it's a transient
 * status while the backoff loop runs.
 */
function wireRemoteListeners(
  remote: RemoteBackend,
  onUnexpectedClose: (e: CloseEvent | { code?: number; reason?: string }) => void,
): void {
  // Mirror engine identity + depth bounds into the session store. These
  // writes used to live inside RemoteBackend's ready handler; the SDK no
  // longer touches app stores, so the app subscribes instead. Attached
  // before connect() on both the initial and reconnect paths, so the
  // "ready" dispatch (fired once the initial buffer lands, just before
  // connect() resolves) is never missed.
  remote.addEventListener("ready", () => {
    const s = useSessionStore.getState();
    s.setCheckpointScale(remote.checkpointScale);
    s.setPipelineDepth(remote.pipelineDepth);
    s.setMaxPipelineDepth(remote.maxPipelineDepth);
    // Backend capability mask (null on pre-Phase-2 servers / replays =
    // ungated). The hand-coded swap/timbre/structure/LoRA panels read
    // it via useCapability.
    s.setCapabilities(remote.capabilities);
    // Activation-steering surface (null on servers without it = tiles
    // hidden). VoiceTile reads these to render the steering racks.
    s.setManualSlotCount(remote.manualSlotCount);
    s.setManualSlotCap(remote.manualSlotCap);
    s.setSteeringAvailable(remote.steeringAvailable);
  });
  remote.addEventListener("depth_applied", (e) => {
    useSessionStore
      .getState()
      .setPipelineDepth((e as CustomEvent<number>).detail);
  });
  remote.addEventListener("manual_slot_count", (e) => {
    useSessionStore
      .getState()
      .setManualSlotCount((e as CustomEvent<number | null>).detail);
  });
  // WS startup telemetry. The SDK only observes its own connection
  // (wsTrace + the optional init_ack echo); persisting the latest trace
  // for the debug surface is the app's job. Attached before connect()
  // (both paths), so traces from attempts that fail before setSession()
  // publishes the remote still land in the store.
  remote.addEventListener("ws_trace_update", (e) => {
    useSessionStore
      .getState()
      .setLastWsTrace((e as CustomEvent<WsTrace>).detail);
  });
  remote.addEventListener("ws_init_ack", () => {
    const s = useSessionStore.getState();
    s.setLastBackendSessionId(remote.backendSessionId);
    s.setLastBackendClientId(remote.backendClientId);
  });

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
    // Knob→ear latency trace: how far AHEAD of the live playhead each
    // fresh slice lands. `window.__demonLatTrace = true` to enable.
    // A steady `lead` ≈ the floor latency before any change is audible.
    if ((window as unknown as { __demonLatTrace?: boolean }).__demonLatTrace) {
      const startSec = startFrame / SAMPLE_RATE;
      const playheadSec = player.positionSec;
      // eslint-disable-next-line no-console
      console.log(
        `[lat] slice numGens=${detail.numGens} ` +
          `startSec=${startSec.toFixed(3)} playheadSec=${playheadSec.toFixed(3)} ` +
          `lead=${(startSec - playheadSec).toFixed(3)} ` +
          `decMs=${detail.decMs?.toFixed?.(1)} tickMs=${detail.tickMs?.toFixed?.(1)} ` +
          `t=${performance.now().toFixed(0)}`,
      );
    }
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
    // closedByUser is set by RemoteBackend.close() — i.e. another
    // session start is tearing this one down. Don't reconnect; just
    // get out of the way.
    if (remote.closedByUser) return;
    // Server preempted this session because a newer connection took
    // the pod (one-session-per-pod). FINAL: reconnecting would just
    // preempt the newer session back and ping-pong the pod through
    // full session rebuilds.
    if (detail?.code === PREEMPTED_CLOSE_CODE) {
      useSessionStore
        .getState()
        .setStatus(
          "closed",
          "Session ended: another connection took over this pod.",
        );
      return;
    }
    onUnexpectedClose(detail ?? { code: undefined, reason: undefined });
  });

  remote.addEventListener("error", () => {
    // The connect() promise rejects on initial-handshake errors; the
    // close listener handles post-ready drops. Nothing else to do here.
  });

  remote.addEventListener("server_error", (e) => {
    // Mid-session structured failure from the backend (e.g. the
    // generation pipeline died: code=pipeline_error). The server closes
    // the WS right after, so the close listener / reconnect loop still
    // runs — but surfacing the message first turns a silently frozen UI
    // into an explained one.
    const detail = (e as CustomEvent<{ code?: string; message?: string }>)
      .detail;
    useSessionStore
      .getState()
      .setStatus(
        "error",
        detail?.message || `Server error: ${detail?.code || "unknown"}`,
      );
  });
}

interface ResolvedFixture {
  fixtureName: string;
  useServerFixture: boolean;
  /** Decoded interleaved PCM, OR an empty Float32Array when the pod can
   *  load the fixture server-side (no audio frame on the wire). */
  interleaved: Float32Array;
  channels: number;
}

/**
 * Resolve the active fixture + decoded audio for the *initial* connect.
 * The result is cached and reused by the reconnect path: re-decoding a
 * fixture (especially a custom upload, which can be a multi-MB blob) on
 * every backoff attempt would burn ~hundreds of ms × N attempts for no
 * benefit — the user can't have changed fixtures while a "Reconnecting…"
 * placard is up.
 *
 * Returns null if there are no fixtures (only possible if the pod is
 * misconfigured), which the caller surfaces as a fatal error.
 */
async function resolveFixtureForConnect(): Promise<ResolvedFixture | null> {
  let fixtureName = usePerformanceStore.getState().fixture;
  if (!fixtureName) {
    const list = await listFixtures();
    fixtureName = pickDefaultFixture(list);
    if (fixtureName) {
      usePerformanceStore.getState().setFixture(fixtureName);
    }
  }
  if (!fixtureName) return null;

  const wsUrl = resolveWsUrl(useSessionStore.getState().wsUrl);
  const serverSideFixtures = await probeServerSideFixtures(wsUrl);
  const useServerFixture = serverSideFixtures.includes(fixtureName);

  let interleaved: Float32Array;
  let channels: number;
  if (useServerFixture) {
    interleaved = new Float32Array(0);
    channels = 2;
  } else {
    const decoded = await loadFixtureAudio(fixtureName);
    interleaved = decoded.interleaved;
    channels = decoded.channels;
  }
  return { fixtureName, useServerFixture, interleaved, channels };
}

/**
 * Re-apply the operator's timbre / structure references to a freshly
 * (re)connected backend. The server session boots with no overrides
 * and refs are NOT part of SessionConfig — so a reconnect would
 * silently drop them without this. Fixture refs re-send by name; clip
 * refs re-resolve their PCM from useCustomTracksStore via
 * loadFixtureAudio. Best-effort: a ref whose source no longer resolves
 * is skipped (the operator can re-pick it).
 *
 * Note: a ref-set that the server REJECTED still leaves its RefSource
 * recorded, so a later reconnect re-attempts it — harmless (it just
 * re-fails with the same status message). Perfect failure-rollback
 * fidelity is a follow-up.
 */
async function restoreRefs(remote: RemoteBackend): Promise<void> {
  const perf = usePerformanceStore.getState();
  const apply = async (
    ref: RefSource | null,
    sendFixture: (name: string) => void,
    sendSource: (i: Float32Array, c: number, n: string) => boolean,
  ): Promise<void> => {
    if (!ref) return;
    if (ref.mode === "fixture") {
      sendFixture(ref.name);
      return;
    }
    try {
      const decoded = await loadFixtureAudio(ref.name);
      sendSource(decoded.interleaved, decoded.channels, ref.name);
    } catch {
      // Source no longer resolvable — skip.
    }
  };
  await apply(
    perf.timbreRef,
    (n) => remote.sendSetTimbreFixture(n),
    (i, c, n) => remote.sendSetTimbreSource(i, c, n),
  );
  await apply(
    perf.structRef,
    (n) => remote.sendSetStructureFixture(n),
    (i, c, n) => remote.sendSetStructureSource(i, c, n),
  );
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
    // A fresh session boots with no timbre / structure override. Drop
    // any RefSource recorded by a prior session so the reconnect path
    // (restoreRefs) can't re-apply a ref the operator didn't set this
    // session. Reconnects go through buildAndConnect, never this hook,
    // so the in-session record is preserved across a recovery.
    usePerformanceStore.getState().setTimbreRef(null);
    usePerformanceStore.getState().setStructRef(null);

    setStatus("loading-fixture", "Loading track…");

    let resolved: ResolvedFixture | null;
    try {
      resolved = await resolveFixtureForConnect();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus("error", `Track failed to load: ${msg}`);
      return;
    }
    if (!resolved) {
      setStatus("error", "No tracks available. Please refresh.");
      return;
    }
    // Snapshot the resolved fixture for the lifetime of this session.
    // The reconnect factory reuses these values rather than re-running
    // resolveFixtureForConnect on every backoff attempt — re-decoding
    // a custom-upload Float32Array (potentially several MB) up to N
    // times during a brief outage is wasted work, and the user can't
    // have swapped fixtures while a "Reconnecting…" placard is up.
    const sessionFixture: ResolvedFixture = resolved;

    setStatus("connecting", "Connecting…");
    if (sessionFixture.fixtureName) {
      const sourceMode = useCustomTracksStore
        .getState()
        .resolveSourceMode(sessionFixture.fixtureName);
      if (sourceMode) {
        useCustomTracksStore
          .getState()
          .setStemStatus(sessionFixture.fixtureName, "processing");
      }
    }

    // Forward-declared so the close handler can reference the same
    // function the initial connect uses.
    let triggerReconnect: (closeInfo?: { code?: number; reason?: string }) => void;

    /**
     * Connect to the pod with current store state. Returns the connected
     * RemoteBackend on success; throws on failure. Used by both the
     * initial Play flow and the reconnect orchestrator — they only
     * differ in what they do with the result (init player vs swap
     * remote into session store).
     *
     * On rejection we close the orphan remote with `closedByUser=true`
     * so its (potentially delayed) close-event dispatch can't sneak
     * past the reconnector and start a fresh recovery loop after
     * we've already given up. The user-facing `triggerReconnect` also
     * guards against this on the read side, but closing the orphan
     * here is the structural fix.
     */
    async function buildAndConnect(): Promise<RemoteBackend> {
      const wsUrl = resolveWsUrl(useSessionStore.getState().wsUrl);
      const config = buildConfig(
        sessionFixture.fixtureName,
        sessionFixture.useServerFixture,
      );
      const remote = new RemoteBackend(
        wsUrl,
        sessionFixture.interleaved,
        sessionFixture.channels,
        config,
        { promptTransform: wirePromptTransform },
      );
      wireRemoteListeners(remote, (detail) => triggerReconnect(detail));
      try {
        await remote.connect();
      } catch (err) {
        remote.close();
        throw err;
      }
      return remote;
    }

    /**
     * Kick off a reconnect attempt. Idempotent: if a reconnector is
     * already running (e.g. the new connection dropped immediately
     * during recovery), this no-ops so we don't stack backoff loops.
     */
    triggerReconnect = (closeInfo) => {
      const state = useSessionStore.getState();
      if (state.reconnector) return;
      // Defense in depth against orphan-WS close events: if we already
      // gave up (status === "error") or the user reset the session
      // (status === "idle" / "closed"), don't resurrect the loop. The
      // structural fix is `buildAndConnect`'s catch closing the failed
      // remote with closedByUser=true, but a late close from somewhere
      // we haven't accounted for still gets filtered here.
      if (state.status === "error" || state.status === "idle" || state.status === "closed") {
        return;
      }
      const reasonStr =
        closeInfo?.reason || `code ${closeInfo?.code ?? "unknown"}`;
      state.setStatus("reconnecting", `Connection lost (${reasonStr}). Reconnecting…`);

      const reconnector = new WsReconnector(
        async () => {
          // Each attempt builds a completely fresh RemoteBackend; the
          // dropped one is already orphaned (its socket is closed, its
          // worker is terminated by its own close() flow when ws.close()
          // landed, or by GC otherwise — the slice listener path
          // gracefully handles the empty case via getState().player).
          const remote = await buildAndConnect();
          if (!remote.initialBuffer) {
            throw new Error("Reconnected but server sent no initial buffer");
          }
          const player = useSessionStore.getState().player;
          if (!player) {
            // Player was torn down between the failure and recovery —
            // a user-initiated session start cancelled the loop, but
            // we got here anyway because the cancel raced the promise
            // resolution. Close the new remote and bail.
            remote.close();
            throw new Error("Session was reset during reconnect");
          }
          // Reset the player's buffer to the fresh server-side source
          // before swapping in the new remote. Slices are encoded as
          // deltas against the server's `client_mirror`, which the new
          // session re-initializes to the clean source. Without this
          // swap the player's mirror still has accumulated generation
          // from the previous session, so `addDelta(generated -
          // source)` lands as `prev_generated + (new_gen - source)` —
          // audible noise. The worklet's swap message preserves the
          // playhead position and crossfades the buffer, so the
          // listener hears a brief return to clean source then the
          // new session's slices stream in normally.
          player.swap(remote.initialBuffer, remote.channels);
          // `player.swap()` bumped `swapCount`, but a fresh
          // RemoteBackend's `_sliceEpoch` starts at 0. The slice
          // listener drops anything where `detail.epoch !==
          // player.swapCount`, so without this realignment every
          // slice from the recovered session would be filtered out
          // — symptom on the user side: source plays, controls
          // appear dead, denoised audio never returns.
          remote.setSliceEpoch(player.swapCount);

          const rawTs = remote.detectedTimeSignature;
          const detectedTs =
            rawTs != null && isTimeSignature(rawTs) ? rawTs : null;
          usePerformanceStore
            .getState()
            .setDetected(remote.detectedBpm, remote.detectedKey, detectedTs);
          if (remote.loraCatalog.length > 0) {
            useLoraStore.getState().setCatalog(remote.loraCatalog);
          }
          // Recompute the duration-aware LoRA cap against the (re)bound
          // source. The reconnect path may land on a different-duration
          // source than the prior session — tiers swap to match.
          // Uses the server-syncing helper so any over-cap LoRAs from
          // the prior session also get a WS disable_lora (avoids
          // ghost-LoRA leak when the reconnected source's tier is
          // tighter than what was previously enabled).
          applyLoraCapWithServerSync(resolveLoraCapForSource(remote.duration));
          useSessionStore.getState().setSession(remote, player);
          // Rebuild the network-quality monitor against the new
          // remote — the old one was bound to the dropped backend's
          // `slice` events and is dead now.
          try {
            useSessionStore.getState().monitor?.stop();
          } catch {}
          useSessionStore
            .getState()
            .setMonitor(createNetworkMonitor(remote, player));
          // Re-apply the timbre / structure references the operator
          // had active. The fresh server session boots with none and
          // refs aren't carried in buildConfig, so without this a
          // reconnect silently drops them.
          await restoreRefs(remote);
        },
        {
          onAttempt: ({ attempt, maxAttempts }) => {
            useSessionStore
              .getState()
              .setStatus(
                "reconnecting",
                `Reconnecting (attempt ${attempt}/${maxAttempts})…`,
              );
          },
          onSuccess: () => {
            useSessionStore.getState().setReconnector(null);
            useSessionStore.getState().setStatus("ready", "Playing");
          },
          onGiveUp: (err) => {
            useSessionStore.getState().setReconnector(null);
            useSessionStore
              .getState()
              .setStatus(
                "error",
                `Reconnect failed after multiple attempts: ${err.message}. Refresh to retry.`,
              );
          },
        },
      );
      useSessionStore.getState().setReconnector(reconnector);
      void reconnector.run();
    };

    const config = buildConfig(resolved.fixtureName, resolved.useServerFixture);
    const wsUrl = resolveWsUrl(useSessionStore.getState().wsUrl);
    const remote = new RemoteBackend(
      wsUrl,
      resolved.interleaved,
      resolved.channels,
      config,
      { promptTransform: wirePromptTransform },
    );
    // Wire engine → store. Bind BEFORE connect() so we never miss the first
    // few slices (server can send within milliseconds of "ready").
    wireRemoteListeners(remote, (detail) => triggerReconnect(detail));

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

    const player = new AudioPlayer({
      loudnessConfig: () => getConfig().audio,
    });
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
    // First cap-recompute against the actual source duration now that
    // ready landed (remote.duration was set by the ready handshake).
    // applyConfig's boot-time setMaxEnabled used the most-permissive
    // tier; this nails the cap to whatever engine workspace this
    // particular source actually loads.
    //
    // Server-syncing helper: if the user-driven default seed enabled
    // more LoRAs than the active source's tier allows, the helper
    // sends disable_lora for each that gets clipped, preventing the
    // ghost-LoRA leak where the server keeps them materialized
    // server-side after they vanish from the UI.
    applyLoraCapWithServerSync(resolveLoraCapForSource(remote.duration));

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
    useSessionStore.getState().setMonitor(createNetworkMonitor(remote, player));
  }, []);
}
