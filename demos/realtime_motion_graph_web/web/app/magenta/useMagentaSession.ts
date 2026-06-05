"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { AudioPlayer, RemoteBackend, SLICE_FLAG_DELTA } from "@demon/client";
import type {
  AudioSlice,
  KnobManifest,
  KnobManifestEntry,
  SessionConfig,
} from "@demon/client";
import { defaultWsUrl } from "@/engine/podUrl";

// Magenta-only session hook. Deliberately standalone: no app stores, no
// fixture decode, no LoRA catalog, no prompt transform — the mrt2 family
// has none of those surfaces, so this is just SDK + the four things a
// magenta session actually speaks: config, params, set_prompt,
// set_prompt_blend.
//
// The knob bank comes from the session's own `ready.knob_manifest`
// (backend-owned, session-resolved), NOT the static /api/knobs probe —
// so the panel renders whatever knobs the mrt2 backend declares, and
// future additions (notes/drums conditioning) appear without UI edits.

export type MagentaStatus = "idle" | "connecting" | "ready" | "error";

export interface MagentaKnob {
  name: string;
  entry: KnobManifestEntry;
}

// The mrt2 session creator ignores the handshake audio entirely (no
// positional source — the frontier starts from silence), but the WS
// adapter still reads one binary frame when no server-side fixture is
// named. Send the smallest honest stub: 0.2 s of stereo zeros, one
// server latent-pool (9600 samples) so every downstream length
// computation stays aligned.
const STUB_FRAMES = 9600;
const STUB_CHANNELS = 2;

const PARAMS_TICK_MS = 80;

function knobList(manifest: KnobManifest): MagentaKnob[] {
  // Insertion order is the registry's declaration order — keep it.
  return Object.entries(manifest)
    .filter(([, entry]) => entry.type === "float" || entry.type === "int")
    .map(([name, entry]) => ({ name, entry }));
}

function defaultValues(knobs: MagentaKnob[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const { name, entry } of knobs) {
    if (typeof entry.default === "number") out[name] = entry.default;
  }
  return out;
}

export function useMagentaSession() {
  const remoteRef = useRef<RemoteBackend | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const tickRef = useRef<number | null>(null);
  const valuesRef = useRef<Record<string, number>>({});

  const [status, setStatus] = useState<MagentaStatus>("idle");
  const [message, setMessage] = useState("");
  const [knobs, setKnobs] = useState<MagentaKnob[]>([]);
  const [values, setValues] = useState<Record<string, number>>({});

  const stop = useCallback(async () => {
    if (tickRef.current != null) {
      window.clearInterval(tickRef.current);
      tickRef.current = null;
    }
    try {
      await playerRef.current?.close();
    } catch {}
    try {
      remoteRef.current?.close();
    } catch {}
    playerRef.current = null;
    remoteRef.current = null;
    setStatus("idle");
    setMessage("");
  }, []);

  useEffect(() => {
    return () => {
      void stop();
    };
  }, [stop]);

  const start = useCallback(
    async (tagsA: string, tagsB: string, blend: number) => {
      await stop();
      setStatus("connecting");
      setMessage("Connecting…");
      try {
        const config: SessionConfig = {
          telemetry_version: 1,
          backend: "mrt2",
          prompt: tagsA,
          prompt_b: tagsB || tagsA,
        };
        const remote = new RemoteBackend(
          defaultWsUrl(),
          new Float32Array(STUB_FRAMES * STUB_CHANNELS),
          STUB_CHANNELS,
          config,
        );
        remoteRef.current = remote;

        remote.addEventListener("slice", (event) => {
          const detail = (event as CustomEvent<AudioSlice>).detail;
          const player = playerRef.current;
          if (!player || detail.epoch !== player.swapCount) return;
          const startFrame = Math.floor(detail.startSample);
          if (detail.flags === SLICE_FLAG_DELTA) {
            player.addDelta(startFrame, detail.audio);
          } else {
            player.patch(startFrame, detail.audio);
          }
        });
        remote.addEventListener("close", () => {
          if (remote.closedByUser) return;
          setStatus("error");
          setMessage("Connection lost.");
        });

        await remote.connect();
        if (!remote.initialBuffer) {
          throw new Error("server sent no initial buffer");
        }

        // Per-session knob bank from ready.knob_manifest. A server old
        // enough to omit it can't run mrt2 sessions anyway, so an empty
        // panel here would mean a contract break — surface it.
        const manifest = remote.knobManifest?.knobs ?? {};
        const list = knobList(manifest);
        const defaults = defaultValues(list);
        setKnobs(list);
        setValues(defaults);
        valuesRef.current = defaults;

        const player = new AudioPlayer();
        playerRef.current = player;
        await player.init(remote.initialBuffer, remote.channels);
        await player.resume();

        if (blend > 0) remote.sendSetPromptBlend(blend);

        // Continuous params flow, same cadence the one-knob POC used:
        // the runner samples knob values at each tick, so keep them
        // streaming rather than edge-triggered.
        tickRef.current = window.setInterval(() => {
          const liveRemote = remoteRef.current;
          const livePlayer = playerRef.current;
          if (!liveRemote || !livePlayer) return;
          liveRemote.sendParams(valuesRef.current, livePlayer.positionSec);
        }, PARAMS_TICK_MS);

        setStatus("ready");
        setMessage("");
      } catch (err) {
        await stop();
        setStatus("error");
        setMessage(err instanceof Error ? err.message : "Start failed");
      }
    },
    [stop],
  );

  const setKnob = useCallback((name: string, value: number) => {
    setValues((prev) => {
      const next = { ...prev, [name]: value };
      valuesRef.current = next;
      return next;
    });
    // Snappy path: ship immediately too; the tick is the steady-state.
    const remote = remoteRef.current;
    const player = playerRef.current;
    if (remote && player) {
      remote.sendParams(
        { ...valuesRef.current, [name]: value },
        player.positionSec,
      );
    }
  }, []);

  const sendTags = useCallback((tagsA: string, tagsB: string) => {
    remoteRef.current?.sendPrompt(
      tagsA,
      undefined,
      undefined,
      tagsB || undefined,
    );
  }, []);

  const setBlend = useCallback((value: number) => {
    remoteRef.current?.sendSetPromptBlend(Math.max(0, Math.min(1, value)));
  }, []);

  return {
    knobs,
    message,
    sendTags,
    setBlend,
    setKnob,
    start,
    status,
    stop,
    values,
  };
}
