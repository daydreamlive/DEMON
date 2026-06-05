"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { AudioPlayer, RemoteBackend, SLICE_FLAG_DELTA } from "@demon/client";
import type {
  AudioSlice,
  KnobManifest,
  KnobManifestEntry,
  SessionConfig,
} from "@demon/client";
import { defaultWsUrl, podHttp } from "@/engine/podUrl";

// SA3-only session hook. Deliberately standalone, like the magenta
// route: no app stores, no LoRA catalog, no timbre/structure — the sa3
// family's v1 surface is prompt + the sa3 knob bank, everything else is
// capability-gated off server-side. What IS sa3-specific:
//
//   * The source anchor matters musically (audio-to-audio: every emit
//     is a partial-denoise cover of it), so the user picks a pod-side
//     fixture and we start with `use_server_fixture` — no PCM upload,
//     the server reads the waveform off its own disk.
//   * `sa3_duration_s` optionally fixes the generation window; absent,
//     the server derives it from the fixture length (capped at 120 s).
//   * No Tags B / blend: sa3 v1 keeps one conditioning bundle at a
//     time; "Send Prompt" re-runs the conditioner server-side.
//
// The knob bank comes from the session's own `ready.knob_manifest`
// (backend-owned, session-resolved), NOT the static /api/knobs probe —
// the panel renders whatever the sa3 backend declares (sa3_denoise,
// seed, steps_override today) and future knobs appear without UI edits.

export type SA3Status = "idle" | "connecting" | "ready" | "error";

export interface SA3Knob {
  name: string;
  entry: KnobManifestEntry;
}

// Constructor stub only: with use_server_fixture set the SDK skips the
// PCM send entirely (and the server never recvs), so these bytes never
// leave the tab.
const STUB_FRAMES = 9600;
const STUB_CHANNELS = 2;

const PARAMS_TICK_MS = 80;

function knobList(manifest: KnobManifest): SA3Knob[] {
  // Insertion order is the backend's declaration order — keep it.
  return Object.entries(manifest)
    .filter(([, entry]) => entry.type === "float" || entry.type === "int")
    .map(([name, entry]) => ({ name, entry }));
}

function defaultValues(knobs: SA3Knob[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const { name, entry } of knobs) {
    if (typeof entry.default === "number") out[name] = entry.default;
  }
  return out;
}

export function useSA3Session() {
  const remoteRef = useRef<RemoteBackend | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const tickRef = useRef<number | null>(null);
  const valuesRef = useRef<Record<string, number>>({});

  const [status, setStatus] = useState<SA3Status>("idle");
  const [message, setMessage] = useState("");
  const [knobs, setKnobs] = useState<SA3Knob[]>([]);
  const [values, setValues] = useState<Record<string, number>>({});
  const [fixtures, setFixtures] = useState<string[]>([]);

  // Pod-side fixture catalog for the source-anchor picker. server-info
  // is the right probe (vs /api/fixtures) because it lists exactly the
  // names the backend accepts for use_server_fixture.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(podHttp("/api/server-info"));
        const info = (await res.json()) as {
          server_side_fixtures?: string[];
        };
        if (!cancelled) setFixtures(info.server_side_fixtures ?? []);
      } catch {
        if (!cancelled) setFixtures([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

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
    async (prompt: string, fixtureName: string, durationS: number | null) => {
      await stop();
      setStatus("connecting");
      setMessage("Connecting…");
      try {
        const config: SessionConfig = {
          telemetry_version: 1,
          backend: "sa3",
          prompt,
          use_server_fixture: true,
          fixture_name: fixtureName,
        };
        if (durationS != null && durationS > 0) {
          config.sa3_duration_s = durationS;
        }
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
        // enough to omit it can't run sa3 sessions anyway, so an empty
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

        // Continuous params flow: the runner samples knob values each
        // tick, so keep them streaming rather than edge-triggered.
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

  const sendPrompt = useCallback((prompt: string) => {
    // No tags_b: sa3 v1 holds one conditioning bundle; the server
    // re-runs prepare_cond for this prompt and swaps it in.
    remoteRef.current?.sendPrompt(prompt);
  }, []);

  return {
    fixtures,
    knobs,
    message,
    sendPrompt,
    setKnob,
    start,
    status,
    stop,
    values,
  };
}
