"use client";

import { useEffect, useRef, useState } from "react";

import {
  decodeAudioFile,
  listFixtures,
  loadFixtureAudio,
  type DecodedFixture,
} from "@/engine/audio/loadFixture";
import { useConfig } from "@/lib/config";
import { LOCAL_MODE } from "@/lib/runtime";
import { trimAudioBuffer } from "@/lib/audio/trimAudioBuffer";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore, type RefSource } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import type { RemoteBackend } from "@demon/client";

import { RefSelect } from "./RefSelect";

// Fallback when engine.max_source_duration_s is unset. Mirrors
// AudioSourceCrate / TrackPicker / LiteTrackCarousel.
const DEFAULT_TRIM_CAP_S = 120;

// Server-side `websockets` library is configured with max_size=100 MiB
// (see demos/realtime_motion_graph_web/server.py). A 48 kHz stereo
// float32 clip longer than ~273 s crosses that cap and the server
// rejects the frame with close code 1009 — the upload silently fails
// for the user. The server also re-trims timbre / structure refs to
// the playback source's duration internally, so anything past
// max_source_duration_s is wire-wasted regardless. Head-trim here
// keeps the WS frame under the cap and saves the round-trip.
function clampRefDuration(
  decoded: DecodedFixture,
  capS: number,
): { decoded: DecodedFixture; trimmedFromS: number | null } {
  const durS = decoded.frames / decoded.sampleRate;
  if (durS <= capS) return { decoded, trimmedFromS: null };
  return {
    decoded: trimAudioBuffer(decoded, 0, capS),
    trimmedFromS: durS,
  };
}

function fmtMMSS(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// Unified picker for the timbre and structure references. Both
// reference axes share the exact same UI shape (caption + dropdown
// with "Input Track" pinned + library + your-tracks optgroups, plus a
// sibling upload icon button) and the exact same upload flow (decode
// locally → push to useCustomTracksStore → ship to the server). The
// only differences are the store field for the displayed name, the
// remote send method, and the status-bar prefix — everything else is
// shared.
//
// Default option at the top:
//   "Input Track"  — clear any active override; server falls back to
//                    encoding against the playback source's own latent.
//
// Upload sits NEXT TO the dropdown, not inside it: opening a file
// picker through the dropdown closes the menu and then the
// AlmostReadyDialog modal hides whatever the user was just comparing
// against. The sibling button lets the user upload without ever
// opening (or hiding) the list.

const VAL_INPUT = "__input__";

export type RefKind = "timbre" | "structure";

interface KindBinding {
  storeName: () => string | null;
  label: string;
  ariaLabel: string;
  statusPrefix: string;
  /** PCM upload path — for custom user tracks that only exist in the
   *  browser. Returns false if the WS isn't open. */
  send: (
    remote: RemoteBackend,
    interleaved: Float32Array,
    channels: number,
    name: string,
  ) => boolean;
  /** Wire-shortcut path for Library fixtures: server reads the WAV
   *  from its local HF cache by name, no PCM round trip. */
  sendFixture: (remote: RemoteBackend, name: string) => void;
  clear: (remote: RemoteBackend) => void;
  /** Record what was just set (or null on clear) so a WS reconnect can
   *  re-apply it — consumed by restoreRefs() in useStartSession. */
  setRef: (ref: RefSource | null) => void;
}

function bindingFor(kind: RefKind): KindBinding {
  if (kind === "timbre") {
    return {
      storeName: () => usePerformanceStore.getState().timbreName,
      label: "timbre ref",
      ariaLabel: "Timbre reference",
      statusPrefix: "timbre",
      send: (r, i, c, n) => r.sendSetTimbreSource(i, c, n),
      sendFixture: (r, n) => r.sendSetTimbreFixture(n),
      clear: (r) => r.sendClearTimbreSource(),
      setRef: (ref) => usePerformanceStore.getState().setTimbreRef(ref),
    };
  }
  return {
    storeName: () => usePerformanceStore.getState().structName,
    label: "structure ref",
    ariaLabel: "Structure reference",
    statusPrefix: "structure",
    send: (r, i, c, n) => r.sendSetStructureSource(i, c, n),
    sendFixture: (r, n) => r.sendSetStructureFixture(n),
    clear: (r) => r.sendClearStructureSource(),
    setRef: (ref) => usePerformanceStore.getState().setStructRef(ref),
  };
}

export function RefControl({ kind }: { kind: RefKind }) {
  // Pick the right store field based on kind. Subscribing through the
  // selector keeps the component reactive — switching kind would be
  // unusual but the binding object below resolves on every render.
  const timbreName = usePerformanceStore((s) => s.timbreName);
  const structName = usePerformanceStore((s) => s.structName);
  const currentName = kind === "timbre" ? timbreName : structName;
  const customNames = useCustomTracksStore((s) => s.names);
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);
  const [fixtures, setFixtures] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [busy, setBusy] = useState(false);

  const bind = bindingFor(kind);
  const value = currentName ?? VAL_INPUT;
  const trimCapS =
    useConfig().engine.max_source_duration_s ?? DEFAULT_TRIM_CAP_S;

  // Same queue-admit gate as AudioSourceCrate: /api/pod/* returns 401
  // pre-admit, so prod waits for wsUrl. Local mode has no queue.
  useEffect(() => {
    if (!sessionWsUrl && !LOCAL_MODE) return;
    void listFixtures()
      .then(setFixtures)
      .catch(() => setFixtures([]));
  }, [sessionWsUrl]);

  function clearActive() {
    const session = useSessionStore.getState();
    if (session.status !== "ready" || !session.remote) return;
    if (currentName) {
      bind.clear(session.remote);
      bind.setRef(null);
    }
  }

  async function pickExisting(name: string) {
    const session = useSessionStore.getState();
    if (session.status !== "ready" || !session.remote) return;

    // Library fixtures live on the server's disk (HF cache). The wire
    // shortcut sends just the name; server reads the WAV and runs the
    // same apply path as a PCM upload. Saves a fetch + decode + ~10×-
    // bigger float32 re-upload that the browser was only doing on the
    // server's behalf. Custom user tracks fall through to the upload
    // path because they only exist in the browser.
    if (fixtures.includes(name)) {
      bind.sendFixture(session.remote, name);
      bind.setRef({ mode: "fixture", name });
      useSessionStore
        .getState()
        .setStatus("ready", `Loading ${bind.statusPrefix} ${name}…`);
      return;
    }

    setBusy(true);
    useSessionStore
      .getState()
      .setStatus("ready", `Loading ${bind.statusPrefix} ${name}…`);
    try {
      // loadFixtureAudio short-circuits to useCustomTracksStore for
      // user uploads — so this branch only handles the custom-track
      // case. (Library picks took the wire shortcut above.)
      const fullDecoded = await loadFixtureAudio(name);
      // Defensive: a custom track that landed in the store via this
      // component's pre-fix uploadAndPick path may exceed the cap.
      // Clamp at the send site so re-picking it doesn't 1009.
      const { decoded } = clampRefDuration(fullDecoded, trimCapS);
      const ok = bind.send(
        session.remote,
        decoded.interleaved,
        decoded.channels,
        name,
      );
      if (ok) {
        bind.setRef({ mode: "clip", name });
      } else {
        useSessionStore
          .getState()
          .setStatus(
            "ready",
            `${bind.statusPrefix[0].toUpperCase()}${bind.statusPrefix.slice(1)} upload failed (socket)`,
          );
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      useSessionStore
        .getState()
        .setStatus(
          "ready",
          `${bind.statusPrefix[0].toUpperCase()}${bind.statusPrefix.slice(1)}: ${msg}`,
        );
    } finally {
      window.setTimeout(() => setBusy(false), 1500);
    }
  }

  async function uploadAndPick(file: File) {
    const session = useSessionStore.getState();
    if (session.status !== "ready" || !session.remote) return;
    const { setStatus } = useSessionStore.getState();
    setBusy(true);
    setStatus("ready", `Loading ${bind.statusPrefix} ${file.name}…`);
    try {
      // decodeAudioFile returns a bare DecodedFixture — no auto-trim.
      // Ref tracks (timbre / structure) intentionally skip the
      // interactive trim dialog the main AudioSourceCrate / Lite flow
      // shows: a reference is conceptually the whole clip the model
      // should imitate, not a slice. The MAX_UPLOAD_DURATION_S
      // browser-memory ceiling inside decodeAudioFile still applies.
      const fullDecoded = await decodeAudioFile(file);
      // ...but the WS server's max_size=100 MiB cap (server.py) hard-
      // rejects frames past ~273 s of 48 kHz stereo float32 with
      // close code 1009. And the server re-trims refs to the playback
      // source's duration anyway, so anything past
      // max_source_duration_s is wire-wasted. Head-trim to the cap.
      const { decoded, trimmedFromS } = clampRefDuration(
        fullDecoded, trimCapS,
      );
      // Mirror AudioSourceCrate's de-dup naming so uploads land in the
      // shared "your tracks" pool without colliding.
      const baseName = file.name;
      let chosen = baseName;
      let i = 1;
      while (useCustomTracksStore.getState().has(chosen)) {
        chosen = `${baseName} (${i++})`;
      }
      useCustomTracksStore.getState().add(chosen, decoded, file);
      if (trimmedFromS !== null) {
        setStatus(
          "ready",
          `${bind.statusPrefix[0].toUpperCase()}${bind.statusPrefix.slice(1)} ${chosen}: trimmed ${fmtMMSS(trimmedFromS)} → first ${fmtMMSS(trimCapS)}`,
        );
      }
      const ok = bind.send(
        session.remote,
        decoded.interleaved,
        decoded.channels,
        chosen,
      );
      if (ok) {
        bind.setRef({ mode: "clip", name: chosen });
      } else {
        setStatus(
          "ready",
          `${bind.statusPrefix[0].toUpperCase()}${bind.statusPrefix.slice(1)} upload failed (socket)`,
        );
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(
        "ready",
        `${bind.statusPrefix[0].toUpperCase()}${bind.statusPrefix.slice(1)}: ${msg}`,
      );
    } finally {
      window.setTimeout(() => setBusy(false), 1500);
    }
  }

  function onSelect(v: string) {
    if (v === VAL_INPUT) {
      clearActive();
      return;
    }
    if (v !== currentName) void pickExisting(v);
  }

  return (
    <>
      <RefSelect
        label={bind.label}
        value={value}
        pinned={[{ value: VAL_INPUT, label: "Input Track" }]}
        groups={[
          {
            label: "Library",
            options: fixtures.map((n) => ({ value: n, label: n })),
          },
          {
            label: "Your tracks",
            options: customNames.map((n) => ({ value: n, label: n })),
          },
        ]}
        onSelect={onSelect}
        disabled={busy}
        ariaLabel={bind.ariaLabel}
        onUpload={() => fileInputRef.current?.click()}
        uploadLabel={`Upload ${bind.label}`}
        tooltip={
          kind === "timbre"
            ? "Optional reference audio for the timbre channel. Picking a track here biases the model's instrument character toward what's in that file, leaving structure (rhythm, sections) free to follow the playing input. Default 'Input Track' uses the playing source's own latent."
            : "Optional reference audio for the structure channel. Picking a track here biases the model's section/rhythm/dynamics layout toward that file, leaving timbre (instrument character) free to follow the playing input. Default 'Input Track' uses the playing source's own latent."
        }
      />
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a,.aac"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) void uploadAndPick(file);
        }}
      />
    </>
  );
}
