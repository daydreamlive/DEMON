// One-shot renders: N whole versions of the session's window in one request
// (the `oneshot` command), outside the streaming ring. Part of the
// demon-client SDK: no host-app imports, no store writes.
//
// The server answers with `oneshot_ready` + ONE binary frame: int16
// little-endian, blocks version-major then stream (in `streams` order), each
// block interleaved [frames, channels]. RemoteBackend turns that pair into a
// `oneshot` CustomEvent whose detail is a decoded `OneshotResult`, or a
// `oneshot_failed` event ({request_id, error}). `renderOneshot` wraps the
// round trip in a promise.
//
// Best on a session created with `render_only: true` in its config (no ring
// runs beside the render); `ready.capabilities.oneshot` says the pod has it.

import type { OneshotCommand, OneshotReadyEvent } from "./types/wireContract.gen";

/** What to render: one version per seed, or a redo of one stream of a kept version. */
export interface OneshotRequest {
  /** Echoed on the answer; minted when absent. */
  requestId?: string;
  /** One version per seed (integers >= 0, at most 16). Ignored when `redo` is set. */
  seeds?: number[];
  /** Knob values for these renders, over the session's current ones (same names as `params`). */
  knobs?: Record<string, number>;
  /** Re-denoise stream `stream` (0-based) of kept version `id` from `seed`; every other stream given from it. */
  redo?: { id: string; stream: number; seed: number };
}

/** One rendered version. `audio[k]` is stream `k`'s interleaved [frames, channels] int16 PCM. */
export interface OneshotVersion {
  /** The kept latent's id: the handle a later redo names. */
  id: string;
  seed: number;
  audio: Record<number, Int16Array>;
}

export interface OneshotResult {
  requestId: string;
  sampleRate: number;
  /** Channels per stream (2). */
  channels: number;
  /** Samples per channel per stream. */
  frames: number;
  /** The stream indices each version carries (every stream, or a redo's one). */
  streams: number[];
  versions: OneshotVersion[];
  /** Server wall time per phase: generate, decode, pack, total. */
  ms: Record<string, number>;
  /** The kept id a redo started from; null for a plain render. */
  redoOf: string | null;
}

let _seq = 0;

/** The wire frame for `req`. */
export function oneshotCommand(req: OneshotRequest): OneshotCommand {
  const msg: OneshotCommand = {
    type: "oneshot",
    request_id: req.requestId ?? `oneshot-${++_seq}-${Date.now()}`,
  };
  if (req.redo) msg.redo = { id: req.redo.id, stream: req.redo.stream, seed: req.redo.seed };
  else msg.seeds = [...(req.seeds ?? [])];
  if (req.knobs) msg.knobs = { ...req.knobs };
  return msg;
}

/** A `oneshot_ready` header and its binary frame as a result (views into `data`, no copy). */
export function decodeOneshot(meta: OneshotReadyEvent, data: ArrayBuffer): OneshotResult {
  const streams = (meta.streams as number[]).map(Number);
  const ids = (meta.ids as string[]).map(String);
  const seeds = (meta.seeds as number[]).map(Number);
  const block = meta.frames * meta.channels;
  const want = ids.length * streams.length * block * 2;
  if (data.byteLength !== want)
    throw new Error(`oneshot: ${data.byteLength} bytes, expected ${want} (${ids.length} x ${streams.length} x ${meta.frames} x ${meta.channels} int16)`);
  const versions = ids.map((id, v) => {
    const audio: Record<number, Int16Array> = {};
    streams.forEach((k, j) => {
      audio[k] = new Int16Array(data, (v * streams.length + j) * block * 2, block);
    });
    return { id, seed: seeds[v], audio };
  });
  const ms: Record<string, number> = {};
  for (const [k, v] of Object.entries(meta.ms ?? {})) ms[k] = Number(v);
  return {
    requestId: meta.request_id,
    sampleRate: meta.sample_rate,
    channels: meta.channels,
    frames: meta.frames,
    streams,
    versions,
    ms,
    redoOf: meta.redo_of ?? null,
  };
}

/** int16 PCM -> float32 in [-1, 1). */
export function int16ToFloat32(a: Int16Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] / 32768;
  return out;
}

/** The surface `renderOneshot` needs (RemoteBackend has it). */
export interface OneshotTarget extends EventTarget {
  sendOneshot(req: OneshotRequest): string | null;
}

/** Send `req` and resolve with its result. Rejects on `oneshot_failed`, a closed socket, or after `timeoutMs`. */
export function renderOneshot(remote: OneshotTarget, req: OneshotRequest, timeoutMs = 120_000): Promise<OneshotResult> {
  return new Promise((resolve, reject) => {
    const id = req.requestId ?? `oneshot-${++_seq}-${Date.now()}`;
    const off = () => {
      clearTimeout(timer);
      remote.removeEventListener("oneshot", on);
      remote.removeEventListener("oneshot_failed", on);
      remote.removeEventListener("close", gone);
    };
    const on = (e: Event) => {
      const d = (e as CustomEvent).detail as (OneshotResult & { request_id?: string; error?: string }) | undefined;
      if ((e.type === "oneshot" ? d?.requestId : d?.request_id) !== id) return;
      off();
      if (e.type === "oneshot_failed") reject(new Error(d?.error || "oneshot failed"));
      else resolve(d as OneshotResult);
    };
    const gone = () => (off(), reject(new Error("connection lost")));
    const timer = setTimeout(() => (off(), reject(new Error("oneshot timed out"))), timeoutMs);
    remote.addEventListener("oneshot", on);
    remote.addEventListener("oneshot_failed", on);
    remote.addEventListener("close", gone);
    if (remote.sendOneshot({ ...req, requestId: id }) === null) (off(), reject(new Error("connection not open")));
  });
}
