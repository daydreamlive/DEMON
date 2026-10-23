// One-shot renders: N whole versions of the session's window in one request
// (the `oneshot` command), outside the streaming ring. Part of the
// demon-client SDK: no host-app imports, no store writes.
//
// The server answers with `oneshot_ready` + ONE binary frame: int16
// little-endian, blocks version-major then stream (in `streams` order), each
// block interleaved [frames, channels]. With `returnLatents`, a SECOND frame
// follows: every version's full latent, raw float16 bits [V, S, C, L].
// RemoteBackend turns that into a `oneshot` CustomEvent whose detail is a
// decoded `OneshotResult`, or a `oneshot_failed` event ({request_id,
// error}). `renderOneshot` wraps the round trip in a promise.
//
// Latents travel as raw float16 bits in a Uint16Array (no Float16Array);
// stored blobs are those bytes, little-endian. A redo may name a kept id or
// hand back such a blob (`redo.latent`), sent as one frame after the header.
//
// Best on a session created with `render_only: true` in its config (no ring
// runs beside the render); `ready.capabilities.oneshot` says the pod has it.

import type { OneshotCommand, OneshotReadyEvent } from "./types/wireContract.gen";

/** A held version's full latent as the client keeps it: float16 bits, C-order [nStreams, channels, frames]. */
export interface OneshotLatent {
  data: Uint16Array;
  nStreams: number;
  channels: number;
  frames: number;
}

/** A redo: regenerate `streams` (or the one `stream`) of a held version from `seed`. */
export interface OneshotRedo {
  /** A kept version's id (from a result). Give this or `latent`, not both. */
  id?: string;
  /** The single-stream form of `streams`. */
  stream?: number;
  /** The streams regenerated (0-based, at least one). */
  streams?: number[];
  seed: number;
  /** 0 < d <= 1, default 1 (re-roll). A number for every regenerated stream, or one per entry of `streams`; d < 1 adjusts that stream from its own held slice. */
  denoise?: number | number[];
  /** Client-held state in place of `id`. */
  latent?: OneshotLatent;
}

/** What to render: one version per seed, or a redo of streams of a held version. */
export interface OneshotRequest {
  /** Echoed on the answer; minted when absent. */
  requestId?: string;
  /** One version per seed (integers >= 0, at most 16). Ignored when `redo` is set. */
  seeds?: number[];
  /** Knob values for these renders, over the session's current ones (same names as `params`). */
  knobs?: Record<string, number>;
  /** Regenerate streams of a held version (kept `id` or client `latent`); every other stream given from it. */
  redo?: OneshotRedo;
  /** Also return every version's full latent (`OneshotResult.latents`). */
  returnLatents?: boolean;
}

/** One rendered version. `audio[k]` is stream `k`'s interleaved [frames, channels] int16 PCM. */
export interface OneshotVersion {
  /** The kept latent's id: the handle a later redo names. */
  id: string;
  seed: number;
  audio: Record<number, Int16Array>;
  /** This version's full latent, float16 bits [S, C, L]: a view into `OneshotResult.latents.data`. */
  latent?: Uint16Array;
}

/** The geometry a held latent must match ([nStreams, channels, frames], frames per stream). */
export interface OneshotLatentHeader {
  nStreams: number;
  channels: number;
  frames: number;
  seconds: number;
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
  /** The kept id a redo started from; null for a plain render (and a redo from a client latent). */
  redoOf: string | null;
  latentHeader: OneshotLatentHeader;
  /** Every version's full latent (when asked): float16 bits, C-order [V, S, C, L]. */
  latents?: { shape: [number, number, number, number]; data: Uint16Array };
}

let _seq = 0;

/** The wire frame for `req`. */
export function oneshotCommand(req: OneshotRequest): OneshotCommand {
  const msg: OneshotCommand = {
    type: "oneshot",
    request_id: req.requestId ?? `oneshot-${++_seq}-${Date.now()}`,
  };
  if (req.redo) {
    const r = req.redo;
    const redo: Record<string, unknown> = {};
    if (r.id !== undefined) redo.id = r.id;
    if (r.stream !== undefined) redo.stream = r.stream;
    if (r.streams !== undefined) redo.streams = [...r.streams];
    redo.seed = r.seed;
    if (r.denoise !== undefined) redo.denoise = Array.isArray(r.denoise) ? [...r.denoise] : r.denoise;
    if (r.latent) {
      const { data, nStreams, channels, frames } = r.latent;
      if (data.length !== nStreams * channels * frames)
        throw new Error(`oneshot: redo.latent has ${data.length} values, expected ${nStreams} x ${channels} x ${frames}`);
      redo.latent = { dtype: "float16", shape: [nStreams, channels, frames] };
    }
    msg.redo = redo;
  } else msg.seeds = [...(req.seeds ?? [])];
  if (req.knobs) msg.knobs = { ...req.knobs };
  if (req.returnLatents) msg.return_latents = true;
  return msg;
}

/** The binary frame that follows the header of `req` (its `redo.latent` bytes), or null when none does. */
export function oneshotLatentFrame(req: OneshotRequest): ArrayBuffer | null {
  const lat = req.redo?.latent;
  if (!lat) return null;
  const d = lat.data;
  return d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength) as ArrayBuffer;
}

/** A `oneshot_ready` header and its binary frame(s) as a result (views into `data` / `latents`, no copy). */
export function decodeOneshot(meta: OneshotReadyEvent, data: ArrayBuffer, latents?: ArrayBuffer): OneshotResult {
  const streams = (meta.streams as number[]).map(Number);
  const ids = (meta.ids as string[]).map(String);
  const seeds = (meta.seeds as number[]).map(Number);
  const block = meta.frames * meta.channels;
  const want = ids.length * streams.length * block * 2;
  if (data.byteLength !== want)
    throw new Error(`oneshot: ${data.byteLength} bytes, expected ${want} (${ids.length} x ${streams.length} x ${meta.frames} x ${meta.channels} int16)`);
  const versions = ids.map((id, v): OneshotVersion => {
    const audio: Record<number, Int16Array> = {};
    streams.forEach((k, j) => {
      audio[k] = new Int16Array(data, (v * streams.length + j) * block * 2, block);
    });
    return { id, seed: seeds[v], audio };
  });
  const ms: Record<string, number> = {};
  for (const [k, v] of Object.entries(meta.ms ?? {})) ms[k] = Number(v);
  const lh = (meta.latent_header ?? {}) as Record<string, unknown>;
  const latentHeader: OneshotLatentHeader = {
    nStreams: Number(lh.n_streams ?? 0),
    channels: Number(lh.channels ?? 0),
    frames: Number(lh.frames ?? 0),
    seconds: Number(lh.seconds ?? meta.frames / meta.sample_rate),
  };
  let lat: OneshotResult["latents"];
  const lm = meta.latents as { shape?: number[] } | undefined;
  if (lm) {
    const shape = (lm.shape ?? []).map(Number) as [number, number, number, number];
    if (shape.length !== 4) throw new Error(`oneshot: latents shape ${JSON.stringify(lm.shape)} is not [V, S, C, L]`);
    const per = shape[1] * shape[2] * shape[3];
    const wantLat = shape[0] * per * 2;
    if (!latents) throw new Error("oneshot: the latents frame is missing");
    if (latents.byteLength !== wantLat)
      throw new Error(`oneshot: latents ${latents.byteLength} bytes, expected ${wantLat} (${shape.join(" x ")} float16)`);
    const bits = new Uint16Array(latents);
    lat = { shape, data: bits };
    versions.forEach((ver, v) => {
      if (v < shape[0]) ver.latent = bits.subarray(v * per, (v + 1) * per);
    });
  }
  return {
    requestId: meta.request_id,
    sampleRate: meta.sample_rate,
    channels: meta.channels,
    frames: meta.frames,
    streams,
    versions,
    ms,
    redoOf: meta.redo_of ?? null,
    latentHeader,
    ...(lat ? { latents: lat } : {}),
  };
}

/** What a finished `oneshot_ready` + frames sequence gave: a result, or why it could not be decoded. */
export interface OneshotAssembled {
  meta: OneshotReadyEvent;
  result?: OneshotResult;
  error?: string;
}

/**
 * Collects the binary frames after a `oneshot_ready` header: ONE (the PCM),
 * or TWO when the header carries `latents` (the PCM, then the latents).
 * RemoteBackend feeds it every binary frame while `waiting`.
 */
export class OneshotFrames {
  private meta: OneshotReadyEvent | null = null;
  private pcm: ArrayBuffer | null = null;

  /** A header is in and its frames are not all here yet. */
  get waiting(): boolean {
    return this.meta !== null;
  }

  /** A new `oneshot_ready` header (drops any half-received one). */
  begin(meta: OneshotReadyEvent): void {
    this.meta = meta;
    this.pcm = null;
  }

  /** One binary frame. Returns the assembled answer once the last frame is in, else null. */
  push(frame: ArrayBuffer): OneshotAssembled | null {
    const meta = this.meta;
    if (meta === null) return null;
    if (meta.latents && this.pcm === null) {
      this.pcm = frame;
      return null;
    }
    const pcm = this.pcm ?? frame;
    const latents = this.pcm !== null ? frame : undefined;
    this.meta = null;
    this.pcm = null;
    try {
      return { meta, result: decodeOneshot(meta, pcm, latents) };
    } catch (e) {
      return { meta, error: String(e) };
    }
  }
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
