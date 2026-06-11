// Wire-frame builders for unit tests: synthesize the server's binary
// frames (initial buffer, RAW/DELTA slices) without a recording.
// Framing must match demos/realtime_motion_graph_web/protocol.py
// (SLICE_HDR_FMT "<BIIHffI") and sdk/types/protocol.ts SLICE_HDR_SIZE.

import * as zlib from "node:zlib";

// zstdCompressSync shipped in Node 22.15 / 23.8 but @types/node@20
// predates it; the runtime (Node 24) has it.
const zstdCompressSync = (
  zlib as unknown as { zstdCompressSync: (data: Uint8Array) => Uint8Array }
).zstdCompressSync;

import {
  SLICE_FLAG_DELTA,
  SLICE_FLAG_RAW,
  SLICE_HDR_SIZE,
} from "@demon/client";

/** float32 -> float16 bits via the native Float16Array (Node >= 23). */
export function f32ToF16Bits(values: Float32Array): Uint16Array {
  const f16 = new Float16Array(values.length);
  for (let i = 0; i < values.length; i++) f16[i] = values[i];
  return new Uint16Array(f16.buffer);
}

/** The ready-phase / swap binary frame: bare interleaved float16. */
export function makeBufferFrame(interleaved: Float32Array): ArrayBuffer {
  const bits = f32ToF16Bits(interleaved);
  const out = new ArrayBuffer(bits.byteLength);
  new Uint8Array(out).set(new Uint8Array(bits.buffer));
  return out;
}

export interface SliceSpec {
  flags?: number;
  startSample: number;
  channels?: number;
  tickMs?: number;
  decMs?: number;
  numGens?: number;
  /** Interleaved float32; numSamples is derived (length / channels). */
  audio: Float32Array;
}

export function makeSliceFrame(spec: SliceSpec): ArrayBuffer {
  const flags = spec.flags ?? SLICE_FLAG_RAW;
  const channels = spec.channels ?? 2;
  const numSamples = spec.audio.length / channels;
  let payload = new Uint8Array(f32ToF16Bits(spec.audio).buffer);
  if (flags === SLICE_FLAG_DELTA) {
    payload = new Uint8Array(zstdCompressSync(payload));
  }
  const buf = new ArrayBuffer(SLICE_HDR_SIZE + payload.byteLength);
  const dv = new DataView(buf);
  let o = 0;
  dv.setUint8(o, flags);
  o += 1;
  dv.setUint32(o, spec.startSample, true);
  o += 4;
  dv.setUint32(o, numSamples, true);
  o += 4;
  dv.setUint16(o, channels, true);
  o += 2;
  dv.setFloat32(o, spec.tickMs ?? 17.0, true);
  o += 4;
  dv.setFloat32(o, spec.decMs ?? 2.5, true);
  o += 4;
  dv.setUint32(o, spec.numGens ?? 1, true);
  o += 4;
  new Uint8Array(buf, SLICE_HDR_SIZE).set(payload);
  return buf;
}

/** Half-exact ramp: values that survive the f32 -> f16 -> f32 round trip
 *  unchanged, so equality assertions stay exact. */
export function f16ExactRamp(n: number, scale = 1 / 1024): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.f16round(((i % 512) - 256) * scale);
  return out;
}
