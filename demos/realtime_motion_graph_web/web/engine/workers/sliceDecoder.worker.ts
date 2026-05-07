// Slice decoder worker — owns fzstd + float16→float32 conversion off the main
// thread. Server delta slices can compress to a few hundred KB; decompressing
// inline blocks RAF and input handling. Each ws binary frame is forwarded
// here and a parsed AudioSlice is posted back, transferring buffers so we
// never copy the audio Float32Array.

import * as fzstd from "fzstd";

import { SLICE_FLAG_DELTA, SLICE_HDR_SIZE } from "@/types/protocol";

interface DecodeRequest {
  id: number;
  buffer: ArrayBuffer;
}

interface DecodeResponse {
  id: number;
  ok: true;
  flags: number;
  startSample: number;
  numSamples: number;
  channels: number;
  tickMs: number;
  decMs: number;
  numGens: number;
  audio: Float32Array;
}

interface DecodeError {
  id: number;
  ok: false;
  error: string;
}

// Reusable scratch overlay for half-precision decode (one allocation, not
// per-sample object churn).
const _fBuf = new ArrayBuffer(4);
const _fU32 = new Uint32Array(_fBuf);
const _fF32 = new Float32Array(_fBuf);

function _half2single(h: number): number {
  const s = (h & 0x8000) << 16;
  let e = (h & 0x7c00) >> 10;
  let f = h & 0x03ff;
  if (e === 0) {
    if (f === 0) {
      _fU32[0] = s;
      return _fF32[0];
    }
    while ((f & 0x0400) === 0) {
      f <<= 1;
      e--;
    }
    e++;
    f &= ~0x0400;
  } else if (e === 31) {
    _fU32[0] = s | 0x7f800000 | (f << 13);
    return _fF32[0];
  }
  e = e + (127 - 15);
  _fU32[0] = s | (e << 23) | (f << 13);
  return _fF32[0];
}

function float16ArrayToFloat32(u16: Uint16Array): Float32Array {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = _half2single(u16[i]);
  return out;
}

self.onmessage = (ev: MessageEvent<DecodeRequest>) => {
  const { id, buffer } = ev.data;
  try {
    if (buffer.byteLength < SLICE_HDR_SIZE) {
      const err: DecodeError = { id, ok: false, error: "slice too short" };
      (self as unknown as Worker).postMessage(err);
      return;
    }
    const dv = new DataView(buffer);
    let o = 0;
    const flags = dv.getUint8(o);
    o += 1;
    const startSample = dv.getUint32(o, true);
    o += 4;
    const numSamples = dv.getUint32(o, true);
    o += 4;
    const channels = dv.getUint16(o, true);
    o += 2;
    const tickMs = dv.getFloat32(o, true);
    o += 4;
    const decMs = dv.getFloat32(o, true);
    o += 4;
    const numGens = dv.getUint32(o, true);
    o += 4;

    let payload: Uint8Array = new Uint8Array(buffer, SLICE_HDR_SIZE);
    if (flags === SLICE_FLAG_DELTA) {
      payload = fzstd.decompress(payload);
    }
    const aligned = new ArrayBuffer(payload.byteLength);
    new Uint8Array(aligned).set(payload);
    const u16 = new Uint16Array(aligned);
    const audio = float16ArrayToFloat32(u16);

    const reply: DecodeResponse = {
      id,
      ok: true,
      flags,
      startSample,
      numSamples,
      channels,
      tickMs,
      decMs,
      numGens,
      audio,
    };
    (self as unknown as Worker).postMessage(reply, [audio.buffer]);
  } catch (e) {
    const err: DecodeError = {
      id,
      ok: false,
      error: e instanceof Error ? e.message : String(e),
    };
    (self as unknown as Worker).postMessage(err);
  }
};

export {};
