// Portable inputs codec — the `SerializedInput(s)` shape plus the pure
// WAV + base64 codec the DemonExport `inputs` field rides on. Lifted out of
// the web app's `inputBundle.ts` (which is store-/DOM-coupled) so every
// client shares ONE codec: the web (browser), the M4L bridge (Node), and a
// future C++/VST consumer must all produce byte-identical `wavBase64`.
//
// Dependency-free and environment-neutral on PURPOSE: no `btoa`/`atob`
// (browser globals) and no `Buffer` (Node) — base64 is hand-rolled over a
// `Uint8Array` so the same bytes come out everywhere. The store/DOM wiring
// that CAPTURES and APPLIES inputs (AudioContext decode, zustand reads)
// stays per-client; only the wire shape + codec are shared here.

/** Which stem of a clip source the server should use. Runtime array is the
 *  single source of truth (the union derives from it) so the C++ contract
 *  emitter walks the option values and the drift guard catches any drift. */
export const STEM_SOURCE_MODES = ["full", "vocals", "instruments"] as const;
export type StemSourceMode = (typeof STEM_SOURCE_MODES)[number];

/** The `kind` discriminant of a SerializedInput. Runtime array so the emitter
 *  derives the `kind` option values instead of hand-mirroring them. */
export const SERIALIZED_INPUT_KINDS = ["fixture", "clip"] as const;

/** One serialized input. `fixture` is a library track the server can load
 *  by name (no audio on the wire). `clip` embeds the trimmed PCM as a base64
 *  WAV so an upload survives the round-trip even on a machine that never had
 *  the original file. */
export type SerializedInput =
  | { kind: "fixture"; name: string }
  | {
      kind: "clip";
      name: string;
      sourceMode?: StemSourceMode;
      /** 16-bit PCM WAV, base64-encoded. Sample rate + channel count ride in
       *  the WAV header, so decode re-derives them. */
      wavBase64: string;
    };

// Compile-time guard: SERIALIZED_INPUT_KINDS must stay exactly in sync with the
// `kind` discriminant above, in both directions — `tsc --noEmit` fails if a kind
// is added/renamed in one place but not the other, so the emitter's runtime
// projection of kind values can never silently drift from the type.
type _AssertKindsExhaustive = SerializedInput["kind"] extends
  (typeof SERIALIZED_INPUT_KINDS)[number] ? true : never;
type _AssertKindsExact = (typeof SERIALIZED_INPUT_KINDS)[number] extends
  SerializedInput["kind"] ? true : never;
const _serializedInputKindGuard: [_AssertKindsExhaustive, _AssertKindsExact] = [
  true,
  true,
];
void _serializedInputKindGuard;

/** The three inputs as captured for export. A field is null when that input
 *  axis simply has nothing active. */
export interface SerializedInputs {
  track?: SerializedInput | null;
  timbre?: SerializedInput | null;
  structure?: SerializedInput | null;
}

// ── WAV codec ──────────────────────────────────────────────────────────

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let i = 0; i < text.length; i++) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

/** Encode already-interleaved float32 PCM as a 16-bit WAV ArrayBuffer. The
 *  source is the interleaved Float32Array a decoded fixture already holds,
 *  so there is no channel de-interleave step. */
export function encodeWavInterleaved(
  interleaved: Float32Array,
  channels: number,
  sampleRate: number,
): ArrayBuffer {
  const frames = Math.floor(interleaved.length / channels);
  const bytesPerSample = 2;
  const dataLen = frames * channels * bytesPerSample;
  const out = new ArrayBuffer(44 + dataLen);
  const view = new DataView(out);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLen, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataLen, true);

  const pcm = new Int16Array(out, 44, frames * channels);
  const n = frames * channels;
  for (let i = 0; i < n; i++) {
    const s = interleaved[i];
    const c = s < -1 ? -1 : s > 1 ? 1 : s;
    // Asymmetric scaling keeps the full negative range without wrapping.
    pcm[i] = c < 0 ? c * 0x8000 : c * 0x7fff;
  }
  return out;
}

// ── base64 (environment-neutral) ───────────────────────────────────────
// Standard RFC 4648 base64 (alphabet `A–Za–z0–9+/`, `=` padding) — the same
// output `btoa`/`Buffer.toString("base64")` produce, so a clip encoded in
// any client decodes identically in every other.

const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64_LOOKUP = (() => {
  const t = new Int16Array(256).fill(-1);
  for (let i = 0; i < B64.length; i++) t[B64.charCodeAt(i)] = i;
  return t;
})();

export function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  const len = bytes.length;
  let out = "";
  let i = 0;
  for (; i + 2 < len; i += 3) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
    out +=
      B64[(n >> 18) & 63] +
      B64[(n >> 12) & 63] +
      B64[(n >> 6) & 63] +
      B64[n & 63];
  }
  const rem = len - i;
  if (rem === 1) {
    const n = bytes[i] << 16;
    out += B64[(n >> 18) & 63] + B64[(n >> 12) & 63] + "==";
  } else if (rem === 2) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8);
    out += B64[(n >> 18) & 63] + B64[(n >> 12) & 63] + B64[(n >> 6) & 63] + "=";
  }
  return out;
}

export function base64ToArrayBuffer(b64: string): ArrayBuffer {
  // Decode 6 bits at a time, flushing a byte every 8. Padding (`=`) and any
  // stray bytes (whitespace, newlines) map to -1 in the lookup and are
  // skipped, the way the browser's atob tolerates our own well-formed
  // payloads. `bi` is therefore the exact decoded length.
  const bytes = new Uint8Array(Math.floor((b64.length * 3) / 4));
  let bi = 0;
  let acc = 0;
  let accBits = 0;
  for (let i = 0; i < b64.length; i++) {
    const v = B64_LOOKUP[b64.charCodeAt(i)];
    if (v < 0) continue;
    acc = (acc << 6) | v;
    accBits += 6;
    if (accBits >= 8) {
      accBits -= 8;
      bytes[bi++] = (acc >> accBits) & 0xff;
    }
  }
  return bytes.buffer.slice(0, bi);
}
