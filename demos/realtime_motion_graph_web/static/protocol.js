// WebSocket wire protocol for the realtime motion-to-music demo.
//
// Mirrors demos/realtime_motion_graph/client/protocol.py:
//   - init: json config, then binary (uint32 channels, uint32 samples) + float32 PCM
//   - server sends: json ready, then binary initial buffer (float16)
//   - streaming: json params/prompt out, binary slices + json params_update in
//
// Binary slice header:  '<BIIHffI'  (little-endian)
//   uint8  flags (0 raw, 1 zstd-delta)
//   uint32 start_sample
//   uint32 num_samples
//   uint16 channels
//   float32 tick_ms
//   float32 dec_ms
//   uint32 num_gens

export const SAMPLE_RATE = 48000;
export const T = 1500;                         // 60s at 25fps latents
export const CROSSFADE_SECONDS = 0.05;

export const SLICE_HDR_SIZE = 1 + 4 + 4 + 2 + 4 + 4 + 4;  // = 23
export const SLICE_FLAG_RAW = 0;
export const SLICE_FLAG_DELTA = 1;

// --- float16 -> float32 ------------------------------------------------
// Browsers don't have native float16, so decode by hand via a reusable
// Uint32Array/Float32Array overlay to avoid per-sample object churn.

const _fBuf = new ArrayBuffer(4);
const _fU32 = new Uint32Array(_fBuf);
const _fF32 = new Float32Array(_fBuf);

function _half2single(h) {
  const s = (h & 0x8000) << 16;
  let e = (h & 0x7c00) >> 10;
  let f = h & 0x03ff;
  if (e === 0) {
    if (f === 0) { _fU32[0] = s; return _fF32[0]; }
    // subnormal -> normalise
    while ((f & 0x0400) === 0) { f <<= 1; e--; }
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

export function float16ArrayToFloat32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = _half2single(u16[i]);
  return out;
}

// --- zstd decompression ------------------------------------------------
// fzstd is loaded as a global via <script> in index.html.  Its UMD build
// exposes ``fzstd.decompress(Uint8Array) -> Uint8Array``.

function _decompressZstd(bytes) {
  if (typeof fzstd === "undefined" || !fzstd.decompress) {
    throw new Error("fzstd library not loaded - check lib/fzstd.min.js or network");
  }
  return fzstd.decompress(bytes);
}

// --- RemoteBackend -----------------------------------------------------

export class RemoteBackend extends EventTarget {
  /**
   * @param {string} url
   * @param {Float32Array} interleaved  (samples * channels,) float32
   * @param {number} channels
   * @param {object} config
   */
  constructor(url, interleaved, channels, config) {
    super();
    this.url = url;
    this._pending = { interleaved, channels, config };
    this.ready = false;
    this.initialBuffer = null;  // Float32Array interleaved
    this.duration = 0;
    this.channels = 0;
    this.sampleRate = SAMPLE_RATE;
    this.loraCatalog = [];
    this.loraDir = "";
  }

  async connect() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;

      let phase = "config";  // "config" -> "init-binary" -> "streaming"

      ws.onopen = () => {
        // Phase 1: send JSON config and then binary audio upload.
        ws.send(JSON.stringify(this._pending.config));
        const { interleaved, channels } = this._pending;
        const samples = interleaved.length / channels;
        const hdr = new ArrayBuffer(8);
        const dv = new DataView(hdr);
        dv.setUint32(0, channels, true);
        dv.setUint32(4, samples, true);
        // Concatenate header + float32 PCM into one ArrayBuffer.
        const pcm = new Uint8Array(interleaved.buffer);
        const combined = new Uint8Array(hdr.byteLength + pcm.byteLength);
        combined.set(new Uint8Array(hdr), 0);
        combined.set(pcm, hdr.byteLength);
        ws.send(combined);
        phase = "ready";
      };

      ws.onmessage = (ev) => {
        if (phase === "ready") {
          // Expect JSON {"type":"ready",...}
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type !== "ready") {
              reject(new Error(`Unexpected init message: ${ev.data}`));
              return;
            }
            this.duration = msg.duration;
            this.channels = msg.channels;
            this.sampleRate = msg.sample_rate;
            this.loraCatalog = msg.lora_catalog || [];
            this.loraDir = msg.lora_dir || "";
            this.detectedBpm = msg.bpm ?? null;
            this.detectedKey = msg.key ?? null;
            phase = "initial-buffer";
          } catch (e) { reject(e); }
          return;
        }

        if (phase === "initial-buffer") {
          // Binary float16 initial buffer.
          const u16 = new Uint16Array(ev.data);
          this.initialBuffer = float16ArrayToFloat32(u16);
          this.ready = true;
          phase = "streaming";
          this._pending = null;
          resolve(this);
          this.dispatchEvent(new CustomEvent("ready"));
          return;
        }

        // The pending-swap state turns the next binary frame into a full
        // buffer replacement (sent right after the swap_ready JSON).
        if (this._pendingSwap && ev.data instanceof ArrayBuffer) {
          const u16 = new Uint16Array(ev.data);
          const interleaved = float16ArrayToFloat32(u16);
          const meta = this._pendingSwap;
          this._pendingSwap = null;
          this.duration = meta.duration;
          this.channels = meta.channels;
          this.dispatchEvent(new CustomEvent("swap_ready", {
            detail: { ...meta, interleaved },
          }));
          return;
        }

        // --- Streaming phase ---
        if (typeof ev.data === "string") {
          let msg;
          try { msg = JSON.parse(ev.data); } catch { return; }
          if (msg.type === "params_update") {
            this.dispatchEvent(new CustomEvent("params", { detail: msg.params }));
          } else if (msg.type === "prompt_applied") {
            this.dispatchEvent(new CustomEvent("prompt_applied", { detail: msg.tags }));
          } else if (msg.type === "lora_catalog") {
            this.loraCatalog = msg.catalog || [];
            this.dispatchEvent(new CustomEvent("lora_catalog", { detail: this.loraCatalog }));
          } else if (msg.type === "swap_ready") {
            // Stage the metadata; the next binary frame is the new buffer.
            this._pendingSwap = msg;
          } else if (msg.type === "swap_failed") {
            this.dispatchEvent(new CustomEvent("swap_failed", { detail: msg.error }));
          } else {
            this.dispatchEvent(new CustomEvent("json", { detail: msg }));
          }
          return;
        }

        // Binary slice.
        try {
          const slice = this._parseSlice(ev.data);
          if (slice) this.dispatchEvent(new CustomEvent("slice", { detail: slice }));
        } catch (e) {
          console.error("[protocol] slice parse failed:", e);
        }
      };

      ws.onerror = (e) => {
        console.error("[protocol] ws error", e);
        if (!this.ready) reject(new Error("WebSocket connection failed (network / port unreachable)"));
        this.dispatchEvent(new CustomEvent("error", { detail: e }));
      };
      ws.onclose = (e) => {
        // If the socket closes before we finished the init handshake,
        // the connect() promise must reject — otherwise the launcher
        // sits on "Uploading..." forever when the server crashes mid-init.
        if (!this.ready) {
          const reason = e.reason || `code ${e.code}`;
          reject(new Error(`WebSocket closed before server sent 'ready' (${reason}). Check the server console.`));
        }
        this.dispatchEvent(new CustomEvent("close", { detail: e }));
      };
    });
  }

  _parseSlice(buf) {
    if (buf.byteLength < SLICE_HDR_SIZE) return null;
    const dv = new DataView(buf);
    let o = 0;
    const flags       = dv.getUint8(o); o += 1;
    const startSample = dv.getUint32(o, true); o += 4;
    const numSamples  = dv.getUint32(o, true); o += 4;
    const channels    = dv.getUint16(o, true); o += 2;
    const tickMs      = dv.getFloat32(o, true); o += 4;
    const decMs       = dv.getFloat32(o, true); o += 4;
    const numGens     = dv.getUint32(o, true); o += 4;

    let payload = new Uint8Array(buf, SLICE_HDR_SIZE);
    if (flags === SLICE_FLAG_DELTA) {
      payload = _decompressZstd(payload);
    }
    // Copy so the Uint16Array is 2-byte aligned regardless of the
    // underlying buffer's origin (zstd output has its own backing).
    const aligned = new ArrayBuffer(payload.byteLength);
    new Uint8Array(aligned).set(payload);
    const u16 = new Uint16Array(aligned);
    const audio = float16ArrayToFloat32(u16);

    return {
      flags, startSample, numSamples, channels,
      tickMs, decMs, numGens, audio,
    };
  }

  sendParams(raw, playbackPos) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(JSON.stringify({
        type: "params",
        raw,
        playback_pos: playbackPos,
      }));
    } catch {}
  }

  sendPrompt(tags, key) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = { type: "prompt", tags };
      if (key) msg.key = key;
      this.ws.send(JSON.stringify(msg));
    } catch {}
  }

  sendEnableLora(id, strength) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = { type: "enable_lora", id };
      // Pass the target strength so the server refits at it in one
      // shot — enabling at 0 and waiting for the next params tick to
      // ramp up produces a one-decode-window glitch where the LoRA
      // sounds missing.
      if (typeof strength === "number") msg.strength = strength;
      this.ws.send(JSON.stringify(msg));
    } catch {}
  }

  sendDisableLora(id) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(JSON.stringify({ type: "disable_lora", id }));
    } catch {}
  }

  /**
   * Replace the source audio in-flight. The server pauses generation,
   * re-runs prepare_source / encode_text on the new waveform, swaps
   * stream.source/conditioning, and replies with a swap_ready event +
   * binary buffer. Caller should hand the buffer to AudioPlayer.swap().
   * @param {Float32Array} interleaved
   * @param {number} channels
   * @param {string} [tags]  current prompt; falls back to server-side
   * @param {string} [key]   override; otherwise server uses detected key
   */
  sendSwapSource(interleaved, channels, tags, key) {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    try {
      const msg = { type: "swap_source" };
      if (tags) msg.tags = tags;
      if (key) msg.key = key;
      this.ws.send(JSON.stringify(msg));
      const samples = interleaved.length / channels;
      const hdr = new ArrayBuffer(8);
      const dv = new DataView(hdr);
      dv.setUint32(0, channels, true);
      dv.setUint32(4, samples, true);
      const pcm = new Uint8Array(interleaved.buffer);
      const combined = new Uint8Array(hdr.byteLength + pcm.byteLength);
      combined.set(new Uint8Array(hdr), 0);
      combined.set(pcm, hdr.byteLength);
      this.ws.send(combined);
      return true;
    } catch (e) {
      console.error("[protocol] sendSwapSource failed:", e);
      return false;
    }
  }

  close() {
    try { this.ws?.close(); } catch {}
  }
}
