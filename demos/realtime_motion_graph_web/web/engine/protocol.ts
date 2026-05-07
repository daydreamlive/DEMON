// WebSocket client for the DEMON realtime motion-to-music backend.
// Direct TS port of DEMON/demos/realtime_motion_graph_web/static/protocol.js.
//
// Phases:
//   1. config   client sends JSON config + binary (uint32 channels, uint32
//               samples) + float32 PCM
//   2. ready    server replies with JSON {type: "ready", ...} then a
//               binary float16 initial buffer (interleaved)
//   3. stream   client sends JSON params/prompt/enable_lora/swap_source;
//               server sends binary slices + JSON params_update / prompt_applied /
//               lora_catalog / swap_ready / swap_failed

import * as fzstd from "fzstd";

import {
  SAMPLE_RATE,
  SLICE_FLAG_DELTA,
  SLICE_HDR_SIZE,
  type AudioSlice,
  type LoraCatalogEntry,
  type SessionConfig,
  type SwapReadyMessage,
} from "@/types/protocol";

export {
  SAMPLE_RATE,
  T,
  CROSSFADE_SECONDS,
  SLICE_FLAG_DELTA,
  SLICE_FLAG_RAW,
  SLICE_HDR_SIZE,
} from "@/types/protocol";

// ── float16 → float32 ──────────────────────────────────────────────────
// Browsers don't have native float16; decode by hand via a reusable
// Uint32Array/Float32Array overlay to avoid per-sample object churn.

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

export function float16ArrayToFloat32(u16: Uint16Array): Float32Array {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = _half2single(u16[i]);
  return out;
}

// ── RemoteBackend ──────────────────────────────────────────────────────

type Phase = "config" | "ready" | "initial-buffer" | "streaming";

interface PendingPayload {
  interleaved: Float32Array;
  channels: number;
  config: SessionConfig;
}

export class RemoteBackend extends EventTarget {
  readonly url: string;
  ws: WebSocket | null = null;
  ready = false;
  initialBuffer: Float32Array | null = null;
  duration = 0;
  channels = 0;
  sampleRate = SAMPLE_RATE;
  loraCatalog: LoraCatalogEntry[] = [];
  loraDir = "";
  detectedBpm: number | null = null;
  detectedKey: string | null = null;

  private _pending: PendingPayload | null;
  private _pendingSwap: SwapReadyMessage | null = null;
  // Slice decoder runs in a worker so fzstd.decompress + float16→float32
  // never block the render loop or input handling. Worker is single-threaded
  // and postMessage is FIFO, so audio slices stay in order.
  private _decoderWorker: Worker | null = null;
  private _nextDecodeId = 1;

  constructor(
    url: string,
    interleaved: Float32Array,
    channels: number,
    config: SessionConfig,
  ) {
    super();
    this.url = url;
    this._pending = { interleaved, channels, config };
    this._initDecoderWorker();
  }

  private _initDecoderWorker(): void {
    if (typeof Worker === "undefined") return;
    try {
      const worker = new Worker(
        new URL("./workers/sliceDecoder.worker.mjs", import.meta.url),
        { type: "module" },
      );
      worker.onmessage = (ev: MessageEvent) => {
        const msg = ev.data;
        if (!msg || typeof msg !== "object") return;
        if (msg.ok === false) {
          console.error("[protocol] slice decode failed:", msg.error);
          return;
        }
        if (msg.ok !== true) return;
        const slice: AudioSlice = {
          flags: msg.flags,
          startSample: msg.startSample,
          numSamples: msg.numSamples,
          channels: msg.channels,
          tickMs: msg.tickMs,
          decMs: msg.decMs,
          numGens: msg.numGens,
          audio: msg.audio,
        };
        this.dispatchEvent(new CustomEvent("slice", { detail: slice }));
      };
      worker.onerror = (e) => {
        console.error("[protocol] slice decoder worker error:", e);
      };
      this._decoderWorker = worker;
    } catch (e) {
      console.warn("[protocol] worker init failed, falling back to main-thread decode:", e);
      this._decoderWorker = null;
    }
  }

  async connect(): Promise<this> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;

      let phase: Phase = "config";

      ws.onopen = () => {
        if (!this._pending) return;
        // Phase 1: JSON config + binary audio upload.
        ws.send(JSON.stringify(this._pending.config));
        const { interleaved, channels } = this._pending;
        const samples = interleaved.length / channels;
        const hdr = new ArrayBuffer(8);
        const dv = new DataView(hdr);
        dv.setUint32(0, channels, true);
        dv.setUint32(4, samples, true);
        const pcm = new Uint8Array(interleaved.buffer);
        const combined = new Uint8Array(hdr.byteLength + pcm.byteLength);
        combined.set(new Uint8Array(hdr), 0);
        combined.set(pcm, hdr.byteLength);
        ws.send(combined);
        phase = "ready";
      };

      ws.onmessage = (ev) => {
        if (phase === "ready") {
          try {
            const msg = JSON.parse(ev.data as string);
            if (msg.type === "error") {
              reject(
                new Error(
                  msg.message || `Server error: ${msg.code || "unknown"}`,
                ),
              );
              return;
            }
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
          } catch (e) {
            reject(e);
          }
          return;
        }

        if (phase === "initial-buffer") {
          const u16 = new Uint16Array(ev.data as ArrayBuffer);
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
          this.dispatchEvent(
            new CustomEvent("swap_ready", {
              detail: { ...meta, interleaved },
            }),
          );
          return;
        }

        if (typeof ev.data === "string") {
          let msg: { type: string; [k: string]: unknown };
          try {
            msg = JSON.parse(ev.data);
          } catch {
            return;
          }
          if (msg.type === "params_update") {
            this.dispatchEvent(
              new CustomEvent("params", { detail: msg.params }),
            );
          } else if (msg.type === "prompt_applied") {
            this.dispatchEvent(
              new CustomEvent("prompt_applied", { detail: msg.tags }),
            );
          } else if (msg.type === "lora_catalog") {
            this.loraCatalog =
              (msg.catalog as LoraCatalogEntry[] | undefined) || [];
            this.dispatchEvent(
              new CustomEvent("lora_catalog", { detail: this.loraCatalog }),
            );
          } else if (msg.type === "swap_ready") {
            this._pendingSwap = msg as unknown as SwapReadyMessage;
          } else if (msg.type === "swap_failed") {
            this.dispatchEvent(
              new CustomEvent("swap_failed", { detail: msg.error }),
            );
          } else {
            this.dispatchEvent(new CustomEvent("json", { detail: msg }));
          }
          return;
        }

        if (this._decoderWorker) {
          const buf = ev.data as ArrayBuffer;
          this._decoderWorker.postMessage(
            { id: this._nextDecodeId++, buffer: buf },
            [buf],
          );
        } else {
          try {
            const slice = this._parseSlice(ev.data as ArrayBuffer);
            if (slice) {
              this.dispatchEvent(new CustomEvent("slice", { detail: slice }));
            }
          } catch (e) {
            console.error("[protocol] slice parse failed:", e);
          }
        }
      };

      ws.onerror = (e) => {
        console.error("[protocol] ws error", e);
        if (!this.ready) {
          reject(
            new Error(
              "WebSocket connection failed (network / port unreachable)",
            ),
          );
        }
        this.dispatchEvent(new CustomEvent("error", { detail: e }));
      };

      ws.onclose = (e) => {
        // If the socket closes before we finished the init handshake, the
        // connect() promise must reject — otherwise the launcher sits on
        // "Uploading..." forever when the server crashes mid-init.
        if (!this.ready) {
          const reason = e.reason || `code ${e.code}`;
          reject(
            new Error(
              `WebSocket closed before server sent 'ready' (${reason}). Check the server console.`,
            ),
          );
        }
        this.dispatchEvent(new CustomEvent("close", { detail: e }));
      };
    });
  }

  private _parseSlice(buf: ArrayBuffer): AudioSlice | null {
    if (buf.byteLength < SLICE_HDR_SIZE) return null;
    const dv = new DataView(buf);
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

    let payload: Uint8Array = new Uint8Array(buf, SLICE_HDR_SIZE);
    if (flags === SLICE_FLAG_DELTA) {
      payload = fzstd.decompress(payload);
    }
    // Copy so the Uint16Array is 2-byte aligned regardless of the underlying
    // buffer's origin (zstd output has its own backing).
    const aligned = new ArrayBuffer(payload.byteLength);
    new Uint8Array(aligned).set(payload);
    const u16 = new Uint16Array(aligned);
    const audio = float16ArrayToFloat32(u16);

    return {
      flags,
      startSample,
      numSamples,
      channels,
      tickMs,
      decMs,
      numGens,
      audio,
    };
  }

  sendParams(
    raw: Record<string, number | string | boolean>,
    playbackPos: number,
  ): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(
        JSON.stringify({
          type: "params",
          raw,
          playback_pos: playbackPos,
        }),
      );
    } catch {}
  }

  sendPrompt(tags: string, key?: string): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg: { type: string; tags: string; key?: string } = {
        type: "prompt",
        tags,
      };
      if (key) msg.key = key;
      this.ws.send(JSON.stringify(msg));
    } catch {}
  }

  sendEnableLora(id: string, strength?: number): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg: { type: string; id: string; strength?: number } = {
        type: "enable_lora",
        id,
      };
      if (typeof strength === "number") msg.strength = strength;
      this.ws.send(JSON.stringify(msg));
    } catch {}
  }

  sendDisableLora(id: string): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(JSON.stringify({ type: "disable_lora", id }));
    } catch {}
  }

  /**
   * Replace the source audio in-flight. Server pauses generation, re-runs
   * prepare_source / encode_text on the new waveform, then replies with
   * swap_ready + a binary buffer (handled in onmessage).
   */
  sendSwapSource(
    interleaved: Float32Array,
    channels: number,
    tags?: string,
    key?: string,
  ): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    try {
      const msg: { type: string; tags?: string; key?: string } = {
        type: "swap_source",
      };
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

  close(): void {
    try {
      this.ws?.close();
    } catch {}
    try {
      this._decoderWorker?.terminate();
    } catch {}
    this._decoderWorker = null;
  }
}
