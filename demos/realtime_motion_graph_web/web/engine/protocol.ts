// WebSocket client for the DEMON realtime motion-to-music backend.
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
  enabledLoraTriggerPrefix,
  stripLeadingTriggers,
} from "@/lib/loraTriggers";
import { useSessionStore } from "@/store/useSessionStore";
import {
  SAMPLE_RATE,
  SLICE_FLAG_DELTA,
  SLICE_HDR_SIZE,
  type AudioSlice,
  type LoraCatalogEntry,
  type SessionConfig,
  type StemAssetsMessage,
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
  detectedTimeSignature: string | null = null;
  /** Active checkpoint identifier (e.g. "acestep-v15-turbo"). Null when
   *  the server didn't ship one (older backend, --no-backend mode). */
  checkpoint: string | null = null;
  /** Model-scale label for the active checkpoint ("2B" | "5B" | null).
   *  Used by the LoRA library UI to hide LoRAs whose trained
   *  ``base_model_scale`` doesn't match. Null = unknown checkpoint;
   *  the UI treats that as "don't filter". */
  checkpointScale: string | null = null;
  /** Current StreamPipeline ring-buffer depth, mirrored from the
   *  server. Set from the ``ready`` message and from ``depth_applied``
   *  acks after a successful runtime retune. */
  pipelineDepth: number | null = null;
  /** Largest depth the server's loaded backend can serve. TRT decoders
   *  report their hidden_states batch_max; eager / compile pin to 4.
   *  Null until ready. */
  maxPipelineDepth: number | null = null;

  private _pending: PendingPayload | null;
  private _pendingSwap: SwapReadyMessage | null = null;
  private _pendingStemAssets: StemAssetsMessage | null = null;
  private _pendingStemBuffers: Partial<Record<"vocals" | "instruments", Float32Array>> = {};
  // Slice decoder runs in a worker so fzstd.decompress + float16→float32
  // never block the render loop or input handling. Worker is single-threaded
  // and postMessage is FIFO, so audio slices stay in order.
  private _decoderWorker: Worker | null = null;
  private _nextDecodeId = 1;
  // Source-buffer epoch. Bumped right before the swap_ready event is
  // dispatched, so any binary slice that arrives at the WS afterwards is
  // tagged for the new buffer. Slices in flight from before the bump
  // (queued in the WS handler ahead of the swap, or sitting in the
  // decoder worker mid-decode) keep their old epoch and get dropped by
  // the listener — without this they'd land in the new track and bleed
  // chunks of the previous song through.
  private _sliceEpoch = 0;

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
      // The .ts extension is intentional: Next.js / Turbopack and modern
      // bundlers transpile worker source files referenced via
      // `new URL(..., import.meta.url)` at build time. The previous .mjs
      // path was a leftover from when this code shipped as a tsup-built
      // npm package whose dist/ contained a pre-compiled .mjs sibling.
      const worker = new Worker(
        new URL("./workers/sliceDecoder.worker.ts", import.meta.url),
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
          epoch: msg.epoch,
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
        // Phase 1: JSON config, then (unless server-side fixture) the
        // binary audio upload. For known fixtures the pod loads the
        // waveform from its own cache, so re-uploading ~20 MB of PCM
        // here is pure waste (~11 s on the measured cold path). When
        // `use_server_fixture` is set the server skips its audio recv,
        // so we must skip the send to match.
        ws.send(JSON.stringify(this._pending.config));
        const useServerFixture =
          (this._pending.config as { use_server_fixture?: boolean })
            .use_server_fixture === true;
        if (!useServerFixture) {
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
        }
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
            this.detectedTimeSignature = msg.time_signature ?? null;
            this.checkpoint = msg.checkpoint ?? null;
            this.checkpointScale = msg.checkpoint_scale ?? null;
            this.pipelineDepth =
              typeof msg.pipeline_depth === "number"
                ? msg.pipeline_depth
                : null;
            this.maxPipelineDepth =
              typeof msg.max_pipeline_depth === "number"
                ? msg.max_pipeline_depth
                : null;
            // Push the scale + depth bounds into the session store so the
            // LoRA library / engine controls can render without subscribing
            // to RemoteBackend instance fields.
            {
              const s = useSessionStore.getState();
              s.setCheckpointScale(this.checkpointScale);
              s.setPipelineDepth(this.pipelineDepth);
              s.setMaxPipelineDepth(this.maxPipelineDepth);
            }
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
          // Bump epoch BEFORE the dispatch so that the synchronous
          // `player.swap()` call inside the listener (which bumps
          // AudioPlayer.swapCount in lockstep) and any subsequent
          // binary slice the WS hands us are all aligned on the new
          // buffer. Stale slices already queued in the worker still
          // carry the previous epoch and will be dropped by the
          // listener.
          this._sliceEpoch++;
          this.dispatchEvent(
            new CustomEvent("swap_ready", {
              detail: { ...meta, interleaved },
            }),
          );
          return;
        }

        // Stem assets are sent as a JSON header followed by one float16
        // binary buffer per listed stem. Consume them before the generic
        // audio-slice path so overlay buffers never get parsed as slices.
        if (this._pendingStemAssets && ev.data instanceof ArrayBuffer) {
          const meta = this._pendingStemAssets;
          const stem = meta.stems[
            Object.keys(this._pendingStemBuffers).length
          ];
          if (stem) {
            const u16 = new Uint16Array(ev.data);
            this._pendingStemBuffers[stem] = float16ArrayToFloat32(u16);
          }
          const complete = meta.stems.every(
            (name) => this._pendingStemBuffers[name],
          );
          if (complete) {
            const buffers = this._pendingStemBuffers as Record<
              "vocals" | "instruments",
              Float32Array
            >;
            this._pendingStemAssets = null;
            this._pendingStemBuffers = {};
            this.dispatchEvent(
              new CustomEvent("stem_assets", {
                detail: { ...meta, buffers },
              }),
            );
          }
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
          } else if (msg.type === "params_echo") {
            // Echo of raw knob values applied by the MCP control bus;
            // useMcpMirror writes these into the perf/lora stores so the
            // browser's UI moves the sliders to match.
            this.dispatchEvent(
              new CustomEvent("params_echo", { detail: msg.raw }),
            );
          } else if (msg.type === "prompt_blend_echo") {
            // Same shape as params_echo but for the dedicated prompt-
            // blend slider, which doesn't ride the generic params
            // channel. useMcpMirror mirrors this through setSlider so
            // the Smooth tween eases the value and usePromptBlendSync
            // ships the tweened sequence back to the server.
            this.dispatchEvent(
              new CustomEvent("prompt_blend_echo", { detail: msg.value }),
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
          } else if (msg.type === "stem_assets") {
            this._pendingStemAssets = msg as unknown as StemAssetsMessage;
            this._pendingStemBuffers = {};
          } else if (msg.type === "stem_failed") {
            this._pendingStemAssets = null;
            this._pendingStemBuffers = {};
            this.dispatchEvent(
              new CustomEvent("stem_failed", { detail: msg }),
            );
          } else if (msg.type === "timbre_set") {
            this.dispatchEvent(
              new CustomEvent("timbre_set", { detail: msg }),
            );
          } else if (msg.type === "timbre_cleared") {
            this.dispatchEvent(new CustomEvent("timbre_cleared"));
          } else if (msg.type === "timbre_failed") {
            this.dispatchEvent(
              new CustomEvent("timbre_failed", { detail: msg.error }),
            );
          } else if (msg.type === "structure_set") {
            this.dispatchEvent(
              new CustomEvent("structure_set", { detail: msg }),
            );
          } else if (msg.type === "structure_cleared") {
            this.dispatchEvent(new CustomEvent("structure_cleared"));
          } else if (msg.type === "structure_failed") {
            this.dispatchEvent(
              new CustomEvent("structure_failed", { detail: msg.error }),
            );
          } else if (msg.type === "depth_applied") {
            const v = typeof msg.value === "number" ? msg.value : null;
            if (v !== null) {
              this.pipelineDepth = v;
              useSessionStore.getState().setPipelineDepth(v);
              this.dispatchEvent(
                new CustomEvent("depth_applied", { detail: v }),
              );
            }
          } else {
            this.dispatchEvent(new CustomEvent("json", { detail: msg }));
          }
          return;
        }

        if (this._decoderWorker) {
          const buf = ev.data as ArrayBuffer;
          this._decoderWorker.postMessage(
            {
              id: this._nextDecodeId++,
              buffer: buf,
              epoch: this._sliceEpoch,
            },
            [buf],
          );
        } else {
          try {
            const slice = this._parseSlice(ev.data as ArrayBuffer);
            if (slice) {
              slice.epoch = this._sliceEpoch;
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
        //
        // Tailor the message by close code: 1011 (server internal error)
        // and 1006 (abnormal closure) are the two shapes operators see
        // most often, both recoverable by reloading. The previous
        // "Check the server console" tail was useless to end users and
        // made the error feel scarier than it is.
        if (!this.ready) {
          let msg: string;
          if (e.code === 1011) {
            msg = "Server restarted to clear memory — refresh the page to retry.";
          } else if (e.code === 1006) {
            msg = "Connection lost — refresh to retry.";
          } else {
            const reason = e.reason || `code ${e.code}`;
            msg = `Connection failed (${reason}) — refresh to retry.`;
          }
          reject(new Error(msg));
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
      // Caller (the WS onmessage fallback path) overwrites this with the
      // current source epoch right before dispatching.
      epoch: 0,
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

  sendPrompt(
    tags: string,
    key?: string,
    timeSignature?: string,
    tagsB?: string,
  ): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      // Inject the enabled-LoRA trigger words on the WIRE only.
      // Callers pass the operator's prompt text (promptA/promptB from
      // the Tags A/B boxes); the trigger prefix is computed fresh here
      // so every send path — Send Tags button, key change, LoRA toggle
      // — carries the current trigger set without the textareas ever
      // showing it.
      //
      // Hard guarantee, independent of upstream state: stripLeadingTriggers
      // removes ANY trigger prefix already on the text (stale, stacked,
      // or belonging to a since-disabled LoRA), then we prepend exactly
      // the de-duped enabled-set prefix. So the wire prompt always
      // carries a disabled LoRA's trigger zero times and an enabled
      // LoRA's trigger exactly once — per tag (A and B alike).
      const prefix = enabledLoraTriggerPrefix();
      const msg: {
        type: string;
        tags: string;
        tags_b?: string;
        key?: string;
        time_signature?: string;
      } = {
        type: "prompt",
        tags: prefix + stripLeadingTriggers(tags),
      };
      if (tagsB) msg.tags_b = prefix + stripLeadingTriggers(tagsB);
      if (key) msg.key = key;
      if (timeSignature) msg.time_signature = timeSignature;
      this.ws.send(JSON.stringify(msg));
    } catch {}
  }

  /**
   * Live prompt A/B blend knob. Backend keeps cached cond pairs for both
   * prompts (encoded by the most recent ``sendPrompt`` that carried a
   * ``tags_b``) and lerps between them by `value` ∈ [0,1] — 0 == A, 1 == B.
   * Same shape as ``sendSetTimbreStrength``; cheap per slider tick.
   */
  sendSetPromptBlend(value: number): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(JSON.stringify({
        type: "set_prompt_blend",
        value: Math.max(0, Math.min(1, value)),
      }));
    } catch {}
  }

  /**
   * Live pipeline_depth retune. The server stages the value and applies
   * it on the next runner-thread before_tick rendezvous, then echoes
   * the (clamped) result back as ``depth_applied``. Shrinking discards
   * in-flight slots beyond the new depth; growing extends with empty
   * slots that warm up over the next ``newDepth - oldDepth`` ticks.
   */
  sendSetDepth(value: number): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    if (!Number.isFinite(value)) return;
    try {
      this.ws.send(JSON.stringify({
        type: "set_depth",
        value: Math.round(value),
      }));
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
   * Live timbre-strength knob. Backend keeps a cached
   * (cond_silence, cond_full) pair and lerp-blends their encoder hidden
   * states by `value` ∈ [0,1] — 1.0 == full timbre reference, 0.0 ==
   * silence-baseline timbre. Cheap enough to send per slider tick.
   */
  sendSetTimbreStrength(value: number): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(JSON.stringify({
        type: "set_timbre_strength",
        value: Math.max(0, Math.min(1, value)),
      }));
    } catch {}
  }

  /**
   * Send a JSON header followed by a binary audio frame. Wire format
   * matches the init handshake / swap_source: <II header (channels,
   * samples) + interleaved float32 PCM. Used by both timbre and
   * structure source uploads.
   */
  private sendAudioFrame(
    messageType: string,
    name: string,
    interleaved: Float32Array,
    channels: number,
  ): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    try {
      this.ws.send(JSON.stringify({ type: messageType, name }));
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
      console.error(`[protocol] ${messageType} failed:`, e);
      return false;
    }
  }

  /**
   * Upload an audio clip as the active timbre reference. Server VAE-
   * encodes it and replaces cond_full with one conditioned on the clip's
   * latent. The clip is capped server-side to the playback source's
   * duration to fit the loaded TRT profile. Replies with timbre_set on
   * success or timbre_failed on error.
   */
  sendSetTimbreSource(
    interleaved: Float32Array,
    channels: number,
    name: string,
  ): boolean {
    return this.sendAudioFrame(
      "set_timbre_source", name, interleaved, channels,
    );
  }

  /**
   * Pick a Library fixture as the active timbre reference. The server
   * resolves the WAV from its local HF cache and runs the same apply
   * path as a PCM upload, so the browser doesn't have to fetch +
   * decode + re-upload a file that already lives on the pod's disk.
   * Replies with timbre_set on success or timbre_failed on error
   * (e.g. unknown fixture name).
   */
  sendSetTimbreFixture(name: string): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(JSON.stringify({ type: "set_timbre_fixture", name }));
    } catch {}
  }

  /**
   * Drop the active timbre reference; server falls back to self-timbre
   * (encode against the playback source's own latent). Replies with
   * timbre_cleared on success.
   */
  sendClearTimbreSource(): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(JSON.stringify({ type: "clear_timbre_source" }));
    } catch {}
  }

  /**
   * Upload an audio clip as the active structure (semantic-hint)
   * reference. Server pads/trims it to match the playback source's
   * exact sample count, runs prepare_source to extract the override's
   * context_latent, and replaces stream.source.context_latent so the
   * runner's hint-strength blend reads the new structure. Replies with
   * structure_set on success or structure_failed on error.
   */
  sendSetStructureSource(
    interleaved: Float32Array,
    channels: number,
    name: string,
  ): boolean {
    return this.sendAudioFrame(
      "set_structure_source", name, interleaved, channels,
    );
  }

  /**
   * Pick a Library fixture as the active structure reference. Server-
   * side counterpart to sendSetTimbreFixture: avoids the wasteful
   * fetch+decode+upload round trip for fixtures that already live on
   * the pod's disk. Replies with structure_set / structure_failed.
   */
  sendSetStructureFixture(name: string): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(JSON.stringify({ type: "set_structure_fixture", name }));
    } catch {}
  }

  /**
   * Drop the active structure reference; server restores the playback
   * source's own context_latent. Replies with structure_cleared.
   */
  sendClearStructureSource(): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(JSON.stringify({ type: "clear_structure_source" }));
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
    fixtureName?: string,
    timeSignature?: string,
    stemSourceMode?: "full" | "vocals" | "instruments",
  ): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    try {
      const msg: {
        type: string;
        tags?: string;
        key?: string;
        fixture_name?: string;
        time_signature?: string;
        stem_source_mode?: "full" | "vocals" | "instruments";
      } = {
        type: "swap_source",
      };
      if (tags) msg.tags = tags;
      if (key) msg.key = key;
      if (fixtureName) msg.fixture_name = fixtureName;
      if (timeSignature) msg.time_signature = timeSignature;
      if (stemSourceMode) msg.stem_source_mode = stemSourceMode;
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
