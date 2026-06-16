// WebSocket client for the DEMON realtime motion-to-music backend.
// Part of the demon-client SDK: no imports from the host app (no "@/"),
// no store writes. The app observes session state via the CustomEvents
// this class dispatches (ready / slice / swap_ready / depth_applied / ...).
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
  PREEMPTED_CLOSE_CODE,
  SAMPLE_RATE,
  SLICE_FLAG_DELTA,
  SLICE_HDR_SIZE,
  type AudioSlice,
  type CapabilityMask,
  type LoraCatalogEntry,
  type ReadyGeometry,
  type SessionConfig,
  type StemAssetsMessage,
  type SwapReadyMessage,
} from "./types/protocol";
import type { KnobManifestResponse } from "./types/knobs";
// Outbound command payloads + the inbound event-name union are GENERATED from
// the backend wire-contract registry (protocol.py) — see
// types/wireContract.gen.ts. Typing the senders/ladder against them means a
// renamed command, a dropped field, or an unregistered event name fails
// `tsc` instead of silently desyncing from the server.
import type {
  ClearStructureSourceCommand,
  ClearTimbreSourceCommand,
  DisableLoraCommand,
  EnableLoraCommand,
  LoopBandCommand,
  ManualSlotAddCommand,
  ManualSlotPopCommand,
  ParamsCommand,
  PromptCommand,
  SetDepthCommand,
  SetInterpMethodCommand,
  SetPromptBlendCommand,
  SetStructureFixtureCommand,
  SetTimbreFixtureCommand,
  SetStructureSourceCommand,
  SetTimbreSourceCommand,
  SetTimbreStrengthCommand,
  SwapSourceCommand,
  WireEvent,
} from "./types/wireContract.gen";

/** Optional behaviors the host app injects into RemoteBackend. */
export interface RemoteBackendOptions {
  /** Applied to `tags` and `tags_b` on every `sendPrompt` before they hit
   *  the wire. The shipped app injects enabled-LoRA trigger prefixes here;
   *  a bare client can omit it and prompts are sent verbatim. */
  promptTransform?: (tags: string) => string;
  /** Where to load the slice-decoder worker from. Omit under a bundler
   *  (Next/Turbopack, esbuild, ...) — the default `new URL(...)` form is
   *  statically analyzed and the worker ships with the app bundle.
   *  REQUIRED when consuming the prebuilt dist/ bundle from a no-build
   *  static page: point it at the sibling `sliceDecoder.worker.js`
   *  (e.g. "/sdk/sliceDecoder.worker.js"). */
  sliceWorkerUrl?: string | URL;
}

// Skip a 125 Hz params tick when this many bytes are already queued in
// the WebSocket send buffer (see sendParams). One params message is
// ~1-2 KB of JSON, so this is roughly 4-8 ticks of backlog — far above
// anything a healthy connection accumulates, low enough that staleness
// stays bounded at a few tens of ms when the uplink degrades.
const PARAMS_BACKPRESSURE_BYTES = 8 * 1024;

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

// ── PCM framing ────────────────────────────────────────────────────────
// The one binary upload frame shape in the protocol: `<II` header
// (channels, samples, little-endian) + interleaved float32 PCM. Shared by
// the init-handshake upload, the timbre/structure reference uploads, and
// swap_source.

function packPcmFrame(interleaved: Float32Array, channels: number): Uint8Array {
  const samples = interleaved.length / channels;
  const hdr = new ArrayBuffer(8);
  const dv = new DataView(hdr);
  dv.setUint32(0, channels, true);
  dv.setUint32(4, samples, true);
  const pcm = new Uint8Array(interleaved.buffer);
  const combined = new Uint8Array(hdr.byteLength + pcm.byteLength);
  combined.set(new Uint8Array(hdr), 0);
  combined.set(pcm, hdr.byteLength);
  return combined;
}

// ── RemoteBackend ──────────────────────────────────────────────────────

type Phase = "config" | "ready" | "initial-buffer" | "streaming";

interface PendingPayload {
  interleaved: Float32Array;
  channels: number;
  config: SessionConfig;
}

export type WsTracePhase =
  | "idle"
  | "connecting"
  | "open"
  | "config_sent"
  | "init_ack"
  | "ready"
  | "streaming"
  | "error"
  | "closed";

export interface WsTrace {
  attemptId: string;
  urlHost: string;
  connectStartAt: number | null;
  openAt: number | null;
  configSentAt: number | null;
  initAckAt: number | null;
  readyAt: number | null;
  closeAt: number | null;
  errorAt: number | null;
  phase: WsTracePhase;
  ready: boolean;
  closedByUser: boolean;
  wsReadyState: number | null;
  closeCode: number | null;
  closeReason: string;
}

function makeAttemptId(): string {
  try {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
  } catch {}
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function hostFromWsUrl(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return "";
  }
}

export class RemoteBackend extends EventTarget {
  readonly url: string;
  ws: WebSocket | null = null;
  ready = false;
  /** True iff `close()` was called from the app (user-initiated session
   *  teardown). Distinguishes a deliberate disconnect from a network drop
   *  / server crash so the close-event listener can decide whether to
   *  trigger automatic reconnect. */
  closedByUser = false;
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
  /** Backend-declared audio geometry from `ready.geometry`. Null on
   *  servers (and recorded replays) from before the backend-seam
   *  contract surface — fall back to the legacy flat ready fields
   *  (duration/channels/sampleRate above) and client constants. */
  geometry: ReadyGeometry | null = null;
  /** Backend capability mask from `ready.capabilities`. Null = older
   *  server / replay: treat as ungated (everything available). */
  capabilities: CapabilityMask | null = null;
  /** Per-session knob manifest from `ready.knob_manifest` — the same
   *  `{version, knobs}` envelope `GET /api/knobs` serves, but resolved
   *  for THIS session (SDE mode, enabled `lora_str_<id>` knobs). Null
   *  on older servers / replays; `/api/knobs` remains the static
   *  pre-session probe. */
  knobManifest: KnobManifestResponse | null = null;
  /** Active manual steering slot count, mirrored from the server
   *  (`ready` + `manual_slot_count` echoes). Null until ready / on
   *  servers without the steering surface. */
  manualSlotCount: number | null = null;
  /** Server-imposed cap on manual steering slots. Null until ready. */
  manualSlotCap: number | null = null;
  /** Whether the session's checkpoint has steering vectors. The host
   *  hides the steering tiles when false. Null until ready. */
  steeringAvailable: boolean | null = null;
  /** Browser-observed WS lifecycle for this concrete connection attempt. */
  wsTrace: WsTrace;
  /** Pod-side session id from the optional init_ack telemetry message. */
  backendSessionId: string | null = null;
  /** Client id echoed in init_ack; mirrors the config client_id. */
  backendClientId: string | null = null;

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
  // Cumulative bytes of binary SLICE frames received on this connection
  // (swap buffers and stem payloads excluded — the server counts the
  // same set on its side). Reported as `slice_bytes_rx` with every
  // params message; the server uses sent-minus-acked as its in-flight
  // window and stops emitting slices when the link can't drain them.
  // Without this, a bandwidth-limited path (SSH tunnel, weak uplink)
  // buffers many seconds of slices in socket/tunnel queues the server
  // can't observe, and every slice lands behind the playhead.
  private _sliceBytesRx = 0;

  private _promptTransform: (tags: string) => string;
  private _sliceWorkerUrl: string | URL | undefined;

  constructor(
    url: string,
    interleaved: Float32Array,
    channels: number,
    config: SessionConfig,
    opts: RemoteBackendOptions = {},
  ) {
    super();
    this.url = url;
    this._pending = { interleaved, channels, config };
    this.wsTrace = {
      attemptId: makeAttemptId(),
      urlHost: hostFromWsUrl(url),
      connectStartAt: null,
      openAt: null,
      configSentAt: null,
      initAckAt: null,
      readyAt: null,
      closeAt: null,
      errorAt: null,
      phase: "idle",
      ready: false,
      closedByUser: false,
      wsReadyState: null,
      closeCode: null,
      closeReason: "",
    };
    this._promptTransform = opts.promptTransform ?? ((tags) => tags);
    this._sliceWorkerUrl = opts.sliceWorkerUrl;
    this._initDecoderWorker();
  }

  private _snapshotTrace(): WsTrace {
    return { ...this.wsTrace };
  }

  private _updateTrace(patch: Partial<WsTrace>): WsTrace {
    this.wsTrace = {
      ...this.wsTrace,
      ...patch,
      ready: this.ready,
      closedByUser: this.closedByUser,
      wsReadyState: this.ws?.readyState ?? null,
    };
    // SDK code never touches app stores: the host app mirrors the trace
    // into its session store by subscribing to this event.
    const snapshot = this._snapshotTrace();
    this.dispatchEvent(new CustomEvent("ws_trace_update", { detail: snapshot }));
    return snapshot;
  }

  getWsTrace(): WsTrace {
    return this._snapshotTrace();
  }

  private _initDecoderWorker(): void {
    if (typeof Worker === "undefined") return;
    try {
      // The .ts extension is intentional: Next.js / Turbopack and modern
      // bundlers transpile worker source files referenced via
      // `new URL(..., import.meta.url)` at build time. The previous .mjs
      // path was a leftover from when this code shipped as a tsup-built
      // npm package whose dist/ contained a pre-compiled .mjs sibling.
      // The literal `new Worker(new URL(...))` form below must stay intact
      // for that static analysis; sliceWorkerUrl consumers (the prebuilt
      // dist/ bundle on a no-build page) take the other branch.
      const worker = this._sliceWorkerUrl !== undefined
        ? new Worker(this._sliceWorkerUrl, { type: "module" })
        : new Worker(
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
      this._updateTrace({
        connectStartAt: Date.now(),
        openAt: null,
        configSentAt: null,
        initAckAt: null,
        readyAt: null,
        closeAt: null,
        errorAt: null,
        phase: "connecting",
        closeCode: null,
        closeReason: "",
      });

      let phase: Phase = "config";

      ws.onopen = () => {
        if (!this._pending) return;
        this._updateTrace({ openAt: Date.now(), phase: "open" });
        // Phase 1: JSON config, then (unless server-side fixture) the
        // binary audio upload. For known fixtures the pod loads the
        // waveform from its own cache, so re-uploading ~20 MB of PCM
        // here is pure waste (~11 s on the measured cold path). When
        // `use_server_fixture` is set the server skips its audio recv,
        // so we must skip the send to match.
        ws.send(JSON.stringify(this._pending.config));
        const useServerFixture =
          this._pending.config.use_server_fixture === true;
        if (!useServerFixture) {
          const { interleaved, channels } = this._pending;
          ws.send(packPcmFrame(interleaved, channels));
        }
        this._updateTrace({ configSentAt: Date.now(), phase: "config_sent" });
        phase = "ready";
      };

      ws.onmessage = (ev) => {
        if (phase === "ready") {
          try {
            // The generated discriminated union narrows the init messages:
            // `init_ack`, `error` and `ready` field reads below are
            // compile-checked against the registry instead of going
            // through `any`.
            const msg = JSON.parse(ev.data as string) as WireEvent;
            if (msg.type === "init_ack") {
              this.backendSessionId =
                typeof msg.session_id === "string" ? msg.session_id : null;
              this.backendClientId =
                typeof msg.client_id === "string" ? msg.client_id : null;
              this._updateTrace({ initAckAt: Date.now(), phase: "init_ack" });
              this.dispatchEvent(new CustomEvent("ws_init_ack", { detail: msg }));
              return;
            }
            if (msg.type === "error") {
              this._updateTrace({ errorAt: Date.now(), phase: "error" });
              reject(
                new Error(
                  msg.message || `Server error: ${msg.code || "unknown"}`,
                ),
              );
              return;
            }
            if (msg.type !== "ready") {
              this._updateTrace({ errorAt: Date.now(), phase: "error" });
              reject(new Error(`Unexpected init message: ${ev.data}`));
              return;
            }
            this.duration = msg.duration;
            this.channels = msg.channels;
            this.sampleRate = msg.sample_rate;
            // The contract types list elements as unknown; LoraCatalogEntry
            // is the client-side refinement of the catalog rows.
            this.loraCatalog = (msg.lora_catalog as LoraCatalogEntry[]) || [];
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
            // Phase-2 contract surface. The generated event types these
            // as plain dicts; the SDK refines them. All three are
            // wire-optional (older servers / recorded replay transcripts
            // omit them) and null means "fall back": legacy flat fields
            // for geometry, ungated panels for capabilities, /api/knobs
            // for the manifest.
            this.geometry =
              (msg.geometry as ReadyGeometry | undefined) ?? null;
            this.capabilities =
              (msg.capabilities as CapabilityMask | undefined) ?? null;
            this.knobManifest =
              (msg.knob_manifest as KnobManifestResponse | undefined) ?? null;
            // Activation-steering surface. Wire-optional like the
            // Phase-2 fields: null hides the steering tiles host-side.
            this.manualSlotCount =
              typeof msg.manual_slot_count === "number"
                ? msg.manual_slot_count
                : null;
            this.manualSlotCap =
              typeof msg.manual_slot_cap === "number"
                ? msg.manual_slot_cap
                : null;
            this.steeringAvailable =
              typeof msg.steering_available === "boolean"
                ? msg.steering_available
                : null;
            // Scale + depth bounds are exposed as instance fields; the host
            // app mirrors them into its own state from the "ready" event
            // listener (the SDK never writes app stores).
            phase = "initial-buffer";
          } catch (e) {
            this._updateTrace({ errorAt: Date.now(), phase: "error" });
            reject(e);
          }
          return;
        }

        if (phase === "initial-buffer") {
          const u16 = new Uint16Array(ev.data as ArrayBuffer);
          this.initialBuffer = float16ArrayToFloat32(u16);
          this.ready = true;
          this._updateTrace({ readyAt: Date.now(), phase: "ready" });
          phase = "streaming";
          this._pending = null;
          resolve(this);
          this.dispatchEvent(new CustomEvent("ready"));
          this._updateTrace({ phase: "streaming" });
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
          let msg: WireEvent;
          try {
            msg = JSON.parse(ev.data) as WireEvent;
          } catch {
            return;
          }
          // `msg` is the generated WireEvent discriminated union
          // (types/wireContract.gen.ts), so each case below narrows to its
          // event's payload type: a case label the backend wire contract
          // doesn't declare, or a field read the registry doesn't carry,
          // fails `tsc`. The Python drift guard
          // (tests/unit/test_wire_contract.py) parses these labels against
          // the registry from its side.
          switch (msg.type) {
            case "params_update":
              this.dispatchEvent(
                new CustomEvent("params", { detail: msg.params }),
              );
              break;
            case "params_echo":
              // Echo of raw knob values applied by the MCP control bus;
              // useMcpMirror writes these into the perf/lora stores so the
              // browser's UI moves the sliders to match.
              this.dispatchEvent(
                new CustomEvent("params_echo", { detail: msg.raw }),
              );
              break;
            case "prompt_blend_echo":
              // Same shape as params_echo but for the dedicated prompt-
              // blend slider, which doesn't ride the generic params
              // channel. useMcpMirror mirrors this through setSlider so
              // the Smooth tween eases the value and usePromptBlendSync
              // ships the tweened sequence back to the server.
              this.dispatchEvent(
                new CustomEvent("prompt_blend_echo", { detail: msg.value }),
              );
              break;
            case "prompt_applied":
              this.dispatchEvent(
                new CustomEvent("prompt_applied", { detail: msg.tags }),
              );
              break;
            case "lora_catalog":
              this.loraCatalog = (msg.catalog as LoraCatalogEntry[]) || [];
              this.dispatchEvent(
                new CustomEvent("lora_catalog", { detail: this.loraCatalog }),
              );
              break;
            case "swap_ready":
              this._pendingSwap = msg;
              break;
            case "swap_failed":
              this.dispatchEvent(
                new CustomEvent("swap_failed", { detail: msg.error }),
              );
              break;
            case "stem_assets":
              // Sole surviving narrowing cast: StemAssetsMessage refines the
              // generated event's `stems: unknown[]` to the literal names.
              this._pendingStemAssets = msg as StemAssetsMessage;
              this._pendingStemBuffers = {};
              break;
            case "stem_failed":
              this._pendingStemAssets = null;
              this._pendingStemBuffers = {};
              this.dispatchEvent(
                new CustomEvent("stem_failed", { detail: msg }),
              );
              break;
            case "timbre_set":
              this.dispatchEvent(
                new CustomEvent("timbre_set", { detail: msg }),
              );
              break;
            case "timbre_cleared":
              this.dispatchEvent(new CustomEvent("timbre_cleared"));
              break;
            case "timbre_failed":
              this.dispatchEvent(
                new CustomEvent("timbre_failed", { detail: msg.error }),
              );
              break;
            case "structure_set":
              this.dispatchEvent(
                new CustomEvent("structure_set", { detail: msg }),
              );
              break;
            case "structure_cleared":
              this.dispatchEvent(new CustomEvent("structure_cleared"));
              break;
            case "structure_failed":
              this.dispatchEvent(
                new CustomEvent("structure_failed", { detail: msg.error }),
              );
              break;
            case "depth_applied": {
              const v = typeof msg.value === "number" ? msg.value : null;
              if (v !== null) {
                this.pipelineDepth = v;
                this.dispatchEvent(
                  new CustomEvent("depth_applied", { detail: v }),
                );
              }
              break;
            }
            case "command_failed":
              // A `requires`-tagged command was rejected because this
              // session's backend lacks the capability (loud failure, never
              // a silent no-op — see protocol.py). Surface it as a typed
              // event so the host app can toast / revert optimistic UI, and
              // log it so the failure is audible even when nothing listens.
              // Without this case it would fall through to the generic
              // `json` event and the rejection would be invisible.
              console.warn(
                `[protocol] command_failed: ${msg.command} needs backend ` +
                  `capability '${msg.requires}'` +
                  (msg.error ? ` — ${msg.error}` : ""),
              );
              this.dispatchEvent(
                new CustomEvent("command_failed", { detail: msg }),
              );
              break;
            case "manual_slot_count": {
              // Echoed after manual_slot_add / manual_slot_pop (success
              // or refusal). The host mirrors it into its own state.
              const v = typeof msg.count === "number" ? msg.count : null;
              this.manualSlotCount = v;
              this.dispatchEvent(
                new CustomEvent("manual_slot_count", { detail: v }),
              );
              break;
            }
            case "error":
              // Mid-session structured failure (e.g. code=pipeline_error
              // when the server's generation loop dies). Handshake-phase
              // errors reject the connect() promise above; this case is
              // the post-ready path. Dispatched as "server_error" (the
              // plain "error" listener name is reserved for the WS-level
              // transport error) so the host can surface the message
              // instead of leaving a silently frozen UI.
              console.error(
                `[protocol] server error: ${msg.code || "unknown"}` +
                  (msg.message ? ` — ${msg.message}` : ""),
              );
              this.dispatchEvent(
                new CustomEvent("server_error", { detail: msg }),
              );
              break;
            default:
              this.dispatchEvent(new CustomEvent("json", { detail: msg }));
          }
          return;
        }

        if (this._decoderWorker) {
          const buf = ev.data as ArrayBuffer;
          // Count BEFORE the transfer detaches the buffer.
          this._sliceBytesRx += buf.byteLength;
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
            this._sliceBytesRx += (ev.data as ArrayBuffer).byteLength;
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
        const trace = this._updateTrace({
          errorAt: Date.now(),
          phase: this.ready ? this.wsTrace.phase : "error",
        });
        if (!this.ready) {
          reject(
            new Error(
              "WebSocket connection failed (network / port unreachable)",
            ),
          );
        }
        this.dispatchEvent(new CustomEvent("ws_connect_error", { detail: trace }));
        this.dispatchEvent(new CustomEvent("error", { detail: e }));
      };

      ws.onclose = (e) => {
        // If the socket closes before we finished the init handshake, the
        // connect() promise must reject — otherwise the launcher sits on
        // "Uploading..." forever when the server crashes mid-init.
        //
        // Tailor the message by close code: 1011 (server internal error)
        // and 1006 (abnormal closure) are the two shapes operators see
        // most often, both recoverable by reloading.
        if (!this.ready) {
          let msg: string;
          if (e.code === PREEMPTED_CLOSE_CODE) {
            msg = "Another connection took over this session.";
          } else if (e.code === 1011) {
            msg = "Session failed while starting — refresh the page to retry.";
          } else if (e.code === 1006) {
            msg = "Connection lost — refresh to retry.";
          } else {
            const reason = e.reason || `code ${e.code}`;
            msg = `Connection failed (${reason}) — refresh to retry.`;
          }
          reject(new Error(msg));
        }
        const trace = this._updateTrace({
          closeAt: Date.now(),
          phase: "closed",
          closeCode: e.code,
          closeReason: e.reason || "",
        });
        this.dispatchEvent(new CustomEvent("ws_close", { detail: trace }));
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

  /** Returns true only when the message was actually handed to `ws.send`.
   *  Callers that consume a one-shot sample (e.g. the worst-slice-lead
   *  tracker, which clears on read) must re-arm it when this returns false,
   *  or the sample is lost on a dropped tick. */
  sendParams(
    raw: Record<string, number | string | boolean>,
    playbackPos: number,
    /** Worst slice landing lead (seconds, folded modulo duration) observed
     *  since the previous params send; see the wire contract's
     *  `slice_lead_s`. Omit when no slice arrived in the interval. */
    sliceLeadS?: number,
  ): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    // Backpressure gate: when the socket can't drain (slow uplink, TCP
    // retransmit storms), queueing more 125 Hz reports only makes every
    // report STALER — the server re-anchors its playhead clock on each
    // arriving playback_pos, so a growing send queue walks its estimate
    // further into the past and freshly rendered slices land behind the
    // listener (heard as the raw source bleeding through). Fresh-or-
    // nothing: skip the tick instead. The server's playhead clock
    // free-runs at 1x while reports are quiet, which is the correct
    // degradation. Threshold is several ticks' worth of params JSON —
    // normal operation never accumulates that much.
    if (this.ws.bufferedAmount > PARAMS_BACKPRESSURE_BYTES) return false;
    try {
      const msg: ParamsCommand = {
        type: "params",
        raw,
        playback_pos: playbackPos,
        // Monotonic send stamp; the server pairs it with arrival time to
        // estimate report staleness for queueing the gate above can't see
        // (middlebox/tunnel buffering, server-side recv backlog).
        client_time: performance.now() / 1000,
        // Flow-control ack: cumulative slice bytes received. The server
        // holds back slice emission while sent-minus-acked exceeds its
        // in-flight window, so a slow link gets fresh slices at link
        // rate instead of an ever-staler backlog.
        slice_bytes_rx: this._sliceBytesRx,
      };
      if (sliceLeadS !== undefined && Number.isFinite(sliceLeadS)) {
        msg.slice_lead_s = sliceLeadS;
      }
      this.ws.send(JSON.stringify(msg));
      return true;
    } catch {
      return false;
    }
  }

  sendPrompt(
    tags: string,
    key?: string,
    timeSignature?: string,
    tagsB?: string,
  ): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      // The host app's promptTransform (RemoteBackendOptions) is applied to
      // both tags on every send. The shipped app injects enabled-LoRA
      // trigger prefixes there, so every send path — Send Tags button, key
      // change, LoRA toggle — carries the current trigger set without the
      // textareas ever showing it. Without a transform, prompts go out
      // verbatim.
      const msg: PromptCommand = {
        type: "prompt",
        tags: this._promptTransform(tags),
      };
      if (tagsB) msg.tags_b = this._promptTransform(tagsB);
      if (key) msg.key = key;
      if (timeSignature) msg.time_signature = timeSignature;
      this.ws.send(JSON.stringify(msg));
      // Opt-in wire-prompt debug. Run `window.__demonPromptLog = true`
      // in the browser console to log exactly what each `prompt` message
      // carries as actually sent to the engine.
      if (
        typeof window !== "undefined" &&
        (window as unknown as { __demonPromptLog?: boolean }).__demonPromptLog
      ) {
        console.log(
          "[demon prompt → engine]\n" +
            `  tags A (wire)  : ${JSON.stringify(msg.tags)}\n` +
            `  tags B (wire)  : ${
              msg.tags_b != null ? JSON.stringify(msg.tags_b) : "(none)"
            }`,
        );
      }
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
      const msg: SetPromptBlendCommand = {
        type: "set_prompt_blend",
        value: Math.max(0, Math.min(1, value)),
      };
      this.ws.send(JSON.stringify(msg));
    } catch {}
  }

  /**
   * Switch the interpolation path for one of the four live blends
   * (prompt / timbre / structure / feedback) between "slerp" and
   * "linear". slerp walks the per-frame geodesic so the blended value's
   * norm stays constant across the sweep; linear is a straight average
   * that dips at the midpoint. The server applies it immediately
   * (prompt/timbre recompute the cached conditioning; structure/feedback
   * are read live each tick), so the change is audible without a
   * restart. Discrete setting, so no smoothing/echo channel.
   */
  sendSetInterpMethod(
    path: SetInterpMethodCommand["path"],
    method: SetInterpMethodCommand["method"],
  ): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg: SetInterpMethodCommand = {
        type: "set_interp_method",
        path,
        method,
      };
      this.ws.send(JSON.stringify(msg));
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
      const msg: SetDepthCommand = {
        type: "set_depth",
        value: Math.round(value),
      };
      this.ws.send(JSON.stringify(msg));
    } catch {}
  }

  /**
   * Mirror the client loop band to the server. The worklet already wraps
   * end→start locally; this tells the pipeline so it wraps its predictive
   * decode target inside the band too, regenerating the seam after `start`
   * before the playhead loops back to it instead of leaving one stale
   * window of pre-change audio at every loop restart. Pass `null`s to
   * clear (linear chase resumes). Seconds, matching `playback_pos`.
   */
  sendLoopBand(startSec: number | null, endSec: number | null): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg: LoopBandCommand = {
        type: "loop_band",
        start_sec: startSec,
        end_sec: endSec,
      };
      this.ws.send(JSON.stringify(msg));
    } catch {}
  }

  sendEnableLora(id: string, strength?: number): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg: EnableLoraCommand = {
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
      const msg: DisableLoraCommand = { type: "disable_lora", id };
      this.ws.send(JSON.stringify(msg));
    } catch {}
  }

  /** Add the next manual steering slot (LIFO). Server echoes
   *  ``manual_slot_count`` on success or refusal. */
  sendManualSlotAdd(): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg: ManualSlotAddCommand = { type: "manual_slot_add" };
      this.ws.send(JSON.stringify(msg));
    } catch {}
  }

  /** Pop the highest-numbered manual steering slot. */
  sendManualSlotPop(): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg: ManualSlotPopCommand = { type: "manual_slot_pop" };
      this.ws.send(JSON.stringify(msg));
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
      const msg: SetTimbreStrengthCommand = {
        type: "set_timbre_strength",
        value: Math.max(0, Math.min(1, value)),
      };
      this.ws.send(JSON.stringify(msg));
    } catch {}
  }

  /**
   * Send a typed JSON header followed by a binary audio frame
   * (packPcmFrame). Used by the timbre/structure source uploads and
   * swap_source; the caller builds the typed command so the header is
   * contract-checked at compile time.
   */
  private sendAudioFrame(
    msg: SetTimbreSourceCommand | SetStructureSourceCommand | SwapSourceCommand,
    interleaved: Float32Array,
    channels: number,
  ): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    try {
      this.ws.send(JSON.stringify(msg));
      this.ws.send(packPcmFrame(interleaved, channels));
      return true;
    } catch (e) {
      console.error(`[protocol] ${msg.type} failed:`, e);
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
      { type: "set_timbre_source", name }, interleaved, channels,
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
      const msg: SetTimbreFixtureCommand = { type: "set_timbre_fixture", name };
      this.ws.send(JSON.stringify(msg));
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
      const msg: ClearTimbreSourceCommand = { type: "clear_timbre_source" };
      this.ws.send(JSON.stringify(msg));
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
      { type: "set_structure_source", name }, interleaved, channels,
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
      const msg: SetStructureFixtureCommand = {
        type: "set_structure_fixture",
        name,
      };
      this.ws.send(JSON.stringify(msg));
    } catch {}
  }

  /**
   * Drop the active structure reference; server restores the playback
   * source's own context_latent. Replies with structure_cleared.
   */
  sendClearStructureSource(): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg: ClearStructureSourceCommand = {
        type: "clear_structure_source",
      };
      this.ws.send(JSON.stringify(msg));
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
    stemSourceMode?: SwapSourceCommand["stem_source_mode"],
  ): boolean {
    const msg: SwapSourceCommand = {
      type: "swap_source",
    };
    if (tags) msg.tags = tags;
    if (key) msg.key = key;
    if (fixtureName) msg.fixture_name = fixtureName;
    if (timeSignature) msg.time_signature = timeSignature;
    if (stemSourceMode) msg.stem_source_mode = stemSourceMode;
    return this.sendAudioFrame(msg, interleaved, channels);
  }

  /**
   * Swap to a source that already lives on the pod (a built-in fixture or
   * a persisted upload), identified by name only — NO PCM is sent. The
   * server loads the waveform off its own disk, which lets the sidecar +
   * stem caches hit instead of re-encoding and re-ripping a re-uploaded
   * buffer. The reply is the same swap_ready + binary buffer as
   * sendSwapSource, so the player gets its crossfade buffer from the
   * server echo.
   */
  sendSwapSourceByName(
    fixtureName: string,
    tags?: string,
    key?: string,
    timeSignature?: string,
    stemSourceMode?: SwapSourceCommand["stem_source_mode"],
  ): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    try {
      const msg: SwapSourceCommand = {
        type: "swap_source",
        use_server_source: true,
        fixture_name: fixtureName,
      };
      if (tags) msg.tags = tags;
      if (key) msg.key = key;
      if (timeSignature) msg.time_signature = timeSignature;
      if (stemSourceMode) msg.stem_source_mode = stemSourceMode;
      this.ws.send(JSON.stringify(msg));
      return true;
    } catch (e) {
      console.error("[protocol] sendSwapSourceByName failed:", e);
      return false;
    }
  }

  close(): void {
    this.closedByUser = true;
    this._updateTrace({ closedByUser: true });
    try {
      this.ws?.close();
    } catch {}
    try {
      this._decoderWorker?.terminate();
    } catch {}
    this._decoderWorker = null;
  }

  /** Align the slice-epoch counter to a target value. Used by the
   *  reconnect path: after `player.swap()` bumps `player.swapCount`
   *  to mark a fresh source buffer, the new remote's `_sliceEpoch`
   *  (which starts at 0 for every new `RemoteBackend` instance) has
   *  to match — otherwise the slice listener's `epoch !== swapCount`
   *  guard drops every incoming slice for the rest of the session.
   *  Safe to call before any WS slice has been posted to the
   *  decoder worker (which is the case during reconnect, since
   *  worker post happens inside `ws.onmessage` after `connect()`
   *  resolves and the slice listener can run). */
  setSliceEpoch(epoch: number): void {
    this._sliceEpoch = epoch;
  }

  /** Test/dev hook: synthesize an abnormal close so the client-side
   *  reconnect path can be exercised without needing real network
   *  failure. The browser maps a TCP RST (the dominant production
   *  cause of 1006 from RunPod / vast.ai tunnels) to a CloseEvent
   *  with code 1006, wasClean:false. We construct the same event
   *  shape and route it through the same `close` listeners the real
   *  socket would, then tear down the underlying ws so no further
   *  frames or events arrive — matching what the OS-level RST does.
   */
  simulateClose(code = 1006, reason = "simulated"): void {
    const ws = this.ws;
    // Detach the real ws callbacks before closing the underlying
    // socket. Without this, `ws.close()` below would fire ws.onclose
    // with a clean code (1005/1000) AND we'd synthesize the "close"
    // CustomEvent below — the reconnect listener would run twice on
    // a single simulated drop. Nulling onerror/onmessage too keeps
    // any in-flight frames from racing the synthetic event.
    if (ws) {
      try {
        ws.onclose = null;
        ws.onerror = null;
        ws.onmessage = null;
      } catch {}
      try {
        ws.close();
      } catch {}
    }
    this.ws = null;
    const trace = this._updateTrace({
      closeAt: Date.now(),
      phase: "closed",
      closeCode: code,
      closeReason: reason,
    });
    this.dispatchEvent(new CustomEvent("ws_close", { detail: trace }));
    // Build a CloseEvent shaped like the real thing. CloseEvent isn't
    // always constructible in older environments, so fall back to a
    // plain Event with the relevant fields glued on.
    let ev: CloseEvent | (Event & { code: number; reason: string; wasClean: boolean });
    try {
      ev = new CloseEvent("close", { code, reason, wasClean: false });
    } catch {
      const e = new Event("close") as Event & {
        code: number;
        reason: string;
        wasClean: boolean;
      };
      e.code = code;
      e.reason = reason;
      e.wasClean = false;
      ev = e;
    }
    this.dispatchEvent(new CustomEvent("close", { detail: ev }));
  }
}
