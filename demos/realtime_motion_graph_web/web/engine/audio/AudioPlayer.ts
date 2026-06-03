// Main-thread wrapper around the realtime-buffer AudioWorklet.
// Falls back to ScriptProcessorNode when AudioWorklet is unavailable
// (non-secure contexts like plain HTTP to a remote IP).

import { SAMPLE_RATE } from "@/engine/protocol";
import { getConfig } from "@/lib/config";

import {
  lufsMakeupGain,
  measureBlock,
  measureLoudness,
  type LoudnessMetric,
} from "./lufs";

type MirrorListener = () => void;
type StemOverlayKind = "vocals" | "instruments";

// Stem overlays (vocals / instruments) are mixed *inside* the audio worklet
// (or _spProcess in the ScriptProcessor fallback) so they share the main
// buffer's playhead and seam-crossfade loop point. The previous design used
// standalone AudioBufferSourceNodes on the main thread; their hard-loop
// didn't match the worklet's seam fade and the two clocks ran free, so the
// overlay drifted out of phase with the inference output after the first
// wrap. The main thread now just holds the buffer + smoothing state for
// the SP fallback path and forwards everything to the worklet otherwise.
interface StemOverlayState {
  interleaved: Float32Array | null;
  channels: number;
  frameCount: number;
  /** UI-set target. Worklet smooths to this; SP fallback smooths via
   *  `volume` below using the same time constant. */
  targetVolume: number;
  /** Live smoothed value. Only used by the SP fallback path (the worklet
   *  owns its own smoothing). */
  volume: number;
}

// 25 ms single-pole IIR for overlay volume smoothing in the SP fallback.
// Matches the worklet's _volRampAlpha so both code paths feel the same.
const SP_OVERLAY_VOL_RAMP_S = 0.025;

// –1 dBTP ceiling, expressed as linear amplitude (10 ** (-1/20)).
const LUFS_PEAK_CEILING = 0.891;
// Headroom multiplier applied to the source's true peak when adapting
// the matcher's ceiling. 4x = +12 dB above source peak, enough to
// fully boost most quieter denoised content into source-loudness
// parity without clipping the gain-modulated output further than the
// source already does. Operator can override via audio.lufs_peak_headroom.
const LUFS_PEAK_HEADROOM_DEFAULT = 4.0;
// Smoothing time constant for gain ramps. Long enough to avoid audible
// clicks when the meter sees a sudden energy drop, short enough that
// the user hears the makeup within ~250 ms.
const LUFS_GAIN_RAMP_TC = 0.08;
// Default sliding window for the meter (BS.1770 short-term). Kept
// for the source-target measurement at init/swap; the live meter
// loop now reads a per-chunk loudness map instead of a sliding
// window so it can react to denoised slices the instant the
// playhead enters them.
const LUFS_METER_WINDOW_DEFAULT_SEC = 3.0;
const LUFS_METER_WINDOW_MIN_SEC = 0.5;
const LUFS_METER_INTERVAL_MS = 100;
// Frame size for the per-chunk loudness map. 0.3 s at 48 kHz matches
// the streaming pipeline's slice size, so each engine-written slice
// updates exactly one or two map entries. Smaller chunks would give
// tighter time resolution but K/A-weighting filter transients would
// dominate the per-chunk reading.
const LUFS_CHUNK_FRAMES = 14400;
// Disengage the matcher when the chunk at the playhead is this many dB
// or more below target. Below the floor, "match loudness" turns into
// "amplify near-zero up to source loudness," which boosts low-level
// model artifacts in silent regions (mid-song silence, end of track,
// start of loop) by tens to hundreds of times. 30 dB is well outside
// the range any real musical content reaches relative to a gated
// integrated target, so normal loudness matching is unaffected.
const LUFS_SILENCE_FLOOR_DB_DEFAULT = 30.0;
// Hysteresis band on the silence floor. Once disengaged, the matcher
// re-engages only when measured rises back to (target - floor +
// hysteresis), i.e. comfortably inside the active band. Without this,
// chunks whose loudness sits near the floor flip every tick and the
// gain swells between 1.0 and the makeup value, audible as volume
// fluctuations.
const LUFS_SILENCE_FLOOR_HYSTERESIS_DB_DEFAULT = 6.0;

interface AudioWorkletNodeWithPort extends AudioNode {
  port: MessagePort;
}

export class AudioPlayer {
  ctx: AudioContext | null = null;
  node: AudioWorkletNode | ScriptProcessorNode | null = null;
  positionSec = 0;
  swapCount = 0;
  channels = 2;
  frameCount = 0;
  // Most recent kick (RMS over a 480-frame window, soft-clipped to [0,1]).
  // Computed by the AudioWorklet on the audio thread and posted alongside
  // position; the main render path reads it via getKick(). On the
  // ScriptProcessor fallback path this stays 0 — kick reactivity degrades
  // gracefully (no flashes on beats) rather than blocking the main thread
  // with a per-frame RMS loop. See PERFORMANCE.md.
  kick = 0;

  private _listeners: Set<MirrorListener> = new Set();
  private _mirror: Float32Array | null = null;
  private _useWorklet = false;
  private _spBuffer: Float32Array | null = null;
  private _spPosition = 0;
  private _recordDest: MediaStreamAudioDestinationNode | null = null;
  private _masterOut: GainNode | null = null;
  private _stemOverlays: Record<StemOverlayKind, StemOverlayState> = {
    vocals: { interleaved: null, channels: 2, frameCount: 0, targetVolume: 0, volume: 0 },
    instruments: { interleaved: null, channels: 2, frameCount: 0, targetVolume: 0, volume: 0 },
  };
  private _spOverlayVolAlpha = 0;

  // Loop + seek state. The worklet path owns its own copy of `loop` (set
  // via postMessage); these fields are the main-thread mirror so the SP
  // fallback path can read them directly. _spEndSignaled is the SP-path
  // equivalent of the worklet's _endSignaled one-shot.
  private _loop = true;
  private _spEndSignaled = false;
  private _endOfBufferListeners: Set<() => void> = new Set();

  // Loudness matching: a GainNode sits between the worklet and
  // destination. We measure the source's integrated loudness once at
  // init() / swap() and lock it as the target. The meter periodically
  // measures the playhead window; if that window is quieter than the
  // source target, we boost it up. We never attenuate. So source
  // plays at unity (it already is the target) and any quieter
  // remix-output at the playhead gets boosted up to source loudness.
  //
  // No running-max, no high-water bookkeeping: the source is the
  // reference, full stop. If the operator's remix happens to be
  // louder than source, "louder side wins" via the never-attenuate
  // clamp -- it plays at unity, source plays at unity, and the
  // matcher does nothing. That's the intended behaviour.
  private _makeupGain: GainNode | null = null;
  private _lufsEnabled = false;
  private _sourceTarget: number | null = null;
  private _meterIntervalId: number | null = null;
  private _meterWindowSec = LUFS_METER_WINDOW_DEFAULT_SEC;
  private _loudnessMetric: LoudnessMetric = "lufs";
  // Effective peak ceiling for the makeup gain. Default is -1 dBTP
  // (0.891), expanded at init/swap to max(default, source_peak *
  // headroom_factor) so a source with hot peaks doesn't pin the
  // never-attenuate clamp at unity for the entire session.
  private _peakCeiling = LUFS_PEAK_CEILING;
  private _silenceFloorDb = LUFS_SILENCE_FLOOR_DB_DEFAULT;
  private _silenceFloorHysteresisDb = LUFS_SILENCE_FLOOR_HYSTERESIS_DB_DEFAULT;
  private _inSilence = false;
  // Per-chunk loudness/peak map of the mirror. The matcher consults
  // these arrays at meter time to know "what's at the playhead right
  // now" without waiting for a sliding window to fill -- which means
  // gain updates the instant the playhead enters a freshly-written
  // denoised chunk, instead of swelling up over the window length.
  // Both arrays are length ceil(totalFrames / LUFS_CHUNK_FRAMES) and
  // are populated at init/swap and refreshed on every patch/addDelta.
  private _chunkLoudness: Float32Array | null = null;
  private _chunkPeak: Float32Array | null = null;
  /** Most recent short-term loudness reading at the playhead (or null
   *  when the window had nothing audible). Units depend on the active
   *  metric (LUFS or dBA). Exposed for UI readouts. */
  lufsMeasured: number | null = null;

  get duration(): number {
    return this.frameCount / SAMPLE_RATE;
  }

  async init(
    initialBufferInterleaved: Float32Array,
    channels: number,
  ): Promise<void> {
    this.ctx = new AudioContext({
      sampleRate: SAMPLE_RATE,
      latencyHint: "interactive",
    });

    this.channels = channels;
    this.frameCount = initialBufferInterleaved.length / channels;
    this._mirror = initialBufferInterleaved.slice();

    this._useWorklet = !!this.ctx.audioWorklet;

    if (this._useWorklet) {
      // Stable URL — worklet ships from public/ so AudioContext can resolve it.
      // Cache-busting query bumped manually when the worklet behaviour or
      // message surface changes (v2 = setLoopBand, v3 = stem overlays
      // inside the worklet, v4 = band-loop seam crossfade, v5 = band loop
      // at the buffer end no longer freezes). The browser caches worklet
      // bytes per-URL and hard refresh doesn't always invalidate.
      await this.ctx.audioWorklet.addModule("/audio-worklet.js?v=5");

      const node = new AudioWorkletNode(this.ctx, "realtime-buffer", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [channels],
      });
      this.node = node;

      node.port.onmessage = (e: MessageEvent) => {
        const msg = e.data as {
          type: string;
          positionSec?: number;
          swapCount?: number;
          kick?: number;
        };
        if (msg.type === "position") {
          this.positionSec = msg.positionSec ?? 0;
          this.swapCount = msg.swapCount ?? this.swapCount;
          if (typeof msg.kick === "number") this.kick = msg.kick;
        } else if (msg.type === "endOfBuffer") {
          for (const fn of this._endOfBufferListeners) fn();
        }
      };

      const send = initialBufferInterleaved.slice();
      node.port.postMessage(
        { type: "init", buffer: send, channels },
        [send.buffer],
      );
    } else {
      // ScriptProcessorNode fallback for non-secure contexts.
      console.warn(
        "[AudioPlayer] AudioWorklet unavailable (non-secure context). Using ScriptProcessor fallback.",
      );
      this._spBuffer = initialBufferInterleaved.slice();
      this._spPosition = 0;
      const BUFFER_SIZE = 4096;
      const sp = this.ctx.createScriptProcessor(BUFFER_SIZE, 0, channels);
      this.node = sp;
      sp.onaudioprocess = (e: AudioProcessingEvent) => this._spProcess(e);
    }

    // Per-frame IIR coefficient for SP-fallback overlay volume smoothing.
    // Computed against the live AudioContext sample rate so it matches the
    // worklet's _volRampAlpha (same 25 ms time constant).
    this._spOverlayVolAlpha =
      1 - Math.exp(-1 / (this.ctx.sampleRate * SP_OVERLAY_VOL_RAMP_S));

    this._masterOut = this.ctx.createGain();
    this._masterOut.gain.value = 1.0;
    this._masterOut.connect(this.ctx.destination);

    this._makeupGain = this.ctx.createGain();
    this._makeupGain.gain.value = 1.0;
    this.node.connect(this._makeupGain);
    this._makeupGain.connect(this._masterOut);

    this._measureSourceTarget();
    if (this._lufsEnabled) this._startMetering();
  }

  /** Overwrite a region of the worklet's buffer. */
  patch(startFrame: number, audioInterleaved: Float32Array): void {
    this._writeMirror(startFrame, audioInterleaved, false);
    this._refreshChunks(startFrame, audioInterleaved.length / this.channels);
    if (this._useWorklet && this.node) {
      const send = audioInterleaved.slice();
      (this.node as AudioWorkletNode).port.postMessage(
        { type: "patch", start: startFrame, audio: send },
        [send.buffer],
      );
    } else {
      this._writeSPBuffer(startFrame, audioInterleaved, false);
    }
  }

  /**
   * Replace the entire loop buffer. The worklet crossfades old → new over
   * CROSSFADE_SECONDS (25 ms); ScriptProcessor fallback does an instant
   * swap (the seam-fade still hides the wrap).
   */
  swap(interleavedBuffer: Float32Array, channels?: number): void {
    this.clearStemOverlays();
    this.channels = channels || this.channels;
    this.frameCount = interleavedBuffer.length / this.channels;
    this._mirror = interleavedBuffer.slice();
    this.swapCount++;
    for (const fn of this._listeners) fn();
    if (this._useWorklet && this.node) {
      const send = interleavedBuffer.slice();
      (this.node as AudioWorkletNode).port.postMessage(
        { type: "swap", buffer: send, channels: this.channels },
        [send.buffer],
      );
    } else {
      this._spBuffer = interleavedBuffer.slice();
      // Preserve playhead phase across the swap, mirroring the worklet
      // `swap` handler (which leaves `this.position` untouched). Clamp into
      // the new buffer so a shorter track can't read past its end. Whether
      // a swap restarts from 0 is owned solely by the restart_song_on_swap
      // gate in useFixtureSwap (which issues seek(0)); hard-zeroing here
      // overrode that flag on the SP fallback path, so `false` never took.
      this._spPosition = Math.max(
        0,
        Math.min(this.frameCount - 1, this._spPosition),
      );
    }
    // Track changed: re-measure source loudness for the new buffer.
    this._sourceTarget = null;
    this.lufsMeasured = null;
    this._inSilence = false;
    this._measureSourceTarget();
  }

  /** Delta-add into a region of the worklet's buffer. */
  addDelta(startFrame: number, deltaInterleaved: Float32Array): void {
    this._writeMirror(startFrame, deltaInterleaved, true);
    this._refreshChunks(startFrame, deltaInterleaved.length / this.channels);
    if (this._useWorklet && this.node) {
      const send = deltaInterleaved.slice();
      (this.node as AudioWorkletNode).port.postMessage(
        { type: "add", start: startFrame, audio: send },
        [send.buffer],
      );
    } else {
      this._writeSPBuffer(startFrame, deltaInterleaved, true);
    }
  }

  setStemOverlay(
    kind: StemOverlayKind,
    interleaved: Float32Array,
    channels: number,
  ): void {
    if (!this.ctx) return;
    const state = this._stemOverlays[kind];
    // Keep an own copy: postMessage with a transfer would zero the
    // caller's buffer (useStemOverlaySync hands us a Float32Array it
    // doesn't own a duplicate of), and the SP fallback also needs the
    // buffer to stay live on the main thread.
    state.interleaved = interleaved.slice();
    state.channels = channels;
    state.frameCount = (interleaved.length / channels) | 0;
    if (this._useWorklet && this.node) {
      const send = interleaved.slice();
      const node = this.node as AudioWorkletNode;
      node.port.postMessage(
        { type: "setOverlayBuffer", kind, buffer: send, channels },
        [send.buffer],
      );
      // Re-assert target volume so a buffer arriving after a clear
      // (e.g. song swap → new stem_assets) picks the user's level back
      // up instead of staying silent until they touch the slider.
      // setOverlayBuffer's volume seed comes from the previous entry,
      // which clearOverlayBuffer just deleted.
      if (state.targetVolume > 0) {
        node.port.postMessage({
          type: "setOverlayVolume",
          kind,
          volume: state.targetVolume,
        });
      }
    }
  }

  clearStemOverlays(): void {
    (Object.keys(this._stemOverlays) as StemOverlayKind[]).forEach((kind) => {
      const state = this._stemOverlays[kind];
      state.interleaved = null;
      state.frameCount = 0;
      state.volume = 0;
      // targetVolume is intentionally preserved across a clear: the
      // user's slider position outlives a song swap, and setStemOverlay
      // re-asserts it to the worklet when the next buffer lands.
      if (this._useWorklet && this.node) {
        (this.node as AudioWorkletNode).port.postMessage({
          type: "clearOverlayBuffer",
          kind,
        });
      }
    });
  }

  setStemOverlayVolume(kind: StemOverlayKind, volume: number): void {
    const state = this._stemOverlays[kind];
    const v = Math.max(0, Math.min(6.0, volume));
    state.targetVolume = v;
    if (this._useWorklet && this.node) {
      (this.node as AudioWorkletNode).port.postMessage({
        type: "setOverlayVolume",
        kind,
        volume: v,
      });
    }
    // SP fallback uses state.volume which is smoothed inside _spProcess
    // against state.targetVolume, so no immediate write is needed here.
  }

  /** Read-only view of the current buffer (for waveform rendering). */
  getMirror(): Float32Array | null {
    return this._mirror;
  }

  onMirrorChange(fn: MirrorListener): () => void {
    this._listeners.add(fn);
    return () => {
      this._listeners.delete(fn);
    };
  }

  async resume(): Promise<void> {
    const ctx = this.ctx;
    if (!ctx || ctx.state !== "suspended") return;
    // Android Chrome quirk: AudioContext.resume() can leave its promise
    // pending forever even after the context has actually transitioned
    // to "running". Awaiting that promise stalled useStartSession past
    // the setStatus("ready", "Playing") line, so the StatusBar stayed
    // stuck on "STARTING AUDIO…" and useParamSync (gated on "ready")
    // never sent slider updates — even though audio was audibly playing.
    //
    // Race the resume() promise against a `statechange` listener and a
    // hard 2s fallback so this method always resolves; callers can
    // re-check ctx.state if they care.
    await new Promise<void>((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        ctx.removeEventListener("statechange", onState);
        resolve();
      };
      const onState = () => {
        if (ctx.state !== "suspended") finish();
      };
      ctx.addEventListener("statechange", onState);
      ctx.resume().then(finish, finish);
      window.setTimeout(finish, 2000);
    });
  }

  /**
   * Lazily create a MediaStream tee'd off the worklet output for recording.
   * Same node graph as the live destination — bit-identical to what the
   * user hears. Stays alive for the rest of the session once created.
   */
  getRecordingStream(): MediaStream | null {
    if (!this.ctx || !this.node) return null;
    if (!this._recordDest) {
      this._recordDest = this.ctx.createMediaStreamDestination();
      // Tap from the makeup-gain output (when present) so recordings
      // reflect what the user hears with LUFS normalization applied.
      // Falls back to the raw node if init somehow ran without creating
      // the gain (defensive — current init() always does).
      const tap = this._masterOut ?? this._makeupGain ?? this.node;
      tap.connect(this._recordDest);
    }
    return this._recordDest.stream;
  }

  /**
   * Toggle loudness matching. When enabled, a periodic meter tracks
   * the running-max short-term LUFS and ramps the makeup gain so
   * quieter passages match the loudest seen (peak-clamped at –1 dBTP).
   * When disabled, the meter stops, the high-water mark resets, and
   * the gain ramps back to 1.0 (mathematically transparent).
   */
  setLufs(enabled: boolean): void {
    this._lufsEnabled = enabled;
    if (!this._makeupGain || !this.ctx) return;
    if (enabled) {
      this._startMetering();
    } else {
      this._stopMetering();
      this.lufsMeasured = null;
      const t = this.ctx.currentTime;
      this._makeupGain.gain.cancelScheduledValues(t);
      this._makeupGain.gain.setTargetAtTime(1.0, t, LUFS_GAIN_RAMP_TC);
    }
  }

  /**
   * Toggle loop-at-end. When true (default), the worklet wraps the
   * playhead via the seam crossfade. When false, the playhead clamps
   * at end-of-buffer, the processor emits silence, and a one-shot
   * endOfBuffer event fires so callers can flip a paused flag.
   */
  setLoop(enabled: boolean): void {
    this._loop = enabled;
    this._spEndSignaled = false;
    // Overlays follow the main playhead inside the worklet / _spProcess,
    // so toggling loop here doesn't need to be propagated to them
    // separately — they share the wrap logic.
    if (this._useWorklet && this.node) {
      (this.node as AudioWorkletNode).port.postMessage({
        type: "setLoop",
        enabled,
      });
    }
  }

  /** Jump the playhead. Seconds are clamped into the current buffer. */
  seek(positionSec: number): void {
    if (this.frameCount === 0) return;
    const target = Math.max(
      0,
      Math.min(this.frameCount - 1, Math.round(positionSec * SAMPLE_RATE)),
    );
    this.positionSec = target / SAMPLE_RATE;
    this._spEndSignaled = false;
    // Overlays follow the main playhead inside the worklet / _spProcess —
    // a seek is automatically reflected at sample boundary on the next
    // process tick, no overlay restart needed.
    if (this._useWorklet && this.node) {
      (this.node as AudioWorkletNode).port.postMessage({
        type: "seek",
        positionFrames: target,
      });
    } else {
      this._spPosition = target;
    }
  }

  /** Lock playback to a sub-region of the buffer. The AudioWorklet wraps
   *  end→start each time the playhead reaches `endSec`. Pure client-side
   *  loop — the engine keeps generating linearly and writing slices into
   *  the same buffer, so what plays inside the band is whatever lives
   *  there now (regenerated audio on subsequent laps; original source
   *  in regions generation hasn't touched yet).
   *
   *  Pass min 30 ms band — anything shorter spins the wrap path tight
   *  enough to be perceived as silence. Out-of-range values are clamped.
   */
  setLoopBand(startSec: number, endSec: number): void {
    if (this.frameCount === 0) return;
    const startFrames = Math.max(
      0,
      Math.min(this.frameCount - 1, Math.round(startSec * SAMPLE_RATE)),
    );
    const endFrames = Math.max(
      startFrames + Math.floor(SAMPLE_RATE * 0.03),
      Math.min(this.frameCount, Math.round(endSec * SAMPLE_RATE)),
    );
    if (this._useWorklet && this.node) {
      (this.node as AudioWorkletNode).port.postMessage({
        type: "setLoopBand",
        startFrames,
        endFrames,
      });
    }
  }

  /** Remove any active band loop; playback resumes wrapping at
   *  end-of-buffer (subject to `setLoop`). */
  clearLoopBand(): void {
    if (this._useWorklet && this.node) {
      (this.node as AudioWorkletNode).port.postMessage({
        type: "clearLoopBand",
      });
    }
  }

  /** Subscribe to end-of-buffer hits (only fires when loop is off).
   *  Returns an unsubscribe. */
  onEndOfBuffer(fn: () => void): () => void {
    this._endOfBufferListeners.add(fn);
    return () => {
      this._endOfBufferListeners.delete(fn);
    };
  }

  async close(): Promise<void> {
    this._stopMetering();
    this.clearStemOverlays();
    try {
      this.node?.disconnect();
    } catch {}
    try {
      this._masterOut?.disconnect();
    } catch {}
    this._recordDest = null;
    try {
      await this.ctx?.close();
    } catch {}
  }

  // ── internals ────────────────────────────────────────────────────────

  private _startMetering(): void {
    if (this._meterIntervalId !== null) return;
    // Snapshot the window length and metric at meter-start so a
    // config reload mid-session can't cause a discontinuity.
    const audioCfg = getConfig().audio;
    const cfgWin = audioCfg.lufs_window_sec;
    this._meterWindowSec = Math.max(
      LUFS_METER_WINDOW_MIN_SEC,
      Number.isFinite(cfgWin) ? cfgWin : LUFS_METER_WINDOW_DEFAULT_SEC,
    );
    this._loudnessMetric =
      audioCfg.lufs_metric === "dba" ? "dba" : "lufs";
    const cfgFloor = audioCfg.lufs_silence_floor_db;
    this._silenceFloorDb =
      Number.isFinite(cfgFloor) && cfgFloor > 0
        ? cfgFloor
        : LUFS_SILENCE_FLOOR_DB_DEFAULT;
    const cfgHyst = audioCfg.lufs_silence_floor_hysteresis_db;
    this._silenceFloorHysteresisDb =
      Number.isFinite(cfgHyst) && cfgHyst >= 0
        ? cfgHyst
        : LUFS_SILENCE_FLOOR_HYSTERESIS_DB_DEFAULT;
    this._inSilence = false;
    this._meterIntervalId = window.setInterval(
      () => this._meterTick(),
      LUFS_METER_INTERVAL_MS,
    );
  }

  private _stopMetering(): void {
    if (this._meterIntervalId === null) return;
    window.clearInterval(this._meterIntervalId);
    this._meterIntervalId = null;
  }

  /**
   * One-shot pass over the source buffer at init() / swap() that
   * captures the two numbers the matcher needs:
   *   1. integrated source loudness  -> _sourceTarget (the boost target)
   *   2. true sample peak            -> drives _peakCeiling
   *
   * Both are stable across the session; the meter loop reads them
   * but never modifies them. Reset and re-measured on swap().
   *
   * Runs synchronously on the main thread (~50-200 ms for a 60 s
   * 48 kHz buffer). Move to a worker if buffer sizes grow into the
   * multi-minute range.
   */
  private _measureSourceTarget(): void {
    if (!this._mirror) return;
    const audioCfg = getConfig().audio;
    const metric: LoudnessMetric =
      audioCfg.lufs_metric === "dba" ? "dba" : "lufs";
    // Snap the metric on init/swap so chunk-map readings stay in
    // the same units as _sourceTarget for the rest of the session.
    this._loudnessMetric = metric;

    const { value, peak } = measureLoudness(
      this._mirror,
      this.channels,
      SAMPLE_RATE,
      metric,
    );
    this._sourceTarget = value;

    const headroomFactor = Number.isFinite(audioCfg.lufs_peak_headroom)
      ? audioCfg.lufs_peak_headroom
      : LUFS_PEAK_HEADROOM_DEFAULT;
    this._peakCeiling = Math.max(
      LUFS_PEAK_CEILING,
      peak * headroomFactor,
    );

    // Allocate + populate the per-chunk loudness map for the whole
    // mirror. Updated incrementally by patch/addDelta as denoised
    // slices replace source content over time.
    const totalFrames = (this._mirror.length / this.channels) | 0;
    const numChunks = Math.max(1, Math.ceil(totalFrames / LUFS_CHUNK_FRAMES));
    this._chunkLoudness = new Float32Array(numChunks);
    this._chunkPeak = new Float32Array(numChunks);
    this._refreshChunks(0, totalFrames);
  }

  /**
   * Re-measure every chunk that overlaps [startFrame, startFrame+frames).
   * Cheap because chunks are 0.3 s; a 0.3 s slice typically touches
   * exactly one or two chunks. Called from patch/addDelta after the
   * mirror is updated, and from _measureSourceTarget for the full
   * initial pass.
   */
  private _refreshChunks(startFrame: number, frames: number): void {
    if (!this._mirror || !this._chunkLoudness || !this._chunkPeak) return;
    const ch = this.channels;
    const totalFrames = (this._mirror.length / ch) | 0;
    if (totalFrames === 0 || frames <= 0) return;
    const numChunks = this._chunkLoudness.length;
    const firstChunk = Math.max(0, Math.floor(startFrame / LUFS_CHUNK_FRAMES));
    const lastChunk = Math.min(
      numChunks - 1,
      Math.floor((startFrame + frames - 1) / LUFS_CHUNK_FRAMES),
    );
    for (let c = firstChunk; c <= lastChunk; c++) {
      const cStart = c * LUFS_CHUNK_FRAMES;
      const cEnd = Math.min(cStart + LUFS_CHUNK_FRAMES, totalFrames);
      const slice = this._mirror.subarray(cStart * ch, cEnd * ch);
      const { loudness, peak } = measureBlock(slice, ch, this._loudnessMetric);
      this._chunkLoudness[c] = loudness;
      this._chunkPeak[c] = peak;
    }
  }

  private _meterTick(): void {
    if (!this._makeupGain || !this.ctx) return;
    const target = this._sourceTarget;
    if (target === null) return;
    const map = this._chunkLoudness;
    const peakMap = this._chunkPeak;
    if (!map || !peakMap || !this._mirror) return;
    const totalFrames = (this._mirror.length / this.channels) | 0;
    if (totalFrames === 0) return;
    // Look up loudness/peak of whatever's currently at the playhead,
    // pre-measured by patch/addDelta when the slice landed. No
    // sliding-window lag.
    const posFramesRaw = (this.positionSec * SAMPLE_RATE) | 0;
    const posFrames =
      ((posFramesRaw % totalFrames) + totalFrames) % totalFrames;
    const chunkIdx = Math.min(
      map.length - 1,
      Math.floor(posFrames / LUFS_CHUNK_FRAMES),
    );
    const measured = map[chunkIdx];
    const peak = peakMap[chunkIdx];
    this.lufsMeasured = Number.isFinite(measured) ? measured : null;
    const t = this.ctx.currentTime;
    this._makeupGain.gain.cancelScheduledValues(t);
    // Silence-floor disengage with Schmitt-trigger hysteresis: when the
    // chunk at the playhead is more than _silenceFloorDb below target
    // (or fully silent), computing makeup gain would amplify near-zero
    // up to source loudness and explode any low-level model artifacts.
    // Ramp toward unity instead. The hysteresis band stops chunks
    // hovering at the threshold from flapping the gain every tick.
    const gap = Number.isFinite(measured) ? target - measured : Infinity;
    if (this._inSilence) {
      if (gap < this._silenceFloorDb - this._silenceFloorHysteresisDb) {
        this._inSilence = false;
      }
    } else {
      if (gap > this._silenceFloorDb) {
        this._inSilence = true;
      }
    }
    let gain: number;
    if (this._inSilence) {
      gain = 1.0;
    } else {
      const matchGain = lufsMakeupGain(measured, target, peak, this._peakCeiling);
      gain = Math.max(1.0, matchGain);
    }
    this._makeupGain.gain.setTargetAtTime(gain, t, LUFS_GAIN_RAMP_TC);
    if ((globalThis as { __LUFS_TRACE__?: boolean }).__LUFS_TRACE__) {
      // Diagnostic trace: window.__LUFS_TRACE__ = true to log one line
      // per tick. Helpful for confirming the matcher snaps to the
      // right gain the instant a denoised slice lands at the playhead.
      // eslint-disable-next-line no-console
      console.log("[LUFS]", {
        positionSec: +this.positionSec.toFixed(2),
        chunk: chunkIdx,
        measured: Number.isFinite(measured) ? +measured.toFixed(2) : null,
        target: +target.toFixed(2),
        peak: +peak.toFixed(4),
        ceiling: +this._peakCeiling.toFixed(3),
        targetGain: +gain.toFixed(3),
        appliedGain: +this._makeupGain.gain.value.toFixed(3),
        disengaged: this._inSilence,
      });
    }
  }

  private _extractRecentWindow(seconds: number): Float32Array | null {
    const mirror = this._mirror;
    if (!mirror) return null;
    const ch = this.channels;
    const totalFrames = (mirror.length / ch) | 0;
    const wantFrames = Math.min(
      Math.round(seconds * SAMPLE_RATE),
      totalFrames,
    );
    if (wantFrames < 1) return null;
    const posFrames = ((this.positionSec * SAMPLE_RATE) | 0) % totalFrames;
    const startFrame =
      (((posFrames - wantFrames) % totalFrames) + totalFrames) % totalFrames;
    const out = new Float32Array(wantFrames * ch);
    if (startFrame + wantFrames <= totalFrames) {
      out.set(
        mirror.subarray(startFrame * ch, (startFrame + wantFrames) * ch),
      );
    } else {
      const tailFrames = totalFrames - startFrame;
      out.set(mirror.subarray(startFrame * ch));
      out.set(mirror.subarray(0, (wantFrames - tailFrames) * ch), tailFrames * ch);
    }
    return out;
  }

  private _writeSPBuffer(
    startFrame: number,
    audioInterleaved: Float32Array,
    add: boolean,
  ): void {
    if (!this._spBuffer) return;
    const ch = this.channels;
    const base = startFrame * ch;
    const n = Math.min(audioInterleaved.length, this._spBuffer.length - base);
    if (n <= 0) return;
    if (add) {
      for (let i = 0; i < n; i++) this._spBuffer[base + i] += audioInterleaved[i];
    } else {
      for (let i = 0; i < n; i++) this._spBuffer[base + i] = audioInterleaved[i];
    }
  }

  private _writeMirror(
    startFrame: number,
    audioInterleaved: Float32Array,
    add: boolean,
  ): void {
    if (!this._mirror) return;
    const ch = this.channels;
    const base = startFrame * ch;
    const n = Math.min(audioInterleaved.length, this._mirror.length - base);
    if (n <= 0) return;
    if (add) {
      for (let i = 0; i < n; i++) this._mirror[base + i] += audioInterleaved[i];
    } else {
      for (let i = 0; i < n; i++) this._mirror[base + i] = audioInterleaved[i];
    }
    // NOTE: do NOT bump swapCount here. swapCount is the *source-buffer
    // epoch* — the slice listener drops any slice whose stamped epoch
    // differs from it, and the encoder bumps it only on a real buffer
    // swap. Incrementing it on every mirror write inflated it between the
    // worklet's ~5 ms position messages (which reset it to the true swap
    // generation), so a slice arriving in that gap — most reliably the
    // seam wrap-decode that fires back-to-back with the primary decode at
    // every loop boundary — was dropped. The server had already advanced
    // its delta baseline, so the dropped slice's region desynced and the
    // error accumulated over loops: "multiple decoded versions stacked."
    // The mirror-changed notification (for waveform recompute) is the
    // listener fire below; it doesn't need a counter.
    for (const fn of this._listeners) fn();
  }

  private _spProcess(e: AudioProcessingEvent): void {
    const output = e.outputBuffer;
    const frames = output.length;
    const ch = this.channels;
    const buf = this._spBuffer;
    if (!buf || this.frameCount === 0 || !this.ctx) {
      for (let c = 0; c < output.numberOfChannels; c++) {
        output.getChannelData(c).fill(0);
      }
      return;
    }
    const nFrames = this.frameCount;
    // Mirror the worklet's loop-seam crossfade so non-secure-context playback
    // (ScriptProcessor fallback) gets the same smooth wrap.
    const seamFadeLen = Math.max(1, Math.floor(this.ctx.sampleRate * 0.05));
    const seam = Math.min(seamFadeLen, Math.floor(nFrames / 4));
    const outChs: Float32Array[] = [];
    for (let c = 0; c < output.numberOfChannels; c++) {
      outChs.push(output.getChannelData(c));
    }
    // Stem overlays — snapshot the active ones so we don't pay an
    // object iterator per frame. Mirrors the worklet's mix logic so
    // the SP fallback behaves identically (same seam fade, same phase
    // lock to the main playhead).
    const overlays: StemOverlayState[] = [];
    {
      const ov = this._stemOverlays.vocals;
      if (ov.interleaved && ov.frameCount > 0) overlays.push(ov);
    }
    {
      const ov = this._stemOverlays.instruments;
      if (ov.interleaved && ov.frameCount > 0) overlays.push(ov);
    }
    const volAlpha = this._spOverlayVolAlpha;
    let pos = this._spPosition;
    for (let i = 0; i < frames; i++) {
      if (!this._loop && pos >= nFrames - 1) {
        for (let c = 0; c < outChs.length; c++) outChs[c][i] = 0;
        if (!this._spEndSignaled) {
          this._spEndSignaled = true;
          for (const fn of this._endOfBufferListeners) fn();
        }
        continue;
      }
      const inSeam = this._loop && seam > 0 && nFrames - pos <= seam;
      const distFromEnd = inSeam ? nFrames - pos : 0;
      const seamT = inSeam ? (seam - distFromEnd) / seam : 0;
      const headPos = inSeam ? seam - distFromEnd : 0;
      if (inSeam) {
        for (let c = 0; c < outChs.length; c++) {
          const cc = Math.min(c, ch - 1);
          const sTail = buf[pos * ch + cc];
          const sHead = buf[headPos * ch + cc];
          outChs[c][i] = sTail * (1 - seamT) + sHead * seamT;
        }
      } else {
        for (let c = 0; c < outChs.length; c++) {
          const cc = Math.min(c, ch - 1);
          outChs[c][i] = buf[pos * ch + cc];
        }
      }
      // Overlay mix — same per-frame IIR smoothing as the worklet, same
      // seam handling, no fading branch (overlays are cleared on swap).
      for (let oi = 0; oi < overlays.length; oi++) {
        const ov = overlays[oi];
        ov.volume += (ov.targetVolume - ov.volume) * volAlpha;
        const ovBuf = ov.interleaved!;
        if (ov.volume < 1e-4) continue;
        if (pos >= ov.frameCount) continue;
        const ovCh = ov.channels;
        const ovVol = ov.volume;
        if (inSeam && headPos < ov.frameCount) {
          for (let c = 0; c < outChs.length; c++) {
            const cc = Math.min(c, ovCh - 1);
            const sTail = ovBuf[pos * ovCh + cc];
            const sHead = ovBuf[headPos * ovCh + cc];
            outChs[c][i] += (sTail * (1 - seamT) + sHead * seamT) * ovVol;
          }
        } else {
          for (let c = 0; c < outChs.length; c++) {
            const cc = Math.min(c, ovCh - 1);
            outChs[c][i] += ovBuf[pos * ovCh + cc] * ovVol;
          }
        }
      }
      pos++;
      if (pos >= nFrames) pos = this._loop ? seam : nFrames;
    }
    this._spPosition = pos;
    this.positionSec = this._spPosition / SAMPLE_RATE;
  }
}
