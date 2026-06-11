// AudioWorklet that plays a looping PCM buffer the main thread can
// patch in place. Mirrors demos/realtime_motion_graph_web/audio_engine.py
// (swap / patch / crossfade behaviour).
//
// Messages from main thread:
//   {type:'init',     buffer:Float32Array (interleaved), channels:int}
//   {type:'patch',    start:int (frame),  audio:Float32Array (interleaved)}
//   {type:'add',      start:int (frame),  audio:Float32Array (interleaved)}  // delta add
//   {type:'swap',     buffer:Float32Array, channels:int}
//   {type:'setLoop',  enabled:bool}                            // loop at end-of-buffer
//   {type:'seek',     positionFrames:int}                      // jump playhead
//   {type:'setLoopBand', startFrames:int, endFrames:int}       // wrap at end→start
//   {type:'clearLoopBand'}                                     // disable band loop
//   {type:'setOverlayBuffer',   kind:string, buffer:Float32Array, channels:int}
//   {type:'clearOverlayBuffer', kind:string}
//   {type:'setOverlayVolume',   kind:string, volume:float}
//
// Stem overlays (vocals / instruments) live in the worklet so they share
// the main buffer's playhead and seam-crossfade loop point — playing them
// as standalone AudioBufferSourceNodes on the main thread drifted because
// their hard-loop didn't match the seam fade and the two clocks ran free.
//
// Messages to main thread:
//   {type:'position',    positionSec:float, swapCount:int, kick:float}
//   {type:'endOfBuffer'} // one-shot when loop=false and playhead reaches end

const CROSSFADE_SECONDS = 0.025;
const SEAM_FADE_SECONDS = 0.05;  // loop-seam crossfade at end-of-buffer
// ~5.3 ms at 48 kHz. Sets the cadence of {position, kick} postMessages
// to the main thread. Two consumers: (1) visual kick reactivity in the
// HUD/graph (190 Hz here is overkill for the eye but cheap), (2) the
// param-sync loop reads `player.positionSec` and ships it to the server
// as `playback_pos`, which the runner uses to set decode_start. Stale
// position → leading edge of the new slice lands behind the listener
// at write-time → wasted bytes + missed leading-edge xfade. 256 keeps
// staleness ≤ ~5 ms; the alternative (1024 / ~21 ms) saves a bit of
// main-thread postMessage overhead but costs decode-position accuracy.
// The actual perf win — moving kick RMS into the worklet — is
// independent of this cadence.
const REPORT_EVERY = 256;
// Kick / RMS window. 480 frames at 48 kHz is ~10 ms — long enough to
// average a kick transient, short enough to track sub-beat dynamics.
// Sliding sum-of-squares is maintained incrementally; periodic refresh
// (every REFRESH frames) re-computes from the buffer to bound floating-
// point drift. KICK_SOFT_GAIN matches the main-thread kickRms() the
// worklet replaces (was: `rms * 1.6`, clamped to [0,1]).
const KICK_WINDOW = 480;
const KICK_REFRESH = 48000;  // ~1 s — drift is microscopic in float32 anyway
const KICK_SOFT_GAIN = 1.6;

class RealtimeBufferProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.current = null;          // Float32Array, interleaved
    this.channels = 2;
    this.frameCount = 0;          // frames in current buffer
    this.position = 0;            // playhead in frames
    this.swapCount = 0;

    this.oldBuffer = null;
    this.oldFrameCount = 0;
    this.fading = false;
    this.fadePos = 0;
    this.crossfadeLen = Math.max(1, Math.floor(sampleRate * CROSSFADE_SECONDS));
    this.seamFadeLen = Math.max(1, Math.floor(sampleRate * SEAM_FADE_SECONDS));

    // Loop / end-of-buffer behaviour. When loop=true (default) the
    // playhead wraps via the seam crossfade as before. When false,
    // the playhead clamps at end-of-buffer, the processor outputs
    // silence, and a one-shot {type:'endOfBuffer'} fires so the main
    // thread can flip its paused flag.
    this.loop = true;
    this._endSignaled = false;

    // Band loop: when both >= 0 and end > start, the wrap path below
    // sends the playhead from loopBandEnd → loopBandStart on each pass
    // instead of letting it run to frameCount. -1 = no band.
    this.loopBandStart = -1;
    this.loopBandEnd = -1;

    this._framesSinceReport = 0;

    // Stem overlays. Each entry holds the interleaved PCM, channel count,
    // a target volume from the UI, and a per-frame-smoothed live volume
    // so toggles/sliders don't zipper. Mixed in process() using the same
    // playhead the main buffer reads from, so the seam crossfade keeps
    // them phase-locked to the main loop instead of free-running on the
    // AudioContext clock. _overlayList is the hot-loop iterable, kept in
    // sync with the keyed object by every setter (avoids `for...in` in
    // the per-frame inner loop).
    this.overlays = Object.create(null);
    this._overlayList = [];
    // 25 ms single-pole IIR for volume smoothing. ~75 ms to 95% of a
    // step at 48 kHz — fast enough to feel instant, slow enough to kill
    // zipper noise on slider drags.
    this._volRampAlpha = 1 - Math.exp(-1 / (sampleRate * 0.025));

    // Kick state: ring of squared frame energies, running sum.
    this._kickRing = new Float32Array(KICK_WINDOW);
    this._kickHead = 0;
    this._kickFilled = 0;
    this._kickSumSq = 0;
    this._lastKick = 0;
    this._kickFramesSinceRefresh = 0;

    this.port.onmessage = (e) => this._onmessage(e.data);
  }

  _onmessage(msg) {
    const { type } = msg;
    if (type === "init") {
      this.current = msg.buffer;
      this.channels = msg.channels || 2;
      this.frameCount = this.current.length / this.channels;
      this.position = 0;
      this.swapCount = 0;
      return;
    }
    if (type === "swap") {
      this.oldBuffer = this.current;
      this.oldFrameCount = this.frameCount;
      this.current = msg.buffer;
      this.channels = msg.channels || this.channels;
      this.frameCount = this.current.length / this.channels;
      this.swapCount++;
      this.fading = true;
      this.fadePos = 0;
      return;
    }
    if (type === "setLoop") {
      this.loop = !!msg.enabled;
      // Re-arm the end-of-buffer one-shot so toggling loop off, hitting
      // end, then toggling on/off again still posts on each transition.
      this._endSignaled = false;
      return;
    }
    if (type === "seek") {
      const target = (msg.positionFrames | 0);
      if (this.frameCount > 0) {
        // Clamp to [0, frameCount-1]; the advance path will wrap or
        // freeze based on `loop` on the next process() block.
        this.position = Math.max(0, Math.min(this.frameCount - 1, target));
      }
      this._endSignaled = false;
      return;
    }
    if (type === "setLoopBand") {
      // Band defined in frames. Refuse if degenerate (≤0 frames between
      // the two markers) so the wrap path doesn't spin in place.
      const s = Math.max(0, msg.startFrames | 0);
      const e = Math.min(this.frameCount, msg.endFrames | 0);
      if (e - s < 1) {
        this.loopBandStart = -1;
        this.loopBandEnd = -1;
        return;
      }
      this.loopBandStart = s;
      this.loopBandEnd = e;
      // If the playhead is outside the new band, snap it to the band
      // start so the next process() block plays from where the user
      // pointed at. Without this, position can sit past loopBandEnd
      // and the wrap below fires every block — fine, but the operator
      // wouldn't hear the band start until naturally cycling past
      // frameCount, which is confusing.
      if (this.position < s || this.position >= e) {
        this.position = s;
      }
      this._endSignaled = false;
      return;
    }
    if (type === "clearLoopBand") {
      this.loopBandStart = -1;
      this.loopBandEnd = -1;
      return;
    }
    if (type === "setOverlayBuffer") {
      const kind = String(msg.kind || "");
      if (!kind) return;
      const channels = msg.channels || 2;
      const buffer = msg.buffer;
      const frameCount = buffer ? (buffer.length / channels) | 0 : 0;
      const prev = this.overlays[kind];
      this.overlays[kind] = {
        buffer,
        channels,
        frameCount,
        targetVolume: prev ? prev.targetVolume : 0,
        // Start at the previous live volume so a buffer swap (e.g. song
        // swap) doesn't pop — the smoothing then resolves to whatever
        // targetVolume the UI has set.
        volume: prev ? prev.volume : 0,
      };
      this._refreshOverlayList();
      return;
    }
    if (type === "clearOverlayBuffer") {
      const kind = String(msg.kind || "");
      if (!kind) return;
      delete this.overlays[kind];
      this._refreshOverlayList();
      return;
    }
    if (type === "setOverlayVolume") {
      const kind = String(msg.kind || "");
      if (!kind) return;
      const v = Math.max(0, Math.min(6.0, Number(msg.volume) || 0));
      if (!this.overlays[kind]) {
        // Volume can land before the buffer (UI default volumes are set
        // up-front). Stash it so the buffer-arrival handler picks it up.
        this.overlays[kind] = {
          buffer: null,
          channels: 2,
          frameCount: 0,
          targetVolume: v,
          volume: 0,
        };
        this._refreshOverlayList();
        return;
      }
      this.overlays[kind].targetVolume = v;
      return;
    }
    if (type === "patch" || type === "add") {
      if (!this.current) return;
      const start = msg.start | 0;
      const slice = msg.audio;            // Float32Array, interleaved
      const ch = this.channels;
      const sliceFrames = (slice.length / ch) | 0;
      if (sliceFrames <= 0) return;

      const end = Math.min(start + sliceFrames, this.frameCount);
      const actual = end - start;
      if (actual <= 0) return;

      const base = start * ch;
      const count = actual * ch;
      if (type === "add") {
        // Delta add: matches ``current[s:e] += data`` in server/client path.
        for (let i = 0; i < count; i++) this.current[base + i] += slice[i];
      } else {
        // Raw patch: matches AudioEngine.patch.  Server only sends raw
        // patches for the initial buffer, so seam xfades are unnecessary.
        for (let i = 0; i < count; i++) this.current[base + i] = slice[i];
      }
      return;
    }
  }

  _refreshOverlayList() {
    const list = [];
    for (const k in this.overlays) list.push(this.overlays[k]);
    this._overlayList = list;
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    const frames = output[0].length;
    const outChannels = output.length;

    if (!this.current || this.frameCount === 0) {
      for (let c = 0; c < outChannels; c++) output[c].fill(0);
      return true;
    }

    const ch = this.channels;
    const nCur = this.frameCount;
    // Active loop region: a band when one is armed, else the whole
    // buffer. The seam crossfade and the wrap target both key off this,
    // so a band loop now gets the SAME click-free seam the end-of-buffer
    // wrap always had — the band wrap used to be a hard cut and clicked
    // on every lap.
    const bandActive =
      this.loopBandStart >= 0 && this.loopBandEnd > this.loopBandStart;
    const regionStart = bandActive ? this.loopBandStart : 0;
    const regionEnd = bandActive ? this.loopBandEnd : nCur;
    const regionLen = regionEnd - regionStart;
    // A band always loops; the full buffer loops only when `loop` is on.
    const wrapping = bandActive || this.loop;
    // Loop-seam crossfade: blend the last `seam` frames before regionEnd
    // with the region head, then wrap the playhead to regionStart + seam
    // so the head we just mixed isn't replayed. Hides the wrap click and
    // softens a source that doesn't loop musically. Shrinks for short
    // regions so even a tiny band still fades cleanly.
    const seam = Math.min(this.seamFadeLen, Math.floor(regionLen / 4));

    for (let i = 0; i < frames; i++) {
      const pos = this.position;

      // Loop-off freeze: position has been clamped at end-of-buffer,
      // emit silence and stop advancing. Fire endOfBuffer once so the
      // main thread can flip the paused flag and suspend the context.
      // Keyed off `wrapping`, not `this.loop`, so a band loop whose end
      // sits exactly at the buffer end still wraps instead of freezing
      // (the band always loops, even with the full-buffer loop off).
      if (!wrapping && pos >= nCur - 1) {
        for (let c = 0; c < outChannels; c++) output[c][i] = 0;
        if (!this._endSignaled) {
          this.port.postMessage({ type: "endOfBuffer" });
          this._endSignaled = true;
        }
        // Skip kick + advance below — the playhead is parked.
        continue;
      }

      // Per-frame seam state, shared by the main read and the overlay
      // mix below so they wrap phase-locked. Active in the last `seam`
      // frames before regionEnd: blend the tail at `pos` toward the
      // region head at `headPos` (regionStart-relative).
      let seamActive = false;
      let seamT = 0;
      let headPos = 0;
      {
        const distFromEnd = regionEnd - pos;
        if (wrapping && seam > 0 && distFromEnd > 0 && distFromEnd <= seam) {
          seamActive = true;
          seamT = (seam - distFromEnd) / seam;
          headPos = regionStart + (seam - distFromEnd);
        }
      }

      if (this.fading && this.oldBuffer) {
        const fadeT = Math.min(1, this.fadePos / this.crossfadeLen);
        const oldPos = this.position % this.oldFrameCount;
        for (let c = 0; c < outChannels; c++) {
          const cc = Math.min(c, ch - 1);
          const sNew = this.current[pos * ch + cc];
          const sOld = this.oldBuffer[oldPos * ch + cc];
          output[c][i] = sOld * (1 - fadeT) + sNew * fadeT;
        }
        this.fadePos++;
        if (this.fadePos >= this.crossfadeLen) {
          this.fading = false;
          this.oldBuffer = null;
        }
      } else if (seamActive) {
        for (let c = 0; c < outChannels; c++) {
          const cc = Math.min(c, ch - 1);
          const sTail = this.current[pos * ch + cc];
          const sHead = this.current[headPos * ch + cc];
          output[c][i] = sTail * (1 - seamT) + sHead * seamT;
        }
      } else {
        for (let c = 0; c < outChannels; c++) {
          const cc = Math.min(c, ch - 1);
          output[c][i] = this.current[pos * ch + cc];
        }
      }

      // Stem overlay mix. Reads each overlay at the SAME `pos` the main
      // buffer just read from, applying the same seam crossfade when the
      // playhead is in the loop-wrap zone — that's what keeps overlays
      // phase-locked instead of free-running on a separate clock.
      // Volume is smoothed per frame with a single-pole IIR (see
      // _volRampAlpha) so UI slider drags don't zipper. The fading
      // (old→new main buffer) branch above has no overlay analogue
      // because AudioPlayer.swap clears overlays before the new main
      // buffer arrives — the brief overlap (~25 ms) plays no overlay.
      const overlays = this._overlayList;
      const volAlpha = this._volRampAlpha;
      for (let oi = 0; oi < overlays.length; oi++) {
        const ov = overlays[oi];
        ov.volume += (ov.targetVolume - ov.volume) * volAlpha;
        const buf = ov.buffer;
        if (!buf || ov.volume < 1e-4) continue;
        if (pos >= ov.frameCount) continue;
        const ovCh = ov.channels;
        const ovVol = ov.volume;
        if (seamActive && headPos < ov.frameCount) {
          for (let c = 0; c < outChannels; c++) {
            const cc = Math.min(c, ovCh - 1);
            const sTail = buf[pos * ovCh + cc];
            const sHead = buf[headPos * ovCh + cc];
            output[c][i] += (sTail * (1 - seamT) + sHead * seamT) * ovVol;
          }
        } else {
          for (let c = 0; c < outChannels; c++) {
            const cc = Math.min(c, ovCh - 1);
            output[c][i] += buf[pos * ovCh + cc] * ovVol;
          }
        }
      }

      // Kick / RMS — sliding sum of squared frame energies. Frame energy
      // is the mean of the output channels at this frame; squaring gives
      // us the input to RMS. We update the ring incrementally (subtract
      // oldest, add newest) so per-frame cost is O(1). Done here on the
      // audio thread so the main render loop never has to read the audio
      // buffer to derive a beat signal — a per-frame mirror walk used to
      // cost a budget slice on every render frame.
      let frameEnergy = 0;
      for (let c = 0; c < outChannels; c++) frameEnergy += output[c][i];
      frameEnergy /= outChannels;
      // Reject non-finite frames at the source. A single NaN sample
      // (uninitialized buffer fragment, mid-swap zero-frame, etc.)
      // poisons _kickSumSq permanently because the incremental update
      // below propagates NaN forever — the consumer (main-thread
      // useRenderLoop → GraphRenderer.draw → addColorStop) then crashes
      // every frame with `rgba(...,NaN)`. Treating non-finite as silent
      // (sq=0) keeps the ring valid and the visuals quiet for that
      // sample, which is the right behaviour for a glitched frame.
      const sq = Number.isFinite(frameEnergy) ? frameEnergy * frameEnergy : 0;
      const oldSq = this._kickRing[this._kickHead];
      this._kickSumSq += sq - oldSq;
      this._kickRing[this._kickHead] = sq;
      this._kickHead = (this._kickHead + 1) % KICK_WINDOW;
      if (this._kickFilled < KICK_WINDOW) this._kickFilled++;

      this.position++;
      // Wrap at regionEnd. A band loop takes precedence over the full
      // buffer and always wraps; the full buffer wraps only when `loop`
      // is on, else it clamps at end so the freeze branch on the next
      // iteration takes over. When wrapping we land at regionStart + seam
      // so the head frames already blended by the crossfade above aren't
      // replayed — same click-free behaviour for band and full buffer.
      if (this.position >= regionEnd) {
        this.position = wrapping ? regionStart + seam : nCur;
      }
    }

    // Periodic refresh of _kickSumSq from the ring to bound float drift.
    // At float32 precision over ~1 s of accumulation drift is well below
    // perceptual threshold, but the recompute is cheap enough.
    this._kickFramesSinceRefresh += frames;
    if (this._kickFramesSinceRefresh >= KICK_REFRESH) {
      this._kickFramesSinceRefresh = 0;
      let s = 0;
      for (let i = 0; i < this._kickFilled; i++) s += this._kickRing[i];
      this._kickSumSq = s;
    }

    // Compute the kick value once per process() block, not per frame.
    // The sliding window already smooths the signal; main-thread visuals
    // poll this at rAF cadence. RMS, soft-clip, clamp.
    const denom = this._kickFilled || 1;
    const rms = Math.sqrt(Math.max(0, this._kickSumSq) / denom);
    const scaled = rms * KICK_SOFT_GAIN;
    // Belt-and-suspenders: even with the per-sample NaN guard above,
    // recompute every frame so a corrupted ring (e.g. from a hostile
    // postMessage) can't keep posting NaN to the main thread. The
    // standard Math.max/Math.min clamp does NOT fix NaN — both calls
    // pass it through untouched.
    this._lastKick = Number.isFinite(scaled)
      ? Math.max(0, Math.min(1, scaled))
      : 0;

    this._framesSinceReport += frames;
    if (this._framesSinceReport >= REPORT_EVERY) {
      this._framesSinceReport = 0;
      this.port.postMessage({
        type: "position",
        positionSec: this.position / sampleRate,
        swapCount: this.swapCount,
        kick: this._lastKick,
      });
    }

    return true;
  }
}

registerProcessor("realtime-buffer", RealtimeBufferProcessor);
