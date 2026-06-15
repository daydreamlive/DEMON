import { useNetworkStore } from "@/store/useNetworkStore";
import { SAMPLE_RATE } from "@demon/client";
import type { AudioPlayer, AudioSlice, RemoteBackend } from "@demon/client";

// Detect "connection is actually broken" from existing WebSocket signals.
// Two inputs, one verdict:
//
//   Stall watchdog — no AudioSlice received in STALL_MS. The one signal
//   that always corresponds to a real user-audible problem: slices
//   stopped arriving, so the AudioPlayer's buffer drains and playback
//   hitches. If the WebSocket dies for any reason (transient blip the
//   server side recovered from, hard close, network drop), slices
//   stop flowing and this catches it.
//
//   Bleed watchdog — slices ARE arriving, but landing BEHIND the
//   playhead. The client's loop buffer always holds the raw source and
//   denoised slices patch over it just ahead of the playhead, so a
//   slice that lands in already-played audio means the listener heard
//   the raw INPUT there. We measure each slice's landing lead (its
//   start position minus the live playhead, folded modulo track
//   duration so the loop-wrap pre-write reads as a small positive lead,
//   not -duration) and flag a sustained negative worst-lead. The stall
//   watchdog misses this — the stream is flowing, just too late — yet
//   it is exactly what a slow / bandwidth-starved link produces and the
//   most direct signal that the user is hearing raw source instead of
//   the processed output. Same hysteresis as the stall path, so only a
//   sustained problem trips it (a brief transient dip self-heals).
//
// What this monitor used to do but no longer does (2026-05-13):
//
//   - **sessionStatus === "error" | "closed" override.** Fired instantly
//     (no debounce) on every WebSocket close — including transient
//     close codes the connection recovered from at the protocol layer.
//     There is no path that flips the store back to "ready" once
//     frames resume, so once stuck the pill stayed on until the next
//     full session-start `reset()`. Redundant with the stall watchdog
//     for the cases that actually matter (real disconnects stop the
//     slice flow on their own).
//
// What this monitor used to do but no longer does (2026-05-12):
//
//   - **Inter-arrival jitter check**. The AudioPlayer's audio worklet
//     buffers slices and smooths their delivery into the actual audio
//     stream. Variance in slice ARRIVAL timing doesn't translate to
//     audible playback jitter unless the worklet's underrun protection
//     fails — and when that happens, the slice cadence collapses
//     entirely (which the stall watchdog catches). The jitter trigger
//     fired on perfectly fine sessions because the slice cadence is
//     bursty by design.
//
//   - **Tick-budget check** (tickMs p95 / measured slice cadence). This
//     was an apples-to-oranges comparison: `tickMs` is per-generation
//     engine time (per a single denoise step), while `medianInterarrival`
//     is wall-clock between SLICE arrivals — and each slice can carry
//     `numGens > 1` generations. The "buffer is one hitch from draining"
//     framing also assumed a video-decode buffer-ahead model that
//     doesn't match how this engine runs: for realtime audio streaming
//     the engine is *expected* to operate near the realtime boundary,
//     so a high tick-vs-cadence ratio is normal operation, not
//     impending degradation.
//
// Net effect: the pill now only fires when something is verifiably
// broken (stalled slices, dead WS). False positives caused users to
// learn to ignore the indicator, which is worse than the missed-edge
// case it was trying to predict.
//
// Asymmetric hysteresis: ~3s of sustained badness before showing,
// ~8s of clean before hiding. Same bias as before (don't flap).

export interface NetworkMonitor {
  stop(): void;
}

const THRESHOLDS = {
  EVAL_INTERVAL_MS: 500,

  /** No slice received in this long → connection is stalled. Upper
   *  end of the VoIP/WebRTC convention for an audio-stream timeout
   *  (1–2s); we use 3s so brief GC/JIT stalls and engine pauses
   *  don't alarm. */
  STALL_MS: 3000,

  /** Consecutive bad ticks before showing (6s @ 500ms). Above the
   *  Zoom/Meet "unstable" debounce range (~1.5–3s) — false alarms
   *  erode trust faster than missed ones in this app, where the
   *  canvas itself visibly degrades during real trouble. */
  ESCALATE_TICKS: 12,
  /** Consecutive clean ticks before hiding (8s @ 500ms). Asymmetric
   *  ~1.3× the show debounce: easy to dismiss, hard to summon. */
  RECOVERY_TICKS: 16,

  /** A slice landing this many seconds (or more) BEHIND the playhead is
   *  counted as a bleed for the interval. Negative = behind. A small
   *  margin (not exactly 0) so sub-perceptual jitter and measurement
   *  noise around the playhead don't register; well inside the healthy
   *  lead (~+0.2s), so a real underrun crosses it decisively. */
  BLEED_LEAD_S: -0.05,
} as const;

/** Landing lead in seconds: how far ahead of the audible playhead a
 *  slice landed, folded modulo track duration into [-dur/2, dur/2).
 *  Positive = ahead (good); negative = it patched already-played audio
 *  (raw source was heard). The fold makes a loop-wrap pre-write — a
 *  slice at the buffer head while the playhead nears the end — read as
 *  a small positive lead instead of ~-duration. `durationSec <= 0`
 *  (unknown duration) skips the fold. Pure; unit-tested. */
export function landingLeadSeconds(
  startSample: number,
  playheadSec: number,
  durationSec: number,
  sampleRate: number = SAMPLE_RATE,
): number {
  let lead = startSample / sampleRate - playheadSec;
  if (durationSec > 0) {
    lead = ((((lead + durationSec / 2) % durationSec) + durationSec)
      % durationSec) - durationSec / 2;
  }
  return lead;
}

export function createNetworkMonitor(
  remote: RemoteBackend,
  player: AudioPlayer,
): NetworkMonitor {
  let lastSliceAt = 0;
  let pendingQuality: "healthy" | "unstable" = "healthy";
  let pendingTicks = 0;
  // Worst (most negative) landing lead observed since the last evaluate()
  // tick, or null when no slice arrived in the interval. Worst-of-interval
  // (not latest) so a single late slice inside an otherwise healthy 500ms
  // still registers. Reset each tick.
  let worstLeadS: number | null = null;

  const onSlice = (e: Event) => {
    const detail = (e as CustomEvent<AudioSlice>).detail;
    if (!detail) return;
    lastSliceAt = performance.now();
    const lead = landingLeadSeconds(
      detail.startSample, player.positionSec, player.duration,
    );
    if (Number.isFinite(lead)) {
      worstLeadS = worstLeadS === null ? lead : Math.min(worstLeadS, lead);
    }
  };
  remote.addEventListener("slice", onSlice);

  const evaluate = () => {
    const now = performance.now();
    const staleMs = lastSliceAt > 0 ? now - lastSliceAt : 0;

    // Take-and-reset the worst lead for this interval. Bleeding = slices
    // arrived but the worst one landed behind the playhead by more than
    // the margin (heard as raw source).
    const intervalWorstLead = worstLeadS;
    worstLeadS = null;
    const bleeding =
      intervalWorstLead !== null
      && intervalWorstLead < THRESHOLDS.BLEED_LEAD_S;

    // Only meaningful once we've seen at least one slice — pre-first-
    // slice we have no baseline and shouldn't flag a "stall" against zero.
    const haveBaseline = lastSliceAt > 0;
    let candidate: "healthy" | "unstable" = "healthy";
    if (haveBaseline && (staleMs >= THRESHOLDS.STALL_MS || bleeding)) {
      candidate = "unstable";
    }

    const current = useNetworkStore.getState().quality;

    if (candidate === current) {
      pendingQuality = current;
      pendingTicks = 0;
    } else if (candidate !== pendingQuality) {
      pendingQuality = candidate;
      pendingTicks = 1;
    } else {
      pendingTicks++;
    }

    const required =
      pendingQuality === "unstable"
        ? THRESHOLDS.ESCALATE_TICKS
        : THRESHOLDS.RECOVERY_TICKS;
    const shouldFlip =
      pendingQuality !== current && pendingTicks >= required;

    useNetworkStore.getState().update({
      ...(shouldFlip ? { quality: pendingQuality } : {}),
      lastSliceAt,
      staleMs,
      // jitterRatio is kept on the store for back-compat with consumers
      // that read it, but we no longer compute it — the trigger was
      // removed (see header).
      jitterRatio: 1,
    });
    if (shouldFlip) {
      // Diagnostic trip-wire. This surface has been patched twice
      // already (2026-05-12 trigger pruning, 2026-05-13 sessionStatus
      // override removal). If the pill ever flips on a healthy session
      // again, the next regression should be observable, not guessed
      // at — a single line in the console gives us staleMs +
      // lastSliceAt + the direction of the flip.
      console.debug("[networkMonitor]", {
        quality: pendingQuality,
        staleMs: Math.round(staleMs),
        lastSliceAt: Math.round(lastSliceAt),
        worstLeadS: intervalWorstLead === null
          ? null : Number(intervalWorstLead.toFixed(3)),
        bleeding,
      });
      pendingTicks = 0;
    }
  };

  const intervalId = window.setInterval(
    evaluate,
    THRESHOLDS.EVAL_INTERVAL_MS,
  );

  return {
    stop() {
      window.clearInterval(intervalId);
      remote.removeEventListener("slice", onSlice);
      useNetworkStore.getState().reset();
    },
  };
}
