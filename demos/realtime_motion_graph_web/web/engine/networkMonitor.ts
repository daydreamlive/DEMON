import { useNetworkStore } from "@/store/useNetworkStore";
import { useSessionStore } from "@/store/useSessionStore";
import type { RemoteBackend } from "@/engine/protocol";
import type { AudioSlice } from "@/types/protocol";

// Detect "connection is actually broken" from existing WebSocket signals.
// Two inputs, one verdict:
//
//   1. Stall watchdog — no AudioSlice received in STALL_MS. This is the
//      one signal that always corresponds to a real user-audible problem:
//      slices stopped arriving, so the AudioPlayer's buffer drains and
//      playback hitches.
//
//   2. Session status — useSessionStore reports "error" / "closed" when
//      the WebSocket has actually torn down. Beats every other gate.
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
} as const;

export function createNetworkMonitor(remote: RemoteBackend): NetworkMonitor {
  let lastSliceAt = 0;
  let pendingQuality: "healthy" | "unstable" = "healthy";
  let pendingTicks = 0;

  const onSlice = (e: Event) => {
    const detail = (e as CustomEvent<AudioSlice>).detail;
    if (!detail) return;
    lastSliceAt = performance.now();
  };
  remote.addEventListener("slice", onSlice);

  const evaluate = () => {
    const now = performance.now();
    const sessionStatus = useSessionStore.getState().status;
    const staleMs = lastSliceAt > 0 ? now - lastSliceAt : 0;

    // Only meaningful once we've seen at least one slice — pre-first-
    // slice we have no baseline and shouldn't flag a "stall" against zero.
    const haveBaseline = lastSliceAt > 0;
    let candidate: "healthy" | "unstable" = "healthy";
    if (haveBaseline && staleMs >= THRESHOLDS.STALL_MS) {
      candidate = "unstable";
    }
    // WS error/close beats every other signal — connection's gone.
    if (sessionStatus === "error" || sessionStatus === "closed") {
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
    if (shouldFlip) pendingTicks = 0;
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
