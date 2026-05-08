import { useNetworkStore } from "@/store/useNetworkStore";
import { useSessionStore } from "@/store/useSessionStore";
import type { RemoteBackend } from "@/engine/protocol";
import type { AudioSlice } from "@/types/protocol";

// Detect "experience is degraded" from existing WebSocket signals,
// without protocol changes. Three inputs, one verdict:
//
//   1. Slice inter-arrival jitter — RemoteBackend already dispatches
//      a CustomEvent("slice") on every audio chunk; we timestamp
//      each one and compute p95/median over a 20-sample ring.
//   2. Server engine stress — each slice carries `tickMs` (engine
//      generation time). High p95 means the pod isn't keeping up.
//   3. Stall watchdog — independent of the listener firing, so we
//      can detect "no slice in N ms" while the connection is silent.
//
// Plus a session-status override: WS error/close forces "unstable".
//
// Runs entirely off the RAF render loop. The slice handler is a few
// microseconds (two ring writes); the evaluator runs on a 500ms
// setInterval. Asymmetric hysteresis (escalate fast, recover slow)
// matches Zoom/Meet UX and prevents flicker at threshold boundaries.

export interface NetworkMonitor {
  stop(): void;
}

const THRESHOLDS = {
  WARMUP_MS: 4000,
  WARMUP_MIN_SAMPLES: 8,
  WINDOW_SIZE: 20,
  EVAL_INTERVAL_MS: 500,

  /** p95 / median of inter-arrival deltas. 1 = steady, 1.8 = noticeable. */
  JITTER_RATIO: 1.8,
  /** Server engine generation time in ms (p95 over the window). */
  TICK_MS_P95: 95,
  /** No slice received in this long → unstable, no matter the jitter. */
  STALL_MS: 1500,

  /** Consecutive bad ticks before showing the indicator (1s @ 500ms). */
  ESCALATE_TICKS: 2,
  /** Consecutive clean ticks before hiding it again (2s @ 500ms). */
  RECOVERY_TICKS: 4,
} as const;

interface RingBuffer {
  buf: Float64Array;
  head: number;
  count: number;
}

function makeRing(size: number): RingBuffer {
  return { buf: new Float64Array(size), head: 0, count: 0 };
}

function pushRing(ring: RingBuffer, value: number): void {
  ring.buf[ring.head] = value;
  ring.head = (ring.head + 1) % ring.buf.length;
  if (ring.count < ring.buf.length) ring.count++;
}

function ringToArray(ring: RingBuffer): number[] {
  const out: number[] = new Array(ring.count);
  // Once full, head points at the oldest slot; before that, samples
  // start at index 0 and run to count-1.
  const start = ring.count < ring.buf.length ? 0 : ring.head;
  for (let i = 0; i < ring.count; i++) {
    out[i] = ring.buf[(start + i) % ring.buf.length];
  }
  return out;
}

function quantile(samples: number[], q: number): number {
  if (samples.length === 0) return 0;
  const sorted = samples.slice().sort((a, b) => a - b);
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.floor(q * (sorted.length - 1))),
  );
  return sorted[idx];
}

export function createNetworkMonitor(remote: RemoteBackend): NetworkMonitor {
  const arrivals = makeRing(THRESHOLDS.WINDOW_SIZE);
  const ticks = makeRing(THRESHOLDS.WINDOW_SIZE);
  let lastSliceAt = 0;
  let readyAt = 0;
  let pendingQuality: "healthy" | "unstable" = "healthy";
  let pendingTicks = 0;

  // If the session is already "ready" when the monitor boots, capture
  // readyAt synchronously. Otherwise the subscribe handler picks it
  // up on the status flip.
  if (useSessionStore.getState().status === "ready") {
    readyAt = performance.now();
  }

  const onSlice = (e: Event) => {
    const detail = (e as CustomEvent<AudioSlice>).detail;
    if (!detail) return;
    const now = performance.now();
    pushRing(arrivals, now);
    pushRing(ticks, detail.tickMs);
    lastSliceAt = now;
  };
  remote.addEventListener("slice", onSlice);

  const unsubSession = useSessionStore.subscribe((state, prev) => {
    if (state.status === "ready" && prev.status !== "ready") {
      readyAt = performance.now();
    }
  });

  const evaluate = () => {
    const now = performance.now();
    const sessionStatus = useSessionStore.getState().status;
    const staleMs = lastSliceAt > 0 ? now - lastSliceAt : 0;

    const arrivalSamples = ringToArray(arrivals);
    const deltas: number[] = [];
    for (let i = 1; i < arrivalSamples.length; i++) {
      deltas.push(arrivalSamples[i] - arrivalSamples[i - 1]);
    }
    const median = quantile(deltas, 0.5) || 1;
    const p95 = quantile(deltas, 0.95);
    const jitterRatio = deltas.length >= 2 ? p95 / median : 1;
    const tickMsP95 = quantile(ringToArray(ticks), 0.95);

    const warmedUp =
      readyAt > 0 &&
      now - readyAt >= THRESHOLDS.WARMUP_MS &&
      deltas.length >= THRESHOLDS.WARMUP_MIN_SAMPLES;

    let candidate: "healthy" | "unstable" = "healthy";
    if (warmedUp) {
      if (
        jitterRatio >= THRESHOLDS.JITTER_RATIO ||
        tickMsP95 >= THRESHOLDS.TICK_MS_P95 ||
        staleMs >= THRESHOLDS.STALL_MS
      ) {
        candidate = "unstable";
      }
    }
    // WS error/close beats every other signal — by then the connection
    // is gone and the warmup gate is the wrong question to ask.
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
      jitterRatio,
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
      unsubSession();
      useNetworkStore.getState().reset();
    },
  };
}
