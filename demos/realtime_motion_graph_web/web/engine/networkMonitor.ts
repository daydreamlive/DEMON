import { useNetworkStore } from "@/store/useNetworkStore";
import { useSessionStore } from "@/store/useSessionStore";
import type { RemoteBackend } from "@/engine/protocol";
import type { AudioSlice } from "@/types/protocol";

// Detect "experience is degraded" from existing WebSocket signals,
// without protocol changes. Three inputs, one verdict:
//
//   1. Slice inter-arrival jitter — RemoteBackend already dispatches a
//      CustomEvent("slice") on every audio chunk; we timestamp arrivals
//      into a 20-sample ring and require BOTH a high p95/median ratio
//      AND an absolute p95 above the audible threshold (~50ms per the
//      VoIP/WebRTC consensus — Cisco, Obkio, MOS literature). Two-gate
//      trigger so a 2.5× ratio of an 8ms cadence (= 20ms p95) doesn't
//      fire — that's not audible jitter, it's just scheduling noise.
//
//   2. Server engine stress — each slice carries `tickMs` (engine
//      generation time). Compared against the *measured* slice cadence
//      (median inter-arrival delta) as a budget ratio: if p95 tickMs is
//      consuming ≥85% of the slot, the buffer is one hitch from
//      draining. Self-calibrating to whatever cadence the engine
//      actually runs at, instead of a hardcoded ms threshold that
//      would over-fire on slow cadences and under-fire on fast ones.
//
//   3. Stall watchdog — independent of the listener firing, so we
//      can detect "no slice in N ms" while the connection is silent.
//      1500ms is squarely in the WebRTC/VoIP convention for an
//      audio-stream timeout (NetEQ concealment runs sub-100ms; the
//      user-facing "stalled" indicator fires at 1–2s).
//
// Plus a session-status override: WS error/close forces "unstable".
//
// Asymmetric hysteresis: ~3s of sustained bad signal before showing
// (don't cry wolf on a single GC pause or Wi-Fi roam), ~8s of clean
// signal before hiding (don't flap). Bias is intentional — false
// positives teach users to ignore the indicator, which is worse than
// missing a real degradation.
//
// Runs entirely off the RAF render loop. The slice handler is a few
// microseconds (two ring writes); the evaluator runs on a 500ms
// setInterval.

export interface NetworkMonitor {
  stop(): void;
}

const THRESHOLDS = {
  WARMUP_MS: 4000,
  WARMUP_MIN_SAMPLES: 8,
  WINDOW_SIZE: 20,
  EVAL_INTERVAL_MS: 500,

  /** p95 / median of inter-arrival deltas. Healthy networks routinely
   *  hit 1.5–2.0× from kernel scheduling alone; 2.5× is where bursts
   *  start to dominate. Required AND with JITTER_ABS_MS below. */
  JITTER_RATIO: 2.5,
  /** Absolute p95 inter-arrival in ms. Audio jitter becomes audibly
   *  perceptible around 30–50ms in the VoIP/WebRTC consensus; use 50
   *  as the conservative trigger so cadences with naturally tight
   *  intervals don't false-positive on a high ratio. */
  JITTER_ABS_MS: 50,
  /** Server tickMs p95 as a fraction of the *measured* slice cadence.
   *  >0.92 = ≤8% headroom = the buffer is genuinely the only thing
   *  saving you from underrun. Self-calibrates: at a 100ms cadence this
   *  fires at 92ms; at a 24ms cadence at ~22ms. Bumped from 0.85 in
   *  2026-05 after the pill was firing too often on otherwise-fine
   *  sessions — 15% headroom is normal scheduler noise. */
  TICK_BUDGET_RATIO: 0.92,
  /** No slice received in this long → unstable, no matter the jitter.
   *  Upper end of the VoIP/WebRTC convention for an audio-stream
   *  timeout (1–2s). Bumped from 1500ms in 2026-05 to absorb brief
   *  GC/JIT stalls without alarming. */
  STALL_MS: 2200,

  /** Consecutive bad ticks before showing (6s @ 500ms). Above the
   *  Zoom/Meet "unstable" debounce range (~1.5–3s) — we lean cautious
   *  because false alarms erode trust faster than missed ones in this
   *  app, where the canvas itself visibly degrades during real
   *  trouble. Bumped from 6 (3s) in 2026-05. */
  ESCALATE_TICKS: 12,
  /** Consecutive clean ticks before hiding (8s @ 500ms). Asymmetric
   *  ~2.7× the show debounce: easy to dismiss, hard to summon. */
  RECOVERY_TICKS: 16,
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
    const medianInterarrival = quantile(deltas, 0.5);
    const p95Interarrival = quantile(deltas, 0.95);
    const jitterRatio =
      deltas.length >= 2 && medianInterarrival > 0
        ? p95Interarrival / medianInterarrival
        : 1;
    const tickMsP95 = quantile(ringToArray(ticks), 0.95);
    const tickBudgetRatio =
      medianInterarrival > 0 ? tickMsP95 / medianInterarrival : 0;

    const warmedUp =
      readyAt > 0 &&
      now - readyAt >= THRESHOLDS.WARMUP_MS &&
      deltas.length >= THRESHOLDS.WARMUP_MIN_SAMPLES;

    let candidate: "healthy" | "unstable" = "healthy";
    if (warmedUp) {
      const jitterTriggered =
        jitterRatio >= THRESHOLDS.JITTER_RATIO &&
        p95Interarrival >= THRESHOLDS.JITTER_ABS_MS;
      const tickTriggered =
        tickBudgetRatio >= THRESHOLDS.TICK_BUDGET_RATIO;
      const stallTriggered = staleMs >= THRESHOLDS.STALL_MS;
      if (jitterTriggered || tickTriggered || stallTriggered) {
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
