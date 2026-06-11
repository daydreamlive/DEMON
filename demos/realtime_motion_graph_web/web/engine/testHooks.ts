// Browser-side test instrumentation, exposed as `window.__demonTest`.
//
// Sibling of debugReconnect's `window.__demonDebug`: installed
// unconditionally at boot (cheap — no listeners or intervals until a
// test calls startProbe()), used by the Playwright e2e suite to observe
// session state, audio-buffer content and streaming health without
// reaching into React internals.
//
// Everything here reads through the public store / engine surfaces the
// app itself uses, so the hooks double as living documentation of the
// observable session contract.

import { SAMPLE_RATE } from "@demon/client";
import type { AudioSlice } from "@demon/client";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

interface SliceRecord {
  /** performance.now() at dispatch. */
  at: number;
  startSample: number;
  numSamples: number;
  flags: number;
  epoch: number;
  /** Server-reported per-slice timings (wire header). */
  decMs: number;
  tickMs: number;
  /** How far AHEAD of the live playhead the slice landed (seconds).
   *  Steady-state p50 of this is the floor latency before any knob
   *  change can be audible. */
  leadS: number;
  /** True when the slice was dropped by the epoch guard (stale source). */
  stale: boolean;
}

interface PositionSample {
  at: number;
  positionSec: number;
  ctxState: string | null;
}

interface ProbeState {
  startedAt: number;
  slices: SliceRecord[];
  positions: PositionSample[];
  /** Playhead stalls: consecutive 100 ms polls with no position advance
   *  while the session claims "ready" (underrun proxy on the worklet
   *  path, where the playhead only advances when audio renders). */
  stalls: number;
  detach: () => void;
}

function quantile(sorted: number[], q: number): number | null {
  if (sorted.length === 0) return null;
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.round(q * (sorted.length - 1))),
  );
  return sorted[idx];
}

function summarize(values: number[]): {
  n: number;
  p50: number | null;
  p95: number | null;
  max: number | null;
  mean: number | null;
} {
  const sorted = [...values].sort((a, b) => a - b);
  const mean =
    values.length > 0
      ? values.reduce((a, v) => a + v, 0) / values.length
      : null;
  return {
    n: values.length,
    p50: quantile(sorted, 0.5),
    p95: quantile(sorted, 0.95),
    max: sorted.length > 0 ? sorted[sorted.length - 1] : null,
    mean,
  };
}

let probe: ProbeState | null = null;

function startProbe(): boolean {
  const { remote, player } = useSessionStore.getState();
  if (!remote || !player) return false;
  stopProbe();

  const slices: SliceRecord[] = [];
  const positions: PositionSample[] = [];
  let stalls = 0;
  let lastPos = -1;

  const onSlice = (e: Event) => {
    const detail = (e as CustomEvent<AudioSlice>).detail;
    const p = useSessionStore.getState().player;
    const positionSec = p?.positionSec ?? 0;
    slices.push({
      at: performance.now(),
      startSample: detail.startSample,
      numSamples: detail.numSamples,
      flags: detail.flags,
      epoch: detail.epoch,
      decMs: detail.decMs,
      tickMs: detail.tickMs,
      leadS: detail.startSample / SAMPLE_RATE - positionSec,
      stale: p ? detail.epoch !== p.swapCount : false,
    });
  };
  remote.addEventListener("slice", onSlice);

  const interval = window.setInterval(() => {
    const s = useSessionStore.getState();
    const p = s.player;
    if (!p) return;
    const ctxState = p.ctx?.state ?? null;
    positions.push({
      at: performance.now(),
      positionSec: p.positionSec,
      ctxState,
    });
    if (
      s.status === "ready" &&
      ctxState === "running" &&
      lastPos >= 0 &&
      p.positionSec === lastPos
    ) {
      stalls++;
    }
    lastPos = p.positionSec;
  }, 100);

  probe = {
    startedAt: performance.now(),
    slices,
    positions,
    // Live view of the closure counter.
    get stalls() {
      return stalls;
    },
    detach: () => {
      remote.removeEventListener("slice", onSlice);
      window.clearInterval(interval);
    },
  };
  return true;
}

function stopProbe(): void {
  probe?.detach();
  probe = null;
}

function probeStats(): Record<string, unknown> | null {
  if (!probe) return null;
  const sl = probe.slices;
  const wallS = (performance.now() - probe.startedAt) / 1000;
  const gaps: number[] = [];
  for (let i = 1; i < sl.length; i++) gaps.push(sl[i].at - sl[i - 1].at);
  return {
    wall_s: wallS,
    n_slices: sl.length,
    slices_per_sec: wallS > 0 ? sl.length / wallS : null,
    audio_seconds_received:
      sl.reduce((a, s) => a + s.numSamples, 0) / SAMPLE_RATE,
    stale_slices_dropped: sl.filter((s) => s.stale).length,
    lead_s: summarize(sl.map((s) => s.leadS)),
    gap_ms: summarize(gaps),
    dec_ms: summarize(sl.map((s) => s.decMs)),
    tick_ms: summarize(sl.map((s) => s.tickMs)),
    playhead_stalls: probe.stalls,
    n_position_samples: probe.positions.length,
  };
}

/** SHA-256 (hex) of the player mirror's interleaved float32 bytes over
 *  [startFrame, endFrame) — byte-layout-identical to how the golden
 *  runner hashes its canonical region (contiguous f32 LE), so the e2e
 *  test can compare the browser-reconstructed buffer against the
 *  reference bundle's recorded `canonical_sha256` without shipping
 *  megabytes of samples out of the page. */
async function bufferRegionSha256(
  startFrame: number,
  endFrame: number,
): Promise<string | null> {
  const player = useSessionStore.getState().player;
  const mirror = player?.getMirror();
  if (!player || !mirror) return null;
  const ch = player.channels;
  const totalFrames = (mirror.length / ch) | 0;
  if (startFrame < 0 || endFrame > totalFrames || endFrame <= startFrame) {
    return null;
  }
  // Standalone copy: digest() wants a clean ArrayBuffer view, and the
  // mirror's backing store outlives/exceeds the region.
  const region = new Float32Array(endFrame * ch - startFrame * ch);
  region.set(mirror.subarray(startFrame * ch, endFrame * ch));
  const digest = await crypto.subtle.digest("SHA-256", region.buffer);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** RMS of the player's mirror over [startSec, endSec) — the test's
 *  "is there actually audio in the buffer" assertion. */
function bufferRms(startSec = 0, endSec = Number.POSITIVE_INFINITY): number {
  const player = useSessionStore.getState().player;
  const mirror = player?.getMirror();
  if (!player || !mirror) return 0;
  const ch = player.channels;
  const totalFrames = (mirror.length / ch) | 0;
  const a = Math.max(0, Math.min(totalFrames, Math.floor(startSec * SAMPLE_RATE)));
  const b = Math.max(a, Math.min(totalFrames, Math.ceil(endSec * SAMPLE_RATE)));
  if (b <= a) return 0;
  let acc = 0;
  for (let i = a * ch; i < b * ch; i++) acc += mirror[i] * mirror[i];
  return Math.sqrt(acc / ((b - a) * ch));
}

export interface DemonTestHooks {
  getStatus: () => string;
  getMessage: () => string;
  getPositionSec: () => number | null;
  getDurationSec: () => number | null;
  getSwapCount: () => number | null;
  getCtxState: () => string | null;
  getWsTrace: () => unknown;
  bufferRms: (startSec?: number, endSec?: number) => number;
  bufferRegionSha256: (
    startFrame: number,
    endFrame: number,
  ) => Promise<string | null>;
  setSlider: (param: string, value: number) => void;
  setFixture: (name: string) => void;
  getFixture: () => string;
  startProbe: () => boolean;
  stopProbe: () => void;
  probeStats: () => Record<string, unknown> | null;
}

declare global {
  interface Window {
    __demonTest?: DemonTestHooks;
  }
}

export function installTestHooks(): void {
  if (typeof window === "undefined") return;
  window.__demonTest = {
    getStatus: () => useSessionStore.getState().status,
    getMessage: () => useSessionStore.getState().message,
    getPositionSec: () =>
      useSessionStore.getState().player?.positionSec ?? null,
    getDurationSec: () => useSessionStore.getState().player?.duration ?? null,
    getSwapCount: () => useSessionStore.getState().player?.swapCount ?? null,
    getCtxState: () =>
      useSessionStore.getState().player?.ctx?.state ?? null,
    getWsTrace: () =>
      useSessionStore.getState().remote?.getWsTrace() ?? null,
    bufferRms,
    bufferRegionSha256,
    // Knob path: same store action the UI ribbons use; useParamSync's
    // 8 ms tick ships the smoothed value to the engine.
    setSlider: (param, value) =>
      usePerformanceStore.getState().setSlider(param, value),
    // Fixture path: same store write the TrackPicker makes; the
    // useFixtureSwap subscription turns it into a swap_source.
    setFixture: (name) => usePerformanceStore.getState().setFixture(name),
    getFixture: () => usePerformanceStore.getState().fixture,
    startProbe,
    stopProbe,
    probeStats,
  };
}
