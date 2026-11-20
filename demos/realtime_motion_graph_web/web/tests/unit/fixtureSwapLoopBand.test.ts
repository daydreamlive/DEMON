// Tier A: useFixtureSwap resets the loop region when (and only when) the
// active track actually changes.
//
// Regression ("Edit loop length on upload"): the loop region (loopBand) is
// drawn in frame/second coordinates measured against the CURRENT source. A
// plain swap to a DIFFERENT track left the band untouched, so the worklet
// kept its stale loopBandStart/End and indexed the new (shorter) buffer out
// of bounds — pops / NaN — or looped a musically wrong span on a longer one.
// The fix clears the store band on a non-forced swap, which makes
// WaveformScrubBox's sync effect push clearLoopBand to the worklet and
// sendLoopBand(null,null) to the server. A source-mode hotswap is the SAME
// song (force=true) and must KEEP the loop.
//
// The fix lives inside run() — a closure created in useFixtureSwap's effect.
// We drive the REAL hook by stubbing React's useEffect (run synchronously)
// and useRef, then exercise it through the real zustand stores and a fake
// RemoteBackend that echoes swap_ready. No React renderer / DOM needed.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Collected effect-cleanup callbacks so each mounted hook can be torn down
// (its store subscriptions removed) between tests.
const hookCleanups = vi.hoisted(() => [] as Array<() => void>);

vi.mock("react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react")>();
  return {
    ...actual,
    // Run effects synchronously and capture their cleanup so tests can
    // unmount. The hook uses a single mount-time effect (deps []).
    useEffect: (fn: () => void | (() => void)) => {
      const cleanup = fn();
      if (typeof cleanup === "function") hookCleanups.push(cleanup);
    },
    useRef: <T,>(initial: T) => ({ current: initial }),
  };
});

// Stub the config surface the swap path reads. Disabling the denoise gate
// keeps run() off the rAF tween path; the LoRA-cap helpers are no-ops so we
// don't drag the LoRA store / WS into a loop-band test.
vi.mock("@/lib/config", () => ({
  getConfig: () => ({
    web: {
      restart_song_on_swap: false,
      denoise_session_gate: { enabled: false, glide_ms: 0 },
    },
  }),
  resolveLoraCapForSource: () => null,
  applyLoraCapWithServerSync: () => {},
  defaultSwapSourceMode: () => "full" as const,
}));

// The server-resident swap path never calls loadFixtureAudio; stub it so the
// module's browser-only transitive imports don't load under the node env.
vi.mock("@/engine/audio/loadFixture", () => ({
  loadFixtureAudio: vi.fn(),
}));

import { useFixtureSwap } from "@/hooks/useFixtureSwap";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore, type LoopBand } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Pristine store snapshots for a clean slate per test.
const perfInitial = usePerformanceStore.getState();
const sessionInitial = useSessionStore.getState();
const tracksInitial = useCustomTracksStore.getState();

// Event carrying a `detail` payload (run() reads it as a CustomEvent).
class DetailEvent<T> extends Event {
  detail: T;
  constructor(type: string, detail: T) {
    super(type);
    this.detail = detail;
  }
}

// Minimal RemoteBackend: synchronously echoes swap_ready so the swap
// promise resolves within the same microtask chain run() awaits.
function makeFakeRemote() {
  const target = new EventTarget();
  const swapDetail = {
    interleaved: new Float32Array(8),
    channels: 1,
    bpm: 120,
    key: "C",
  };
  return Object.assign(target, {
    duration: 60,
    sendSwapSourceByName: vi.fn(() => {
      target.dispatchEvent(new DetailEvent("swap_ready", swapDetail));
      return true;
    }),
    sendSwapSource: vi.fn(() => {
      target.dispatchEvent(new DetailEvent("swap_ready", swapDetail));
      return true;
    }),
    sendPrompt: vi.fn(),
    sendLoopBand: vi.fn(),
    sendDisableLora: vi.fn(),
  });
}

function makeFakePlayer() {
  return { swap: vi.fn(), seek: vi.fn() };
}

// Yield enough microtask turns for run()'s post-await continuation to finish.
async function flush() {
  for (let i = 0; i < 25; i++) await Promise.resolve();
}

beforeEach(() => {
  // Fake timers swallow the stem-processing watchdog setTimeout that a
  // source-mode swap arms, so it never leaks past the test.
  vi.useFakeTimers();
  usePerformanceStore.setState(perfInitial, true);
  useSessionStore.setState(sessionInitial, true);
  useCustomTracksStore.setState(tracksInitial, true);
  useCustomTracksStore.setState({ names: [], tracks: new Map() });
});

afterEach(() => {
  while (hookCleanups.length) hookCleanups.pop()?.();
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.clearAllMocks();
});

const BAND: LoopBand = { start: 1.5, end: 3.0 };

describe("useFixtureSwap loop-band reset on track change", () => {
  it("clears loopBand after a non-forced swap to a different track", async () => {
    const remote = makeFakeRemote();
    const player = makeFakePlayer();
    usePerformanceStore.setState({ fixture: "songA", loopBand: BAND });
    useSessionStore.setState({
      status: "ready",
      remote: remote as never,
      player: player as never,
    });

    useFixtureSwap();

    // User picks a different track -> non-forced swap.
    usePerformanceStore.getState().setFixture("songB");
    await flush();

    expect(remote.sendSwapSourceByName).toHaveBeenCalledTimes(1);
    expect(player.swap).toHaveBeenCalledTimes(1);
    expect(usePerformanceStore.getState().loopBand).toBeNull();
  });

  it("keeps loopBand after a forced source-mode (same-song) hotswap", async () => {
    const remote = makeFakeRemote();
    const player = makeFakePlayer();
    useCustomTracksStore.getState().addPersisted("upload1", "full");
    usePerformanceStore.setState({ fixture: "upload1", loopBand: BAND });
    useSessionStore.setState({
      status: "ready",
      remote: remote as never,
      player: player as never,
    });

    useFixtureSwap();

    // Same song, different stem feeding inference -> force=true.
    useCustomTracksStore.getState().setSourceMode("upload1", "vocals");
    await flush();

    expect(remote.sendSwapSourceByName).toHaveBeenCalledTimes(1);
    expect(player.swap).toHaveBeenCalledTimes(1);
    expect(usePerformanceStore.getState().loopBand).toEqual(BAND);
  });
});
