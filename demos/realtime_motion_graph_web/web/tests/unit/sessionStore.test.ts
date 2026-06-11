// Tier A: useSessionStore lifecycle transitions and reset semantics.

import { beforeEach, describe, expect, it, vi } from "vitest";

import { useSessionStore } from "@/store/useSessionStore";

// Snapshot of the pristine store so every test starts clean (the store
// is module-level state shared across the file).
const initialState = useSessionStore.getState();

beforeEach(() => {
  useSessionStore.setState(initialState, true);
});

describe("useSessionStore", () => {
  it("starts idle with no session objects", () => {
    const s = useSessionStore.getState();
    expect(s.status).toBe("idle");
    expect(s.message).toBe("");
    expect(s.remote).toBeNull();
    expect(s.player).toBeNull();
    expect(s.monitor).toBeNull();
    expect(s.reconnector).toBeNull();
  });

  it("setStatus carries the message and clears it by default", () => {
    const s = useSessionStore.getState();
    s.setStatus("connecting", "Connecting...");
    expect(useSessionStore.getState().status).toBe("connecting");
    expect(useSessionStore.getState().message).toBe("Connecting...");
    s.setStatus("ready");
    expect(useSessionStore.getState().status).toBe("ready");
    expect(useSessionStore.getState().message).toBe("");
  });

  it("walks the happy-path lifecycle", () => {
    const s = useSessionStore.getState();
    for (const status of [
      "loading-fixture",
      "connecting",
      "ready",
    ] as const) {
      s.setStatus(status);
      expect(useSessionStore.getState().status).toBe(status);
    }
  });

  it("reset stops the monitor and cancels the reconnector", () => {
    const stop = vi.fn();
    const cancel = vi.fn();
    const s = useSessionStore.getState();
    s.setMonitor({ stop } as never);
    s.setReconnector({ cancel } as never);
    s.setStatus("ready", "Playing");
    s.reset();
    expect(stop).toHaveBeenCalledTimes(1);
    expect(cancel).toHaveBeenCalledTimes(1);
    const after = useSessionStore.getState();
    expect(after.status).toBe("idle");
    expect(after.monitor).toBeNull();
    expect(after.reconnector).toBeNull();
    expect(after.remote).toBeNull();
    expect(after.player).toBeNull();
    expect(after.pipelineDepth).toBeNull();
    expect(after.maxPipelineDepth).toBeNull();
    expect(after.lastWsTrace).toBeNull();
    expect(after.lastBackendSessionId).toBeNull();
    expect(after.lastBackendClientId).toBeNull();
  });

  it("reset survives a monitor/reconnector that throws", () => {
    const s = useSessionStore.getState();
    s.setMonitor({
      stop: () => {
        throw new Error("already dead");
      },
    } as never);
    s.setReconnector({
      cancel: () => {
        throw new Error("already dead");
      },
    } as never);
    expect(() => s.reset()).not.toThrow();
    expect(useSessionStore.getState().status).toBe("idle");
  });

  it("checkpointScale deliberately survives reset", () => {
    // The server's checkpoint doesn't change across sessions; the LoRA
    // library filter needs the scale before the next session starts.
    const s = useSessionStore.getState();
    s.setCheckpointScale("2B");
    s.reset();
    expect(useSessionStore.getState().checkpointScale).toBe("2B");
  });

  it("wsUrl survives reset (reset only clears per-session state)", () => {
    const s = useSessionStore.getState();
    s.setWsUrl("ws://pod.example:1318/session");
    s.reset();
    expect(useSessionStore.getState().wsUrl).toBe(
      "ws://pod.example:1318/session",
    );
  });

  it("tracks pipeline depth bounds", () => {
    const s = useSessionStore.getState();
    s.setPipelineDepth(4);
    s.setMaxPipelineDepth(8);
    expect(useSessionStore.getState().pipelineDepth).toBe(4);
    expect(useSessionStore.getState().maxPipelineDepth).toBe(8);
  });
});
