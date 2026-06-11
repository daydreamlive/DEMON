// Tier A: WsReconnector backoff / cancel / give-up semantics.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { WsReconnector, type ReconnectAttempt } from "@demon/client";

describe("WsReconnector", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    // Pin jitter to its upper bound (factor 1.0) so delays are exact.
    vi.spyOn(Math, "random").mockReturnValue(1);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("resolves on first success and fires onSuccess once", async () => {
    const connect = vi.fn().mockResolvedValue(undefined);
    const onSuccess = vi.fn();
    const onGiveUp = vi.fn();
    const r = new WsReconnector(connect, { onSuccess, onGiveUp });
    const done = r.run();
    await vi.advanceTimersByTimeAsync(500);
    await done;
    expect(connect).toHaveBeenCalledTimes(1);
    expect(onSuccess).toHaveBeenCalledTimes(1);
    expect(onGiveUp).not.toHaveBeenCalled();
  });

  it("doubles the backoff and clamps at maxDelayMs", async () => {
    const connect = vi.fn().mockRejectedValue(new Error("nope"));
    const delays: number[] = [];
    const onGiveUp = vi.fn();
    const r = new WsReconnector(connect, {
      onAttempt: (a: ReconnectAttempt) => delays.push(a.delayMs),
      onGiveUp,
    });
    const done = r.run();
    await vi.advanceTimersByTimeAsync(60_000);
    await done;
    // base 500ms doubling, clamped to 4000: 500, 1000, 2000, 4000, 4000
    expect(delays).toEqual([500, 1000, 2000, 4000, 4000]);
    expect(connect).toHaveBeenCalledTimes(5);
    expect(onGiveUp).toHaveBeenCalledTimes(1);
    expect(onGiveUp.mock.calls[0][0]).toBeInstanceOf(Error);
    expect((onGiveUp.mock.calls[0][0] as Error).message).toBe("nope");
  });

  it("jitter stays within [base/2, base]", async () => {
    (Math.random as ReturnType<typeof vi.fn>).mockReturnValue(0);
    const connect = vi.fn().mockResolvedValue(undefined);
    const delays: number[] = [];
    const r = new WsReconnector(connect, {
      onAttempt: (a) => delays.push(a.delayMs),
    });
    const done = r.run();
    await vi.advanceTimersByTimeAsync(500);
    await done;
    expect(delays).toEqual([250]); // random=0 -> lower bound base/2
  });

  it("recovers after transient failures", async () => {
    const connect = vi
      .fn()
      .mockRejectedValueOnce(new Error("blip 1"))
      .mockRejectedValueOnce(new Error("blip 2"))
      .mockResolvedValueOnce(undefined);
    const onSuccess = vi.fn();
    const onGiveUp = vi.fn();
    const attempts: number[] = [];
    const r = new WsReconnector(connect, {
      onAttempt: (a) => attempts.push(a.attempt),
      onSuccess,
      onGiveUp,
    });
    const done = r.run();
    await vi.advanceTimersByTimeAsync(10_000);
    await done;
    expect(attempts).toEqual([1, 2, 3]);
    expect(onSuccess).toHaveBeenCalledTimes(1);
    expect(onGiveUp).not.toHaveBeenCalled();
  });

  it("cancel during the backoff sleep stops the loop silently", async () => {
    const connect = vi.fn().mockResolvedValue(undefined);
    const onSuccess = vi.fn();
    const onGiveUp = vi.fn();
    const r = new WsReconnector(connect, { onSuccess, onGiveUp });
    const done = r.run();
    // Cancel mid-sleep: the sleep resolves immediately and the loop
    // exits before invoking connect.
    await vi.advanceTimersByTimeAsync(100);
    r.cancel();
    await done;
    expect(connect).not.toHaveBeenCalled();
    expect(onSuccess).not.toHaveBeenCalled();
    expect(onGiveUp).not.toHaveBeenCalled();
  });

  it("cancel during an in-flight connect suppresses onSuccess", async () => {
    let resolveConnect: () => void = () => {};
    const connect = vi.fn(
      () =>
        new Promise<void>((res) => {
          resolveConnect = res;
        }),
    );
    const onSuccess = vi.fn();
    const r = new WsReconnector(connect, { onSuccess });
    const done = r.run();
    await vi.advanceTimersByTimeAsync(500); // sleep elapses, connect starts
    expect(connect).toHaveBeenCalledTimes(1);
    r.cancel();
    resolveConnect();
    await done;
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it("cancel is idempotent and safe before run", () => {
    const r = new WsReconnector(vi.fn());
    r.cancel();
    r.cancel();
  });
});
