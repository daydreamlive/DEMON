// Regression guard for: "Audio source mode reconnect keeps playing the
// stale track."
//
// Repro from the tracker: in audio-source mode with a track loaded, the
// socket drops; the user quickly switches to a different source while the
// "Reconnecting…" placard is up; when the session recovers it keeps
// playing the PREVIOUS track even though the UI now shows the new one.
//
// Root cause: the reconnect path (useStartSession) rebinds the fixture
// snapshotted at session start. The mid-outage track change is dropped
// because useFixtureSwap.run() bails while `status !== "ready"`, and
// nothing re-applies it once the session recovers. So the recovered
// backend + AudioPlayer stay bound to the stale track (session store's
// `boundFixture`) while the perf store's `fixture` shows the new pick.
//
// The fix adds a reconnect reconcile: on the "reconnecting" → "ready"
// edge, if the live `boundFixture` diverges from the selected `fixture`,
// re-run the swap exactly once. `needsFixtureReconcile` is that decision;
// these tests pin its matrix so the heal fires for the bug scenario and
// for nothing else (no double-swap on fresh Play or a clean reconnect).

import { describe, expect, it } from "vitest";

import { needsFixtureReconcile } from "@/hooks/useFixtureSwap";
import type { SessionStatus } from "@/store/useSessionStore";

const A = "inside_confusion_loop_60s_gsm.wav"; // track loaded at start
const B = "low_fi_Gm_loop_60s_gnm.wav"; // track picked during the outage

describe("needsFixtureReconcile", () => {
  it("heals the reported bug: track switched during the outage", () => {
    // Recovered session is bound to A (the snapshot); the UI selection is
    // B (chosen while the socket was down). The reconnect completed
    // (reconnecting → ready) → reconcile must fire.
    expect(
      needsFixtureReconcile({
        prevStatus: "reconnecting",
        status: "ready",
        selectedFixture: B,
        boundFixture: A,
      }),
    ).toBe(true);
  });

  it("does NOT fire on a clean reconnect (selection unchanged)", () => {
    // The common case: the socket blipped, the user touched nothing. The
    // bound track still matches the selection — no redundant swap.
    expect(
      needsFixtureReconcile({
        prevStatus: "reconnecting",
        status: "ready",
        selectedFixture: A,
        boundFixture: A,
      }),
    ).toBe(false);
  });

  it("does NOT fire on a fresh Play (idle/connecting → ready)", () => {
    // useStartSession already resolves the CURRENT selection and records
    // it as boundFixture, so a fresh start never needs a reconcile — even
    // if boundFixture hasn't been written yet at the instant of the edge.
    for (const prevStatus of [
      "idle",
      "loading-fixture",
      "connecting",
    ] as const satisfies readonly SessionStatus[]) {
      expect(
        needsFixtureReconcile({
          prevStatus,
          status: "ready",
          selectedFixture: B,
          boundFixture: null,
        }),
      ).toBe(false);
      expect(
        needsFixtureReconcile({
          prevStatus,
          status: "ready",
          selectedFixture: B,
          boundFixture: A,
        }),
      ).toBe(false);
    }
  });

  it("ignores non-ready destinations of the reconnect edge", () => {
    // Entering reconnect, giving up, and re-render churn must not swap.
    for (const status of [
      "reconnecting",
      "error",
      "closed",
      "idle",
    ] as const satisfies readonly SessionStatus[]) {
      expect(
        needsFixtureReconcile({
          prevStatus: "reconnecting",
          status,
          selectedFixture: B,
          boundFixture: A,
        }),
      ).toBe(false);
    }
  });

  it("never fires with an empty selection", () => {
    // No track selected yet → nothing to reconcile to.
    expect(
      needsFixtureReconcile({
        prevStatus: "reconnecting",
        status: "ready",
        selectedFixture: "",
        boundFixture: A,
      }),
    ).toBe(false);
  });
});
