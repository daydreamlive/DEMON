// Tier A: per-session checkpoint selection, client half.
//
// The wire payload must be byte-identical to today when no checkpoint is
// picked (every current pod and client), and carry `checkpoint` only when
// one is.

import { describe, expect, it } from "vitest";

import {
  DEFAULT_CONFIG,
  rtmgConfigToSessionConfig,
  type SessionConfigRuntime,
} from "@demon/client";

const runtime: SessionConfigRuntime = {
  enabledLoras: [],
  loraStrengths: {},
  promptA: "a",
  promptB: "b",
  fixtureName: "fx",
  useServerFixture: false,
};

describe("rtmgConfigToSessionConfig checkpoint", () => {
  it("omits the key when unset, null or empty (byte-identical payload)", () => {
    const base = JSON.stringify(rtmgConfigToSessionConfig(DEFAULT_CONFIG, runtime));
    for (const checkpoint of [undefined, null, ""]) {
      const cfg = rtmgConfigToSessionConfig(DEFAULT_CONFIG, { ...runtime, checkpoint });
      expect("checkpoint" in cfg).toBe(false);
      expect(JSON.stringify(cfg)).toBe(base);
    }
  });

  it("emits the key when set", () => {
    const cfg = rtmgConfigToSessionConfig(DEFAULT_CONFIG, {
      ...runtime,
      checkpoint: "sa3-small",
    });
    expect(cfg.checkpoint).toBe("sa3-small");
  });
});
