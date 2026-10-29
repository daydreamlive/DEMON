// Tier A: per-session checkpoint selection, client half.
//
// The wire payload must be byte-identical to today when no checkpoint is
// picked (every current pod and client), and carry `checkpoint` only when
// one is.

import { beforeEach, describe, expect, it } from "vitest";

import {
  DEFAULT_CONFIG,
  rtmgConfigToSessionConfig,
  type SessionConfigRuntime,
} from "@demon/client";
import {
  parseServedCheckpoints,
  showCheckpointPicker,
} from "@/lib/servedCheckpoints";
import { useSessionStore, type ServedCheckpoint } from "@/store/useSessionStore";

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

// ── Store + server-info parse ──────────────────────────────────────────

const pristine = useSessionStore.getState();

const TWO: ServedCheckpoint[] = [
  { name: "sa3-medium", family: "sa3", model_id: "medium", scale: null, default: true },
  { name: "sa3-small", family: "sa3", model_id: "small-music", scale: null, default: false },
];

describe("served checkpoints store", () => {
  beforeEach(() => useSessionStore.setState(pristine, true));

  it("starts empty with the pod default selected", () => {
    const s = useSessionStore.getState();
    expect(s.servedCheckpoints).toEqual([]);
    expect(s.selectedCheckpoint).toBeNull();
  });

  it("setters write, and both fields survive reset()", () => {
    const s = useSessionStore.getState();
    s.setServedCheckpoints(TWO);
    s.setSelectedCheckpoint("sa3-small");
    useSessionStore.getState().reset();
    const after = useSessionStore.getState();
    expect(after.servedCheckpoints).toEqual(TWO);
    expect(after.selectedCheckpoint).toBe("sa3-small");
  });

  it("a new list that no longer serves the selection falls back to default", () => {
    const s = useSessionStore.getState();
    s.setServedCheckpoints(TWO);
    s.setSelectedCheckpoint("sa3-small");
    s.setServedCheckpoints([TWO[0]]);
    expect(useSessionStore.getState().selectedCheckpoint).toBeNull();
  });
});

describe("parseServedCheckpoints", () => {
  it("returns [] for old pods and malformed payloads", () => {
    expect(parseServedCheckpoints({})).toEqual([]);
    expect(parseServedCheckpoints(null)).toEqual([]);
    expect(parseServedCheckpoints({ checkpoints: "x" })).toEqual([]);
    expect(parseServedCheckpoints({ checkpoints: [{ family: "sa3" }, 3] })).toEqual([]);
  });

  it("parses the §3.2 shape", () => {
    expect(parseServedCheckpoints({ checkpoint: "sa3-medium", checkpoints: TWO })).toEqual(TWO);
  });
});

describe("picker visibility", () => {
  it("hidden below two entries, shown at two", () => {
    expect(showCheckpointPicker([])).toBe(false);
    expect(showCheckpointPicker([TWO[0]])).toBe(false);
    expect(showCheckpointPicker(TWO)).toBe(true);
  });
});
