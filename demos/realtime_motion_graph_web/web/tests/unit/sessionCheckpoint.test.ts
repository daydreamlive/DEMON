// Tier A: per-session checkpoint selection, client half.
//
// The wire payload must be byte-identical to today when no checkpoint is
// picked (every current pod and client), and carry `checkpoint` only when
// one is.

import { beforeEach, describe, expect, it, vi } from "vitest";

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

describe("lorasPath", () => {
  beforeEach(() => useSessionStore.setState(pristine, true));

  it("is the bare path with no pick, and carries the pick otherwise", async () => {
    const { lorasPath } = await import("@/engine/lora/listLoras");
    expect(lorasPath()).toBe("/api/loras");
    useSessionStore.getState().setSelectedCheckpoint("sa3-small");
    expect(lorasPath()).toBe("/api/loras?checkpoint=sa3-small");
  });
});

describe("boot-time server-info population", () => {
  beforeEach(() => useSessionStore.setState(pristine, true));

  it("gates like the pre-Start LoRA fetch", async () => {
    const { shouldFetchServerInfoAtBoot } = await import("@/lib/servedCheckpoints");
    expect(shouldFetchServerInfoAtBoot(null, false)).toBe(false);
    expect(shouldFetchServerInfoAtBoot(null, true)).toBe(true);
    expect(shouldFetchServerInfoAtBoot("ws://pod:1/", false)).toBe(true);
  });

  it("writes servedCheckpoints to the store from server-info, no Start needed", async () => {
    const { fetchServerInfo, serverInfoUrl } = await import("@/lib/servedCheckpoints");
    const seen: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      seen.push(String(url));
      return new Response(JSON.stringify({ checkpoints: TWO }), { status: 200 });
    });
    try {
      await fetchServerInfo("wss://pod.example/session/?token=x");
    } finally {
      vi.unstubAllGlobals();
    }
    expect(seen).toEqual([serverInfoUrl("wss://pod.example/session/?token=x")]);
    expect(seen[0]).toBe("https://pod.example/session/api/server-info");
    expect(useSessionStore.getState().servedCheckpoints).toEqual(TWO);
    expect(useSessionStore.getState().status).toBe("idle");
  });

  it("a failed fetch leaves the store alone", async () => {
    const { fetchServerInfo } = await import("@/lib/servedCheckpoints");
    useSessionStore.getState().setServedCheckpoints(TWO);
    vi.stubGlobal("fetch", async () => { throw new Error("down"); });
    try {
      expect(await fetchServerInfo("ws://pod/")).toBeNull();
    } finally {
      vi.unstubAllGlobals();
    }
    expect(useSessionStore.getState().servedCheckpoints).toEqual(TWO);
  });
});
