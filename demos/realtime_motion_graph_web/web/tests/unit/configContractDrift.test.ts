// Drift guard for the generated C++ config + controls contracts.
//
// The portable RtmgConfig preset shape and the controls/ copy layer are
// hand-authored in TypeScript, and the rtmg-vst plugin (C++, can't link the TS
// SDK) consumes a generated projection of them:
//   packages/demon-client/types/configContract.gen.hpp
//   packages/demon-client/types/controlsContract.gen.hpp
// produced by packages/demon-client/scripts/genConfigTypes.mjs.
//
// render*ContractHpp() are pure functions of the SDK module, so this test
// regenerates each header in-memory from the SDK SOURCE (the @demon/client
// alias) and byte-compares the committed file — the config-side analogue of
// pytest's test_wire_contract.py for the wire header. A stale committed copy
// fails CI; regenerate with `npm run build && npm run gen:config` in
// packages/demon-client.
//
// The TS/Node tier (not pytest) is deliberate: the config source of truth is
// TypeScript, not the Python wire registry, so the guard lives where the source
// lives. See plan 06.

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import * as sdk from "@demon/client";
// The generator is a plain .mjs that imports only node builtins at module load
// and guards main() behind an argv check, so importing it here runs no I/O.
import {
  renderConfigContractHpp,
  renderConfigDefaultsJson,
  renderControlsContractHpp,
} from "../../../../../packages/demon-client/scripts/genConfigTypes.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const TYPES_DIR = path.resolve(
  here,
  "../../../../../packages/demon-client/types",
);

function readCommitted(name: string): string {
  return readFileSync(path.join(TYPES_DIR, name), "utf-8");
}

describe("generated C++ config/controls contracts", () => {
  it("configContract.gen.hpp matches the SDK (regenerate if this fails)", () => {
    const expected = renderConfigContractHpp(sdk);
    expect(readCommitted("configContract.gen.hpp")).toBe(expected);
  });

  it("controlsContract.gen.hpp matches the SDK (regenerate if this fails)", () => {
    const expected = renderControlsContractHpp(sdk);
    expect(readCommitted("controlsContract.gen.hpp")).toBe(expected);
  });

  it("configDefaults.gen.json matches the SDK (regenerate if this fails)", () => {
    const expected = renderConfigDefaultsJson(sdk);
    expect(readCommitted("configDefaults.gen.json")).toBe(expected);
  });

  // ── Coverage assertions (the caveat plan 06 flags: "pick one and test it") ──

  it("emits the COMPLETE top-level key surface (KNOWN_TOP_LEVEL_KEYS + inputs)", () => {
    const hpp = renderConfigContractHpp(sdk);
    // Top-level constants are complete-by-construction from KNOWN_TOP_LEVEL_KEYS
    // (so curves/loop, absent from DEFAULT_CONFIG, are still emitted) plus the
    // DemonExport `inputs` extension. A missing one here means the VST would
    // have no constant for a key it must round-trip.
    for (const key of sdk.KNOWN_TOP_LEVEL_KEYS) {
      expect(hpp).toContain(`inline constexpr const char* k`);
      expect(hpp).toContain(`= "${key}";`);
    }
    expect(hpp).toContain('kInputs = "inputs";');
    expect(hpp).toContain(
      `inline constexpr int kKnownTopLevelKeyCount = ${sdk.KNOWN_TOP_LEVEL_KEYS.size};`,
    );
  });

  it("emits a constant for EVERY control key in DEFAULT_CONFIG (no silent drop)", () => {
    // Nested coverage == DEFAULT_CONFIG coverage. controls.* is the load-bearing
    // surface (the VST APVTS param IDs key off these), so guard it explicitly:
    // adding a control to DEFAULT_CONFIG without regenerating fails here.
    const hpp = renderConfigContractHpp(sdk);
    for (const key of Object.keys(sdk.DEFAULT_CONFIG.controls)) {
      expect(hpp).toContain(`= "${key}";`);
    }
  });

  it("projects enum option values from the SDK's runtime arrays (no hand-mirror)", () => {
    // These option sets were once hand-mirrored in the generator, so the byte
    // compare couldn't catch an upstream addition (both sides shared the same
    // literal). They now derive from runtime const arrays the SDK exports —
    // assert each option actually lands in the header so the guard has teeth.
    const hpp = renderConfigContractHpp(sdk);
    for (const v of sdk.SWAP_SOURCE_MODES) expect(hpp).toContain(`= "${v}";`);
    for (const v of sdk.STEM_SOURCE_MODES) expect(hpp).toContain(`= "${v}";`);
    for (const v of sdk.SERIALIZED_INPUT_KINDS) expect(hpp).toContain(`= "${v}";`);
  });

  it("emits a typed defaults constant AND a table row for EVERY control", () => {
    // The defaults:: namespace is the numeric-facts surface (plan 7.3): C++
    // hosts consume DEFAULT_CONFIG values from it instead of re-declaring
    // them, so a control missing from the table would silently fall back to a
    // host-local literal again.
    const hpp = renderConfigContractHpp(sdk);
    for (const [key, value] of Object.entries(sdk.DEFAULT_CONFIG.controls)) {
      expect(hpp).toContain(`{ ${JSON.stringify(key)}, ValueKind::`);
      if (typeof value === "string") {
        expect(hpp).toContain(`= ${JSON.stringify(value)};`);
      }
    }
    expect(hpp).toContain(
      `inline constexpr int kValueCount = ${Object.keys(sdk.DEFAULT_CONFIG.controls).length};`,
    );
  });

  it("emits every channel_ranges row with its min/max/reverse facts", () => {
    const hpp = renderConfigContractHpp(sdk);
    for (const ch of Object.keys(sdk.DEFAULT_CONFIG.channel_ranges)) {
      expect(hpp).toMatch(
        new RegExp(`\\{ ${JSON.stringify(ch)}, [0-9.]+, [0-9.]+, (true|false) \\},`),
      );
    }
    expect(hpp).toContain(
      `inline constexpr int kRangeCount = ${Object.keys(sdk.DEFAULT_CONFIG.channel_ranges).length};`,
    );
  });

  it("configDefaults.gen.json is exactly serializeConfig(DEFAULT_CONFIG)", () => {
    // The JSON artifact doubles as a byte-exact serialization fixture in
    // downstream repos (rtmg-vst pins its DemonExport capture against it), so
    // guard the CONTENT as well as the committed bytes.
    const artifact = renderConfigDefaultsJson(sdk);
    expect(artifact.endsWith("\n")).toBe(true);
    expect(JSON.parse(artifact)).toEqual(sdk.serializeConfig(sdk.DEFAULT_CONFIG));
  });

  it("projects the controls lexicon + display-name overrides verbatim", () => {
    const hpp = renderControlsContractHpp(sdk);
    for (const [, label] of Object.entries(sdk.TERMS)) {
      expect(hpp).toContain(`"${label}";`);
    }
    // The live drift the lexicon exists to kill: SDK renamed these.
    expect(hpp).toContain('kHintStrength = "structure";');
    expect(hpp).toContain('kTimbreStrength = "timbre";');
  });
});
