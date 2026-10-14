// Generate the C++ config + controls contracts from the TypeScript SDK.
//
// The portable `RtmgConfig` preset shape (and the `controls/` user-facing copy
// layer) is hand-authored in TypeScript, NOT in the Python wire registry. So,
// unlike the wire emitter (scripts/gen_wire_types.py, sourced from protocol.py),
// the CONFIG emitter is Node/TS: it imports the BUILT SDK and walks RUNTIME data
// (DEFAULT_CONFIG, the config enums, KNOWN_TOP_LEVEL_KEYS, TERMS, the control
// copy maps) — never erased types — and projects them into two committed C++
// headers consumed by the JUCE/C++ rtmg-vst plugin (which cannot link the TS SDK):
//
//   types/configContract.gen.hpp    — config field-key/enum/version constants
//                                      (demon::config::*) the VST keys its
//                                      DemonExport (de)serialization off of,
//                                      plus the DEFAULT_CONFIG VALUES as typed
//                                      constexpr facts (demon::config::defaults)
//                                      so C++ hosts consume the defaults instead
//                                      of re-declaring them.
//   types/controlsContract.gen.hpp  — TERMS + CONTROL_DISPLAY_NAMES +
//                                      CONTROL_DESCRIPTIONS (demon::controls::*)
//                                      the VST APVTS param names key off of.
//   types/configDefaults.gen.json   — serializeConfig(DEFAULT_CONFIG), the same
//                                      value facts as one machine-readable JSON
//                                      artifact for JS/TS consumers (a web UI's
//                                      hand-synced tables, test fixtures).
//
// Both headers are CONSTANTS-ONLY and JSON-library-agnostic, exactly like the
// wire header: no structs, no nlohmann. The VST keeps its own juce::var
// (de)serialization and references these generated names instead of literals.
//
// render*ContractHpp() are PURE functions of the SDK module object, so the drift
// guard (demos/realtime_motion_graph_web/web/tests/unit/configContractDrift.test.ts)
// can regenerate each in-memory from the SDK SOURCE and byte-compare the committed
// file — no build needed for the test. main() regenerates the committed files
// from the BUILT bundle.
//
// Regenerate after any change to config/ or controls/:
//   (cd packages/demon-client && npm run build && npm run gen:config)
//
// Coverage note (the caveat plan 06 flags): nested field-key coverage ==
// DEFAULT_CONFIG coverage — a nested field absent from DEFAULT_CONFIG (e.g.
// engine.depth_xl, engine.max_concurrent_loras, prompts.a_xl) emits NO constant.
// That is intentional: the VST does not model those fields and round-trips them
// opaquely inside their parent object (parse the whole `engine` var, mutate only
// modeled keys, write it back), so it needs no constant for them. The TOP-LEVEL
// surface, by contrast, is complete-by-construction from KNOWN_TOP_LEVEL_KEYS
// (so `curves`/`loop`, optional and absent from DEFAULT_CONFIG, still emit a
// top-level constant). The drift guard asserts both invariants.

import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import { writeFileSync } from "node:fs";

// ── String helpers (mirror gen_wire_types.py so the two headers read alike) ──

/** snake_case / digits -> PascalCase. "ch_g0" -> "ChG0", "ch13" -> "Ch13",
 *  "2" -> "2". */
function pascal(name) {
  return String(name)
    .split("_")
    .map((p) => (p ? p[0].toUpperCase() + p.slice(1) : ""))
    .join("");
}

/** Field/option name -> constant identifier ("at_s" -> "kAtS"). */
function kName(name) {
  return "k" + pascal(name);
}

/** A valid, escaped C++ double-quoted string literal. JSON.stringify yields a
 *  correct literal for the ASCII identifiers + editorial prose here, and defends
 *  against a stray quote/backslash/newline in a value. */
function cppStr(value) {
  return JSON.stringify(String(value));
}

/** Sanitize free text for a single-line /** ... *\/ doc comment. Escape EVERY
 *  `*\/` (not just the first) so a description with two of them can't close the
 *  comment early and corrupt the header. */
function cppComment(text) {
  return String(text).split(/\s+/).join(" ").replaceAll("*/", "*\\/");
}

function indent(block, n) {
  const pad = " ".repeat(n);
  return block
    .split("\n")
    .map((line) => (line ? pad + line : line))
    .join("\n");
}

/** A `namespace <ns> { <body> }` block with the body indented two spaces. */
function ns(name, body) {
  return `namespace ${name} {\n\n${indent(body, 2)}\n\n}  // namespace ${name}`;
}

/** One `inline constexpr const char* kFoo = "foo";` line. */
function strConst(name, value) {
  return `inline constexpr const char* ${kName(name)} = ${cppStr(value)};`;
}

/** A C++ `double` literal from a JS number. JSON has one number type, so every
 *  numeric default is emitted as double (no int/double flip when a default
 *  moves between integral and fractional); integral values get a `.0` so the
 *  literal reads as floating-point. */
function cppNum(value) {
  const s = String(value);
  return Number.isInteger(value) && !/[eE.]/.test(s) ? `${s}.0` : s;
}

/** One typed `inline constexpr <bool|double|const char*> kFoo = ...;` line,
 *  the C++ type picked from the JS scalar's runtime type. */
function typedConst(name, value) {
  if (typeof value === "boolean") {
    return `inline constexpr bool ${kName(name)} = ${value};`;
  }
  if (typeof value === "number") {
    return `inline constexpr double ${kName(name)} = ${cppNum(value)};`;
  }
  return strConst(name, value);
}

/** A flat namespace of string-key constants from an array of names, each
 *  constant's VALUE being the name itself (field-key vocabulary). */
function keyNamespace(name, keys) {
  return ns(name, keys.map((k) => strConst(k, k)).join("\n"));
}

/** A flat namespace mapping each id to an arbitrary string VALUE
 *  (display names / descriptions), constant identifier derived from the id. */
function valueNamespace(name, entries) {
  return ns(
    name,
    entries.map(([id, value]) => strConst(id, value)).join("\n"),
  );
}

// ── Enum option sets ────────────────────────────────────────────────────────
// All enum option lists are walked from runtime const arrays the SDK exports
// (SWAP_SOURCE_MODES, STEM_SOURCE_MODES, SERIALIZED_INPUT_KINDS, DCW_MODES, …),
// each the single source of truth for its TS union. Nothing is hand-mirrored
// here, so the drift guard genuinely fails if any option set changes upstream
// without a regenerate.

// ── demon::config::defaults — DEFAULT_CONFIG value facts ────────────────────

/** The `defaults` namespace body: DEFAULT_CONFIG's VALUES as typed constexpr
 *  constants (bool / double / const char* by runtime type), plus two constexpr
 *  POD tables — the ordered controls list and the channel_ranges rows — so a
 *  consumer can iterate the facts as well as reference them by name. Coverage:
 *  engine scalars, prompts, controls, channel_ranges, and the top-level
 *  seed / swap_source_mode. Not emitted: web.* (the browser demo's own block)
 *  and enabled_loras (empty array in DEFAULT_CONFIG). */
function renderDefaultsNamespace(cfg) {
  const controlEntries = Object.entries(cfg.controls);
  const controlRow = ([key, value]) => {
    if (typeof value === "boolean") {
      return `    { ${cppStr(key)}, ValueKind::Boolean, 0.0, ${value}, nullptr },`;
    }
    if (typeof value === "number") {
      return `    { ${cppStr(key)}, ValueKind::Number, ${cppNum(value)}, false, nullptr },`;
    }
    return `    { ${cppStr(key)}, ValueKind::String, 0.0, false, ${cppStr(value)} },`;
  };

  const controls = ns(
    "controls",
    [
      "// DEFAULT_CONFIG.controls values, one typed constant per knob.",
      ...controlEntries.map(([k, v]) => typedConst(k, v)),
      "",
      "// The same values as one ordered table (DEFAULT_CONFIG insertion order,",
      "// which is also JSON.stringify emission order), for consumers that",
      "// iterate the control set instead of naming each constant.",
      "enum class ValueKind { Number, Boolean, String };",
      "struct ControlValue {",
      "  const char* key;",
      "  ValueKind kind;",
      "  double number;       // valid when kind == Number",
      "  bool boolean;        // valid when kind == Boolean",
      "  const char* string;  // valid when kind == String",
      "};",
      `inline constexpr ControlValue kValues[] = {\n${controlEntries
        .map(controlRow)
        .join("\n")}\n};`,
      `inline constexpr int kValueCount = ${controlEntries.length};`,
    ].join("\n"),
  );

  const rangeEntries = Object.entries(cfg.channel_ranges);
  const channelRanges = ns(
    "channel_ranges",
    [
      "// DEFAULT_CONFIG.channel_ranges rows ({ min, max, reverse } per channel),",
      "// in DEFAULT_CONFIG insertion order.",
      "struct ChannelRange {",
      "  const char* channel;",
      "  double min;",
      "  double max;",
      "  bool reverse;",
      "};",
      `inline constexpr ChannelRange kRanges[] = {\n${rangeEntries
        .map(
          ([ch, r]) =>
            `    { ${cppStr(ch)}, ${cppNum(r.min)}, ${cppNum(r.max)}, ${r.reverse} },`,
        )
        .join("\n")}\n};`,
      `inline constexpr int kRangeCount = ${rangeEntries.length};`,
    ].join("\n"),
  );

  return ns(
    "defaults",
    [
      "// Top-level scalar defaults.",
      typedConst("seed", cfg.seed),
      typedConst("swap_source_mode", cfg.swap_source_mode),
      "",
      "// engine.* scalar defaults (enabled_loras is [] in DEFAULT_CONFIG; an",
      "// empty array has no value constant).",
      ns(
        "engine",
        Object.entries(cfg.engine)
          .filter(([, v]) => !Array.isArray(v))
          .map(([k, v]) => typedConst(k, v))
          .join("\n"),
      ),
      "",
      ns(
        "prompts",
        Object.entries(cfg.prompts)
          .map(([k, v]) => typedConst(k, v))
          .join("\n"),
      ),
      "",
      controls,
      "",
      channelRanges,
    ].join("\n"),
  );
}

// ── configContract.gen.hpp ──────────────────────────────────────────────────

const CONFIG_HPP_HEADER = `\
// AUTO-GENERATED — do not edit by hand.
//
// Projected from the TypeScript config SDK
//   packages/demon-client/config/{defaults,types,enums}.ts + inputs.ts
// by packages/demon-client/scripts/genConfigTypes.mjs.
//
// Regenerate after any config change:
//   (cd packages/demon-client && npm run build && npm run gen:config)
// Drift-guarded by web/tests/unit/configContractDrift.test.ts
// (a stale copy fails CI).
//
// Constants-ONLY and JSON-library-agnostic: this header pulls in NO JSON
// dependency and declares no serialization machinery. It provides the config
// VOCABULARY as string constants — JSON field keys, enum option values, and the
// schema version — so a C++ client (the rtmg-vst plugin) references generated
// names instead of hand-copied literals while keeping its own juce::var
// (de)serialization. It also provides the DEFAULT_CONFIG VALUES as typed
// constexpr facts (demon::config::defaults), including two small constexpr POD
// tables for iteration, so C++ hosts consume the defaults instead of
// re-declaring them. The same value facts ship as machine-readable JSON in
// types/configDefaults.gen.json for JS/TS consumers.
//
// The preset file IS a web DemonExport: an RtmgConfig plus an optional top-level
// \`inputs\` (SerializedInputs). Field-key coverage of NESTED objects mirrors
// DEFAULT_CONFIG; optional fields absent from defaults (engine._xl, prompt._xl,
// max_concurrent_loras*) are NOT emitted — they round-trip opaquely inside their
// parent object. The top-level key surface is complete (KNOWN_TOP_LEVEL_KEYS).`;

function renderConfigContractHpp(sdk) {
  const cfg = sdk.DEFAULT_CONFIG;
  const known = Array.from(sdk.KNOWN_TOP_LEVEL_KEYS);

  // Top-level field keys: complete-by-construction from KNOWN_TOP_LEVEL_KEYS
  // (so curves/loop, absent from DEFAULT_CONFIG, still get a constant) plus the
  // DemonExport `inputs` extension.
  const topLevel = [
    "// ── Top-level keys ──",
    "// A DemonExport is an RtmgConfig (these known keys) plus an optional",
    "// top-level `inputs`. Unknown top-level keys round-trip untouched",
    "// (preserve-unknown); kKnownTopLevelKeys is that boundary.",
    ...known.map((k) => strConst(k, k)),
    "// DemonExport extension: serialized track/timbre/structure inputs.",
    strConst("inputs", "inputs"),
    "",
    "// The preserve-unknown boundary: top-level keys the schema models. A",
    "// loaded config's other top-level keys are re-emitted untouched on write.",
    `inline constexpr const char* kKnownTopLevelKeys[] = {\n${known
      .map((k) => `    ${cppStr(k)},`)
      .join("\n")}\n};`,
    `inline constexpr int kKnownTopLevelKeyCount = ${known.length};`,
  ].join("\n");

  const blocks = [
    "// Shared-config schema version (RtmgConfig.version / DEFAULT_CONFIG.version).\n" +
      "// Bumped on a breaking shape change; compare against a loaded preset's\n" +
      "// `version` to detect a cross-frontend mismatch.\n" +
      `inline constexpr int kSchemaVersion = ${cfg.version};`,
    topLevel,
    "// ── Nested object field keys (coverage == DEFAULT_CONFIG) ──",
    ns(
      "engine",
      [
        Object.keys(cfg.engine)
          .map((k) => strConst(k, k))
          .join("\n"),
        "",
        "// One enabled_loras[] entry. The array is empty in DEFAULT_CONFIG so its",
        "// element shape (EnabledLoraEntry's object form) is projected explicitly",
        "// from config/types.ts. A bare-string entry (enable at default strength)",
        "// has no keys.",
        keyNamespace("enabled_lora", ["name", "strength"]),
      ].join("\n"),
    ),
    keyNamespace("prompts", Object.keys(cfg.prompts)),
    "// controls.* — per-knob initial values; keys are the engine param ids\n" +
      "// (also the MIDI-map / config-file names). The VST APVTS param IDs key\n" +
      "// off these. Friendly display names live in controlsContract.gen.hpp.",
    keyNamespace("controls", Object.keys(cfg.controls)),
    "// channel_ranges.* — keyed by channel name (a subset of controls keys);\n" +
      "// each value is a { min, max, reverse } object (the `range` sub-keys).",
    ns(
      "channel_ranges",
      [
        Object.keys(cfg.channel_ranges)
          .map((k) => strConst(k, k))
          .join("\n"),
        "",
        keyNamespace("range", Object.keys(Object.values(cfg.channel_ranges)[0])),
      ].join("\n"),
    ),
    "// web.* — the browser demo's own block. Other frontends ignore it but must\n" +
      "// round-trip it untouched (it is a known top-level key, not an unknown).",
    ns(
      "web",
      [
        // Every web.* key — scalars AND the nested-object keys (effects / audio /
        // denoise_session_gate), so a consumer can navigate to a child by its
        // generated key constant (kAudio) as well as read the child's fields.
        ...Object.keys(cfg.web).map((k) => strConst(k, k)),
        "",
        keyNamespace("effects", Object.keys(cfg.web.effects)),
        "",
        keyNamespace("audio", Object.keys(cfg.web.audio)),
        "",
        keyNamespace(
          "denoise_session_gate",
          Object.keys(cfg.web.denoise_session_gate),
        ),
      ].join("\n"),
    ),
    "// ── Enum option values ──",
    keyNamespace("swap_source_mode", Array.from(sdk.SWAP_SOURCE_MODES)),
    keyNamespace("time_signature", Array.from(sdk.VALID_TIME_SIGNATURES)),
    keyNamespace("dcw_mode", Array.from(sdk.DCW_MODES)),
    keyNamespace("dcw_wavelet", Array.from(sdk.DCW_WAVELETS)),
    keyNamespace("rcfg_mode", Array.from(sdk.RCFG_MODES)),
    keyNamespace("loop_grid", Array.from(sdk.LOOP_GRID_ORDER)),
    "// ── inputs.* — the DemonExport `inputs` block (SerializedInputs codec) ──",
    ns(
      "inputs",
      [
        "// The three input axes.",
        strConst("track", "track"),
        strConst("timbre", "timbre"),
        strConst("structure", "structure"),
        "",
        ns(
          "input",
          [
            "// SerializedInput fields. `kind` discriminates fixture vs clip;",
            "// a clip embeds trimmed PCM as a base64 16-bit WAV in `wavBase64`.",
            strConst("kind", "kind"),
            strConst("name", "name"),
            strConst("source_mode", "sourceMode"),
            strConst("wav_base64", "wavBase64"),
            "",
            keyNamespace("kind_value", Array.from(sdk.SERIALIZED_INPUT_KINDS)),
            "",
            keyNamespace("source_mode_value", Array.from(sdk.STEM_SOURCE_MODES)),
          ].join("\n"),
        ),
      ].join("\n"),
    ),
    "// ── DEFAULT_CONFIG values (defaults::*) ──\n" +
      "// The VALUE facts of DEFAULT_CONFIG — control defaults, prompts, engine\n" +
      "// scalars, channel-range min/max/reverse — as typed constexpr constants,\n" +
      "// so a C++ host CONSUMES the defaults instead of re-declaring them.",
    renderDefaultsNamespace(cfg),
  ];

  const body = blocks.join("\n\n");
  return (
    [
      CONFIG_HPP_HEADER,
      "#pragma once",
      "namespace demon::config {\n\n" +
        body +
        "\n\n}  // namespace demon::config",
    ].join("\n\n") + "\n"
  ).replace(/\r\n/g, "\n");
}

// ── controlsContract.gen.hpp ────────────────────────────────────────────────

const CONTROLS_HPP_HEADER = `\
// AUTO-GENERATED — do not edit by hand.
//
// Projected from the TypeScript controls/ copy layer
//   packages/demon-client/controls/{lexicon,copy}.ts
// by packages/demon-client/scripts/genConfigTypes.mjs.
//
// Regenerate after any controls/ change:
//   (cd packages/demon-client && npm run build && npm run gen:config)
// Drift-guarded by web/tests/unit/configContractDrift.test.ts
// (a stale copy fails CI).
//
// The FOURTH contract: user-facing control copy — the labels/tooltips a frontend
// renders ON its knobs/inputs, distinct from config + wire + the terse
// agent-facing /api/knobs descriptions. JS hosts (web demo, M4L generator, VST
// webview) import @demon/client/controls directly; the C++ VST cannot, so its
// DAW-facing APVTS param names key off these generated constants.
//
// terms::    — the product-noun lexicon (TERMS): flip a noun in ONE place.
// display::  — friendly per-param labels (CONTROL_DISPLAY_NAMES); only the
//              SDK's explicit overrides (others fall back to snake->space).
// describe:: — rich tooltip prose (CONTROL_DESCRIPTIONS) + the runtime-family
//              shared strings (LoRA / manual-steering slots).`;

function renderControlsContractHpp(sdk) {
  const terms = ns(
    "terms",
    [
      "// Product-facing nouns the team renames over time, keyed by a stable id.",
      ...Object.entries(sdk.TERMS).map(([id, label]) => strConst(id, label)),
    ].join("\n"),
  );

  const display = valueNamespace(
    "display",
    Object.entries(sdk.CONTROL_DISPLAY_NAMES),
  );

  const describeLines = Object.entries(sdk.CONTROL_DESCRIPTIONS).map(
    ([id, prose]) =>
      `/** ${cppComment(prose)} */\n${strConst(id, prose)}`,
  );
  // Runtime-generated knob families share one string each (matched by prefix in
  // describeControl); the VST has no per-instance copy for them either.
  const families = [
    ["lora_strength", sdk.LORA_STRENGTH_DESCRIPTION],
    ["lora_blend", sdk.LORA_BLEND_DESCRIPTION],
    ["manual_src", sdk.MANUAL_SRC_DESCRIPTION],
    ["manual_layer", sdk.MANUAL_LAYER_DESCRIPTION],
    ["manual_step", sdk.MANUAL_STEP_DESCRIPTION],
    ["manual_alpha", sdk.MANUAL_ALPHA_DESCRIPTION],
  ];
  const describe = ns(
    "describe",
    [
      "// Per-param tooltip prose (CONTROL_DESCRIPTIONS).",
      ...describeLines,
      "",
      "// Runtime-family shared copy (LoRA strength / blend, manual slots).",
      ns(
        "family",
        families.map(([id, prose]) => `/** ${cppComment(prose)} */\n${strConst(id, prose)}`).join("\n\n"),
      ),
    ].join("\n"),
  );

  const body = [terms, display, describe].join("\n\n");
  return (
    [
      CONTROLS_HPP_HEADER,
      "#pragma once",
      "namespace demon::controls {\n\n" +
        body +
        "\n\n}  // namespace demon::controls",
    ].join("\n\n") + "\n"
  ).replace(/\r\n/g, "\n");
}

// ── configDefaults.gen.json ─────────────────────────────────────────────────

/** The DEFAULT_CONFIG value facts as one machine-readable JSON artifact:
 *  exactly `JSON.stringify(serializeConfig(DEFAULT_CONFIG))` + newline — the
 *  same bytes a frontend captures when it serializes an untouched default
 *  config, so downstream repos can use one file as BOTH a consumable defaults
 *  table (web UIs) and a byte-exact serialization fixture (drift tests). */
function renderConfigDefaultsJson(sdk) {
  return JSON.stringify(sdk.serializeConfig(sdk.DEFAULT_CONFIG)) + "\n";
}

// ── CLI ─────────────────────────────────────────────────────────────────────

function typesDir() {
  return join(dirname(fileURLToPath(import.meta.url)), "..", "types");
}

function main() {
  const require = createRequire(import.meta.url);
  const sdk = require("../dist/demon-client.node.cjs");

  const out = typesDir();
  const configText = renderConfigContractHpp(sdk);
  const configPath = join(out, "configContract.gen.hpp");
  writeFileSync(configPath, configText, { encoding: "utf-8" });
  console.log(`wrote ${configPath} (${configText.length} bytes)`);

  const controlsText = renderControlsContractHpp(sdk);
  const controlsPath = join(out, "controlsContract.gen.hpp");
  writeFileSync(controlsPath, controlsText, { encoding: "utf-8" });
  console.log(`wrote ${controlsPath} (${controlsText.length} bytes)`);

  const defaultsText = renderConfigDefaultsJson(sdk);
  const defaultsPath = join(out, "configDefaults.gen.json");
  writeFileSync(defaultsPath, defaultsText, { encoding: "utf-8" });
  console.log(`wrote ${defaultsPath} (${defaultsText.length} bytes)`);
}

export {
  renderConfigContractHpp,
  renderControlsContractHpp,
  renderConfigDefaultsJson,
};

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  main();
}
