// `config.json` operator-defaults: the portable schema + pure transforms +
// loader, shared across DEMON frontends. This is a CLIENT-SIDE,
// hand-authored format — distinct from the backend-served `/api/knobs`
// (knob manifest) and `/api/protocol` (wire contract) the rest of the SDK
// projects from the Python registry. See ../README.md "config.json".
//
// What lives here (portable): the schema types, DEFAULT_CONFIG, the
// config-semantic enums, the pure transforms (merge / variant / lora-cap /
// preserve-unknown), the parameterized loader, and the wire mapping +
// neutral state adapters. What stays per-client: the store/DOM wiring that
// reads/writes these (web keeps its zustand apply/capture as thin wrappers).

export * from "./enums";
export * from "./types";
export { DEFAULT_CONFIG } from "./defaults";
export {
  KNOWN_TOP_LEVEL_KEYS,
  getUnknownKeys,
  withUnknownKeys,
  serializeConfig,
  isSwapSourceMode,
  normalizeConfigShape,
  mergeConfig,
  resolveLoraCapForSource,
  selectVariant,
} from "./transforms";
export { loadConfig } from "./load";
export {
  rtmgConfigToSessionConfig,
  applyConfigToState,
  captureConfigFromState,
} from "./wire";
export type {
  SessionConfigRuntime,
  ConfigStateSnapshot,
  ConfigApplyPatch,
} from "./wire";
