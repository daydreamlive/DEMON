// Backend WebSocket wire contract — the shape served by `GET /api/protocol`
// and the MCP `describe_protocol` tool, projected from the single registry in
// demos/realtime_motion_graph_web/protocol.py (`wire_contract`). This is the
// backend-owned vocabulary a re-skinned or vibecoded UI builds against:
// every command it may send and every event it will receive, instead of
// hand-copying message shapes out of engine/protocol.ts.
//
// The per-command `params` payload (the knob set) is described separately by
// the /api/knobs manifest (see ./knobs.ts). Binary framing (PCM uploads, the
// float16 slice stream) is documented per-entry in `description` rather than
// in this JSON schema.

/** One field of a command/event JSON payload. */
export interface WireFieldSpec {
  /** Scalar/structural kind. */
  type: "str" | "float" | "int" | "bool" | "dict" | "list" | "enum";
  /** True = the sender must include this field. */
  required: boolean;
  /** Neutral/reset value, when the field has one. Container-typed config
   *  fields (list/dict) surface their empty-container defaults. */
  default?: number | string | boolean | unknown[] | Record<string, unknown>;
  /** Allowed values for `enum` fields. */
  options?: Array<string | boolean>;
  /** True = explicit JSON null is a valid value (e.g. loop_band clears). */
  nullable?: boolean;
  /** Agent/human-facing one-liner. */
  description?: string;
}

/** One client → server message type. */
export interface WireCommandSpec {
  /** Payload fields keyed by name. */
  fields: Record<string, WireFieldSpec>;
  /** True = a trailing binary audio frame follows the JSON. */
  binary: boolean;
  /** True = the binary frame is present only in some variants (e.g.
   *  `swap_source` omits it when `use_server_source` is set). */
  binary_optional: boolean;
  /** True = an EXTERNAL (MCP / control-bus) send is NOT applied; the server
   *  echoes it back on `echo_event` for the session's own UI to mirror and
   *  re-send through its smoothing tween. Commands without this flag apply
   *  identically from any origin and ack via their normal events. */
  origin_sensitive: boolean;
  /** The event that mirrors an EXTERNAL send of this command (`params` →
   *  `params_echo`, `set_prompt_blend` → `prompt_blend_echo`). Present only
   *  on origin-sensitive commands. */
  echo_event?: string;
  description?: string;
}

/** One server → client message type. */
export interface WireEventSpec {
  fields: Record<string, WireFieldSpec>;
  /** True = one or more binary frames follow the JSON (e.g. `ready`,
   *  `swap_ready`, `stem_assets`). */
  binary_follow: boolean;
  description?: string;
}

export interface WireContract {
  /** Contract schema version; bump = the vocabulary changed shape. */
  version: number;
  commands: Record<string, WireCommandSpec>;
  events: Record<string, WireEventSpec>;
  /** Session-init payload schema, derived server-side from the
   *  SessionConfig dataclass. Optional: older backends don't ship it. */
  config?: Record<string, WireFieldSpec>;
  /** Init-phase upload sub-protocol (upload_track / upload_ok /
   *  upload_failed), dispatched before a session exists. Optional:
   *  older backends don't ship it. */
  handshake?: {
    commands: Record<string, WireCommandSpec>;
    events: Record<string, WireEventSpec>;
  };
}
