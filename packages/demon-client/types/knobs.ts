// Backend knob manifest — the shape served by `GET /api/knobs` and the
// MCP `list_knobs` tool, projected from the single registry in
// acestep/streaming/knobs.py (`knob_catalog`). This is the backend-owned
// contract a re-skinned or vibecoded UI builds against: render by `type`,
// group by `group`, read ranges/options/defaults from here instead of
// re-declaring them on the client.

export interface KnobManifestEntry {
  /** Control kind. Drives which widget renders the knob. */
  type: "float" | "int" | "enum" | "bool";
  /** Neutral/reset value. Numeric for float/int, string for enum, bool. */
  default: number | string | boolean;
  /** Registry group (core / groups / keystones / guidance / dcw / …). */
  group: string;
  /** True = KnobState-backed continuous param; false = raw-param knob
   *  (rides the params channel; the runner reads it straight from the wire
   *  dict). Its default is seeded server-side, so it still reports a live
   *  value in a session's `knob_values`. */
  bank: boolean;
  /** Numeric floor (float/int only). Always emitted by current backends
   *  (0 when the registry leaves it unset); optional only for older
   *  backends that omitted the implicit-zero floor. */
  min?: number;
  /** Numeric ceiling (float/int only). */
  max?: number;
  /** Allowed values for enum/bool knobs. */
  options?: Array<string | boolean>;
  /** Server-side slew ceiling in knob units per second (continuous knobs
   *  only). The backend rate-limits the applied value toward a freshly
   *  set target at this rate so a fast sweep ramps instead of jumping;
   *  a contract-built UI can mirror the same ramp in its own tween.
   *  Absent for discrete (int/enum/bool) knobs — they are never slewed —
   *  and for backends predating KNOB_SCHEMA_VERSION 2. */
  slew_max_per_s?: number;
  /** Agent/human-facing one-liner. */
  description?: string;
}

export type KnobManifest = Record<string, KnobManifestEntry>;

/** The full `GET /api/knobs` response envelope. */
export interface KnobManifestResponse {
  /** Knob-manifest schema version (KNOB_SCHEMA_VERSION server-side; the
   *  generated `KNOB_SCHEMA_VERSION` constant is the client's expected
   *  value). Absent from older backends that predate versioning. */
  version?: number;
  knobs: KnobManifest;
}
