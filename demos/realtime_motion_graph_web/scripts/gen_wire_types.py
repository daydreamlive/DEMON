"""Generate the TypeScript wire-contract types from the Python registry.

The single source of truth for the WebSocket vocabulary is
``demos/realtime_motion_graph_web/protocol.py`` (:func:`wire_contract`). This
script projects that contract into committed TypeScript types
(``web/sdk/types/wireContract.gen.ts``, part of the demon-client SDK) so a
re-skinned / vibecoded UI builds against generated, always-in-sync types
instead of hand-copying message shapes out of ``sdk/protocol.ts``.

``render_wire_types_ts(contract)`` is a pure function of the contract dict, so
the drift guard (``tests/unit/test_wire_contract.py``) can regenerate in-memory
and assert the committed file matches — no running server, no GPU.

Regenerate after any change to the registry::

    python demos/realtime_motion_graph_web/scripts/gen_wire_types.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HEADER = """\
// AUTO-GENERATED — do not edit by hand.
//
// Projected from the Python wire-contract registry
//   demos/realtime_motion_graph_web/protocol.py :: wire_contract()
// by demos/realtime_motion_graph_web/scripts/gen_wire_types.py.
//
// Regenerate after any registry change:
//   python demos/realtime_motion_graph_web/scripts/gen_wire_types.py
// Drift-guarded by tests/unit/test_wire_contract.py
// (test_generated_wire_types_match_contract) — a stale copy fails CI.
//
// The `params` command's `raw` payload (the knob set) is described separately
// by the /api/knobs manifest. Binary framing (PCM uploads, the float16 slice
// stream) is documented per-entry in the source registry, not typed here.
"""


def _ts_lit(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # json.dumps gives a valid (escaped) TS double-quoted string literal,
        # so a quote or backslash in a registry value can't break the output.
        return json.dumps(value)
    return str(value)


def _ts_comment(text: str) -> str:
    """Sanitize free text for a single-line ``/** ... */`` doc comment:
    collapse whitespace/newlines and defang a literal ``*/``."""
    return " ".join(str(text).split()).replace("*/", "*\\/")


def _ts_field_type(field: dict) -> str:
    t = field.get("type")
    if t == "str":
        return "string"
    if t in ("float", "int"):
        return "number"
    if t == "bool":
        return "boolean"
    if t == "dict":
        return "Record<string, unknown>"
    if t == "list":
        return "unknown[]"
    if t == "enum":
        opts = field.get("options") or []
        return " | ".join(_ts_lit(o) for o in opts) if opts else "string"
    return "unknown"


def _pascal(name: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in name.split("_"))


def _emit_interface(
    iface: str,
    fields: dict,
    *,
    type_literal: str | None = None,
    index_signature: bool = False,
) -> str:
    lines = [f"export interface {iface} {{"]
    if type_literal is not None:
        lines.append(f'  type: "{type_literal}";')
    for fname, fspec in fields.items():
        desc = fspec.get("description")
        if desc:
            lines.append(f"  /** {_ts_comment(desc)} */")
        opt = "" if fspec.get("required") else "?"
        ts_type = _ts_field_type(fspec)
        if fspec.get("nullable"):
            ts_type += " | null"
        lines.append(f"  {fname}{opt}: {ts_type};")
    if index_signature:
        lines.append("  // SessionConfig is permissive; extras pass through.")
        lines.append("  [k: string]: unknown;")
    lines.append("}")
    return "\n".join(lines)


def _name_union(type_name: str, names) -> str:
    body = "\n".join(f"  | {_ts_lit(n)}" for n in names)
    return f"export type {type_name} =\n{body};"


def _name_array(const_name: str, type_name: str, names) -> str:
    body = "\n".join(f"  {_ts_lit(n)}," for n in names)
    return (
        f"export const {const_name}: readonly {type_name}[] = [\n{body}\n] as const;"
    )


def render_wire_types_ts(contract: dict, knob_schema_version: int) -> str:
    """Render the full generated module from a ``wire_contract()`` dict plus
    the knob-manifest schema version (``acestep.streaming.knobs.
    KNOB_SCHEMA_VERSION``), passed in so this stays a pure function of its
    inputs and the drift guard can regenerate in-memory.

    Pure + deterministic (registry insertion order), so a byte-for-byte string
    compare against the committed file is a sound drift check.
    """
    commands = contract["commands"]
    events = contract["events"]
    config = contract["config"]
    handshake = contract["handshake"]
    hs_commands = handshake["commands"]
    hs_events = handshake["events"]

    blocks: list = [
        _HEADER.rstrip(),
        f"export const PROTOCOL_VERSION = {contract['version']};",
        "// Knob-manifest schema version (the `version` field served by GET\n"
        "// /api/knobs and the MCP list_knobs tool). Compare against the live\n"
        "// manifest to detect a stale build, exactly like PROTOCOL_VERSION.\n"
        f"export const KNOB_SCHEMA_VERSION = {knob_schema_version};",
        _name_union("CommandName", commands.keys()),
        _name_array("COMMAND_NAMES", "CommandName", commands.keys()),
        _name_union("EventName", events.keys()),
        _name_array("EVENT_NAMES", "EventName", events.keys()),
        _name_union("HandshakeCommandName", hs_commands.keys()),
        _name_union("HandshakeEventName", hs_events.keys()),
        "// ── Command payloads (client → server) ──",
        *[
            _emit_interface(_pascal(name) + "Command", spec["fields"],
                            type_literal=name)
            for name, spec in commands.items()
        ],
        "// ── Event payloads (server → client) ──",
        *[
            _emit_interface(_pascal(name) + "Event", spec["fields"],
                            type_literal=name)
            for name, spec in events.items()
        ],
        "// ── Session-init config (client → server, sent at handshake) ──",
        _emit_interface("SessionConfigPayload", config, index_signature=True),
        "// ── Init-phase upload handshake ──",
        *[
            _emit_interface(_pascal(name) + "Command", spec["fields"],
                            type_literal=name)
            for name, spec in hs_commands.items()
        ],
        *[
            _emit_interface(_pascal(name) + "Event", spec["fields"],
                            type_literal=name)
            for name, spec in hs_events.items()
        ],
        "// ── Discriminated unions ──",
        "export type WireCommand =\n"
        + "\n".join(f"  | {_pascal(n)}Command" for n in commands) + ";",
        "export type WireEvent =\n"
        + "\n".join(f"  | {_pascal(n)}Event" for n in events) + ";",
        "export type HandshakeCommand =\n"
        + "\n".join(f"  | {_pascal(n)}Command" for n in hs_commands) + ";",
        "export type HandshakeEvent =\n"
        + "\n".join(f"  | {_pascal(n)}Event" for n in hs_events) + ";",
    ]
    return "\n\n".join(blocks) + "\n"


def output_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "web" / "sdk" / "types" / "wireContract.gen.ts"
    )


def main() -> None:
    # Force the repo root to the front of sys.path so we import THIS repo's
    # acestep (config_catalog reaches it), not a sibling ACE-Step checkout.
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from acestep.streaming.knobs import KNOB_SCHEMA_VERSION
    from demos.realtime_motion_graph_web.protocol import wire_contract

    text = render_wire_types_ts(wire_contract(), KNOB_SCHEMA_VERSION)
    out = output_path()
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
