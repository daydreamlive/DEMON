"""Generate the wire-contract types from the Python registry.

The single source of truth for the WebSocket vocabulary is
``demos/realtime_motion_graph_web/protocol.py`` (:func:`wire_contract`). This
script projects that contract into two committed, always-in-sync artifacts:

* ``packages/demon-client/types/wireContract.gen.ts`` — TypeScript types, so a
  re-skinned / vibecoded UI builds against generated types instead of
  hand-copying message shapes out of ``packages/demon-client/protocol.ts``.
* ``packages/demon-client/types/wireContract.gen.hpp`` — a C++ header of wire
  string constants (contract-types-only, JSON-library-agnostic), consumed by
  the JUCE/C++ rtmg-vst plugin, which cannot link the TS SDK.

``render_wire_types_ts(contract, ...)`` / ``render_wire_types_hpp(contract,
...)`` are pure functions of the contract dict, so the drift guard
(``tests/unit/test_wire_contract.py``) can regenerate each in-memory and assert
the committed file matches — no running server, no GPU.

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
        union = field.get("of")
        return f"{_pascal(union)}[]" if union else "unknown[]"
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


def _item_kind_blocks(item_kinds: dict) -> list:
    """TS for every registered item union: one interface per kind
    (``NoteAnalysisItem``), the union type (``AnalysisItem``) that typed
    list fields reference, and a ``<Union>ByKind`` map from the envelope's
    ``kind`` value to its element type."""
    blocks: list = []
    for union, kinds in item_kinds.items():
        union_t = _pascal(union)
        for kind, spec in kinds.items():
            blocks.append(_emit_interface(_pascal(kind) + union_t, spec["fields"]))
        blocks.append(
            f"export type {union_t} =\n"
            + "\n".join(f"  | {_pascal(k)}{union_t}" for k in kinds) + ";"
        )
        body = "\n".join(f"  {k}: {_pascal(k)}{union_t};" for k in kinds)
        blocks.append(f"export interface {union_t}ByKind {{\n{body}\n}}")
    return blocks


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
    item_kinds = contract.get("item_kinds") or {}

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
        *(["// ── Item kinds (elements of typed list fields) ──"]
          + _item_kind_blocks(item_kinds) if item_kinds else []),
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


# ---------------------------------------------------------------------------
# C++ emit target (the rtmg-vst plugin)
# ---------------------------------------------------------------------------
# The VST is a JUCE/C++ client of the same contract. It cannot link the
# TypeScript SDK, so it consumes a generated C++ HEADER projected from this
# same registry. The header is contract-types-ONLY and JSON-library-agnostic
# by design: it declares NO structs and pulls in NO JSON dependency — only the
# wire VOCABULARY as string constants (message "type" names, JSON field keys,
# enum option values). The VST keeps its own juce::var (de)serialization and
# references these generated names instead of hand-copied string literals, so
# the only thing that can drift between the clients (the contract surface) is
# generated for C++ too. See plan 04_vst_client_sdk_wire_contract.md.

_HPP_HEADER = """\
// AUTO-GENERATED — do not edit by hand.
//
// Projected from the Python wire-contract registry
//   demos/realtime_motion_graph_web/protocol.py :: wire_contract()
// by demos/realtime_motion_graph_web/scripts/gen_wire_types.py.
//
// Regenerate after any registry change:
//   python demos/realtime_motion_graph_web/scripts/gen_wire_types.py
// Drift-guarded by tests/unit/test_wire_contract.py
// (test_generated_wire_types_hpp_match_contract) — a stale copy fails CI.
//
// Contract-types-ONLY and JSON-library-agnostic: this header declares NO
// structs and pulls in NO JSON dependency. It provides the wire VOCABULARY as
// string constants — message "type" names, JSON field keys, and enum option
// values — so a C++ client (the rtmg-vst plugin) references generated names
// instead of hand-copied literals while keeping its own (de)serialization.
//
// The `params` command's `raw` payload (the knob set) is described separately
// by the /api/knobs manifest. Binary framing (PCM uploads, the float16 slice
// stream) is documented per-entry in the source registry, not encoded here.
"""


def _cpp_comment(text: str) -> str:
    """Sanitize free text for a single-line ``/** ... */`` doc comment:
    collapse whitespace/newlines and defang a literal ``*/``."""
    return " ".join(str(text).split()).replace("*/", "*\\/")


def _cpp_str_lit(value: str) -> str:
    # json.dumps gives a valid (escaped) double-quoted literal; for the ASCII
    # wire identifiers in the registry this is also a valid C++ string literal,
    # and it defends against a stray quote/backslash in a registry value.
    return json.dumps(value)


def _cpp_const_name(name: str) -> str:
    """Field/option name -> constant identifier (``at_s`` -> ``kAtS``)."""
    return "k" + _pascal(name)


def _indent(block: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in block.split("\n"))


def _emit_cpp_message(ns: str, fields: dict, *, type_literal: str | None) -> str:
    """One message -> a ``namespace`` of string constants: ``kType`` (when the
    message carries a ``type`` discriminator), one ``k<Field>`` per field, and
    a nested ``namespace <field>`` of ``k<Option>`` constants for each enum
    field. No types, no structs."""
    lines = [f"namespace {ns} {{"]
    if type_literal is not None:
        lines.append(
            f"  inline constexpr const char* kType = {_cpp_str_lit(type_literal)};"
        )
    enum_blocks: list = []
    for fname, fspec in fields.items():
        desc = fspec.get("description")
        if desc:
            lines.append(f"  /** {_cpp_comment(desc)} */")
        lines.append(
            f"  inline constexpr const char* {_cpp_const_name(fname)} = "
            f"{_cpp_str_lit(fname)};"
        )
        if fspec.get("type") == "enum" and fspec.get("options"):
            opt = [f"namespace {fname} {{"]
            for o in fspec["options"]:
                opt.append(
                    f"  inline constexpr const char* {_cpp_const_name(o)} = "
                    f"{_cpp_str_lit(o)};"
                )
            opt.append(f"}}  // namespace {fname}")
            enum_blocks.append("\n".join(opt))
    for blk in enum_blocks:
        lines.append("")
        lines.append(_indent(blk, 2))
    lines.append(f"}}  // namespace {ns}")
    return "\n".join(lines)


def _emit_cpp_group(group: str, items) -> str:
    """A group namespace (``command`` / ``event``) wrapping one message
    namespace per ``(ns, fields, type_literal)`` item."""
    inner = "\n\n".join(
        _emit_cpp_message(ns, fields, type_literal=tl) for ns, fields, tl in items
    )
    return (
        f"namespace {group} {{\n\n{_indent(inner, 2)}\n\n"
        f"}}  // namespace {group}"
    )


def render_wire_types_hpp(contract: dict, knob_schema_version: int) -> str:
    """Render the generated C++ header from a ``wire_contract()`` dict plus the
    knob-manifest schema version. Parallel to :func:`render_wire_types_ts`:
    pure + deterministic (registry insertion order), so the drift guard can
    regenerate in-memory and byte-compare the committed file."""
    commands = contract["commands"]
    events = contract["events"]
    config = contract["config"]
    handshake = contract["handshake"]
    hs_commands = handshake["commands"]
    hs_events = handshake["events"]

    command_group = _emit_cpp_group(
        "command",
        [(name, spec["fields"], name) for name, spec in commands.items()],
    )
    event_group = _emit_cpp_group(
        "event",
        [(name, spec["fields"], name) for name, spec in events.items()],
    )
    config_ns = _emit_cpp_message("config", config, type_literal=None)
    hs_command_group = _emit_cpp_group(
        "command",
        [(name, spec["fields"], name) for name, spec in hs_commands.items()],
    )
    hs_event_group = _emit_cpp_group(
        "event",
        [(name, spec["fields"], name) for name, spec in hs_events.items()],
    )
    item_kinds = contract.get("item_kinds") or {}
    item_blocks: list = []
    if item_kinds:
        unions = "\n\n".join(
            _emit_cpp_group(
                union,
                [(kind, spec["fields"], None) for kind, spec in kinds.items()],
            )
            for union, kinds in item_kinds.items()
        )
        item_blocks = [
            "// ── Item kinds (elements of typed list fields) ──",
            f"namespace item {{\n\n{_indent(unions, 2)}\n\n}}  // namespace item",
        ]
    handshake_ns = (
        "namespace handshake {\n\n"
        f"{_indent(hs_command_group, 2)}\n\n"
        f"{_indent(hs_event_group, 2)}\n\n"
        "}  // namespace handshake"
    )

    body = "\n\n".join(
        [
            "inline constexpr int kProtocolVersion = "
            f"{contract['version']};",
            "// Knob-manifest schema version (the `version` field served by GET\n"
            "// /api/knobs and the MCP list_knobs tool). Compare against the live\n"
            "// manifest to detect a stale build, exactly like kProtocolVersion.\n"
            f"inline constexpr int kKnobSchemaVersion = {knob_schema_version};",
            "// The JSON discriminator key carried by every command and event.\n"
            'inline constexpr const char* kTypeKey = "type";',
            *item_blocks,
            "// ── Command payloads (client → server) ──",
            command_group,
            "// ── Event payloads (server → client) ──",
            event_group,
            "// ── Session-init config (client → server, sent at handshake) ──",
            config_ns,
            "// ── Init-phase upload handshake ──",
            handshake_ns,
        ]
    )

    blocks = [
        _HPP_HEADER.rstrip(),
        "#pragma once",
        "namespace demon::wire {\n\n" + body + "\n\n}  // namespace demon::wire",
    ]
    return "\n\n".join(blocks) + "\n"


def output_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "packages" / "demon-client" / "types" / "wireContract.gen.ts"
    )


def output_path_hpp() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "packages" / "demon-client" / "types" / "wireContract.gen.hpp"
    )


def main() -> None:
    # Force the repo root to the front of sys.path so we import THIS repo's
    # acestep (config_catalog reaches it), not a sibling ACE-Step checkout.
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from acestep.streaming.knobs import KNOB_SCHEMA_VERSION
    from demos.realtime_motion_graph_web.protocol import wire_contract

    contract = wire_contract()

    text = render_wire_types_ts(contract, KNOB_SCHEMA_VERSION)
    out = output_path()
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(text)} bytes)")

    hpp = render_wire_types_hpp(contract, KNOB_SCHEMA_VERSION)
    out_hpp = output_path_hpp()
    out_hpp.write_text(hpp, encoding="utf-8", newline="\n")
    print(f"wrote {out_hpp} ({len(hpp)} bytes)")


if __name__ == "__main__":
    main()
