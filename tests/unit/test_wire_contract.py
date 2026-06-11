"""Drift guard for the WebSocket wire-contract registry.

The registry in ``demos/realtime_motion_graph_web/protocol.py`` is the single
source of truth for the command/event vocabulary served at ``/api/protocol``
and by the MCP ``describe_protocol`` tool. These tests fail if the ws_adapter
dispatcher or the browser protocol client grows a message type the registry
doesn't declare (or vice versa) — so a new verb cannot be added on either side
without registering it in the contract.

Pure source parsing (AST for the Python dispatcher, regex for the TS client):
no torch, no GPU, no running server.
"""

import ast
import re
from pathlib import Path

from demos.realtime_motion_graph_web.protocol import (
    COMMAND_NAMES,
    EVENT_NAMES,
    HANDSHAKE_COMMAND_NAMES,
    HANDSHAKE_EVENT_NAMES,
    coerce_command_payload,
    command_catalog,
    config_catalog,
    event_catalog,
    handshake_contract,
    wire_contract,
)

_DEMO = (
    Path(__file__).resolve().parents[2]
    / "demos"
    / "realtime_motion_graph_web"
)


def _dispatcher_command_types() -> set:
    """Every ``mtype == "<x>"`` literal in ws_adapter's dispatcher.

    AST-parsed from source so we never import ws_adapter (which pulls the
    torch-heavy engine). Matches the ``if/elif mtype == "..."`` chain that
    routes each client message to a session call.
    """
    src = (_DEMO / "ws_adapter.py").read_text(encoding="utf-8")
    found = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Compare):
            continue
        if not (isinstance(node.left, ast.Name) and node.left.id == "mtype"):
            continue
        for comp in node.comparators:
            if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                found.add(comp.value)
    return found


def _balanced_braces(src: str, open_idx: int) -> str:
    """The text of the ``{...}`` block opening at ``open_idx`` (inclusive),
    by brace counting. Good enough for the ladder body, which contains no
    string literals with unbalanced braces."""
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[open_idx:i + 1]
    raise AssertionError("unbalanced braces — ladder parse drifted?")


def _client_handled_events() -> set:
    """Every event name the browser protocol client (the demon-client SDK's
    RemoteBackend) handles: the streaming ladder's ``case "<x>":`` labels
    (scanned only inside ``switch (msg.type)`` blocks, so an unrelated
    future switch can't inject false positives) plus the init-phase
    ``msg.type === "<x>"`` checks."""
    src = (_DEMO / "web" / "sdk" / "protocol.ts").read_text(encoding="utf-8")
    handled = set(re.findall(r'msg\.type === "([^"]+)"', src))
    for m in re.finditer(r"switch \(msg\.type\)", src):
        block = _balanced_braces(src, src.index("{", m.end()))
        handled |= set(re.findall(r'case "([^"]+)":', block))
    return handled


# Files that serialize server -> client event frames as dict literals. The
# ws adapter owns the JSON envelopes; audio_codec owns the stem_assets header.
_EVENT_EMITTER_FILES = ("ws_adapter.py", "audio_codec.py")


def _emitted_event_fields(valid_names) -> dict:
    """Map ``event name -> set(field names)`` actually emitted on the wire.

    AST-walks the emitter modules for ``{"type": "<event>", ...}`` dict
    literals (the shape every ``ws.send(json.dumps({...}))`` / ``_send_json``
    call uses) and, for each literal whose ``type`` is in ``valid_names``,
    unions in its other string-literal keys. Source-parsed, so it never
    imports the torch-heavy adapter. Dict literals whose ``type`` isn't a
    known event are ignored by design.
    """
    valid = set(valid_names)
    emitted: dict = {}
    for fname in _EVENT_EMITTER_FILES:
        src = (_DEMO / fname).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Dict):
                continue
            str_keys = {
                k.value: v
                for k, v in zip(node.keys, node.values)
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
            tval = str_keys.get("type")
            if not (isinstance(tval, ast.Constant)
                    and isinstance(tval.value, str)):
                continue
            ename = tval.value
            if ename not in valid:
                continue
            fields = {k for k in str_keys if k != "type"}
            emitted.setdefault(ename, set()).update(fields)
    return emitted


def _dict_field_reads(stmts, varname: str) -> set:
    """Every string key read off ``<varname>`` within a list of AST
    statements: ``v.get("x")`` / ``v.get("x", default)`` / ``v["x"]``."""
    fields: set = set()
    for stmt in stmts:
        for sub in ast.walk(stmt):
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "get"
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == varname
                    and sub.args
                    and isinstance(sub.args[0], ast.Constant)
                    and isinstance(sub.args[0].value, str)):
                fields.add(sub.args[0].value)
            elif (isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == varname
                    and isinstance(sub.slice, ast.Constant)
                    and isinstance(sub.slice.value, str)):
                fields.add(sub.slice.value)
    fields.discard("type")
    return fields


def _dispatcher_command_fields() -> dict:
    """Map ``command name -> set(fields the dispatcher reads)``.

    For each ``if/elif mtype == "<cmd>":`` arm in ws_adapter, collect the
    ``data`` keys read inside that arm's body only (not the elif chain that
    hangs off its ``orelse``). Source-parsed; no import of the torch-heavy
    adapter.
    """
    src = (_DEMO / "ws_adapter.py").read_text(encoding="utf-8")
    out: dict = {}
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name) and test.left.id == "mtype"
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and isinstance(test.comparators[0].value, str)):
            continue
        cmd = test.comparators[0].value
        out.setdefault(cmd, set()).update(_dict_field_reads(node.body, "data"))
    return out


def _upload_handler_header_fields() -> set:
    """String keys the init-phase ``_handle_upload_track`` reads off its
    ``header`` arg — the upload_track command payload, dispatched via
    ``config_dict.get("type")`` rather than the ``mtype`` chain."""
    src = (_DEMO / "ws_adapter.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "_handle_upload_track":
            return _dict_field_reads(node.body, "header")
    return set()


def _command_dict_type(node: ast.Dict):
    """The ``"type"`` value of a dict literal, if it's a known command."""
    type_node = next(
        (v for k, v in zip(node.keys, node.values)
         if isinstance(k, ast.Constant) and k.value == "type"),
        None,
    )
    if (isinstance(type_node, ast.Constant)
            and isinstance(type_node.value, str)
            and type_node.value in COMMAND_NAMES):
        return type_node.value
    return None


def _mcp_command_dicts() -> dict:
    """Map ``command name -> set(fields)`` from the envelopes the MCP tools
    build (third hand copy of the vocabulary). Two shapes are covered:

    * ``{"type": "<cmd>", ...}`` dict literals anywhere in the module, and
    * fields added to such a dict after the fact via subscript assignment
      (``msg["tags_b"] = ...``), which the tools use for optional fields.
      Tracked per function: an Assign/AnnAssign binding a name to a command
      dict literal registers that name, and later ``name["field"] = ...``
      stores in the same function are attributed to that command.
    """
    src = (_DEMO / "mcp_server.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: dict = {}

    # Pass 1: every command dict literal, wherever it appears.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        cmd = _command_dict_type(node)
        if cmd is None:
            continue
        out.setdefault(cmd, set()).update(
            k.value
            for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
            and k.value != "type"
        )

    # Pass 2: subscript-assigned fields on names bound to a command dict.
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Sub-pass A: names bound to a command dict literal anywhere in the
        # function (`msg = {...}` / `msg: dict[...] = {...}`). Collected
        # before sub-pass B so the guard is insensitive to ast.walk order.
        var_cmd: dict = {}
        for node in ast.walk(fn):
            target = None
            value = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            if isinstance(target, ast.Name) and isinstance(value, ast.Dict):
                cmd = _command_dict_type(value)
                if cmd is not None:
                    var_cmd[target.id] = cmd
        # Sub-pass B: `msg["field"] = ...` stores on tracked names.
        for node in ast.walk(fn):
            if (isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Subscript)
                    and isinstance(node.targets[0].value, ast.Name)
                    and node.targets[0].value.id in var_cmd
                    and isinstance(node.targets[0].slice, ast.Constant)
                    and isinstance(node.targets[0].slice.value, str)):
                field = node.targets[0].slice.value
                if field != "type":
                    out.setdefault(
                        var_cmd[node.targets[0].value.id], set(),
                    ).add(field)
    return out


def test_commands_match_dispatcher_exactly():
    # Every command the registry declares must be handled by the dispatcher,
    # and every type the dispatcher handles must be in the registry.
    assert _dispatcher_command_types() == set(COMMAND_NAMES)


def test_client_events_are_declared():
    handled = _client_handled_events()
    assert handled, "regex found no msg.type === handlers — pattern drifted?"
    missing = handled - set(EVENT_NAMES)
    assert not missing, (
        f"events handled by the browser client but absent from the wire "
        f"contract registry: {sorted(missing)}"
    )


def test_emitted_event_fields_are_declared():
    # Every field the server actually puts on an event frame must be declared
    # in that event's EventSpec — otherwise a UI built purely from the wire
    # contract would silently miss it (e.g. ready.session_id, swap_ready.
    # sample_rate). The name-level guards above can't see this; this one
    # AST-extracts the emitted dict-literal keys and checks them field-by-field.
    # Streaming events + the init-phase handshake events (upload_ok/_failed).
    evts = {**event_catalog(), **handshake_contract()["events"]}
    names = set(EVENT_NAMES) | set(HANDSHAKE_EVENT_NAMES)
    emitted = _emitted_event_fields(names)
    assert emitted, "no event dict literals found — emitter parse drifted?"
    undeclared = {
        ename: sorted(fields - set(evts[ename]["fields"]))
        for ename, fields in emitted.items()
        if fields - set(evts[ename]["fields"])
    }
    assert not undeclared, (
        f"event fields emitted by the server but absent from the wire "
        f"contract registry: {undeclared}"
    )


def test_dispatcher_command_fields_are_declared():
    # Every field the ws_adapter dispatcher reads off an inbound command must
    # be declared in that command's CommandSpec — so the dispatcher can't grow
    # a dependency on a wire field the contract (and a contract-built UI / MCP)
    # doesn't know to send. Mirror of the event-field guard, command side.
    cmds = command_catalog()
    read = _dispatcher_command_fields()
    assert read, "no `mtype == ...` arms parsed — dispatcher shape drifted?"
    assert set(read) <= set(COMMAND_NAMES), (
        f"dispatcher arms for unregistered commands: "
        f"{sorted(set(read) - set(COMMAND_NAMES))}"
    )
    undeclared = {
        c: sorted(f - set(cmds[c]["fields"]))
        for c, f in read.items()
        if c in cmds and f - set(cmds[c]["fields"])
    }
    assert not undeclared, (
        f"command fields read by the dispatcher but absent from the wire "
        f"contract registry: {undeclared}"
    )


def test_mcp_command_fields_are_declared():
    # The MCP tools are the third hand copy of the command vocabulary. Every
    # field they put on a `{"type": "<cmd>", ...}` envelope must be declared in
    # that command's CommandSpec, so the agent surface can't drift either.
    cmds = command_catalog()
    built = _mcp_command_dicts()
    assert built, "no MCP command dict literals found — parse drifted?"
    # Self-check: the subscript-assignment pass must be seeing the fields the
    # tools add incrementally (set_prompt does `msg["tags_b"] = ...`). If this
    # fails, the AST helper went blind to that shape and the guard is hollow.
    assert "tags_b" in built.get("prompt", set()), (
        "MCP subscript-assigned fields not detected — _mcp_command_dicts "
        "pass 2 drifted from mcp_server.py's envelope-building style"
    )
    undeclared = {
        c: sorted(f - set(cmds[c]["fields"]))
        for c, f in built.items()
        if f - set(cmds[c]["fields"])
    }
    assert not undeclared, (
        f"command fields built by the MCP tools but absent from the wire "
        f"contract registry: {undeclared}"
    )


def test_coerce_command_payload_coerces_and_passes_through():
    # Declared scalar coerced to its type; unknown field rides through; the
    # free-form params.raw knob dict is type-checked but not recursed into.
    clean, errors = coerce_command_payload(
        "params",
        {"type": "params", "raw": {"denoise": 0.4}, "playback_pos": "1.2",
         "extra": 7},
    )
    assert errors == []
    assert clean["raw"] == {"denoise": 0.4}
    assert clean["playback_pos"] == 1.2 and isinstance(clean["playback_pos"], float)
    assert clean["extra"] == 7  # unknown field passes through verbatim


def test_coerce_command_payload_required_and_enum_and_unknown():
    _, missing = coerce_command_payload("prompt", {"type": "prompt"})
    assert any("tags" in e and "required" in e for e in missing)

    _, bad_enum = coerce_command_payload(
        "set_interp_method", {"path": "bogus", "method": "slerp"},
    )
    assert any("path" in e for e in bad_enum)

    clean, ok = coerce_command_payload(
        "set_interp_method", {"path": "prompt", "method": "slerp"},
    )
    assert ok == [] and clean["path"] == "prompt"

    _, unknown = coerce_command_payload("does_not_exist", {})
    assert any("unknown command" in e for e in unknown)


def test_coerce_command_payload_drops_bad_type():
    # A bool-typed field given a non-bool is dropped from clean AND recorded,
    # so a malformed value can't reach the session as a truthy string.
    clean, errors = coerce_command_payload(
        "swap_source", {"type": "swap_source", "use_server_source": "yes"},
    )
    assert "use_server_source" not in clean
    assert any("use_server_source" in e for e in errors)
    # ...but JSON 0/1 is tolerated as a bool.
    clean, errors = coerce_command_payload(
        "swap_source", {"type": "swap_source", "use_server_source": 1},
    )
    assert clean["use_server_source"] is True and errors == []


def test_coerce_command_payload_dispatcher_equivalence():
    # The ws_adapter dispatcher now coerces every inbound envelope before its
    # arms run. These pin the behavior-preserving cases the live client relies
    # on, so the runtime adoption can't drift from the old arm semantics.

    # loop_band clear: the client sends explicit JSON nulls. nullable=True
    # passes them through error-free, so set_loop_band(None, None) still fires
    # (and no spurious warning logs at every loop clear).
    clean, errors = coerce_command_payload(
        "loop_band",
        {"type": "loop_band", "start_sec": None, "end_sec": None},
    )
    assert errors == []
    assert clean["start_sec"] is None and clean["end_sec"] is None

    # params with a malformed playback_pos: dropped, so the arm's
    # data.get("playback_pos", 0.0) default applies — same as the old inline
    # try/float/except. raw must survive untouched.
    clean, errors = coerce_command_payload(
        "params", {"type": "params", "raw": {"denoise": 0.2},
                   "playback_pos": "abc"},
    )
    assert "playback_pos" not in clean and clean["raw"] == {"denoise": 0.2}
    assert any("playback_pos" in e for e in errors)

    # prompt without tags: the field stays absent from clean, preserving the
    # arm's data["tags"] KeyError -> ws_dispatch_error -> message-ignored path.
    clean, _ = coerce_command_payload("prompt", {"type": "prompt"})
    assert "tags" not in clean

    # set_depth tolerance: numeric strings now coerce (round-to-int) instead
    # of being skipped — clamp-style, consistent with the knob channel.
    clean, _ = coerce_command_payload(
        "set_depth", {"type": "set_depth", "value": "3.5"},
    )
    assert clean["value"] == 4

    # enable_lora with a junk strength: dropped -> arm passes strength=None,
    # exactly the old except-branch behavior.
    clean, _ = coerce_command_payload(
        "enable_lora", {"type": "enable_lora", "id": "x", "strength": "junk"},
    )
    assert clean["id"] == "x" and "strength" not in clean


def test_config_payload_matches_dataclass():
    # config_catalog() is DERIVED from SessionConfig via get_type_hints; this
    # pins that derivation so a new config field can't slip into the
    # session-init contract untyped — which would also desync the generated TS
    # SessionConfigPayload. A field with an annotation the projection can't
    # handle raises TypeError from config_catalog itself (covered below).
    from dataclasses import fields as dc_fields
    from typing import get_type_hints

    import pytest

    from acestep.streaming.config import SessionConfig
    from demos.realtime_motion_graph_web.protocol import _project_config_type

    cat = config_catalog()
    assert set(cat) == {f.name for f in dc_fields(SessionConfig)}

    wire_vocab = {"bool", "int", "float", "str", "list", "dict"}
    hints = get_type_hints(SessionConfig)
    for name, entry in cat.items():
        assert entry["type"] in wire_vocab, (name, entry)
        # Optional fields project to nullable; plain fields must not.
        is_optional = hints[name] != type(None) and type(None) in getattr(
            hints[name], "__args__", (),
        )
        assert entry.get("nullable", False) == is_optional, (name, entry)
    # The str | None family really is marked nullable.
    assert cat["prompt_b"]["nullable"] is True
    assert cat["fixture_name"]["nullable"] is True
    assert "nullable" not in cat["sde"]

    # An annotation with no wire projection fails loudly, not as "str".
    with pytest.raises(TypeError):
        _project_config_type(complex)
    with pytest.raises(TypeError):
        _project_config_type(int | str)  # non-Optional union


def test_handshake_upload_is_registered_and_handled():
    # The init-phase upload sub-protocol is dispatched via
    # config_dict.get("type") == "upload_track", NOT the mtype chain, so the
    # streaming command guard can't see it. Pin it here: the registered
    # handshake command must be matched in the adapter, and its header fields
    # must be declared.
    hs = handshake_contract()
    assert set(hs["commands"]) == set(HANDSHAKE_COMMAND_NAMES)
    assert set(hs["events"]) == set(HANDSHAKE_EVENT_NAMES)

    src = (_DEMO / "ws_adapter.py").read_text(encoding="utf-8")
    for cmd in HANDSHAKE_COMMAND_NAMES:
        assert f'"{cmd}"' in src, (
            f"handshake command {cmd!r} registered but never matched in "
            f"ws_adapter.py"
        )

    read = _upload_handler_header_fields()
    declared = set(hs["commands"]["upload_track"]["fields"])
    assert read, "no header reads parsed from _handle_upload_track"
    assert read <= declared, (
        f"upload_track header fields read by the adapter but absent from the "
        f"handshake contract: {sorted(read - declared)}"
    )


def _load_codegen_module():
    import importlib.util

    path = _DEMO / "scripts" / "gen_wire_types.py"
    spec = importlib.util.spec_from_file_location("gen_wire_types", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_wire_types_match_contract():
    # The committed web/types/wireContract.gen.ts must be a byte-for-byte
    # projection of the live contract. Regenerate with
    # `python demos/realtime_motion_graph_web/scripts/gen_wire_types.py`.
    from acestep.streaming.knobs import KNOB_SCHEMA_VERSION

    gen = _load_codegen_module()
    expected = gen.render_wire_types_ts(wire_contract(), KNOB_SCHEMA_VERSION)
    committed = (
        _DEMO / "web" / "sdk" / "types" / "wireContract.gen.ts"
    ).read_text(encoding="utf-8")
    # Normalize EOLs: git autocrlf may rewrite the committed file on checkout.
    assert expected.replace("\r\n", "\n") == committed.replace("\r\n", "\n"), (
        "web/types/wireContract.gen.ts is stale — regenerate with "
        "`python demos/realtime_motion_graph_web/scripts/gen_wire_types.py`"
    )


def test_catalog_projection_shapes():
    wc = wire_contract()
    assert wc["version"] >= 1
    assert set(wc["commands"]) == set(COMMAND_NAMES)
    assert set(wc["events"]) == set(EVENT_NAMES)

    cmds = command_catalog()
    # required-field flag and binary framing survive the projection
    assert cmds["prompt"]["fields"]["tags"]["required"] is True
    assert cmds["set_timbre_source"]["binary"] is True
    assert cmds["swap_source"]["binary_optional"] is True
    assert cmds["params"]["origin_sensitive"] is True

    evts = event_catalog()
    assert evts["ready"]["binary_follow"] is True
    assert evts["timbre_cleared"]["fields"] == {}


def _session_capability_gates() -> dict:
    """Map ``command name -> capability`` from the
    ``@requires_capability("<cap>", "<command>")`` decorators on
    StreamingSession's operations. Source-parsed (session.py imports
    torch); the registry's ``requires`` tags must match this mapping
    exactly, so a tagged command can't ship without a session-side
    gate and a gate can't exist for an untagged command."""
    src = (
        Path(__file__).resolve().parents[2]
        / "acestep" / "streaming" / "session.py"
    ).read_text(encoding="utf-8")
    out: dict = {}
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Name)
                    and dec.func.id == "requires_capability"
                    and len(dec.args) == 2
                    and all(isinstance(a, ast.Constant) for a in dec.args)):
                cap, cmd = dec.args[0].value, dec.args[1].value
                assert cmd not in out, f"duplicate gate for {cmd!r}"
                out[cmd] = cap
    return out


def test_requires_tags_have_defined_failure_behavior():
    # Plan §3.4: every `requires`-tagged command must have defined
    # failure behavior. Concretely: (a) the tag names a real
    # Capabilities field; (b) the typed command_failed event is
    # registered with the fields the session's CommandFailed event
    # carries; (c) the session gates exactly the tagged command set via
    # @requires_capability, mapping each command to the same capability
    # the registry declares.
    from dataclasses import fields as dc_fields

    from acestep.streaming.generator_backend import Capabilities

    cap_names = {f.name for f in dc_fields(Capabilities)}
    tagged = {
        name: spec["requires"]
        for name, spec in command_catalog().items()
        if "requires" in spec
    }
    assert tagged, "no requires-tagged commands — registry drifted?"
    bogus = {c: r for c, r in tagged.items() if r not in cap_names}
    assert not bogus, f"requires tags naming unknown capabilities: {bogus}"

    assert "command_failed" in EVENT_NAMES
    cf_fields = set(event_catalog()["command_failed"]["fields"])
    assert cf_fields == {"command", "requires", "error"}

    gates = _session_capability_gates()
    assert gates == tagged, (
        f"session capability gates and registry requires tags drifted:\n"
        f"  gated but untagged: "
        f"{sorted(set(gates) - set(tagged))}\n"
        f"  tagged but ungated: "
        f"{sorted(set(tagged) - set(gates))}\n"
        f"  capability mismatches: "
        f"{ {c: (gates[c], tagged[c]) for c in set(gates) & set(tagged) if gates[c] != tagged[c]} }"
    )


def test_origin_sensitive_matches_session_semantics():
    # origin_sensitive means "an EXTERNAL (MCP) send is echoed via echo_event
    # instead of applied". StreamingSession implements that for exactly two
    # verbs (set_knobs -> ParamsEcho, set_prompt_blend -> PromptBlendEcho);
    # every other command applies identically for both origins. Pin the
    # registry to those semantics so the contract can't drift back to
    # flagging commands the session doesn't actually gate.
    cmds = command_catalog()
    flagged = {n for n, c in cmds.items() if c["origin_sensitive"]}
    assert flagged == {"params", "set_prompt_blend"}
    assert cmds["params"]["echo_event"] == "params_echo"
    assert cmds["set_prompt_blend"]["echo_event"] == "prompt_blend_echo"
    # The echo events are themselves registered, and non-gated commands
    # carry no echo_event.
    assert {"params_echo", "prompt_blend_echo"} <= set(EVENT_NAMES)
    for name, c in cmds.items():
        if name not in flagged:
            assert "echo_event" not in c, name
