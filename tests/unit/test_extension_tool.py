"""extension_tool: a model-extension tool call from the page or the control
bus, through the session and the SA3 backend to ``runtime.call_tool``, and
its tool_result back to the caller. Tier 1, no GPU.

The ws_adapter dispatcher is a closure inside the per-connection handler,
so its extension_tool arm is pinned two ways: the module-level helper it
calls is driven directly from both origins, and the closure's wiring (the
arm, the reply choice, the control drain passing its reply) is checked on
the source, the way test_wire_contract checks the rest of the dispatcher.
The control bus is driven for real: an HTTP server on an ephemeral port,
a registered session whose inject answers on another thread, and the MCP
tool posting to it.
"""

from __future__ import annotations

import ast
import json
import queue
import struct
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

import pytest
import torch

from acestep.plugins import ModelExtensionRuntime, ModelExtensionSpec, ToolSpec
from acestep.plugins.api import LoadedPlugin, PluginManifest
from acestep.plugins.selection import select_model_extension
from acestep.streaming import registry as session_registry
from acestep.streaming.commands import CommandOrigin
from acestep.streaming.session import StreamingSession
from demos.realtime_motion_graph_web import control_http
from demos.realtime_motion_graph_web.protocol import (
    COMMAND_NAMES,
    EVENT_NAMES,
    coerce_command_payload,
    command_catalog,
    event_catalog,
)
from demos.realtime_motion_graph_web.ws_adapter import _dispatch_extension_tool
from tests.unit.test_sa3_extension_integration import (
    PLUGIN_ID,
    _backend,
    _ProbeExtension,
    _ProbeRuntime,
)

_DEMO = Path(__file__).resolve().parents[2] / "demos" / "realtime_motion_graph_web"

TOOL = "dynamics_curve"
ARGS = {"stem": "guitar", "bpm": 90, "points": [{"bar": 4, "db": -24},
                                               {"bar": 8, "db": 0}]}


# ---- the wire contract (pinned in runbook 19h) ------------------------------


def test_command_and_event_are_registered_with_the_pinned_shape():
    assert "extension_tool" in COMMAND_NAMES
    fields = command_catalog()["extension_tool"]["fields"]
    assert {k: (v["type"], v["required"]) for k, v in fields.items()} == {
        "name": ("str", True), "args": ("dict", True), "token": ("str", True),
    }
    assert not command_catalog()["extension_tool"]["origin_sensitive"]
    assert "tool_result" in EVENT_NAMES
    ev = event_catalog()["tool_result"]["fields"]
    assert set(ev) == {"token", "name", "ok", "message", "state"}
    assert ev["ok"]["type"] == "bool" and ev["state"]["nullable"]


def test_coercion_keeps_a_good_call_and_flags_a_bad_one():
    good = {"type": "extension_tool", "name": TOOL, "args": ARGS, "token": "t1"}
    clean, errors = coerce_command_payload("extension_tool", good)
    assert errors == [] and clean == good
    clean, errors = coerce_command_payload(
        "extension_tool", {"type": "extension_tool", "name": TOOL, "args": [1]},
    )
    assert "args" not in clean
    assert any("token" in e for e in errors) and any("args" in e for e in errors)


# ---- the dispatcher arm, from both origins ---------------------------------


class _Streaming:
    def __init__(self, result=None, raises=None):
        self.calls = []
        self.result = result or {"ok": True, "message": "applied",
                                 "state": {"points": 2}}
        self.raises = raises

    def call_extension_tool(self, name, args, *, origin):
        self.calls.append((name, args, origin))
        if self.raises:
            raise self.raises
        return self.result


@pytest.mark.parametrize("origin", [CommandOrigin.PRIMARY, CommandOrigin.EXTERNAL])
def test_tool_reaches_the_session_and_the_reply_carries_the_token(origin):
    streaming, sent = _Streaming(), []
    _dispatch_extension_tool(
        {"type": "extension_tool", "name": TOOL, "args": ARGS, "token": "abc"},
        streaming, sent.append, origin,
    )
    assert streaming.calls == [(TOOL, ARGS, origin)]
    assert sent == [{"type": "tool_result", "token": "abc", "name": TOOL,
                     "ok": True, "message": "applied", "state": {"points": 2}}]


@pytest.mark.parametrize("data, message", [
    ({"args": {}, "token": "t"}, "needs a tool name"),
    ({"name": TOOL, "token": "t"}, "args must be an object"),
    ({"name": TOOL, "args": [1], "token": "t"}, "args must be an object"),
])
def test_malformed_call_answers_ok_false_without_the_session(data, message):
    streaming, sent = _Streaming(), []
    _dispatch_extension_tool(
        {"type": "extension_tool", **data}, streaming, sent.append,
        CommandOrigin.PRIMARY,
    )
    assert streaming.calls == []
    assert len(sent) == 1 and sent[0]["ok"] is False
    assert message in sent[0]["message"] and sent[0]["token"] == "t"
    assert sent[0]["state"] is None


def test_session_error_still_answers_once():
    sent = []
    _dispatch_extension_tool(
        {"name": TOOL, "args": {}, "token": "t"},
        _Streaming(raises=RuntimeError("boom")), sent.append,
        CommandOrigin.EXTERNAL,
    )
    assert [(m["ok"], "boom" in m["message"]) for m in sent] == [(False, True)]


def _fn(tree, name):
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def test_dispatcher_wires_both_origins_to_the_helper_and_the_caller():
    tree = ast.parse((_DEMO / "ws_adapter.py").read_text(encoding="utf-8"))
    dispatch = _fn(tree, "_dispatch_message")
    assert [a.arg for a in dispatch.args.args] == [
        "data", "recv_audio", "source", "reply",
    ]
    arm = next(
        n for n in ast.walk(dispatch)
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name) and n.test.left.id == "mtype"
        and getattr(n.test.comparators[0], "value", None) == "extension_tool"
    )
    call = next(n for n in ast.walk(ast.Module(body=arm.body, type_ignores=[]))
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") == "_dispatch_extension_tool")
    assert ast.unparse(call.args[2]) == (
        "reply if reply is not None else _send_json"
    )
    assert ast.unparse(call.args[3]) == "origin"
    # The control drain hands each injected command's reply to the router.
    recv = _fn(tree, "recv_loop")
    control_calls = [
        n for n in ast.walk(recv)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_dispatch_message"
        and any(getattr(a, "value", None) == "control" for a in n.args)
    ]
    assert len(control_calls) == 1
    assert {k.arg: ast.unparse(k.value) for k in control_calls[0].keywords} == {
        "reply": "creply",
    }
    inject = _fn(tree, "inject_control")
    assert [a.arg for a in inject.args.args] == ["data", "audio", "reply"]


# ---- the session --------------------------------------------------------------


def _session(backend):
    return types.SimpleNamespace(
        state=types.SimpleNamespace(last_activity_ts=0.0), backend=backend,
    )


def test_session_passes_the_call_to_the_backend_from_either_origin():
    seen = []

    class _B:
        name = "sa3"

        def call_extension_tool(self, name, args):
            seen.append((name, args))
            return {"ok": True, "message": "ok", "state": None}

    for origin in (CommandOrigin.PRIMARY, CommandOrigin.EXTERNAL):
        sess = _session(_B())
        out = StreamingSession.call_extension_tool(sess, TOOL, ARGS, origin=origin)
        assert out["ok"] is True
        assert sess.state.last_activity_ts > 0
    assert seen == [(TOOL, ARGS), (TOOL, ARGS)]


def test_session_on_a_backend_without_tools_refuses():
    out = StreamingSession.call_extension_tool(
        _session(types.SimpleNamespace(name="ace")), TOOL, {},
    )
    assert out == {"ok": False, "message": "backend 'ace' has no extension tools",
                   "state": None}


# ---- the backend passthrough ------------------------------------------------


class _ToolRuntime(_ProbeRuntime):
    def __init__(self, result=None, raises=None):
        super().__init__()
        self.tool_calls = []
        self.result = result
        self.raises = raises

    def call_tool(self, name, args):
        self.tool_calls.append((name, dict(args)))
        if self.raises:
            raise self.raises
        return self.result or {"ok": True, "message": f"{name} applied",
                               "state": {"digest": "d1"}}


def _selected_with_tools():
    spec = ModelExtensionSpec(
        name="probe", family="sa3", create=_ProbeExtension,
        tool_specs=(ToolSpec(TOOL, {"type": "object"}, "Set a level curve."),),
    )
    plugins = {PLUGIN_ID: LoadedPlugin(
        manifest=PluginManifest(id=PLUGIN_ID, version="0.1.0", plugin_api=1),
        model_extensions={"probe": spec},
    )}
    return select_model_extension(f"{PLUGIN_ID}.probe", family="sa3",
                                  plugins=plugins)


def test_backend_hands_a_registered_tool_to_the_runtime():
    runtime = _ToolRuntime()
    b = _backend(extension=_selected_with_tools(), runtime=runtime)
    out = b.call_extension_tool(TOOL, ARGS)
    assert runtime.tool_calls == [(TOOL, ARGS)]
    assert out == {"ok": True, "message": f"{TOOL} applied",
                   "state": {"digest": "d1"}}


def test_backend_refuses_an_unknown_tool_without_calling_the_runtime():
    runtime = _ToolRuntime()
    b = _backend(extension=_selected_with_tools(), runtime=runtime)
    out = b.call_extension_tool("not_a_tool", {})
    assert out["ok"] is False and "unknown tool 'not_a_tool'" in out["message"]
    assert TOOL in out["message"] and out["state"] is None
    assert runtime.tool_calls == []


def test_backend_refuses_without_a_runtime_or_without_call_tool():
    stock = _backend()
    assert stock.call_extension_tool(TOOL, {})["ok"] is False

    class _NoTools:
        def apply_controls(self, values):
            pass

        def decorate_conditioning(self, bundle, source=None):
            return bundle

    b = _backend(extension=_selected_with_tools(), runtime=_NoTools())
    out = b.call_extension_tool(TOOL, {})
    assert out == {"ok": False,
                   "message": "this session's model has no extension tools",
                   "state": None}


def test_backend_turns_a_raise_or_a_bad_result_into_ok_false():
    sel = _selected_with_tools()
    out = _backend(extension=sel, runtime=_ToolRuntime(
        raises=ValueError("point before bar 0"),
    )).call_extension_tool(TOOL, {})
    assert out["ok"] is False and "point before bar 0" in out["message"]

    out = _backend(extension=sel, runtime=_ToolRuntime(
        result={"ok": True, "message": "m", "state": {"t": torch.zeros(1)}},
    )).call_extension_tool(TOOL, {})
    assert out == {"ok": True, "message": "m", "state": None}

    class _Odd(_ToolRuntime):
        def call_tool(self, name, args):
            return "yes"

    out = _backend(extension=sel, runtime=_Odd()).call_extension_tool(TOOL, {})
    assert out["ok"] is False


def test_base_runtime_without_an_override_refuses_through_the_backend():
    class _Plain(ModelExtensionRuntime):
        pass

    out = _backend(extension=_selected_with_tools(),
                   runtime=_Plain()).call_extension_tool(TOOL, {})
    assert out["ok"] is False and TOOL in out["message"]


# ---- the control bus caller gets the answer --------------------------------


class _FakeSession:
    """A registered session whose dispatch thread runs the real helper."""

    def __init__(self, sid, answer=True):
        self.sid = sid
        self.answer = answer
        self.streaming = _Streaming()
        self.injected = []
        self._q = queue.Queue()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        session_registry.register(session_registry.SessionHandle(
            id=sid, started_at=time.time(), inject=self.inject,
            snapshot=lambda: {"knob_values": {}},
        ))

    def inject(self, data, audio=None, reply=None):
        self.injected.append((data, reply is not None))
        self._q.put((data, reply))

    def _loop(self):
        while True:
            data, reply = self._q.get()
            if data is None:
                return
            if data.get("type") == "extension_tool" and self.answer:
                _dispatch_extension_tool(
                    data, self.streaming, reply, CommandOrigin.EXTERNAL,
                )

    def close(self):
        self._q.put((None, None))
        session_registry.unregister(self.sid)


@pytest.fixture
def control_server():
    srv = control_http.start_control_server("127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _post(base, sid, data, timeout=5.0):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/sessions/{sid}/cmd",
        data=struct.pack("<I", len(body)) + body, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_control_bus_returns_the_tool_result_to_its_caller(control_server):
    sess = _FakeSession("toolsA")
    try:
        status, body = _post(control_server, "toolsA", {
            "type": "extension_tool", "name": TOOL, "args": ARGS, "token": "k9",
        })
    finally:
        sess.close()
    assert status == 200
    assert body["result"] == {"type": "tool_result", "token": "k9", "name": TOOL,
                              "ok": True, "message": "applied",
                              "state": {"points": 2}}
    assert sess.streaming.calls == [(TOOL, ARGS, CommandOrigin.EXTERNAL)]


def test_control_bus_other_commands_do_not_wait(control_server):
    sess = _FakeSession("toolsB")
    try:
        status, body = _post(control_server, "toolsB", {"type": "manual_slot_add"})
    finally:
        sess.close()
    assert status == 200 and "result" not in body
    assert sess.injected == [({"type": "manual_slot_add"}, False)]


def test_control_bus_times_out_when_no_answer_comes(control_server, monkeypatch):
    monkeypatch.setattr(control_http, "_ANSWER_TIMEOUT_S", 0.2)
    sess = _FakeSession("toolsC", answer=False)
    try:
        status, body = _post(control_server, "toolsC", {
            "type": "extension_tool", "name": TOOL, "args": {}, "token": "x",
        })
    finally:
        sess.close()
    assert status == 504 and "extension_tool" in body["error"]


# ---- the MCP tool --------------------------------------------------------------


@pytest.fixture
def mcp_server(monkeypatch):
    """Import mcp_server against a stub FastMCP (this venv ships mcp 2.x,
    which renamed it); the tool functions are plain coroutines."""
    stub = types.ModuleType("mcp.server.fastmcp")

    class _FastMCP:
        def __init__(self, *a, **k):
            pass

        def tool(self, *a, **k):
            return lambda fn: fn

        def __getattr__(self, name):
            return lambda *a, **k: (lambda fn: fn)

    stub.FastMCP = _FastMCP
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", stub)
    monkeypatch.delitem(
        sys.modules, "demos.realtime_motion_graph_web.mcp_server", raising=False,
    )
    import importlib

    yield importlib.import_module("demos.realtime_motion_graph_web.mcp_server")
    # Never leave the stub-built module behind for another test to import.
    sys.modules.pop("demos.realtime_motion_graph_web.mcp_server", None)


def test_mcp_call_tool_posts_extension_tool_and_returns_the_result(
    control_server, mcp_server, monkeypatch,
):
    import asyncio

    monkeypatch.setattr(mcp_server, "CONTROL_HTTP", control_server)
    sess = _FakeSession("toolsD")
    try:
        out = asyncio.run(mcp_server.call_tool(TOOL, ARGS, session_id="toolsD"))
    finally:
        sess.close()
    assert out["type"] == "tool_result" and out["ok"] is True
    assert out["name"] == TOOL and out["token"]
    (sent, had_reply), = sess.injected
    assert sent == {"type": "extension_tool", "name": TOOL, "args": ARGS,
                    "token": out["token"]}
    assert had_reply
