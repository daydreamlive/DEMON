"""/api/server-info advertises the selected model extension's stacked-stream
layout (read from the extension module's ``STEMS`` / ``N_STREAMS`` by its
qualified id), and null on the stock path, with no session."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

from demos.realtime_motion_graph_web import server


def _server_info():
    response = server._process_request(
        SimpleNamespace(remote_address=("127.0.0.1", 0)),
        SimpleNamespace(headers={}, path="/api/server-info"),
    )
    return json.loads(response.body)


def _fake_module(monkeypatch, name="fake_stream_ext", **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return name


def test_null_without_extension(monkeypatch):
    monkeypatch.setattr(server, "_STREAM_LAYOUT", server._stream_layout_info(None))
    assert _server_info()["stream_layout"] is None
    assert server._stream_layout_info("") is None


def test_null_when_module_declares_no_layout(monkeypatch):
    qid = _fake_module(monkeypatch)
    monkeypatch.setattr(server, "_STREAM_LAYOUT", server._stream_layout_info(qid))
    assert _server_info()["stream_layout"] is None


def test_null_when_module_missing():
    assert server._stream_layout_info("no_such_stream_ext_module_xyz") is None


def test_layout_from_extension_module(monkeypatch):
    qid = _fake_module(monkeypatch, STEMS=("drums", "bass", "keys"), N_STREAMS=3)
    monkeypatch.setattr(server, "_STREAM_LAYOUT", server._stream_layout_info(qid))
    assert _server_info()["stream_layout"] == {
        "n_streams": 3, "stems": ["drums", "bass", "keys"],
    }


def test_n_streams_falls_back_to_stem_count(monkeypatch):
    qid = _fake_module(monkeypatch, STEMS=("drums", "bass"))
    assert server._stream_layout_info(qid) == {"n_streams": 2, "stems": ["drums", "bass"]}
