"""/api/server-info advertises the loaded model extension's stacked-stream
layout (n_streams from its ``stream_layout()`` contract, stem names from
its module's ``STEMS``), and null on the stock path, with no session."""

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


def _fake_selected(monkeypatch, *, layout, stems=("drums", "bass", "keys")):
    module = types.ModuleType("fake_stream_ext")
    module.STEMS = stems

    class Runtime:
        def stream_layout(self):
            return layout

    Runtime.__module__ = module.__name__
    module.Runtime = Runtime
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return SimpleNamespace(extension=Runtime())


def test_null_without_extension(monkeypatch):
    monkeypatch.setattr(server, "_STREAM_LAYOUT", server._stream_layout_info(None))
    assert _server_info()["stream_layout"] is None


def test_null_when_runtime_has_no_layout(monkeypatch):
    selected = _fake_selected(monkeypatch, layout=None)
    monkeypatch.setattr(server, "_STREAM_LAYOUT", server._stream_layout_info(selected))
    assert _server_info()["stream_layout"] is None


def test_layout_from_extension_contract(monkeypatch):
    selected = _fake_selected(monkeypatch, layout=SimpleNamespace(n_streams=3))
    monkeypatch.setattr(server, "_STREAM_LAYOUT", server._stream_layout_info(selected))
    assert _server_info()["stream_layout"] == {
        "n_streams": 3, "stems": ["drums", "bass", "keys"],
    }
