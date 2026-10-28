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


# ---- model_extension: the selected extension's server_info() at boot ----


class _InfoExt:
    def server_info(self):
        return {"checkpoints": ["fs1_s012000", "s064000"]}


class _RaisingExt:
    def server_info(self):
        raise RuntimeError("boom")


def test_model_extension_info_published(monkeypatch):
    monkeypatch.setattr(server, "_MODEL_EXTENSION_INFO", server._model_extension_info(_InfoExt()))
    assert _server_info()["model_extension"] == {"checkpoints": ["fs1_s012000", "s064000"]}


def test_model_extension_info_null_without_hook(monkeypatch):
    from acestep.plugins.model_extensions import ModelExtension

    assert server._model_extension_info(ModelExtension()) is None
    assert server._model_extension_info(object()) is None
    monkeypatch.setattr(server, "_MODEL_EXTENSION_INFO", server._model_extension_info(None))
    assert _server_info()["model_extension"] is None


def test_model_extension_info_exception_is_null(monkeypatch):
    monkeypatch.setattr(server, "_MODEL_EXTENSION_INFO", server._model_extension_info(_RaisingExt()))
    info = _server_info()
    assert info["model_extension"] is None
    assert "stream_layout" in info


def test_model_extension_info_non_dict_is_null():
    class _ListExt:
        def server_info(self):
            return ["fs1_s012000"]

    assert server._model_extension_info(_ListExt()) is None
