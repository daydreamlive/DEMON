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


# ---- checkpoint allow-list on server-info and /api/loras ----

from acestep.streaming.families import resolve_served


def _serve(monkeypatch, names, *, no_backend=False):
    served = resolve_served(names)
    monkeypatch.setattr(server, "_SERVED_CHECKPOINTS", served)
    monkeypatch.setattr(server, "_CHECKPOINT", served[0].model_id)
    monkeypatch.setattr(server, "_CHECKPOINT_NAME", served[0].name)
    monkeypatch.setattr(server, "_BACKEND_FAMILY", served[0].family)
    monkeypatch.setattr(server, "_NO_BACKEND", no_backend)


def test_server_info_lists_served_checkpoints(monkeypatch):
    _serve(monkeypatch, ["sa3-medium", "sa3-small"])
    info = _server_info()
    assert info["checkpoint"] == "sa3-medium"
    assert info["checkpoint_scale"] is None
    assert info["checkpoints"] == [
        {"name": "sa3-medium", "family": "sa3", "model_id": "medium", "scale": None, "default": True},
        {"name": "sa3-small", "family": "sa3", "model_id": "small-music", "scale": None, "default": False},
    ]


def test_server_info_single_value_keeps_existing_keys(monkeypatch):
    _serve(monkeypatch, ["xl"])
    info = _server_info()
    for key in ("no_backend", "kiosk", "default_mode", "warmup",
                "server_side_fixtures", "stream_layout", "model_extension"):
        assert key in info
    assert info["checkpoint"] == "xl"
    assert info["checkpoint_scale"] == "5B"
    assert [c["model_id"] for c in info["checkpoints"]] == ["acestep-v15-xl-turbo"]


def test_server_info_no_backend_lists_nothing(monkeypatch):
    _serve(monkeypatch, ["acestep-v15-turbo"], no_backend=True)
    info = _server_info()
    assert info["checkpoints"] == []
    assert info["checkpoint"] == "acestep-v15-turbo"


def _loras(monkeypatch, query=""):
    import acestep.paths as paths

    monkeypatch.setattr(paths, "discover_all_loras", lambda: [])
    response = server._process_request(
        SimpleNamespace(remote_address=("127.0.0.1", 0)),
        SimpleNamespace(headers={}, path="/api/loras" + query),
    )
    return response.status_code, json.loads(response.body)


def test_loras_default_checkpoint_unchanged(monkeypatch):
    _serve(monkeypatch, ["acestep-v15-turbo", "xl"])
    status, body = _loras(monkeypatch)
    assert status == 200
    assert body["checkpoint"] == "acestep-v15-turbo"
    assert body["checkpoint_scale"] == "2B"


def test_loras_query_selects_served_model(monkeypatch):
    _serve(monkeypatch, ["acestep-v15-turbo", "xl"])
    for q in ("?checkpoint=xl", "?checkpoint=acestep-v15-xl-turbo"):
        status, body = _loras(monkeypatch, q)
        assert status == 200
        assert body["checkpoint"] == "acestep-v15-xl-turbo"
        assert body["checkpoint_scale"] == "5B"


def test_loras_query_unknown_is_400(monkeypatch):
    _serve(monkeypatch, ["acestep-v15-turbo"])
    status, body = _loras(monkeypatch, "?checkpoint=xl")
    assert status == 400
    assert body["code"] == "checkpoint_unavailable"
    assert body["message"] == (
        "checkpoint 'xl' is not served by this pod; served: ['acestep-v15-turbo']"
    )
