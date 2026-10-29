"""Per-session checkpoint: the session config names one of the pod's
allow-listed checkpoints, the adapter resolves it (or refuses it before
any upload or preempt), and a model switch preempts every live session."""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import pytest

import demos.realtime_motion_graph_web.ws_adapter as wa
from acestep.streaming.config import SessionConfig
from acestep.streaming.families import ServedCheckpoint, resolve_served

SA3 = resolve_served(["sa3-medium", "sa3-small"])
ACE = resolve_served(["acestep-v15-turbo", "xl"])


def _resolve(config, served, *, default=None, family=None):
    default = default if default is not None else served[0].model_id
    family = family if family is not None else served[0].family
    return wa._resolve_session_checkpoint(
        config, default=default, backend_family=family, served=served,
    )


# ---- SessionConfig field ----


def test_config_field_parses():
    assert SessionConfig.from_dict({"checkpoint": "xl"}).checkpoint == "xl"


def test_config_field_absent_is_none():
    assert SessionConfig.from_dict({}).checkpoint is None


# ---- resolution table ----


def test_absent_is_default():
    assert _resolve({}, SA3) == ("medium", None)


def test_null_is_default():
    assert _resolve({"checkpoint": None}, SA3) == ("medium", None)


def test_by_name():
    assert _resolve({"checkpoint": "sa3-small"}, SA3) == ("small-music", None)
    assert _resolve({"checkpoint": "xl"}, ACE) == ("acestep-v15-xl-turbo", None)


def test_by_model_id_echoed_from_ready_frame():
    assert _resolve({"checkpoint": "small-music"}, SA3) == ("small-music", None)
    assert _resolve({"checkpoint": "acestep-v15-xl-turbo"}, ACE) == (
        "acestep-v15-xl-turbo", None,
    )


def test_default_by_name():
    assert _resolve({"checkpoint": "sa3-medium"}, SA3) == ("medium", None)


def test_unknown_is_refused():
    model, err = _resolve({"checkpoint": "sa3-sfx"}, SA3)
    assert model == "medium"
    assert err == (
        "checkpoint 'sa3-sfx' is not served by this pod; "
        "served: ['sa3-medium', 'sa3-small']"
    )


def test_non_string_is_refused():
    _, err = _resolve({"checkpoint": 7}, SA3)
    assert err is not None and "not served" in err
    _, err = _resolve({"checkpoint": "  "}, SA3)
    assert err is not None


def test_wrong_family_is_refused():
    # An explicit backend key naming another family than the served entry.
    _, err = _resolve({"checkpoint": "sa3-small", "backend": "acestep"}, SA3)
    assert err is not None and "family 'sa3'" in err


def test_extension_truncated_list_refuses_the_rest():
    truncated = SA3[:1]
    assert _resolve({"checkpoint": "sa3-medium"}, truncated) == ("medium", None)
    _, err = _resolve({"checkpoint": "sa3-small"}, truncated)
    assert err == (
        "checkpoint 'sa3-small' is not served by this pod; served: ['sa3-medium']"
    )


def test_empty_served_is_default_only():
    # Callers predating allow-lists (golden local_server, replay_server).
    assert wa._resolve_session_checkpoint(
        {}, default="acestep-v15-turbo", backend_family="acestep",
    ) == ("acestep-v15-turbo", None)
    assert wa._resolve_session_checkpoint(
        {"checkpoint": "acestep-v15-turbo"},
        default="acestep-v15-turbo", backend_family="acestep",
    ) == ("acestep-v15-turbo", None)
    _, err = wa._resolve_session_checkpoint(
        {"checkpoint": "xl"}, default="acestep-v15-turbo", backend_family="acestep",
    )
    assert err is not None


# ---- preempt rule ----


@pytest.fixture
def _active_slot():
    with wa._ACTIVE_SLOT_LOCK:
        prev = list(wa._ACTIVE_SESSIONS)
        wa._ACTIVE_SESSIONS.clear()
    yield wa._ACTIVE_SESSIONS
    with wa._ACTIVE_SLOT_LOCK:
        wa._ACTIVE_SESSIONS[:] = prev


def _live(model_id):
    return wa._ActiveSession("s", SimpleNamespace(checkpoint=model_id), None)


def test_preempt_keep_same_model_uses_cap(_active_slot):
    _active_slot.append(_live("small-music"))
    assert wa._preempt_keep(3, "small-music") == 2
    assert wa._preempt_keep(1, "small-music") == 0


def test_preempt_keep_empty_uses_cap(_active_slot):
    assert wa._preempt_keep(3, "medium") == 2


def test_preempt_keep_model_switch_is_total(_active_slot):
    _active_slot.append(_live("small-music"))
    assert wa._preempt_keep(3, "medium") == 0


# ---- the handler refuses before any upload or preempt ----


class _FakeWs:
    def __init__(self, config):
        self._frames = [json.dumps(config)]
        self.sent = []
        self.closed = None

    def recv(self):
        if not self._frames:
            raise AssertionError("handler read past the config frame")
        return self._frames.pop(0)

    def send(self, data):
        self.sent.append(json.loads(data))

    def close(self, code=1000, reason=""):
        self.closed = (code, reason)


def _run_body(ws, served, monkeypatch):
    monkeypatch.delenv("DEMON_WIPE_USER_UPLOADS", raising=False)
    calls = []
    monkeypatch.setattr(
        wa, "_preempt_active_session", lambda *a, **k: calls.append((a, k)),
    )
    with contextlib.ExitStack() as stack:
        wa._handle_client_body(
            ws, stack,
            decoder_backend="eager",
            vae_backend="eager",
            checkpoint=served[0].model_id,
            backend_family=served[0].family,
            offload_text_encoder=False,
            checkpoint_dir=None,
            model_extension=None,
            served_checkpoints=served,
        )
    return calls


def test_handler_refuses_unserved_checkpoint(monkeypatch):
    ws = _FakeWs({"checkpoint": "sa3-sfx", "prompt": "x"})
    preempts = _run_body(ws, SA3, monkeypatch)
    assert ws.sent == [{
        "type": "error",
        "code": "checkpoint_unavailable",
        "message": (
            "checkpoint 'sa3-sfx' is not served by this pod; "
            "served: ['sa3-medium', 'sa3-small']"
        ),
    }]
    assert ws.closed == (1008, "checkpoint unavailable")
    assert preempts == []


def test_served_checkpoint_type_is_frozen():
    entry = ServedCheckpoint(name="xl", family="acestep", model_id="acestep-v15-xl-turbo")
    with pytest.raises(Exception):
        entry.name = "other"
