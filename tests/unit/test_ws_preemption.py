"""Single-active-session preemption + the 4001 close-code contract.

Covers ``_preempt_active_session`` (stop the old runner, close its
socket with PREEMPTED_CLOSE_CODE, wait for teardown, clear the slot),
the late ``stem_assets``/``stem_failed`` push to the live session, and
a cross-language drift guard pinning the 4001 constant between
ws_adapter.py and the web SDK (which treats it as a FINAL close — the
guard keeps a value drift from silently re-enabling the reconnect
ping-pong the code exists to prevent).
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import demos.realtime_motion_graph_web.ws_adapter as wa
from acestep.streaming.events import StemAssets, StemFailed

_DEMO = Path(wa.__file__).resolve().parent
_REPO = _DEMO.parent.parent
_SDK = _REPO / "packages" / "demon-client"


# ---------------------------------------------------------------------------
# Cross-language constant guard
# ---------------------------------------------------------------------------


def test_preempted_close_code_matches_web_sdk():
    ts_src = (_SDK / "types" / "protocol.ts").read_text(
        encoding="utf-8",
    )
    match = re.search(
        r"export const PREEMPTED_CLOSE_CODE\s*=\s*(\d+)", ts_src,
    )
    assert match, "PREEMPTED_CLOSE_CODE missing from packages/demon-client/types/protocol.ts"
    assert int(match.group(1)) == wa.PREEMPTED_CLOSE_CODE


def test_preempted_close_code_is_an_application_code():
    # 4000-4999 is the private-use range; anything else collides with
    # registered WebSocket close semantics.
    assert 4000 <= wa.PREEMPTED_CLOSE_CODE <= 4999


def test_web_client_handles_preempted_close():
    # Both client surfaces must reference the shared constant: the
    # reconnect loop (useStartSession) and the pre-ready error mapper
    # (packages/demon-client/protocol.ts).
    for path in (
        _DEMO / "web" / "hooks" / "useStartSession.ts",
        _SDK / "protocol.ts",
    ):
        src = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO)
        assert "PREEMPTED_CLOSE_CODE" in src, f"{rel} ignores preemption"


# ---------------------------------------------------------------------------
# _preempt_active_session
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_active_slot():
    with wa._ACTIVE_SLOT_LOCK:
        prev = wa._ACTIVE_SESSION[0]
        wa._ACTIVE_SESSION[0] = None
    yield
    with wa._ACTIVE_SLOT_LOCK:
        wa._ACTIVE_SESSION[0] = prev


class _FakeWs:
    def __init__(self, raise_on_close=False):
        self.closes: list[tuple[int, str]] = []
        self._raise = raise_on_close

    def close(self, code, reason=""):
        self.closes.append((code, reason))
        if self._raise:
            raise RuntimeError("socket already gone")


def _fake_streaming(closed: bool):
    streaming = SimpleNamespace(
        state=SimpleNamespace(running=True),
        closed=threading.Event(),
    )
    if closed:
        streaming.closed.set()
    return streaming


def _register(session_id="old", *, closed=True, raise_on_close=False):
    ws = _FakeWs(raise_on_close=raise_on_close)
    streaming = _fake_streaming(closed)
    with wa._ACTIVE_SLOT_LOCK:
        wa._ACTIVE_SESSION[0] = wa._ActiveSession(session_id, streaming, ws)
    return streaming, ws


def test_preempt_noop_without_active_session():
    wa._preempt_active_session("new")  # must not raise
    with wa._ACTIVE_SLOT_LOCK:
        assert wa._ACTIVE_SESSION[0] is None


def test_preempt_stops_runner_closes_with_4001_and_clears_slot():
    streaming, ws = _register(closed=True)
    wa._preempt_active_session("new")
    assert streaming.state.running is False
    assert ws.closes == [(wa.PREEMPTED_CLOSE_CODE, "preempted by a newer session")]
    with wa._ACTIVE_SLOT_LOCK:
        assert wa._ACTIVE_SESSION[0] is None


def test_preempt_waits_for_teardown_signal():
    streaming, _ = _register(closed=False)
    released = []

    def _close_later():
        streaming.closed.set()
        released.append(True)

    timer = threading.Timer(0.05, _close_later)
    timer.start()
    try:
        wa._preempt_active_session("new")
    finally:
        timer.cancel()
    assert released == [True]  # preempt returned only after the signal
    with wa._ACTIVE_SLOT_LOCK:
        assert wa._ACTIVE_SESSION[0] is None


def test_preempt_proceeds_after_teardown_timeout(monkeypatch):
    monkeypatch.setattr(wa, "_PREEMPT_TEARDOWN_TIMEOUT_S", 0.05)
    streaming, ws = _register(closed=False)
    wa._preempt_active_session("new")
    # Bounded wait, then create proceeds anyway: runner stopped, socket
    # closed, slot cleared.
    assert streaming.state.running is False
    assert ws.closes[0][0] == wa.PREEMPTED_CLOSE_CODE
    with wa._ACTIVE_SLOT_LOCK:
        assert wa._ACTIVE_SESSION[0] is None


def test_preempt_survives_socket_close_failure():
    streaming, _ = _register(closed=True, raise_on_close=True)
    wa._preempt_active_session("new")  # close() raising must be swallowed
    assert streaming.state.running is False
    with wa._ACTIVE_SLOT_LOCK:
        assert wa._ACTIVE_SESSION[0] is None


# ---------------------------------------------------------------------------
# Late stem push to the live session
# ---------------------------------------------------------------------------


def _register_with_bus():
    events = []
    streaming = SimpleNamespace(bus=SimpleNamespace(publish=events.append))
    with wa._ACTIVE_SLOT_LOCK:
        wa._ACTIVE_SESSION[0] = wa._ActiveSession("live", streaming, _FakeWs())
    return events


def test_publish_stems_noop_without_active_session():
    stems = {"vocals": torch.zeros((2, 8))}
    assert wa._publish_stems_to_active_session("track.wav", stems) is False


def test_publish_stems_pushes_overlay_only_stem_assets():
    events = _register_with_bus()
    stems = {
        "vocals": torch.zeros((2, 480)),
        "instruments": torch.zeros((2, 480)),
    }
    assert wa._publish_stems_to_active_session("track.wav", stems) is True
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, StemAssets)
    assert event.fixture_name == "track.wav"
    # Empty source_mode = overlay-only push, never a mode change.
    assert event.source_mode == ""
    assert event.channels == 2
    assert event.frames == 480
    assert event.stems is stems


def test_publish_stems_failure_pushes_stem_failed():
    events = _register_with_bus()
    assert wa._publish_stems_to_active_session(
        "track.wav", None, error="separator exploded",
    ) is True
    event = events[0]
    assert isinstance(event, StemFailed)
    assert event.fixture_name == "track.wav"
    assert event.error == "separator exploded"


def test_publish_stems_swallows_bus_errors():
    streaming = SimpleNamespace(
        bus=SimpleNamespace(
            publish=lambda e: (_ for _ in ()).throw(RuntimeError("bus closed")),
        ),
    )
    with wa._ACTIVE_SLOT_LOCK:
        wa._ACTIVE_SESSION[0] = wa._ActiveSession("live", streaming, _FakeWs())
    stems = {"vocals": torch.zeros((2, 8))}
    assert wa._publish_stems_to_active_session("track.wav", stems) is False
