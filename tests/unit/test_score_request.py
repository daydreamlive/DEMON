"""A scoring request hands the card to another tool.

Covers ``release_sa3`` (decoders dropped, then the eviction), the SA3
family's ``release``, the ``score_request`` socket (register before the
release, release at open and at close, a live session preempted and waited
for, the tool route accepting the request's id, unregister last so it fires
the reset), scoring requests taking turns (a waiter is told ``queued``,
stays out of the registry and touches nothing until the holder has
unregistered; a waiter that goes leaves the line), and ``DEMON_TOOL_YIELDS`` (POSTed before a model session is
created, nothing when unset). No GPU.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import demos.realtime_motion_graph_web.ws_adapter as wa
from acestep.engine import sa3_context
from acestep.streaming import registry
from acestep.streaming import sa3_session
from acestep.streaming.families import FAMILY_SPECS
from demos.realtime_motion_graph_web import tool_proxy


@pytest.fixture(autouse=True)
def _isolated():
    with wa._ACTIVE_SLOT_LOCK:
        prev = list(wa._ACTIVE_SESSIONS)
        wa._ACTIVE_SESSIONS.clear()
    with wa._SCORE_TURN:
        wa._SCORE_LINE.clear()
    yield
    with wa._SCORE_TURN:
        wa._SCORE_LINE.clear()
    with wa._ACTIVE_SLOT_LOCK:
        wa._ACTIVE_SESSIONS[:] = prev
    tool_proxy.configure({}, poll_health=False)


class _Tool(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def _reply(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).seen.append((self.command, self.path))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_POST = _reply


@pytest.fixture
def tool():
    seen: list = []
    handler = type("_RecTool", (_Tool,), {"seen": seen})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], seen
    srv.shutdown()


# ---------------------------------------------------------------------------
# release_sa3 and the family field
# ---------------------------------------------------------------------------


def test_release_sa3_drops_the_decoders_then_evicts(monkeypatch):
    monkeypatch.setattr(sa3_context, "_SAME_L_DECODERS",
                        {"x.trt": (object(), threading.Lock())})
    seen = []
    monkeypatch.setattr(sa3_session, "evict_sa3_contexts",
                        lambda: seen.append(dict(sa3_context._SAME_L_DECODERS)) or 2)
    assert sa3_session.release_sa3() == 2
    assert seen == [{}]  # decoders already gone when the eviction runs


def test_sa3_family_release_is_release_sa3(monkeypatch):
    calls = []
    monkeypatch.setattr(sa3_session, "release_sa3", lambda: calls.append(1) or 0)
    FAMILY_SPECS["sa3"].release()
    assert calls == [1]
    assert FAMILY_SPECS["acestep"].release is None


# ---------------------------------------------------------------------------
# The score_request socket
# ---------------------------------------------------------------------------


class _ScoreWs:
    """Yields nothing until ``during`` has run, then the socket closes."""

    def __init__(self, during=None):
        self.sent: list[dict] = []
        self.closes: list[tuple] = []
        self._during = during

    def send(self, text):
        self.sent.append(json.loads(text))

    def close(self, code=1000, reason=""):
        self.closes.append((code, reason))

    def __iter__(self):
        if self._during is not None:
            self._during(self)
        return iter(())


def _sa3_release(monkeypatch, log):
    spec = SimpleNamespace(release=lambda: log.append(
        ("release", [h.id for h in registry.list_handles()])))
    monkeypatch.setitem(FAMILY_SPECS, "fake-sa3", spec)


def test_score_request_order(monkeypatch, tool):
    port, seen = tool
    tool_proxy.configure({"other": ("127.0.0.1", port)},
                         resets={"other": "/load"}, poll_health=False)
    log: list = []
    _sa3_release(monkeypatch, log)

    def during(ws):
        sid = ws.sent[-1]["session_id"]
        log.append(("open", registry.get(sid) is not None, list(seen)))

    ws = _ScoreWs(during)
    wa._handle_score_request(ws, "fake-sa3")

    assert ws.sent == [{"type": "ready", "session_id": ws.sent[0]["session_id"]}]
    sid = ws.sent[0]["session_id"]
    # Registered before the first release and still registered at the
    # second one; no reset while the request is open.
    assert log == [("release", [sid]), ("open", True, []), ("release", [sid])]
    # Unregistered last, which fired the reset once the registry was empty.
    assert registry.get(sid) is None
    assert seen == [("POST", "/load")]


def test_score_request_preempts_a_live_session_and_waits(monkeypatch):
    log: list = []
    _sa3_release(monkeypatch, log)
    streaming = SimpleNamespace(state=SimpleNamespace(running=True),
                                closed=threading.Event())
    band_ws = SimpleNamespace(closes=[], close=lambda c, r="": band_ws.closes.append(c))
    with wa._ACTIVE_SLOT_LOCK:
        wa._ACTIVE_SESSIONS.append(wa._ActiveSession("band", streaming, band_ws))

    def closed_later():
        log.append(("band_closed",))
        streaming.closed.set()

    timer = threading.Timer(0.05, closed_later)
    timer.start()
    ws = _ScoreWs()
    try:
        wa._handle_score_request(ws, "fake-sa3")
    finally:
        timer.cancel()
    assert streaming.state.running is False
    assert band_ws.closes == [wa.PREEMPTED_CLOSE_CODE]
    assert log[0] == ("band_closed",)  # the release waited for the close
    assert [e[0] for e in log] == ["band_closed", "release", "release"]
    with wa._ACTIVE_SLOT_LOCK:
        assert wa._ACTIVE_SESSIONS == []


def test_score_request_preempts_a_session_left_open_at_close(monkeypatch):
    log: list = []
    _sa3_release(monkeypatch, log)
    streaming = SimpleNamespace(state=SimpleNamespace(running=True),
                                closed=threading.Event())
    streaming.closed.set()

    def during(ws):  # the page's SA3 session is still open when the tab dies
        with wa._ACTIVE_SLOT_LOCK:
            wa._ACTIVE_SESSIONS.append(wa._ActiveSession(
                "sa3", streaming, SimpleNamespace(close=lambda c, r="": None)))

    wa._handle_score_request(_ScoreWs(during), "fake-sa3")
    assert streaming.state.running is False
    with wa._ACTIVE_SLOT_LOCK:
        assert wa._ACTIVE_SESSIONS == []


def test_score_request_unsupported_family(monkeypatch):
    monkeypatch.setitem(FAMILY_SPECS, "no-release", SimpleNamespace(release=None))
    before = [h.id for h in registry.list_handles()]
    ws = _ScoreWs()
    wa._handle_score_request(ws, "no-release")
    assert ws.sent[0]["type"] == "error"
    assert ws.sent[0]["code"] == "score_request_unsupported"
    assert ws.closes == [(1008, "score_request unsupported")]
    assert [h.id for h in registry.list_handles()] == before


def test_score_request_takes_no_commands():
    with pytest.raises(RuntimeError):
        wa._score_request_takes_no_commands({"type": "prompt"}, None)


def test_tool_route_accepts_the_request_id(monkeypatch, tool):
    from websockets.datastructures import Headers
    from websockets.http11 import Response
    from websockets.sync.server import serve as ws_serve

    port, seen = tool
    log: list = []
    _sa3_release(monkeypatch, log)
    srv = ws_serve(lambda ws: None, "127.0.0.1", 0,
                   process_request=lambda c, r: Response(200, "OK", Headers(), b""))
    tool_proxy.configure({"other": ("127.0.0.1", port)}, poll_health=False)
    tool_proxy.wrap_server(srv)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    statuses = []

    def during(ws):
        conn = http.client.HTTPConnection("127.0.0.1", srv.socket.getsockname()[1], timeout=10)
        conn.request("POST", "/tools/other/work", body=b"{}",
                     headers={tool_proxy.SESSION_HEADER: ws.sent[-1]["session_id"]})
        statuses.append(conn.getresponse().status)
        conn.close()

    try:
        wa._handle_score_request(_ScoreWs(during), "fake-sa3")
    finally:
        srv.shutdown()
    assert statuses == [200]
    assert seen == [("POST", "/work")]


# ---------------------------------------------------------------------------
# Scoring requests take turns
# ---------------------------------------------------------------------------


class _HeldWs(_ScoreWs):
    """A score socket that stays open until ``done`` is set; ``state`` and
    ``fail_send`` stand in for a socket that has gone away."""

    def __init__(self):
        super().__init__(during=lambda ws: ws.done.wait(10))
        self.done = threading.Event()
        self.state = None
        self.fail_send = False

    def send(self, text):
        if self.fail_send:
            raise ConnectionError("gone")
        super().send(text)


def _wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def _start(ws, family="fake-sa3"):
    t = threading.Thread(target=wa._handle_score_request, args=(ws, family),
                         daemon=True)
    t.start()
    return t


def _types(ws):
    return [m["type"] for m in ws.sent]


def test_second_request_waits_its_turn(monkeypatch, tool):
    port, seen = tool
    tool_proxy.configure({"other": ("127.0.0.1", port)},
                         resets={"other": "/load"}, poll_health=False)
    log: list = []
    _sa3_release(monkeypatch, log)
    a, b = _HeldWs(), _HeldWs()
    ta = _start(a)
    _wait_until(lambda: _types(a) == ["ready"])
    tb = _start(b)
    _wait_until(lambda: b.sent)
    time.sleep(0.05)
    sid_a = a.sent[0]["session_id"]
    assert b.sent == [{"type": "queued", "position": 1}]
    # Nothing ran for the waiter: one release (the first's open), the
    # waiter is not in the registry, no reset yet.
    assert log == [("release", [sid_a])]
    assert [h.id for h in registry.list_handles()] == [sid_a]
    assert seen == []
    b.done.set()  # the waiter's socket has nothing to say yet
    a.done.set()
    ta.join(5)
    tb.join(5)
    assert _types(b) == ["queued", "ready"]
    sid_b = b.sent[1]["session_id"]
    # The registry was empty between the runs: the second registered only
    # after the first unregistered, and the reset fired once per run.
    assert log == [("release", [sid_a]), ("release", [sid_a]),
                   ("release", [sid_b]), ("release", [sid_b])]
    assert seen == [("POST", "/load"), ("POST", "/load")]
    assert registry.get(sid_a) is None and registry.get(sid_b) is None


def test_three_requests_run_in_order(monkeypatch):
    log: list = []
    _sa3_release(monkeypatch, log)
    a, b, c = _HeldWs(), _HeldWs(), _HeldWs()
    ta = _start(a)
    _wait_until(lambda: _types(a) == ["ready"])
    tb = _start(b)
    _wait_until(lambda: b.sent)
    tc = _start(c)
    _wait_until(lambda: c.sent)
    assert b.sent == [{"type": "queued", "position": 1}]
    assert c.sent == [{"type": "queued", "position": 2}]
    a.done.set()
    ta.join(5)
    _wait_until(lambda: _types(b) == ["queued", "ready"])
    _wait_until(lambda: len(c.sent) == 2)
    assert c.sent[1] == {"type": "queued", "position": 1}
    assert "ready" not in _types(c)
    b.done.set()
    tb.join(5)
    c.done.set()
    tc.join(5)
    assert _types(c) == ["queued", "queued", "ready"]
    sids = [ws.sent[-1]["session_id"] for ws in (a, b, c)]
    assert [e[1] for e in log] == [[sids[0]]] * 2 + [[sids[1]]] * 2 + [[sids[2]]] * 2


@pytest.mark.parametrize("how", ["closed", "failed_send"])
def test_a_waiter_that_goes_leaves_the_line(monkeypatch, how):
    from websockets.protocol import State

    monkeypatch.setattr(wa, "_SCORE_QUEUE_PING_S", 0.05)
    log: list = []
    _sa3_release(monkeypatch, log)
    a, b, c = _HeldWs(), _HeldWs(), _HeldWs()
    ta = _start(a)
    _wait_until(lambda: _types(a) == ["ready"])
    tb = _start(b)
    _wait_until(lambda: b.sent)
    tc = _start(c)
    _wait_until(lambda: c.sent)
    if how == "closed":
        b.state = State.CLOSED
    else:
        b.fail_send = True
    tb.join(5)
    assert not tb.is_alive()  # left the line without a turn
    _wait_until(lambda: {"type": "queued", "position": 1} in c.sent)
    a.done.set()
    ta.join(5)
    _wait_until(lambda: "ready" in _types(c))
    c.done.set()
    tc.join(5)
    assert "ready" not in _types(b)
    sid_a, sid_c = a.sent[0]["session_id"], c.sent[-1]["session_id"]
    assert [e[1] for e in log] == [[sid_a]] * 2 + [[sid_c]] * 2
    with wa._SCORE_TURN:
        assert wa._SCORE_LINE == []


def test_waiter_is_pinged_while_its_position_holds(monkeypatch):
    monkeypatch.setattr(wa, "_SCORE_QUEUE_PING_S", 0.02)
    log: list = []
    _sa3_release(monkeypatch, log)
    a, b = _HeldWs(), _HeldWs()
    ta = _start(a)
    _wait_until(lambda: _types(a) == ["ready"])
    tb = _start(b)
    _wait_until(lambda: len(b.sent) >= 3)
    assert all(m == {"type": "queued", "position": 1} for m in b.sent)
    a.done.set()
    b.done.set()
    ta.join(5)
    tb.join(5)
    assert _types(b)[-1] == "ready"


# ---------------------------------------------------------------------------
# DEMON_TOOL_YIELDS
# ---------------------------------------------------------------------------


def test_yields_are_posted(tool):
    port, seen = tool
    tool_proxy.configure_from_env("other=http://127.0.0.1:%d" % port, env={
        "DEMON_TOOL_YIELDS": "other=/park",
        "DEMON_TOOL_RESETS": "other=/load",
    })
    tool_proxy.yield_tools()
    assert [r for r in seen if r[0] == "POST"] == [("POST", "/park")]


def test_no_yields_when_unset(tool):
    port, seen = tool
    tool_proxy.configure_from_env("other=http://127.0.0.1:%d" % port, env={})
    tool_proxy.yield_tools()
    assert [r for r in seen if r[0] == "POST"] == []


def test_a_refused_or_down_yield_never_raises(tool):
    tool_proxy.configure({"other": ("127.0.0.1", 1)}, yields={"other": "/park"},
                         poll_health=False, timeout_s=1)
    tool_proxy.yield_tools()  # nothing listening: logged, create goes on


# ---------------------------------------------------------------------------
# an evicted model the server froze at boot is collected
# ---------------------------------------------------------------------------

class _Cycle:
    """A module-like object in a reference cycle (as a model's modules are)."""

    def __init__(self):
        self.me = self


def test_evict_collects_a_context_frozen_at_boot(monkeypatch):
    import gc
    import weakref

    from acestep.streaming import sa3_session

    monkeypatch.setattr(sa3_session, "_CONTEXTS", {})
    gc.collect()
    model = _Cycle()
    ref = weakref.ref(model)
    gc.freeze()  # the server, once ready: boot's objects never collected again
    try:
        del model
        gc.collect()
        assert ref() is not None  # frozen: a plain collection cannot free it
        assert sa3_session.evict_sa3_contexts() == 0
        assert ref() is None  # the eviction reached it
        assert gc.get_freeze_count() > 0  # and what is still alive is frozen again
    finally:
        gc.unfreeze()


def test_evict_leaves_an_unfrozen_process_unfrozen(monkeypatch):
    import gc

    from acestep.streaming import sa3_session

    monkeypatch.setattr(sa3_session, "_CONTEXTS", {})
    gc.unfreeze()
    sa3_session.evict_sa3_contexts()
    assert gc.get_freeze_count() == 0
