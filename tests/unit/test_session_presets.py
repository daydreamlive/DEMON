"""Session presets (``--presets <file>``): the table load, a session opened
by a preset name, the sealed wire (what goes out, what comes in), the
sealed HTTP router and the tool proxy's gate. The session runs against a
fake backend, so no model and no GPU."""

from __future__ import annotations

import contextlib
import json
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from websockets.exceptions import ConnectionClosed

import demos.realtime_motion_graph_web.ws_adapter as wa
from acestep.streaming import registry as session_registry
from acestep.streaming.events import (
    OneshotDone,
    OneshotFailed,
    OneshotReady,
    OneshotSlice,
    SwapFailed,
    SwapReady,
)
from acestep.streaming.families import resolve_served

SERVED = resolve_served(["sa3-medium"])
EMPTY_FRAME = struct.pack("<II", 2, 0)


# ---- a fake socket and a fake backend ----


class FakeWs:
    """Frames the client sends, in order; ``sent`` is every frame the
    server sent, as sent (str or bytes). Out of frames = the client left."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.sent = []
        self.closed = None

    def recv(self, timeout=None):
        if not self._frames:
            raise ConnectionClosed(None, None)
        return self._frames.pop(0)

    def send(self, data):
        if not isinstance(data, (str, bytes, bytearray)):
            data = b"".join(bytes(c) for c in data)
        self.sent.append(data)

    def close(self, code=1000, reason=""):
        self.closed = (code, reason)

    def __iter__(self):
        while True:
            try:
                yield self.recv()
            except ConnectionClosed:
                return

    def json_sent(self):
        return [json.loads(f) for f in self.sent if isinstance(f, str)]


class FakeStreaming:
    """What ``_handle_client_body`` reads off a ``StreamingSession``.
    ``oneshot``/``swap_source`` answer through the bus subscriber the way
    the real session does."""

    def __init__(self, seconds=2.0):
        n = int(seconds * 48000)
        self.initial_buffer = np.zeros((n, 2), dtype=np.float32)
        self.state = SimpleNamespace(
            n_channels=2, bpm=None, key=None, time_signature=None,
            current_depth=1, source_epoch=0, running=True,
            params={}, _lock=threading.Lock(),
        )
        self.initial_enable_ids = []
        self.max_pipeline_depth = 2
        self.initial_upload_stems = None
        self.initial_stem_error = None
        self.use_lora = False
        self.subscriber = None
        self.oneshots = []
        self.swaps = 0
        self.bus = SimpleNamespace(subscribe=self._subscribe)

    def _subscribe(self, fn, name=None):
        self.subscriber = fn

    def render_only(self):
        return True

    def lora_catalog_payload(self):
        return [{"id": "secret-lora", "path": "/secret/dir/lora.bin"}]

    def geometry_payload(self):
        return {"sample_rate": 48000, "channels": 2, "chunk_rate_hz": 21.5}

    def capabilities_payload(self):
        return {"oneshot": True, "swap": True}

    def knob_manifest_payload(self):
        return {"version": 1, "knobs": {"secret_knob": {"min": 0, "max": 1}}}

    def steering_payload(self):
        return {"manual_slot_count": 0, "manual_slot_cap": 0,
                "steering_available": False}

    def snapshot(self):
        return {}

    def run(self):
        deadline = time.monotonic() + 10
        while self.state.running and time.monotonic() < deadline:
            time.sleep(0.005)

    def close(self):
        pass

    def oneshot(self, data, origin=None):
        self.oneshots.append(dict(data))
        rid = data.get("request_id")
        if data.get("knobs", {}).get("boom"):
            self.subscriber(OneshotFailed(
                rid, "RuntimeError: secret_engine.plan missing at /secret/dir",
            ))
            return
        if data.get("knobs", {}).get("crowd"):
            self.subscriber(OneshotFailed(rid, "4 queued", reason="queue_full"))
            return
        if data.get("knobs", {}).get("windows"):
            pcm = np.zeros((4, 2), dtype=np.float16)
            self.subscriber(OneshotSlice(
                rid, "v0", 7, 0, 1, 0, 4, 2, 48000, pcm,
                peaks={"mix": 0.5, "streams": [0.5]}, mix=False,
            ))
            self.subscriber(OneshotDone(
                rid, "v0", 7, 4, 2, 48000, 1, "ab" * 32,
                {"generate": 1.0, "decode": 2.0},
            ))
            return
        pcm = np.zeros((1, 1, 4, 2), dtype=np.int16)
        self.subscriber(OneshotReady(
            rid, pcm, 48000, ("v0",), tuple(data.get("seeds") or ()), (0,),
            {"generate": 158.2, "decode": 775.5},
            latent_header={"n_streams": 1, "channels": 64, "frames": 4,
                           "seconds": 1.0},
        ))

    def swap_source(self, audio, **kw):
        self.swaps += 1
        if self.swaps > 1:
            self.subscriber(SwapFailed("engine secret_engine.plan missing",
                                       build_command="build --secret"))
            return
        self.subscriber(SwapReady(
            duration=2.0, sample_rate=48000, channels=2, bpm=None, key=None,
            time_signature=None, fixture_name="secret_fixture",
            initial_buffer=np.zeros((4, 2), dtype=np.float32), source_epoch=1,
        ))


@pytest.fixture
def run_session(monkeypatch):
    """Run ``_handle_client_body`` on ``frames`` against a FakeStreaming.
    Returns (ws, streaming, create_calls)."""
    monkeypatch.delenv("DEMON_WIPE_USER_UPLOADS", raising=False)
    monkeypatch.setattr(wa, "_preempt_active_session", lambda *a, **k: None)
    monkeypatch.setattr(wa, "_log_session_vram", lambda *a, **k: None)
    monkeypatch.setattr(wa, "_server_capabilities", lambda caps: caps)
    monkeypatch.setattr(wa, "_capabilities_with_analysis", lambda caps: caps)
    monkeypatch.setattr(wa, "loras_dir", lambda: Path("/secret/loras"))
    monkeypatch.setattr(wa, "checkpoint_scale", lambda ckpt: "scale-x")
    monkeypatch.setattr(wa.tool_proxy, "healthy_tools", lambda: ["secret-tool"])
    monkeypatch.setattr(wa.tool_proxy, "yield_tools", lambda: None)

    def run(frames, streaming=None):
        streaming = streaming or FakeStreaming()
        created = []

        def create(**kw):
            created.append(kw)
            return streaming

        monkeypatch.setattr(wa, "StreamingSession", SimpleNamespace(create=create))
        ws = FakeWs(frames)
        with contextlib.ExitStack() as stack:
            wa._handle_client_body(
                ws, stack,
                decoder_backend="eager", vae_backend="eager",
                checkpoint=SERVED[0].model_id, backend_family="sa3",
                offload_text_encoder=False, checkpoint_dir=None,
                model_extensions={}, served_checkpoints=SERVED,
            )
        return ws, streaming, created

    return run


# ---- 8: without --presets nothing changes ----


def test_default_ready_is_byte_for_byte_unchanged(run_session):
    config = {"telemetry_version": 1, "text_only": True, "sa3_duration_s": 2.0,
              "prompt": "a plain config", "client_id": "c1"}
    ws, _, created = run_session([json.dumps(config)])
    assert created and created[0]["config"].prompt == "a plain config"
    frames = [f for f in ws.sent if isinstance(f, str)]
    assert json.loads(frames[0])["type"] == "init_ack"
    sid = json.loads(frames[1])["session_id"]
    expected = json.dumps({
        "type": "ready",
        "duration": 2.0,
        "sample_rate": 48000,
        "channels": 2,
        "lora_dir": str(Path("/secret/loras")),
        "lora_catalog": [{"id": "secret-lora", "path": "/secret/dir/lora.bin"}],
        "lora_pending_enable": [],
        "bpm": None,
        "key": None,
        "time_signature": None,
        "checkpoint": "medium",
        "checkpoint_scale": "scale-x",
        "pipeline_depth": 1,
        "max_pipeline_depth": 2,
        "session_id": sid,
        "source_epoch": 0,
        "geometry": {"sample_rate": 48000, "channels": 2, "chunk_rate_hz": 21.5},
        "capabilities": {"oneshot": True, "swap": True},
        "tools": ["secret-tool"],
        "knob_manifest": {"version": 1, "knobs": {"secret_knob": {"min": 0, "max": 1}}},
        "manual_slot_count": 0,
        "manual_slot_cap": 0,
        "steering_available": False,
    })
    assert frames[1] == expected
    # The silent stand-in frame follows ready, as today.
    assert ws.sent[2] == np.zeros((1, 2), dtype=np.float16).tobytes()


# ---- the table ----

from demos.realtime_motion_graph_web import presets  # noqa: E402

TABLE_RAW = {
    "sealed": True,
    "request_tools": ["kiwi"],
    "tool_routes": {
        "kiwi": ["POST /run", "GET /job/*", "DELETE /job/*", "GET /audio/*"],
        "plum": ["POST /in"],
    },
    "presets": {
        "q9x": {
            "config": {"text_only": True, "steps": 5, "checkpoint": "sa3-medium",
                       "prompt": "Zebra lantern. Length: {seconds} seconds"},
            "values": {"seconds": {"field": "sa3_duration_s", "min": 1, "max": 30}},
            "intake": {"tool": "plum", "path": "/in"},
            "refresh": True,
            "versions": [
                {"id": "1", "knobs": {"walrus_gain": 0.25}, "seed": 11},
                {"id": "2", "knobs": {"walrus_gain": 0.6}, "seed": 12},
                {"id": "3", "knobs": {"boom": 1.0}, "seed": 13},
                {"id": "4", "knobs": {"crowd": 1.0}, "seed": 14},
                {"id": "5", "knobs": {"windows": 1.0}, "seed": 15},
            ],
            "params": {"amount": {"knob": "otter_mix", "min": 0, "max": 1,
                                  "default": 0.35}},
            "tools": ["plum"],
        },
    },
}

#: Strings of the table, the session and the backend a client must never see.
DENY = ["Zebra", "lantern", "walrus_gain", "otter_mix", "sa3", "steps", "checkpoint",
        "secret", "boom", "crowd", "medium", "knob", "seed", "\"ms\"", "latent",
        "Error", "build", "scale-x", "kiwi", "init_ack"]


@pytest.fixture
def table():
    t = presets.parse_table(json.loads(json.dumps(TABLE_RAW)), proxies=["kiwi", "plum"])
    presets.configure(t)
    yield t
    presets.configure(None)


def _bad(**patch):
    raw = json.loads(json.dumps(TABLE_RAW))
    p = raw["presets"]["q9x"]
    for k, v in patch.items():
        if k.startswith("top_"):
            raw[k[4:]] = v
        else:
            p[k] = v
    return raw


@pytest.mark.parametrize("raw", [
    _bad(top_extra=1),
    _bad(config={"prompt": "x"}),                       # would read an audio frame
    _bad(config={"text_only": True, "preset": "q9x"}),
    _bad(versions=[]),
    _bad(versions=[{"id": "1", "label": "x"}]),          # no labels
    _bad(versions=[{"id": "1"}, {"id": "1"}]),
    _bad(params={"amount": {"knob": "k", "min": 0, "max": 1, "default": 2}}),
    _bad(values={"seconds": {"field": "f", "min": 5, "max": 1}}),
    _bad(tools=["fig"]),                                 # no tool_routes
    _bad(intake={"tool": "kiwi", "path": "/in"}),         # not the preset's tool
    _bad(top_tool_routes={"kiwi": ["PATCH /x"], "plum": ["POST /in"]}),
    _bad(top_presets={}),
])
def test_bad_tables_fail(raw):
    with pytest.raises(presets.TableError):
        presets.parse_table(raw, proxies=["kiwi", "plum"])


def test_table_tools_must_be_configured():
    with pytest.raises(presets.TableError):
        presets.parse_table(json.loads(json.dumps(TABLE_RAW)), proxies=["kiwi"])


def test_load_table_reads_a_file_and_fails_on_junk(tmp_path):
    good = tmp_path / "t.json"
    good.write_text(json.dumps(TABLE_RAW))
    assert set(presets.load_table(str(good)).presets) == {"q9x"}
    bad = tmp_path / "b.json"
    bad.write_text("{not json")
    with pytest.raises(presets.TableError):
        presets.load_table(str(bad))


def test_a_bad_file_fails_the_boot(tmp_path, monkeypatch):
    import demos.realtime_motion_graph_web.server as server

    bad = tmp_path / "b.json"
    bad.write_text(json.dumps(_bad(versions=[])))
    monkeypatch.setattr(server, "load_repo_env_defaults", lambda: None)
    monkeypatch.setattr(server, "configure_logging", lambda: None)
    monkeypatch.setattr("sys.argv", ["server", "--no-backend", "--presets", str(bad)])
    monkeypatch.delenv("DEMON_TOOL_PROXIES", raising=False)
    with pytest.raises(SystemExit, match="invalid --presets file"):
        server.main()
    assert presets.current() is None


# ---- 2: the session config ----


def test_expand_applies_values_and_fills_the_prompt(table):
    preset, config = presets.expand(table, {"preset": "q9x", "values": {"seconds": 7.4}})
    assert preset.name == "q9x"
    assert config["sa3_duration_s"] == 7.4
    assert config["prompt"] == "Zebra lantern. Length: 7 seconds"
    assert config["text_only"] is True and config["telemetry_version"] == 1
    assert "values" not in config and "preset" not in config


@pytest.mark.parametrize("msg,code", [
    ({"preset": "nope"}, "bad_request"),
    ({"prompt": "x"}, "bad_request"),
    ({"preset": "q9x"}, "bad_request"),                          # no default
    ({"preset": "q9x", "values": {"seconds": 31}}, "too_long"),
    ({"preset": "q9x", "values": {"seconds": 0.5}}, "bad_request"),
    ({"preset": "q9x", "values": {"seconds": "7"}}, "bad_request"),
    ({"preset": "q9x", "values": {"seconds": 7, "other": 1}}, "bad_request"),
])
def test_expand_refusals(table, msg, code):
    with pytest.raises(presets.Refused) as e:
        presets.expand(table, msg)
    assert e.value.code == code


def test_sealed_refuses_a_config_without_a_preset(table, run_session):
    for config in ({"prompt": "x", "text_only": True, "telemetry_version": 1},
                   {"preset": "nope"},
                   {"type": "upload_track"}):
        ws, _, created = run_session([json.dumps(config)])
        assert created == []
        [frame] = ws.json_sent()
        assert set(frame) == {"type", "code", "message", "ref"}
        assert frame["code"] == "bad_request"
        assert frame["message"] == presets.SENTENCES["bad_request"]
        assert ws.closed == (1008, "")


# ---- 3, 4, 5: a preset session end to end against the fake backend ----


def _oneshot(rid, version, **kw):
    return json.dumps({"type": "oneshot", "request_id": rid, "version": version, **kw})


def test_preset_session_end_to_end(table, run_session):
    frames = [
        json.dumps({"telemetry_version": 1, "preset": "q9x", "values": {"seconds": 2}}),
        _oneshot("r1", "1"),
        _oneshot("r2", "2", values={"amount": 5}),
        _oneshot("r3", "1", knobs={"walrus_gain": 1}),
        _oneshot("r4", "9"),
        _oneshot("r5", "1", values={"nope": 1}),
        json.dumps({"type": "params", "walrus_gain": 1}),
        json.dumps({"type": "set_prompt", "prompt": "x"}),
        "not json",
        b"stray binary",
        json.dumps({"type": "swap_source", "fixture_name": "x"}), EMPTY_FRAME,
        json.dumps({"type": "swap_source"}), struct.pack("<II", 2, 1) + b"\0" * 8,
        _oneshot("r6", "3"),
        _oneshot("r7", "4"),
        _oneshot("r8", "5"),
        json.dumps({"type": "swap_source"}), EMPTY_FRAME,
    ]
    ws, streaming, created = run_session(frames)
    cfg = created[0]["config"]
    assert cfg.prompt == "Zebra lantern. Length: 2 seconds" and cfg.steps == 5

    # What reached the session: versions expanded, params clamped.
    assert streaming.oneshots == [
        {"type": "oneshot", "request_id": "r1", "seeds": [11],
         "knobs": {"walrus_gain": 0.25, "otter_mix": 0.35}},
        {"type": "oneshot", "request_id": "r2", "seeds": [12],
         "knobs": {"walrus_gain": 0.6, "otter_mix": 1.0}},
        {"type": "oneshot", "request_id": "r6", "seeds": [13],
         "knobs": {"boom": 1.0, "otter_mix": 0.35}},
        {"type": "oneshot", "request_id": "r7", "seeds": [14],
         "knobs": {"crowd": 1.0, "otter_mix": 0.35}},
        {"type": "oneshot", "request_id": "r8", "seeds": [15],
         "knobs": {"windows": 1.0, "otter_mix": 0.35}},
    ]
    assert streaming.swaps == 2

    sent = ws.json_sent()
    ready = sent[0]
    assert ready == {
        "type": "ready", "session_id": ready["session_id"], "sample_rate": 48000,
        "channels": 2, "duration": 2.0,
        "plan": {"intake": "/tools/plum/in", "refresh": True,
                 "versions": [{"id": i} for i in "12345"],
                 "params": [{"id": "amount", "min": 0, "max": 1, "default": 0.35}]},
    }
    assert ws.sent[1] == np.zeros((1, 2), dtype=np.float16).tobytes()

    by_type = {}
    for f in sent[1:]:
        by_type.setdefault(f["type"], []).append(f)
    assert set(by_type) == {"oneshot_ready", "oneshot_failed", "error", "swap_ready",
                            "swap_failed", "oneshot_slice", "oneshot_done"}
    assert by_type["oneshot_ready"][0] == {
        "type": "oneshot_ready", "request_id": "r1", "ids": ["v0"], "streams": [0],
        "sample_rate": 48000, "channels": 2, "frames": 4, "encoding": "pcm_s16le",
        "redo_of": None,
    }
    failed = {f["request_id"]: f["code"] for f in by_type["oneshot_failed"]}
    assert failed == {"r3": "bad_request", "r4": "bad_request", "r5": "bad_request",
                      "r6": "failed", "r7": "busy"}
    for f in by_type["oneshot_failed"] + by_type["error"] + by_type["swap_failed"]:
        assert set(f) - {"request_id"} == {"type", "code", "message", "ref"}
        assert f["message"] == presets.SENTENCES[f["code"]]
    assert [f["code"] for f in by_type["error"]] == ["bad_request"] * 3
    assert [f["code"] for f in by_type["swap_failed"]] == ["bad_request", "failed"]
    assert by_type["swap_ready"] == [{"type": "swap_ready", "duration": 2.0,
                                      "sample_rate": 48000, "channels": 2,
                                      "source_epoch": 1}]
    assert by_type["oneshot_slice"] == [{
        "type": "oneshot_slice", "request_id": "r8", "version": "v0", "index": 0,
        "count": 1, "start_frame": 0, "frames": 4, "channels": 2,
        "sample_rate": 48000, "encoding": "slice_f16_zstd"}]
    assert by_type["oneshot_done"] == [{
        "type": "oneshot_done", "request_id": "r8", "version": "v0", "frames": 4,
        "channels": 2, "sample_rate": 48000, "windows": 1, "sha256": "ab" * 32}]

    # Binary frames: the ready stand-in, two oneshot_ready PCMs, the slice,
    # the swapped buffer; nothing else.
    assert sum(isinstance(f, bytes) for f in ws.sent) == 5

    wire = "".join(f if isinstance(f, str) else f.decode("latin1") for f in ws.sent)
    for word in DENY:
        assert word not in wire, word


def test_request_socket_passes_only_turn_frames(table):
    inner = FakeWs([])
    sock = presets.PresetSocket(inner, None)
    sock.send(json.dumps({"type": "queued", "position": 2}))
    sock.send(json.dumps({"type": "ready", "session_id": "s1", "tools": ["kiwi"]}))
    sock.send(json.dumps({"type": "error", "code": "score_request_unsupported",
                          "message": "family 'sa3' cannot release its model"}))
    sock.send(json.dumps({"type": "init_ack", "client_id": "x"}))
    sock.send(b"binary")
    sock.close(1008, "score_request unsupported")
    sent = inner.json_sent()
    assert sent[:2] == [{"type": "queued", "position": 2},
                        {"type": "ready", "session_id": "s1"}]
    assert sent[2]["code"] == "unavailable" and len(sent) == 3
    assert b"binary" not in inner.sent and inner.closed == (1008, "")


# ---- 6: the sealed HTTP router ----


def _http(path, upgrade=None):
    import demos.realtime_motion_graph_web.server as server

    headers = {"Upgrade": upgrade} if upgrade else {}
    req = SimpleNamespace(path=path, headers=headers)
    conn = SimpleNamespace(remote_address=("1.2.3.4", 1))
    return server._process_request(conn, req)


def test_sealed_router_answers_404_except_the_socket(table):
    for path in ("/api/server-info", "/api/loras", "/api/protocol", "/", "/sdk/x.js"):
        resp = _http(path)
        assert resp.status_code == 404 and resp.body == b"404 not found\n"
    assert _http("/", upgrade="websocket") is None


def test_open_table_router_is_unchanged():
    presets.configure(presets.parse_table(
        {**json.loads(json.dumps(TABLE_RAW)), "sealed": False}))
    try:
        assert _http("/api/protocol").status_code == 200
    finally:
        presets.configure(None)


# ---- 7: the tool proxy with a table ----


@pytest.fixture
def proxy(table):
    import http.client
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from websockets.datastructures import Headers
    from websockets.http11 import Response
    from websockets.sync.server import serve as ws_serve

    from demos.realtime_motion_graph_web import tool_proxy

    class Echo(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            out = json.dumps({"path": self.path, "upstream": "kiwi-upstream"}).encode()
            self.send_response(207)
            self.send_header("Content-Type", "application/x-up")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        do_GET = do_POST = do_PUT = do_DELETE = _reply

    up = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    srv = ws_serve(lambda ws: None, "127.0.0.1", 0, process_request=lambda c, r: Response(
        200, "OK", Headers([("Content-Length", "2")]), b"ws"))
    tool_proxy.configure({"kiwi": ("127.0.0.1", up.server_address[1]),
                          "plum": ("127.0.0.1", dead),
                          "fig": ("127.0.0.1", up.server_address[1])},
                         timeout_s=5, poll_health=False)
    tool_proxy.wrap_server(srv)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for sid in ("p-sid", "r-sid", "u-sid"):
        session_registry.register(session_registry.SessionHandle(
            sid, time.time(), lambda d, a: None, lambda: {}))
    presets.bind("p-sid", table.presets["q9x"])
    presets.bind("r-sid", presets.REQUEST)
    port = srv.socket.getsockname()[1]

    def req(method, path, sid=None, body=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        headers = {tool_proxy.SESSION_HEADER: sid} if sid else {}
        c.request(method, path, body=body, headers=headers)
        r = c.getresponse()
        out = (r.status, r.read())
        c.close()
        return out

    yield req
    for sid in ("p-sid", "r-sid", "u-sid"):
        session_registry.unregister(sid)
        presets.unbind(sid)
    tool_proxy.configure({}, poll_health=False)
    srv.shutdown()
    up.shutdown()


def _fixed(out, code):
    body = json.loads(out)
    assert set(body) == {"code", "message", "ref"}
    assert body["code"] == code and body["message"] == presets.SENTENCES[code]


def test_proxy_allows_only_listed_routes_of_the_sessions_tools(proxy):
    status, out = proxy("POST", "/tools/kiwi/run", "r-sid", b"{}")
    # An allowed answer passes through untouched (the upstream owns it).
    assert status == 207 and json.loads(out) == {"path": "/run", "upstream": "kiwi-upstream"}
    assert proxy("GET", "/tools/kiwi/job/abc", "r-sid")[0] == 207
    assert proxy("DELETE", "/tools/kiwi/job/abc", "r-sid")[0] == 207
    denied = [
        ("GET", "/tools/kiwi/health", "r-sid"),       # not listed
        ("GET", "/tools/kiwi/status", "r-sid"),
        ("POST", "/tools/kiwi/job/abc", "r-sid"),     # wrong method
        ("POST", "/tools/kiwi/run", "p-sid"),         # not the preset's tool
        ("GET", "/tools/fig/x", "r-sid"),             # configured, not in the table
        ("POST", "/tools/kiwi/run", "u-sid"),         # unbound session, sealed
        ("POST", "/tools/kiwi/run", None),            # no session
        ("POST", "/tools/kiwi/run", "gone"),          # dead session
        ("POST", "/tools/nope/run", None),            # unknown tool
        ("POST", "/tools/nope/run", "r-sid"),
        ("GET", "/tools/kiwi/job/%2e%2e/x", "r-sid"),
    ]
    for method, path, sid in denied:
        status, out = proxy(method, path, sid)
        assert status == 404, (method, path, sid)
        _fixed(out, "bad_request")


def test_proxy_own_errors_are_fixed_sentences(proxy):
    status, out = proxy("POST", "/tools/plum/in", "p-sid", b"{}")
    assert status == 503
    _fixed(out, "unavailable")
