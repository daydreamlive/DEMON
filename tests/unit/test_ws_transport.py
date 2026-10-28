"""ws_transport: inbound message cap and per-recv read size of the demo server."""

import threading
import time

import pytest
from websockets.sync.client import connect
from websockets.sync.server import ServerConnection, serve

from demos.realtime_motion_graph_web import ws_transport as W


def test_defaults_fit_a_300s_multichannel_upload(monkeypatch):
    monkeypatch.delenv("DEMON_WS_MAX_MESSAGE_BYTES", raising=False)
    monkeypatch.delenv("DEMON_WS_RECV_BUFSIZE", raising=False)
    assert W.max_message_bytes() == 1024 * 1024 * 1024 > 300 * 48000 * 14 * 4 + 8
    opts = W.server_options()
    assert opts["max_size"] == W.DEFAULT_MAX_MESSAGE_BYTES
    assert issubclass(opts["create_connection"], ServerConnection)
    assert opts["create_connection"].recv_bufsize == 4 * 1024 * 1024
    assert ServerConnection.recv_bufsize == 65536            # the base class is untouched


def test_env_overrides_and_bad_values(monkeypatch):
    monkeypatch.setenv("DEMON_WS_MAX_MESSAGE_BYTES", str(512 * 1024 * 1024))
    monkeypatch.setenv("DEMON_WS_RECV_BUFSIZE", "1048576")
    assert W.max_message_bytes() == 512 * 1024 * 1024
    assert W.server_options()["create_connection"].recv_bufsize == 1048576
    monkeypatch.setenv("DEMON_WS_RECV_BUFSIZE", "12")          # floored to 64 KiB
    assert W.recv_bufsize() == 64 * 1024
    monkeypatch.setenv("DEMON_WS_MAX_MESSAGE_BYTES", "lots")   # unparseable: default
    assert W.max_message_bytes() == W.DEFAULT_MAX_MESSAGE_BYTES


def _transfer_seconds(conn_cls, nbytes, hogs=3):
    got, stop = {}, threading.Event()

    def handler(ws):
        try:
            got["n"] = len(ws.recv(timeout=60))
        except Exception as exc:  # noqa: BLE001
            got["err"] = repr(exc)
        got["t"] = time.monotonic()

    def hog():
        x = 0
        while not stop.is_set():
            x += 1

    srv = serve(handler, "127.0.0.1", 0, compression=None, max_size=None,
                create_connection=conn_cls)
    port = srv.socket.getsockname()[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for _ in range(hogs):
        threading.Thread(target=hog, daemon=True).start()
    try:
        with connect(f"ws://127.0.0.1:{port}/", max_size=None) as c:
            t0 = time.monotonic()
            c.send(b"Z" * nbytes)
            while "t" not in got and time.monotonic() - t0 < 65:
                time.sleep(0.02)
        assert got.get("n") == nbytes, got
        return got["t"] - t0
    finally:
        stop.set()
        srv.shutdown()


def test_large_reads_keep_an_upload_fast_while_other_threads_hold_the_gil():
    """The same upload with busy Python threads in the process: the
    large-read connection receives it several times faster than the stock
    64 KiB reads (each recv needs a GIL turn)."""
    n = 24 * 1024 * 1024
    fast = _transfer_seconds(W.connection_class(4 * 1024 * 1024), n)
    slow = _transfer_seconds(ServerConnection, n)
    assert fast * 3 < slow, (fast, slow)
