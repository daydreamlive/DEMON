"""Transcript replay server for browser-client (Tier C) tests.

Serves a recorded golden-session transcript (``transcript.jsonl`` +
``blobs/``, captured by :mod:`tests.golden.client` and cached by
:mod:`tests.golden.refs_store`) over a real WebSocket so the full web
app can boot and stream CPU-only — no GPU, no model load, no engine.

The HTTP side mimics just enough of
``demos/realtime_motion_graph_web/server.py`` for the app to come up:
``/api/server-info`` (advertising the transcript's fixtures as
server-side loadable, so the client skips the PCM upload exactly like
the recording did), ``/api/fixtures``, ``/api/loras``,
``/api/user_uploads``, and the self-describing capability probes
``/api/knobs`` / ``/api/protocol`` (served from the real registries —
both are torch-free, so the replay stays CPU-only). WebSocket upgrades
on any path replay the transcript.

Replay semantics:

* ``recv`` frames are sent to the client paced by the recorded
  inter-frame deltas (divided by ``--speed``, capped at ``--max-gap``
  so the model-load pause compresses away).
* Recorded *client* sends are causality gates: when the walker reaches
  a recorded ``swap_source`` / ``prompt`` / ``enable_lora`` /
  ``disable_lora`` send, it WAITS until the live client actually sends
  a message of that type before continuing — so a ``swap_ready`` can
  never arrive before the browser asked for the swap. High-rate
  ``params`` sends are not gated (the live client free-runs its own
  8 ms param sync).
* Everything else the live client sends is drained and ignored.

Usage:
    python -m tests.golden.replay_server --scenario swap_fixture --port 18931
"""

import argparse
import asyncio
import http
import json
import sys
import time
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response

DEFAULT_REFS_DIR = Path.home() / ".cache" / "demon" / "test-refs"

# Client message types the replay walker blocks on (1:1 with a recorded
# send of the same type). Everything else is fire-and-forget.
GATING_TYPES = {"swap_source", "prompt", "enable_lora", "disable_lora"}


def _log(msg: str) -> None:
    # ASCII only: Windows consoles default to cp1252.
    print(f"[replay-server] {msg}", flush=True)


class Transcript:
    def __init__(self, scenario_dir: Path):
        self.dir = scenario_dir
        lines = (scenario_dir / "transcript.jsonl").read_text(
            encoding="utf-8").splitlines()
        self.entries = [json.loads(ln) for ln in lines if ln.strip()]
        first = self.entries[0]
        if first.get("dir") != "send" or first.get("kind") != "json":
            raise RuntimeError("transcript does not start with the config")
        self.config = first["data"]

    def blob(self, entry: dict) -> bytes:
        return (self.dir / entry["blob"]).read_bytes()

    def fixtures(self) -> list[str]:
        names = set()
        if self.config.get("fixture_name"):
            names.add(self.config["fixture_name"])
        for e in self.entries:
            if e.get("kind") == "json" and e.get("dir") == "send":
                name = e["data"].get("fixture_name")
                if name:
                    names.add(name)
        return sorted(names)


def _json_response(payload: object) -> Response:
    body = json.dumps(payload).encode()
    return Response(
        200, "OK",
        Headers([
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            # Same posture as the real server: public read-only probes,
            # fetched cross-origin by the app before the WS handshake.
            ("Access-Control-Allow-Origin", "*"),
            ("Cache-Control", "no-store"),
        ]),
        body,
    )


def make_process_request(transcript: Transcript):
    def _process_request(connection, request):
        upgrade = request.headers.get("Upgrade", "") or ""
        if upgrade.lower() == "websocket":
            return None  # proceed with the WS handshake
        path_only = request.path.split("?", 1)[0].split("#", 1)[0]
        if path_only == "/api/server-info":
            return _json_response({
                "no_backend": False,
                "kiosk": False,
                "default_mode": None,
                "warmup": {"state": "ready"},
                "replay": True,
                "server_side_fixtures": transcript.fixtures(),
            })
        if path_only == "/api/fixtures":
            return _json_response(transcript.fixtures())
        if path_only in ("/api/loras", "/api/user_uploads", "/api/videos"):
            return _json_response([])
        # Capability probes: serve the REAL knob manifest and wire
        # contract (same modules server.py uses — pure dataclasses /
        # numpy, no torch) so the app boots against replay exactly as it
        # does against a live pod and the e2e clean-console bar holds.
        if path_only == "/api/knobs":
            from acestep.streaming.knobs import (
                KNOB_SCHEMA_VERSION, knob_catalog)
            query = request.path.split("?", 1)[1] if "?" in request.path else ""
            sde = "sde=1" in query or "sde=true" in query
            return _json_response({
                "version": KNOB_SCHEMA_VERSION,
                "knobs": knob_catalog(sde=sde, loras=[]),
            })
        if path_only == "/api/protocol":
            from demos.realtime_motion_graph_web.protocol import wire_contract
            return _json_response(wire_contract())
        body = b"not found (replay server)"
        return Response(
            404, "Not Found",
            Headers([
                ("Content-Type", "text/plain"),
                ("Content-Length", str(len(body))),
                ("Access-Control-Allow-Origin", "*"),
            ]),
            body,
        )

    return _process_request


class SessionReplay:
    """One live client connection driving one pass over the transcript."""

    def __init__(self, ws, transcript: Transcript, speed: float,
                 max_gap_s: float, gate_timeout_s: float):
        self.ws = ws
        self.tr = transcript
        self.speed = speed
        self.max_gap_s = max_gap_s
        self.gate_timeout_s = gate_timeout_s
        # type -> count of live client messages received; gates wait on
        # the counter so order of (gate reached / client sent) is free.
        self._recv_counts: dict[str, int] = {}
        self._recv_event = asyncio.Event()
        self._client_msgs = 0
        self._closed = False

    async def run(self) -> None:
        # Phase 1: config handshake (mirrors ws_adapter's recv order).
        cfg_raw = await self.ws.recv()
        cfg = json.loads(cfg_raw)
        _log(f"client config: fixture={cfg.get('fixture_name')} "
             f"use_server_fixture={cfg.get('use_server_fixture')}")
        if not cfg.get("use_server_fixture"):
            pcm = await self.ws.recv()
            _log(f"client uploaded {len(pcm)} bytes of PCM (discarded)")

        reader = asyncio.create_task(self._reader())
        t0 = time.monotonic()
        try:
            await self._walk()
            _log(f"transcript exhausted in {time.monotonic() - t0:.1f}s "
                 f"(client sent {self._client_msgs} messages); holding "
                 f"connection open")
            await self.ws.wait_closed()
        except ConnectionClosed:
            _log("client disconnected mid-replay")
        finally:
            reader.cancel()

    async def _reader(self) -> None:
        try:
            async for msg in self.ws:
                self._client_msgs += 1
                if isinstance(msg, (bytes, bytearray)):
                    continue
                try:
                    mtype = json.loads(msg).get("type")
                except Exception:
                    continue
                if mtype in GATING_TYPES:
                    self._recv_counts[mtype] = (
                        self._recv_counts.get(mtype, 0) + 1)
                    self._recv_event.set()
        except ConnectionClosed:
            pass

    async def _wait_for_client(self, mtype: str, needed: int) -> None:
        deadline = time.monotonic() + self.gate_timeout_s
        while self._recv_counts.get(mtype, 0) < needed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"gate timeout: client never sent {mtype!r}")
            self._recv_event.clear()
            try:
                await asyncio.wait_for(
                    self._recv_event.wait(), timeout=min(remaining, 1.0))
            except asyncio.TimeoutError:
                continue

    async def _walk(self) -> None:
        prev_t = self.tr.entries[0]["t"]
        gates_needed: dict[str, int] = {}
        for entry in self.tr.entries[1:]:
            if entry["dir"] == "send":
                if entry["kind"] != "json":
                    continue  # pcm_upload: consumed in the handshake
                mtype = entry["data"].get("type")
                if mtype in GATING_TYPES:
                    gates_needed[mtype] = gates_needed.get(mtype, 0) + 1
                    _log(f"gate: waiting for client {mtype!r}")
                    await self._wait_for_client(mtype, gates_needed[mtype])
                    _log(f"gate: {mtype!r} satisfied")
                continue
            # recv frame: pace by the recorded delta, then send.
            dt = max(0.0, (entry["t"] - prev_t) / 1000.0) / self.speed
            prev_t = entry["t"]
            if dt > 0:
                await asyncio.sleep(min(dt, self.max_gap_s))
            if entry["kind"] == "json":
                await self.ws.send(json.dumps(entry["data"]))
            else:
                await self.ws.send(self.tr.blob(entry))


async def amain(args: argparse.Namespace) -> None:
    scenario_dir = Path(args.refs_dir) / args.scenario
    if not (scenario_dir / "transcript.jsonl").exists():
        _log(f"ERROR: no transcript at {scenario_dir}. Fetch refs first: "
             f"python -m tests.golden.refs_store fetch")
        sys.exit(2)
    transcript = Transcript(scenario_dir)
    n_bin = sum(1 for e in transcript.entries if e.get("kind") == "bin")
    _log(f"scenario={args.scenario} entries={len(transcript.entries)} "
         f"binary={n_bin} fixtures={transcript.fixtures()}")

    async def handler(ws):
        replay = SessionReplay(
            ws, transcript, speed=args.speed, max_gap_s=args.max_gap,
            gate_timeout_s=args.gate_timeout)
        try:
            await replay.run()
        except TimeoutError as e:
            _log(f"ERROR: {e}")
            await ws.close(code=1011, reason=str(e))
        except ConnectionClosed:
            pass

    async with serve(
        handler, args.host, args.port,
        process_request=make_process_request(transcript),
        max_size=200 * 1024 * 1024,
    ):
        _log(f"listening on {args.host}:{args.port} "
             f"(speed x{args.speed}, max-gap {args.max_gap}s)")
        await asyncio.Future()  # run until killed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="baseline_stream")
    ap.add_argument("--refs-dir", default=str(DEFAULT_REFS_DIR))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18931)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="divide recorded inter-frame gaps by this")
    ap.add_argument("--max-gap", type=float, default=0.3,
                    help="cap any single inter-frame sleep (seconds); "
                         "compresses the recorded model-load pause")
    ap.add_argument("--gate-timeout", type=float, default=120.0,
                    help="max seconds to wait at a client-send gate")
    args = ap.parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
