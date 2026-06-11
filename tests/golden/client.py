"""Recording WebSocket client for the golden/latency harness.

A self-contained, torch-free driver for the realtime streaming server
(``demos/realtime_motion_graph_web/ws_adapter.py``). Unlike the minimal
``RemoteBackend`` at the top of ``demos/realtime_motion_graph_web/
protocol.py`` it speaks the parts of the wire vocabulary the golden
scenarios need (server-side fixture load, swap_source, enable_lora) and
records every frame it sends or receives (with monotonic timestamps)
into a session transcript that later replay-based app tests can feed to
the browser client.

Deliberately black-box: it imports only the binary-framing constants
from the demo's protocol module, never server or app internals, so the
same harness runs unchanged against any server build that speaks the
wire protocol.
"""

import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from demos.realtime_motion_graph_web.protocol import (
    SAMPLE_RATE,
    SLICE_FLAG_DELTA,
    SLICE_HDR_FMT,
    SLICE_HDR_SIZE,
)

__all__ = ["GoldenClient", "Recorder", "Slice", "SAMPLE_RATE"]


@dataclass
class Slice:
    """One decoded audio slice plus its receive-side timing."""

    flags: int
    start_sample: int
    num_samples: int
    channels: int
    tick_ms: float
    dec_ms: float
    num_gens: int
    audio: np.ndarray  # float32, shape (num_samples, channels)
    recv_at: float  # seconds since connect (monotonic)


class Recorder:
    """Append-only wire transcript: ``transcript.jsonl`` + ``blobs/``.

    Each line is ``{"t": <ms since connect>, "dir": "send"|"recv",
    "kind": "json"|"bin", ...}``. JSON frames embed their payload;
    binary frames reference a blob file and carry a ``role`` tag
    (initial / slice / swap_buffer / stem / pcm_upload) so a replay
    server can re-frame them without re-deriving the state machine.
    """

    def __init__(self, out_dir: Path, save_blobs: bool = True):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._blobs = self.dir / "blobs"
        self._save_blobs = save_blobs
        if save_blobs:
            self._blobs.mkdir(exist_ok=True)
        self._fh = open(self.dir / "transcript.jsonl", "w", encoding="utf-8")
        self._n = 0
        self._t0 = time.monotonic()

    def _t(self) -> float:
        return round((time.monotonic() - self._t0) * 1000.0, 3)

    def json_frame(self, direction: str, payload: dict) -> None:
        self._fh.write(json.dumps(
            {"t": self._t(), "dir": direction, "kind": "json",
             "data": payload}) + "\n")
        self._fh.flush()

    def bin_frame(self, direction: str, data: bytes, role: str) -> None:
        entry: dict = {"t": self._t(), "dir": direction, "kind": "bin",
                       "role": role, "bytes": len(data)}
        if self._save_blobs:
            name = f"{self._n:06d}.bin"
            (self._blobs / name).write_bytes(data)
            entry["blob"] = f"blobs/{name}"
        self._n += 1
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class GoldenClient:
    """Drive one streaming session and mirror its audio state.

    Mirrors the browser client's buffer semantics exactly: the buffer is
    initialized from the binary initial frame, RAW slices overwrite their
    region, DELTA slices add into it, and a ``swap_ready`` binary buffer
    replaces it wholesale. Binary frames are routed by the same pending-
    state machine the web client uses (post-``swap_ready`` buffer, then
    ``stem_assets``-announced stem buffers, everything else is a slice).
    """

    def __init__(self, url: str, recorder: Recorder | None = None):
        from websockets.sync.client import connect as ws_connect

        self.url = url
        self.rec = recorder
        self._t0 = time.monotonic()
        self.ws = ws_connect(url, max_size=100 * 1024 * 1024,
                             open_timeout=30)
        self.connected_at = self._now()

        self.ready: dict = {}
        self.ready_at: float | None = None
        self.buffer: np.ndarray | None = None  # (samples, channels) f32
        self.channels = 2
        self.slices: list[Slice] = []
        self.events: list[tuple[float, dict]] = []  # (recv_at, json msg)
        self.gen_samples = 0
        # Buffer regions written by slices since the last (re)init,
        # merged intervals in frame space. Position-aligned coverage is
        # what the canonical comparison region is cut from.
        self.coverage: list[list[int]] = []
        # Binary routing: roles queued by the JSON frames that announce
        # binary follow-ups. Empty queue == the frame is a slice.
        self._expect_bin: list[str] = []

    # ── time ──────────────────────────────────────────────────────────

    def _now(self) -> float:
        return time.monotonic() - self._t0

    # ── sends ─────────────────────────────────────────────────────────

    def _send_json(self, msg: dict) -> None:
        self.ws.send(json.dumps(msg))
        if self.rec:
            self.rec.json_frame("send", msg)

    def send_config(self, config: dict) -> None:
        self._send_json(config)

    def send_pcm(self, interleaved: np.ndarray, channels: int,
                 role: str = "pcm_upload") -> None:
        """``<II`` header + interleaved float32 PCM (the one upload frame
        shape in the protocol)."""
        samples = interleaved.shape[0]
        hdr = struct.pack("<II", channels, samples)
        frame = hdr + interleaved.astype(np.float32).tobytes()
        self.ws.send(frame)
        if self.rec:
            self.rec.bin_frame("send", frame, role)

    def send_params(self, raw: dict, playback_pos: float) -> None:
        self._send_json({"type": "params", "raw": raw,
                         "playback_pos": playback_pos})

    def send_prompt(self, tags: str, tags_b: str | None = None) -> None:
        msg: dict = {"type": "prompt", "tags": tags}
        if tags_b is not None:
            msg["tags_b"] = tags_b
        self._send_json(msg)

    def send_swap_to_fixture(self, fixture_name: str) -> None:
        self._send_json({"type": "swap_source", "use_server_source": True,
                         "fixture_name": fixture_name})

    def send_enable_lora(self, lora_id: str,
                         strength: float | None = None) -> None:
        msg: dict = {"type": "enable_lora", "id": lora_id}
        if strength is not None:
            msg["strength"] = strength
        self._send_json(msg)

    # ── handshake ─────────────────────────────────────────────────────

    def wait_ready(self, timeout: float = 300.0) -> dict:
        """Block until the ``ready`` JSON + binary initial buffer land.
        Raises RuntimeError on a structured ``error`` event."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                msg = self.ws.recv(timeout=5.0)
            except TimeoutError:
                continue  # server still loading models / building session
            if isinstance(msg, str):
                data = json.loads(msg)
                if self.rec:
                    self.rec.json_frame("recv", data)
                if data.get("type") == "error":
                    raise RuntimeError(
                        f"server error during init: {data.get('code')}: "
                        f"{data.get('message')}")
                self.events.append((self._now(), data))
                if data.get("type") == "ready":
                    self.ready = data
                    self.ready_at = self._now()
                    self.channels = int(data["channels"])
            else:
                if self.rec:
                    self.rec.bin_frame("recv", msg, "initial")
                if not self.ready:
                    raise RuntimeError(
                        "binary frame before ready JSON: protocol drift?")
                buf = np.frombuffer(msg, dtype=np.float16).astype(np.float32)
                self.buffer = buf.reshape(-1, self.channels).copy()
                return self.ready
        raise TimeoutError(f"no ready within {timeout}s")

    # ── streaming receive ─────────────────────────────────────────────

    def pump(self, timeout: float = 0.05) -> bool:
        """Receive and process at most one frame. Returns False on
        timeout (nothing pending), True when a frame was handled."""
        try:
            msg = self.ws.recv(timeout=timeout)
        except TimeoutError:
            return False
        if isinstance(msg, str):
            self._on_json(json.loads(msg))
        else:
            self._on_binary(msg)
        return True

    def _on_json(self, data: dict) -> None:
        if self.rec:
            self.rec.json_frame("recv", data)
        self.events.append((self._now(), data))
        mtype = data.get("type")
        if mtype == "swap_ready":
            self._expect_bin.append("swap_buffer")
        elif mtype == "stem_assets":
            self._expect_bin.extend(
                "stem" for _ in data.get("stems", []))

    def _on_binary(self, msg: bytes) -> None:
        role = self._expect_bin.pop(0) if self._expect_bin else "slice"
        if self.rec:
            self.rec.bin_frame("recv", msg, role)
        if role == "swap_buffer":
            buf = np.frombuffer(msg, dtype=np.float16).astype(np.float32)
            self.buffer = buf.reshape(-1, self.channels).copy()
            self.coverage = []  # new source: coverage restarts
            return
        if role == "stem":
            return  # recorded for replay; not part of the mixdown state
        self._on_slice(msg)

    def _on_slice(self, msg: bytes) -> None:
        hdr = struct.unpack(SLICE_HDR_FMT, msg[:SLICE_HDR_SIZE])
        flags = hdr[0]
        payload = msg[SLICE_HDR_SIZE:]
        if flags == SLICE_FLAG_DELTA:
            import zstandard as zstd
            payload = zstd.decompress(payload)
        audio = (np.frombuffer(payload, dtype=np.float16)
                 .astype(np.float32)
                 .reshape(hdr[2], hdr[3]))
        sl = Slice(flags=flags, start_sample=hdr[1], num_samples=hdr[2],
                   channels=hdr[3], tick_ms=hdr[4], dec_ms=hdr[5],
                   num_gens=hdr[6], audio=audio, recv_at=self._now())
        self.slices.append(sl)
        self.gen_samples += sl.num_samples
        # Mirror the browser AudioPlayer: RAW overwrites, DELTA adds.
        if self.buffer is not None:
            s, n = sl.start_sample, sl.num_samples
            if s + n <= self.buffer.shape[0]:
                if flags == SLICE_FLAG_DELTA:
                    self.buffer[s:s + n] += audio
                else:
                    self.buffer[s:s + n] = audio
                self._cover(s, s + n)

    def _cover(self, start: int, end: int) -> None:
        """Merge [start, end) into the coverage interval list."""
        merged = []
        for a, b in self.coverage:
            if end < a or start > b:
                merged.append([a, b])
            else:
                start, end = min(start, a), max(end, b)
        merged.append([start, end])
        merged.sort()
        self.coverage = merged

    def covered_run_from(self, frame: int) -> tuple[int, int] | None:
        """The contiguous covered interval containing ``frame``, or the
        first one starting after it. None when nothing qualifies."""
        for a, b in self.coverage:
            if a <= frame < b:
                return (frame, b)
            if a > frame:
                return (a, b)
        return None

    # ── teardown ──────────────────────────────────────────────────────

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass
        if self.rec:
            self.rec.close()
