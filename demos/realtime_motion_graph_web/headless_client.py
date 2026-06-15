"""Headless PRIMARY client for the realtime motion-graph backend.

Drives a full streaming session over the same WebSocket wire protocol
the browser uses — init handshake, binary slice decode, the periodic
``params`` tick that carries ``playback_pos`` — but with a *simulated*
audio clock instead of an AudioWorklet. This makes the frontend's
realtime behavior reproducible from an agent (the demo's MCP server)
with no browser:

- The session registers in the server-side registry like any other, so
  every existing MCP tool (``set_prompt``, ``set_knob``,
  ``swap_to_fixture``, ...) drives it unchanged.
- Knob persistence works because this client implements the same
  ``params_echo`` mirror the web UI does: EXTERNAL (control-bus) knob
  changes are echoed by the server, merged into this client's raw dict,
  and carried back on the next PRIMARY params tick so they stick.
- The simulated playhead advances at wall-clock rate and is reported to
  the server exactly like the browser reports its audible position.

On top of the transport it measures the two quantities that define
"generation lagging behind the playhead":

- **lead_s** per received slice: how far ahead of the live playhead the
  slice's first sample landed (folded to the circular buffer; negative
  = the server wrote audio *behind* where the listener already is).
- **staleness_s** per params tick: how long ago the audio currently
  under the playhead was last refreshed. Healthy streaming keeps this
  near the playback lead (<~1.5 s); a lagging generator lets the
  playhead run into audio last written a full buffer-lap ago.

Torch-free (numpy + zstandard + websockets), mirroring
:class:`~demos.realtime_motion_graph_web.protocol.RemoteBackend`, so it
can run inside the lightweight MCP stdio process.
"""

from __future__ import annotations

import json
import math
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import zstandard as zstd

from .protocol import (
    SAMPLE_RATE,
    SLICE_FLAG_DELTA,
    SLICE_FLAG_RAW,
    SLICE_HDR_FMT,
    SLICE_HDR_SIZE,
)


# ---------------------------------------------------------------------------
# Simulated playhead
# ---------------------------------------------------------------------------


class PlayheadSim:
    """Wall-clock simulation of the browser's audible playhead.

    Anchor-based like the server's ``_RemotePlayheadClock``: position is
    ``anchor + elapsed`` (samples), modulo the buffer length, advancing
    at wall-clock rate. Reported to the server exactly like the browser
    reports its audible position.
    """

    def __init__(
        self,
        duration_samples: int,
        *,
        sample_rate: int = SAMPLE_RATE,
        now_fn=time.monotonic,
    ):
        self._now = now_fn
        self.sample_rate = int(sample_rate)
        self._lock = threading.Lock()
        self._n = max(1, int(duration_samples))
        self._anchor_sample = 0.0
        self._anchor_wall = self._now()

    def _position_locked(self, now: float) -> float:
        elapsed = max(0.0, now - self._anchor_wall)
        return (
            self._anchor_sample + elapsed * self.sample_rate
        ) % self._n

    @property
    def duration_samples(self) -> int:
        return self._n

    def sample(self) -> int:
        with self._lock:
            return int(self._position_locked(self._now()))

    def seconds(self) -> float:
        with self._lock:
            return self._position_locked(self._now()) / self.sample_rate

    def reset(self, duration_samples: int) -> None:
        """New source buffer (swap): playhead restarts at 0, matching
        the server's ``audio_eng.position = 0`` on swap commit."""
        with self._lock:
            self._n = max(1, int(duration_samples))
            self._anchor_sample = 0.0
            self._anchor_wall = self._now()


# ---------------------------------------------------------------------------
# Lag measurement
# ---------------------------------------------------------------------------


def fold_lead_samples(start_sample: int, playhead_sample: int, n: int) -> int:
    """Signed circular distance from the playhead to a slice start.

    Positive: the slice landed AHEAD of the playhead (healthy).
    Negative: it landed behind — the listener already passed it.
    Folded to ``[-n/2, n/2)``.
    """
    if n <= 0:
        return 0
    lead = (int(start_sample) - int(playhead_sample)) % n
    if lead >= n // 2:
        lead -= n
    return lead


@dataclass
class SliceRecord:
    wall: float
    start_sample: int
    num_samples: int
    lead_s: float
    interval_s: float
    tick_ms: float
    dec_ms: float
    num_gens: int


@dataclass
class TickRecord:
    wall: float
    playhead_s: float
    staleness_s: float


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(arr.min()), 4),
        "p5": round(float(np.percentile(arr, 5)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": round(float(arr.max()), 4),
        "mean": round(float(arr.mean()), 4),
    }


class LagTracker:
    """Per-slice lead + per-tick staleness accounting.

    ``bucket_s``-wide buckets over the circular buffer record the wall
    time each region was last patched; staleness at the playhead is the
    age of the bucket under it. The whole initial buffer counts as
    fresh at construction (the encoded source is playable from t=0).
    """

    def __init__(
        self,
        duration_samples: int,
        *,
        sample_rate: int = SAMPLE_RATE,
        bucket_s: float = 0.25,
        maxlen: int = 100_000,
        now_fn=time.monotonic,
    ):
        self._now = now_fn
        self.sample_rate = int(sample_rate)
        self.bucket_s = float(bucket_s)
        self._lock = threading.Lock()
        self.slices: deque[SliceRecord] = deque(maxlen=maxlen)
        self.ticks: deque[TickRecord] = deque(maxlen=maxlen)
        self._reset_locked(duration_samples)

    def _reset_locked(self, duration_samples: int) -> None:
        self._n = max(1, int(duration_samples))
        n_buckets = max(
            1, math.ceil(self._n / (self.bucket_s * self.sample_rate)),
        )
        self._bucket_stamp = np.full(n_buckets, self._now(), dtype=np.float64)
        self._last_slice_wall: float | None = None
        self.slices.clear()
        self.ticks.clear()

    def reset(self, duration_samples: int) -> None:
        with self._lock:
            self._reset_locked(duration_samples)

    def _bucket_index(self, sample: int) -> int:
        per_bucket = self.bucket_s * self.sample_rate
        return min(
            len(self._bucket_stamp) - 1, int(sample / per_bucket),
        )

    def on_slice(
        self,
        *,
        start_sample: int,
        num_samples: int,
        playhead_sample: int,
        tick_ms: float = 0.0,
        dec_ms: float = 0.0,
        num_gens: int = 0,
    ) -> SliceRecord:
        now = self._now()
        with self._lock:
            lead = fold_lead_samples(start_sample, playhead_sample, self._n)
            interval = (
                0.0 if self._last_slice_wall is None
                else now - self._last_slice_wall
            )
            self._last_slice_wall = now
            b0 = self._bucket_index(start_sample)
            b1 = self._bucket_index(
                min(start_sample + max(0, num_samples - 1), self._n - 1),
            )
            self._bucket_stamp[b0:b1 + 1] = now
            rec = SliceRecord(
                wall=now,
                start_sample=int(start_sample),
                num_samples=int(num_samples),
                lead_s=lead / self.sample_rate,
                interval_s=interval,
                tick_ms=float(tick_ms),
                dec_ms=float(dec_ms),
                num_gens=int(num_gens),
            )
            self.slices.append(rec)
            return rec

    def on_tick(self, playhead_sample: int) -> TickRecord:
        now = self._now()
        with self._lock:
            stamp = self._bucket_stamp[self._bucket_index(playhead_sample)]
            rec = TickRecord(
                wall=now,
                playhead_s=playhead_sample / self.sample_rate,
                staleness_s=max(0.0, now - float(stamp)),
            )
            self.ticks.append(rec)
            return rec

    def report(
        self,
        *,
        window_s: float | None = 30.0,
        stale_threshold_s: float = 3.0,
        include_timeline: bool = False,
        timeline_step_s: float = 1.0,
    ) -> dict:
        """Aggregate lag statistics over the trailing ``window_s``.

        The two verdict-shaped fields:
        - ``slices.lead_s``: distribution of where fresh audio landed
          relative to the playhead. p5 < 0 means slices regularly land
          behind the listener.
        - ``ticks.staleness_s`` + ``stale_ticks``: how old the audio
          under the playhead was. Sustained values near the buffer
          duration mean the generator is a full lap behind.
        """
        now = self._now()
        cutoff = -math.inf if window_s is None else now - float(window_s)
        with self._lock:
            srecs = [r for r in self.slices if r.wall >= cutoff]
            trecs = [r for r in self.ticks if r.wall >= cutoff]
            buffer_s = self._n / self.sample_rate
        behind = [r for r in srecs if r.lead_s < 0.0]
        stale = [r for r in trecs if r.staleness_s > stale_threshold_s]
        out: dict = {
            "window_s": window_s,
            "buffer_duration_s": round(buffer_s, 3),
            "slices": {
                "count": len(srecs),
                "behind_playhead": len(behind),
                "lead_s": _percentiles([r.lead_s for r in srecs]),
                "interval_s": _percentiles(
                    [r.interval_s for r in srecs if r.interval_s > 0.0],
                ),
                "tick_ms": _percentiles([r.tick_ms for r in srecs]),
                "dec_ms": _percentiles([r.dec_ms for r in srecs]),
            },
            "ticks": {
                "count": len(trecs),
                "stale_threshold_s": stale_threshold_s,
                "stale_ticks": len(stale),
                "staleness_s": _percentiles(
                    [r.staleness_s for r in trecs],
                ),
            },
        }
        if srecs:
            out["slices"]["num_gens_last"] = srecs[-1].num_gens
        if stale:
            worst = max(stale, key=lambda r: r.staleness_s)
            out["ticks"]["worst_stale"] = {
                "ago_s": round(now - worst.wall, 2),
                "playhead_s": round(worst.playhead_s, 3),
                "staleness_s": round(worst.staleness_s, 3),
            }
        if include_timeline:
            out["timeline"] = self._timeline(srecs, trecs, timeline_step_s, now)
        return out

    @staticmethod
    def _timeline(
        srecs: list[SliceRecord],
        trecs: list[TickRecord],
        step_s: float,
        now: float,
    ) -> list[dict]:
        """Downsampled per-``step_s`` rollup: worst lead / staleness and
        slice count per step, oldest first."""
        if not srecs and not trecs:
            return []
        t0 = min(
            [r.wall for r in srecs] + [r.wall for r in trecs],
        )
        n_steps = max(1, math.ceil((now - t0) / step_s))
        rows: list[dict] = [
            {"t_ago_s": round(now - (t0 + (i + 1) * step_s), 2)}
            for i in range(n_steps)
        ]
        for r in srecs:
            row = rows[min(n_steps - 1, int((r.wall - t0) / step_s))]
            row["slices"] = row.get("slices", 0) + 1
            row["min_lead_s"] = round(
                min(row.get("min_lead_s", math.inf), r.lead_s), 3,
            )
        for r in trecs:
            row = rows[min(n_steps - 1, int((r.wall - t0) / step_s))]
            row["max_staleness_s"] = round(
                max(row.get("max_staleness_s", 0.0), r.staleness_s), 3,
            )
            row["playhead_s"] = round(r.playhead_s, 2)
        return rows


# ---------------------------------------------------------------------------
# Slice decoding (client-side mirror of SliceCodec)
# ---------------------------------------------------------------------------


@dataclass
class DecodedSlice:
    flags: int
    start_sample: int
    num_samples: int
    channels: int
    tick_ms: float
    dec_ms: float
    num_gens: int
    audio: np.ndarray  # (num_samples, channels) float32


def decode_slice_frame(msg: bytes) -> DecodedSlice:
    """Decode one binary slice frame (header + float16 payload,
    optionally zstd-compressed delta) into a typed record. Inverse of
    :meth:`demos.realtime_motion_graph_web.audio_codec.SliceCodec.encode`.
    """
    if len(msg) < SLICE_HDR_SIZE:
        raise ValueError(f"slice frame too short: {len(msg)} bytes")
    flags, ss, n, ch, tick_ms, dec_ms, num_gens = struct.unpack(
        SLICE_HDR_FMT, msg[:SLICE_HDR_SIZE],
    )
    payload = msg[SLICE_HDR_SIZE:]
    if flags == SLICE_FLAG_DELTA:
        payload = zstd.decompress(payload, max_output_size=n * ch * 2)
    audio = (
        np.frombuffer(payload, dtype=np.float16)
        .astype(np.float32)
        .reshape(n, ch)
    )
    return DecodedSlice(
        flags=flags, start_sample=ss, num_samples=n, channels=ch,
        tick_ms=tick_ms, dec_ms=dec_ms, num_gens=num_gens, audio=audio,
    )


def apply_slice(mirror: np.ndarray, sl: DecodedSlice) -> None:
    """Apply a decoded slice to the client mirror in place. Delta frames
    add (the server encodes against the float16-quantized reconstruction
    we hold, so += converges); raw frames overwrite."""
    ss = sl.start_sample
    se = min(ss + sl.num_samples, len(mirror))
    if se <= ss:
        return
    region = sl.audio[: se - ss]
    if sl.flags == SLICE_FLAG_RAW:
        mirror[ss:se] = region
    else:
        mirror[ss:se] += region


# ---------------------------------------------------------------------------
# The headless client
# ---------------------------------------------------------------------------


class HeadlessClientError(RuntimeError):
    pass


@dataclass
class _BinaryExpectation:
    """The next binary frame is NOT a slice: it's the payload announced
    by a preceding JSON frame (swap buffer / stem audio)."""
    kind: str  # "swap_buffer" | "stem"
    meta: dict = field(default_factory=dict)


class HeadlessClient:
    """A full PRIMARY streaming client with a simulated audio clock.

    Lifecycle: construct → :meth:`start` (blocking handshake) →
    background recv + params-tick threads run until :meth:`stop` or the
    server closes. All public accessors are thread-safe.
    """

    def __init__(
        self,
        url: str,
        config: dict,
        waveform: np.ndarray | None,
        *,
        params_hz: float = 30.0,
        now_fn=time.monotonic,
    ):
        self.url = url
        self.config = dict(config)
        # (channels, samples) float32, or None when the server loads the
        # fixture itself (config.use_server_fixture).
        self._waveform = waveform
        self.params_hz = max(1.0, float(params_hz))
        self._now = now_fn

        self.ws = None
        self.ready: dict = {}
        self.session_id: str | None = None
        self.channels = 2
        self.mirror: np.ndarray | None = None  # (samples, channels) f32
        self.player: PlayheadSim | None = None
        self.tracker: LagTracker | None = None

        self.running = False
        self.closed_reason: str | None = None
        self._threads: list[threading.Thread] = []
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._expectations: deque[_BinaryExpectation] = deque()

        # PRIMARY raw dict carried on every params tick. Starts empty
        # (server-side defaults rule); grows by merging params_echo so
        # MCP-driven knob changes persist exactly like the web UI's
        # useMcpMirror makes them persist.
        self._raw: dict = {}

        # Recent server JSON events (type + wall + payload extract) for
        # status/debugging.
        self.events: deque[dict] = deque(maxlen=200)
        self.last_params_update: dict = {}
        self.slice_count = 0

    # ---- lifecycle -------------------------------------------------------

    def start(self, timeout_s: float = 180.0) -> dict:
        from websockets.sync.client import connect as ws_connect

        deadline = self._now() + float(timeout_s)
        self.ws = ws_connect(
            self.url,
            max_size=100 * 1024 * 1024,
            open_timeout=min(30.0, timeout_s),
        )
        try:
            self.ws.send(json.dumps(self.config))
            if self._waveform is not None:
                wav = np.ascontiguousarray(self._waveform.T, dtype=np.float32)
                hdr = struct.pack("<II", wav.shape[1], wav.shape[0])
                self.ws.send(hdr + wav.tobytes())

            # First JSON must be ready (or a typed error). init_ack only
            # ships when the config carries telemetry_version, which we
            # don't send.
            while True:
                remaining = deadline - self._now()
                if remaining <= 0:
                    raise HeadlessClientError(
                        "timed out waiting for the ready frame (is the "
                        "backend warm? a cold TRT engine load can take "
                        "minutes on first session)",
                    )
                try:
                    msg = self.ws.recv(timeout=min(remaining, 10.0))
                except TimeoutError:
                    # Session create can block for minutes on a cold
                    # backend; keep waiting until the caller's deadline.
                    continue
                if isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("type") == "ready":
                        self.ready = data
                        break
                    if data.get("type") == "error":
                        raise HeadlessClientError(
                            f"session init failed: {data.get('code')}: "
                            f"{data.get('message')}",
                        )
                    # init_ack or other pre-ready frames: ignore.

            geometry = self.ready.get("geometry") or {}
            sr = int(geometry.get("sample_rate") or self.ready["sample_rate"])
            if sr != SAMPLE_RATE:
                raise HeadlessClientError(
                    f"unsupported sample rate {sr} (client assumes "
                    f"{SAMPLE_RATE})",
                )
            self.channels = int(self.ready["channels"])
            self.session_id = self.ready.get("session_id")

            # Binary initial buffer (float16, fragmented sends are
            # reassembled by the websockets message layer).
            init_bytes = self._recv_binary_blocking(deadline)
            self.mirror = (
                np.frombuffer(init_bytes, dtype=np.float16)
                .astype(np.float32)
                .reshape(-1, self.channels)
            )
        except BaseException:
            try:
                self.ws.close()
            except Exception:
                pass
            raise

        n = len(self.mirror)
        self.player = PlayheadSim(n, now_fn=self._now)
        self.tracker = LagTracker(n, now_fn=self._now)
        self.running = True
        for fn, name in (
            (self._recv_loop, "headless-recv"),
            (self._tick_loop, "headless-params-tick"),
        ):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return self.ready

    def _recv_binary_blocking(self, deadline: float) -> bytes:
        while True:
            remaining = deadline - self._now()
            if remaining <= 0:
                raise HeadlessClientError(
                    "timed out waiting for a binary frame during init",
                )
            try:
                msg = self.ws.recv(timeout=min(remaining, 10.0))
            except TimeoutError:
                continue
            if isinstance(msg, (bytes, bytearray)):
                return bytes(msg)
            # Stray JSON between ready and the buffer (e.g. lora_catalog)
            # is legal; record and keep waiting.
            try:
                self._handle_json(json.loads(msg))
            except Exception:
                pass

    def stop(self) -> None:
        self.running = False
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads.clear()

    # ---- background loops --------------------------------------------------

    def _recv_loop(self) -> None:
        from websockets.exceptions import ConnectionClosed

        while self.running:
            try:
                msg = self.ws.recv(timeout=0.25)
            except TimeoutError:
                continue
            except ConnectionClosed as exc:
                self.closed_reason = f"connection closed: {exc}"
                self.running = False
                break
            except Exception as exc:
                self.closed_reason = f"recv failed: {exc}"
                self.running = False
                break
            try:
                if isinstance(msg, str):
                    self._handle_json(json.loads(msg))
                else:
                    self._handle_binary(bytes(msg))
            except Exception as exc:
                # One bad frame must not kill the transport.
                self.events.append({
                    "wall": self._now(),
                    "type": "_client_decode_error",
                    "error": str(exc),
                })

    def _tick_loop(self) -> None:
        from websockets.exceptions import ConnectionClosed

        interval = 1.0 / self.params_hz
        while self.running:
            t0 = self._now()
            player = self.player
            tracker = self.tracker
            if player is not None and tracker is not None:
                pos = player.sample()
                with self._state_lock:
                    raw = dict(self._raw)
                try:
                    with self._send_lock:
                        self.ws.send(json.dumps({
                            "type": "params",
                            "raw": raw,
                            "playback_pos": pos / SAMPLE_RATE,
                        }))
                except ConnectionClosed as exc:
                    self.closed_reason = f"connection closed: {exc}"
                    self.running = False
                    break
                except Exception:
                    pass
                tracker.on_tick(pos)
            elapsed = self._now() - t0
            time.sleep(max(0.0, interval - elapsed))

    # ---- frame handling ------------------------------------------------------

    def _handle_json(self, data: dict) -> None:
        mtype = data.get("type")
        # params_update rides every slice (~15-25 Hz) and would drown
        # the event log; it's kept separately in last_params_update.
        if mtype != "params_update":
            self.events.append({
                "wall": self._now(),
                "type": mtype,
                **{
                    k: data[k] for k in (
                        "error", "code", "message", "command",
                        "fixture_name", "duration", "tags", "value",
                        "count", "source_epoch",
                    ) if k in data
                },
            })
        if mtype == "params_echo":
            # Adopt EXTERNAL knob changes so the next PRIMARY tick
            # carries them back and they persist server-side — the
            # web UI's useMcpMirror contract.
            raw = data.get("raw") or {}
            with self._state_lock:
                self._raw.update(raw)
        elif mtype == "params_update":
            self.last_params_update = data.get("params") or {}
        elif mtype == "swap_ready":
            self._expectations.append(_BinaryExpectation(
                kind="swap_buffer", meta=data,
            ))
        elif mtype == "stem_assets":
            for name in data.get("stems") or []:
                self._expectations.append(_BinaryExpectation(
                    kind="stem", meta={"stem": name},
                ))

    def _handle_binary(self, msg: bytes) -> None:
        if self._expectations:
            exp = self._expectations.popleft()
            if exp.kind == "swap_buffer":
                self._adopt_swap_buffer(msg, exp.meta)
            # "stem": measurement client has no use for stem PCM; drop.
            return
        sl = decode_slice_frame(msg)
        mirror = self.mirror
        if mirror is None:
            return
        apply_slice(mirror, sl)
        self.slice_count += 1
        player = self.player
        tracker = self.tracker
        if player is not None and tracker is not None:
            tracker.on_slice(
                start_sample=sl.start_sample,
                num_samples=sl.num_samples,
                playhead_sample=player.sample(),
                tick_ms=sl.tick_ms,
                dec_ms=sl.dec_ms,
                num_gens=sl.num_gens,
            )

    def _adopt_swap_buffer(self, msg: bytes, meta: dict) -> None:
        channels = int(meta.get("channels") or self.channels)
        new_mirror = (
            np.frombuffer(msg, dtype=np.float16)
            .astype(np.float32)
            .reshape(-1, channels)
        )
        self.channels = channels
        self.mirror = new_mirror
        n = len(new_mirror)
        if self.player is not None:
            self.player.reset(n)
        if self.tracker is not None:
            self.tracker.reset(n)

    # ---- public accessors -----------------------------------------------

    def send_command(self, data: dict) -> None:
        """Send a raw JSON command on the PRIMARY socket (no binary
        follow-up). For commands with binary payloads keep using the
        control bus."""
        with self._send_lock:
            self.ws.send(json.dumps(data))

    def status(self) -> dict:
        player = self.player
        tracker = self.tracker
        out = {
            "running": self.running,
            "closed_reason": self.closed_reason,
            "session_id": self.session_id,
            "url": self.url,
            "slice_count": self.slice_count,
            "params_hz": self.params_hz,
        }
        if player is not None:
            out.update({
                "playhead_s": round(player.seconds(), 3),
                "buffer_duration_s": round(
                    player.duration_samples / SAMPLE_RATE, 3,
                ),
            })
        if tracker is not None:
            out["lag"] = tracker.report(window_s=10.0)
        recent = list(self.events)[-8:]
        now = self._now()
        out["recent_events"] = [
            {**{k: v for k, v in e.items() if k != "wall"},
             "ago_s": round(now - e["wall"], 1)}
            for e in recent
        ]
        return out
