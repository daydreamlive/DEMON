"""``bounce``: full-window render of a generation fresher than the request.

CPU-only: the real SA3Backend + StreamPipeline (mock DiT, fake codec,
via the test_sa3_backend helpers) for the freshness signal and the
render, a fake backend for the generic tiler, and a bare
StreamingSession for the request lifecycle.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from acestep.streaming import session as session_mod
from acestep.streaming.diffusion_backend import DiffusionBackend
from acestep.streaming.events import BounceFailed, BounceReady
from acestep.streaming.generator_backend import (
    AudioChunk,
    AudioGeometry,
    Capabilities,
)
from acestep.streaming.sa3_backend import DELIVERY_SAMPLE_RATE
from acestep.streaming.session import StreamingSession
from demos.realtime_motion_graph_web.audio_codec import send_bounce_payload

from tests.unit.test_sa3_backend import CTX, N44, _backend, _knobs


# ---- freshness --------------------------------------------------------------


def test_bounce_waits_for_a_generation_started_after_the_change():
    b = _backend(steps=3)
    for _ in range(8):
        b.produce(_knobs(b, seed=1), CTX, "generate")
    assert b.has_renderable_state()

    # The change lands, THEN the mark is taken (session order).
    knobs = _knobs(b, seed=2)
    mark = b.bounce_mark()
    assert not b.bounce_ready(mark)

    stale_after_mark = 0
    for _ in range(20):
        if not b.produce(knobs, CTX, "generate"):
            continue
        if b.bounce_ready(mark):
            break
        # Staggered pipeline: results started before the change keep
        # emerging after it, and must not satisfy the bounce.
        assert b._emerged_request.seed == 1
        stale_after_mark += 1
    else:
        pytest.fail("no fresh generation emerged")
    assert stale_after_mark >= 1
    assert b._emerged_request.seed == 2
    assert b._emerged_request.submit_seq > mark


def test_reuse_does_not_count_as_fresh():
    b = _backend()
    for _ in range(8):
        b.produce(_knobs(b), CTX, "generate")
    mark = b.bounce_mark()
    for _ in range(5):
        b.produce(_knobs(b), CTX, "reuse")  # DiT-pause re-adopt
    assert not b.bounce_ready(mark)


# ---- render ------------------------------------------------------------------


def test_sa3_bounce_is_the_full_decode():
    b = _backend()
    for _ in range(8):
        b.produce(_knobs(b), CTX, "generate")
    chunk = b.render_bounce()
    total = round(b.playable_duration_s() * DELIVERY_SAMPLE_RATE)
    assert chunk.start_sample == 0
    assert chunk.pcm.shape == (total, 2)
    full = b._rendered_audio(b._last_result_latent)[:total]
    np.testing.assert_allclose(chunk.pcm, full, atol=1e-6)
    assert b.codec.decodes == 1


class _WindowCodec:
    """decode_window over the same ramp the fake full decode returns."""

    def __init__(self):
        ramp = torch.linspace(-0.5, 0.5, N44)
        self.audio = torch.stack([ramp, -ramp])
        self.windows = 0

    def decode_window(self, latent_bct, start, num):
        self.windows += 1
        out = self.audio[:, start:start + num]
        return torch.nn.functional.pad(out, (0, num - out.shape[-1]))


def test_windowed_codec_bounce_tiles_the_whole_window():
    b = _backend()
    b.codec = _WindowCodec()
    b._windowed_codec = True
    for _ in range(8):
        b.produce(_knobs(b), CTX, "generate")
    chunk = b.render_bounce()
    total = round(b.playable_duration_s() * DELIVERY_SAMPLE_RATE)
    assert chunk.pcm.shape == (total, 2)
    assert b.codec.windows > 1
    import torchaudio

    expect = torchaudio.functional.resample(
        b.codec.audio, 44100, DELIVERY_SAMPLE_RATE,
    ).numpy().T[:total]
    # Interior only: the fake ramp ends in a hard step, where a whole-
    # buffer and a windowed resample ring differently.
    np.testing.assert_allclose(chunk.pcm[:-64], expect[:-64], atol=2e-3)


class _Tiled(DiffusionBackend):
    """Fake whose render_window returns slices of a known signal, with
    the song-end clamp the real backends apply."""

    SR = 48000

    def __init__(self, dur_s, win_s):
        super().__init__()
        self.vae_window = win_s
        self._dur = dur_s
        self._last_result_latent = object()
        n = round(dur_s * self.SR)
        self.signal = np.stack(
            [np.sin(np.arange(n) / 50.0), np.cos(np.arange(n) / 70.0)], axis=1,
        ).astype(np.float32)

    def geometry(self):
        return AudioGeometry(self.SR, 2, 25.0, self._dur)

    def playable_duration_s(self):
        return self._dur

    def render_window(self, t_start_s):
        n = round(self.vae_window * self.SR)
        start = max(0, min(round(t_start_s * self.SR), len(self.signal) - n))
        return AudioChunk(pcm=self.signal[start:start + n].copy(), start_sample=start)


def test_generic_tiler_reassembles_the_window():
    b = _Tiled(dur_s=1.037, win_s=0.1)
    chunk = b.render_bounce()
    np.testing.assert_allclose(chunk.pcm, b.signal, atol=1e-6)


def test_no_fixed_window_cannot_bounce():
    b = _Tiled(dur_s=1.0, win_s=0.1)
    b.playable_duration_s = lambda: None  # walk mode
    assert b.render_bounce() is None


# ---- session lifecycle ---------------------------------------------------------


class _FakeBackend:
    name = "fake"

    def __init__(self, bounce=True, ready=False):
        self._caps = Capabilities(bounce=bounce)
        self.ready = ready
        self.marks = 0

    def capabilities(self):
        return self._caps

    def geometry(self):
        return AudioGeometry(48000, 2, 25.0, 1.0)

    def bounce_mark(self):
        self.marks += 1
        return self.marks

    def bounce_ready(self, mark):
        return self.ready

    def render_bounce(self):
        return AudioChunk(pcm=np.zeros((48000, 2), np.float32), start_sample=0)


def _session(backend):
    s = StreamingSession.__new__(StreamingSession)
    s.backend = backend
    s._bounces = []
    s.state = SimpleNamespace(
        _lock=threading.RLock(), last_activity_ts=0.0,
        pending_register=[], pending_enable=[], pending_disable=[],
        pending_depth=None, swap_pending={"waveform": None},
        params={"num_gens": 7},
    )
    events: list = []
    s.bus = SimpleNamespace(publish=events.append)
    return s, events


def test_unsupported_backend_fails_immediately():
    s, events = _session(_FakeBackend(bounce=False))
    s.bounce("a")
    assert len(events) == 1 and isinstance(events[0], BounceFailed)
    assert events[0].request_id == "a"
    assert s._bounces == []


def test_ready_bounce_publishes_audio():
    backend = _FakeBackend()
    s, events = _session(backend)
    s.bounce("a")
    s._service_bounces()
    assert events == []  # marked, but no fresh generation yet
    backend.ready = True
    s._service_bounces()
    (ev,) = events
    assert isinstance(ev, BounceReady)
    assert (ev.request_id, ev.sample_rate, ev.num_gens) == ("a", 48000, 7)
    assert ev.audio.shape == (48000, 2)
    assert s._bounces == []


def test_mark_waits_for_pending_work():
    backend = _FakeBackend(ready=True)
    s, events = _session(backend)
    s.state.pending_enable.append(("lora", 1.0))
    s.bounce("a")
    s._service_bounces()
    assert events == [] and backend.marks == 0
    s.state.pending_enable.clear()
    s._service_bounces()
    assert backend.marks == 1 and isinstance(events[0], BounceReady)


def test_timeout_fails(monkeypatch):
    monkeypatch.setattr(session_mod, "BOUNCE_TIMEOUT_S", 0.0)
    s, events = _session(_FakeBackend(ready=False))
    s.bounce("a")
    s._service_bounces()
    (ev,) = events
    assert isinstance(ev, BounceFailed) and ev.request_id == "a"
    assert s._bounces == []


# ---- wire framing ----------------------------------------------------------------


def test_bounce_payload_framing():
    sent: list = []
    ws = SimpleNamespace(
        send=lambda m: sent.append(m if isinstance(m, (str, bytes)) else b"".join(m)),
    )
    audio = np.random.default_rng(0).uniform(-1, 1, (96000, 2)).astype(np.float32)
    send_bounce_payload(
        ws, request_id="take-1", audio=audio, sample_rate=48000, num_gens=3,
    )
    header, payload = sent
    assert json.loads(header) == {
        "type": "bounce_ready", "request_id": "take-1", "sample_rate": 48000,
        "channels": 2, "frames": 96000, "num_gens": 3,
    }
    pcm = np.frombuffer(payload, dtype=np.float16).reshape(-1, 2)
    np.testing.assert_allclose(pcm, audio, atol=1e-3)
