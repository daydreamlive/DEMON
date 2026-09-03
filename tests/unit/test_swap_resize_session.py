"""Session-side adoption of a backend-owned swap resize.

``StreamingSession._apply_swap_backend_owned`` is the half of the
swap-resize path that lives above the backend: it gates the request on
the ``swap_resize`` capability, lifts the truncation ceiling so a longer
source survives long enough for the backend to see it, and adopts the
playback length the hook hands back (``duration`` / ``playback_samples``
/ ``max_seconds`` / the client buffer). Driven here with a stub backend
and unbound session methods — no GPU, no model load, same pattern as
test_lora_facade_session.
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.streaming.generator_backend import Capabilities
from acestep.streaming.session import SAMPLE_RATE, StreamingSession

# 48 kHz delivery: the create-time window and what a resize grows it to.
OLD_PLAY = 12 * SAMPLE_RATE
NEW_PLAY = 30 * SAMPLE_RATE


class _StubBackend:
    """Backend-owned-swap surface: records the hook call, returns a new
    playback length when asked to resize."""

    name = "sa3-stub"

    def __init__(self, *, swap_resize=True, new_playback=None,
                 max_duration_s=None, raises=None):
        self._caps = Capabilities(swap=True, swap_resize=swap_resize)
        self._new_playback = new_playback
        self._max_duration_s = max_duration_s
        self._raises = raises
        self.calls: list = []
        self.playable_s = OLD_PLAY / SAMPLE_RATE

    def capabilities(self):
        return self._caps

    def playable_duration_s(self):
        return self.playable_s

    def handle_swap_source(self, waveform, sample_rate, duration_s=None):
        self.calls.append({
            "samples": int(waveform.shape[-1]),
            "channels": int(waveform.shape[0]),
            "sample_rate": int(sample_rate),
            "duration_s": duration_s,
        })
        if self._raises is not None:
            raise self._raises
        if duration_s is None or self._new_playback is None:
            return None
        self.playable_s = self._new_playback / SAMPLE_RATE
        return self._new_playback


class _MaxDurationBackend(_StubBackend):
    """Declares the optional family ceiling hook."""

    def max_duration_s(self):
        return self._max_duration_s


class _Bus:
    def __init__(self):
        self.events: list = []

    def publish(self, event):
        self.events.append(event)


class _AudioEng:
    def __init__(self, n):
        self.current = np.zeros((n, 2), dtype=np.float32)
        self.position = 999
        self.loop_band = (1.0, 2.0)

    def swap(self, data):
        self.current = data


class _SwapSession:
    """Just enough StreamingSession surface for the backend-owned swap."""

    _apply_swap_backend_owned = StreamingSession._apply_swap_backend_owned

    def __init__(self, backend, *, max_seconds=OLD_PLAY / SAMPLE_RATE):
        self.backend = backend
        self.max_seconds = max_seconds
        self.audio_eng = _AudioEng(OLD_PLAY)
        self.bus = _Bus()
        self.state = types.SimpleNamespace(
            _lock=threading.Lock(),
            playback_samples=OLD_PLAY,
            duration=OLD_PLAY / SAMPLE_RATE,
            n_channels=2,
            source_epoch=3,
            bpm=None, key=None, time_signature=None,
        )

    def swap(self, seconds, *, duration_s=None, channels=2):
        wf = torch.ones(channels, int(seconds * SAMPLE_RATE))
        self._apply_swap_backend_owned(
            self.backend.handle_swap_source, wf, None, duration_s,
        )
        return wf


def _ready(ss):
    assert len(ss.bus.events) == 1, ss.bus.events
    return ss.bus.events[0]


def test_resize_adopts_the_returned_playback_length():
    """The hook's return value IS the new session geometry: duration,
    playback_samples, the later-swap ceiling, and the client buffer all
    follow it, and SwapReady carries the resized shape."""
    be = _StubBackend(new_playback=NEW_PLAY)
    ss = _SwapSession(be)

    ss.swap(30.0, duration_s=30.0)

    assert be.calls[0]["duration_s"] == 30.0
    assert ss.state.playback_samples == NEW_PLAY
    assert abs(ss.state.duration - 30.0) < 1e-6
    # The ceiling for LATER swaps follows the new window, not the
    # create-time one.
    assert abs(ss.max_seconds - 30.0) < 1e-6
    assert len(ss.audio_eng.current) == NEW_PLAY
    # A band from the previous song is meaningless against the new
    # buffer, and playback restarts at its head.
    assert ss.audio_eng.loop_band is None
    assert ss.audio_eng.position == 0
    assert ss.state.source_epoch == 4

    ready = _ready(ss)
    assert abs(ready.duration - 30.0) < 1e-6
    assert len(ready.initial_buffer) == NEW_PLAY
    assert ready.channels == 2


def test_resize_to_a_shorter_window_truncates_the_buffer():
    be = _StubBackend(new_playback=6 * SAMPLE_RATE)
    ss = _SwapSession(be)

    ss.swap(6.0, duration_s=6.0)

    assert ss.state.playback_samples == 6 * SAMPLE_RATE
    assert len(ss.audio_eng.current) == 6 * SAMPLE_RATE
    assert abs(ss.max_seconds - 6.0) < 1e-6


def test_short_source_is_padded_out_to_the_resized_window():
    """The client buffer always matches the render geometry: a source
    shorter than the requested window is zero-padded, not left ragged."""
    be = _StubBackend(new_playback=NEW_PLAY)
    ss = _SwapSession(be)

    ss.swap(10.0, duration_s=30.0)

    buf = ss.audio_eng.current
    assert len(buf) == NEW_PLAY
    assert buf[: 10 * SAMPLE_RATE].any()
    assert not buf[10 * SAMPLE_RATE:].any()   # the pad


def test_plain_swap_keeps_the_session_geometry():
    """No duration_s = the legacy fixed-geometry swap: the hook is called
    without the kwarg and nothing about the session's shape moves."""
    be = _StubBackend(new_playback=NEW_PLAY)
    ss = _SwapSession(be)

    ss.swap(30.0)

    assert be.calls == [{
        "samples": OLD_PLAY, "channels": 2,
        "sample_rate": SAMPLE_RATE, "duration_s": None,
    }]
    assert ss.state.playback_samples == OLD_PLAY
    assert abs(ss.max_seconds - OLD_PLAY / SAMPLE_RATE) < 1e-6
    assert len(_ready(ss).initial_buffer) == OLD_PLAY


def test_resize_is_ignored_without_the_capability():
    """A new client against a backend that doesn't declare swap_resize:
    the field is dropped (loudly) and the swap runs the legacy path —
    the hook never sees a duration_s it might not accept."""
    be = _StubBackend(swap_resize=False, new_playback=NEW_PLAY)
    ss = _SwapSession(be)

    ss.swap(30.0, duration_s=30.0)

    assert be.calls[0]["duration_s"] is None
    assert be.calls[0]["samples"] == OLD_PLAY   # cut at the old ceiling
    assert ss.state.playback_samples == OLD_PLAY


def test_resize_lifts_the_truncation_ceiling_for_a_longer_source():
    """Without the lift, a 30 s upload would be cut to the 12 s session
    window before the backend ever saw it and the resize would have
    nothing to grow into."""
    be = _StubBackend(new_playback=NEW_PLAY)
    ss = _SwapSession(be)

    ss.swap(30.0, duration_s=30.0)

    assert be.calls[0]["samples"] == 30 * SAMPLE_RATE


def test_backend_ceiling_bounds_the_lifted_truncation():
    """The ceiling is backend-owned: a request past the family's cap is
    bounded by max_duration_s, not by a constant in the session layer."""
    be = _MaxDurationBackend(new_playback=NEW_PLAY, max_duration_s=20.0)
    ss = _SwapSession(be)

    ss.swap(60.0, duration_s=60.0)

    assert be.calls[0]["samples"] == 20 * SAMPLE_RATE
    # The request itself still reaches the backend unclamped — it owns
    # the conditioning decision.
    assert be.calls[0]["duration_s"] == 60.0


def test_mono_upload_is_upmixed_before_the_backend_sees_it():
    be = _StubBackend(new_playback=NEW_PLAY)
    ss = _SwapSession(be)

    ss.swap(30.0, duration_s=30.0, channels=1)

    assert be.calls[0]["channels"] == 2


def test_backend_failure_publishes_swap_failed_and_moves_nothing():
    be = _StubBackend(new_playback=NEW_PLAY, raises=RuntimeError("cuda oom"))
    ss = _SwapSession(be)

    ss.swap(30.0, duration_s=30.0)

    assert ss.state.playback_samples == OLD_PLAY
    assert abs(ss.state.duration - OLD_PLAY / SAMPLE_RATE) < 1e-6
    assert abs(ss.max_seconds - OLD_PLAY / SAMPLE_RATE) < 1e-6
    assert ss.state.source_epoch == 3
    assert len(ss.audio_eng.current) == OLD_PLAY
    failed = _ready(ss)
    assert "cuda oom" in failed.error
