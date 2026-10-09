"""``set_x0_target_prompt``: session lifecycle around the SA3 backend.

CPU-only: the real SA3Backend + StreamPipeline (mock DiT, via the
test_sa3_backend helpers) behind a bare StreamingSession, checking the
capability gate and that progress reaches the bus as X0TargetState.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import torch

from acestep.streaming.events import CommandFailed, X0TargetState
from acestep.streaming.session import StreamingSession

from tests.unit.test_sa3_backend import C, CTX, T, _backend, _knobs, _x0_backend


def _session(backend):
    s = StreamingSession.__new__(StreamingSession)
    s.backend = backend
    s.state = SimpleNamespace(_lock=threading.RLock(), last_activity_ts=0.0)
    events: list = []
    s.bus = SimpleNamespace(publish=events.append)
    return s, events


def test_unsupported_backend_fails_loudly():
    s, events = _session(_backend())  # no prompt rebuilder
    s.set_x0_target_prompt("song")
    (ev,) = events
    assert isinstance(ev, CommandFailed)


def test_progress_reaches_the_bus():
    b = _x0_backend(source_latent_bct=torch.randn(1, C, T))
    s, events = _session(b)

    s.set_x0_target_prompt("song")
    assert events == [X0TargetState(status="generating", tags="song")]

    knobs = _knobs(b)
    for _ in range(20):
        b.produce(knobs, CTX, "generate")
        s._service_x0_target()
        if len(events) > 1:
            break
    assert events[-1] == X0TargetState(status="ready", tags="song")

    s.set_x0_target_prompt("")
    assert events[-1] == X0TargetState(status="cleared")


def test_service_keeps_the_session_awake_while_generating():
    b = _x0_backend(source_latent_bct=torch.randn(1, C, T))
    s, _ = _session(b)
    s.set_x0_target_prompt("song")
    s.state.last_activity_ts = 0.0
    s._service_x0_target()
    assert s.state.last_activity_ts > 0.0


def test_backend_error_publishes_failed():
    def rebuilder(tags, steps, duration_s):
        raise RuntimeError("encoder exploded")

    s, events = _session(_backend(prompt_rebuilder=rebuilder))
    s.set_x0_target_prompt("song")
    (ev,) = events
    assert ev.status == "failed" and "encoder exploded" in ev.error
