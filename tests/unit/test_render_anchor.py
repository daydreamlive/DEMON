"""Focused coverage for explicit stationary render placement."""

from types import SimpleNamespace
import threading

import numpy as np
import pytest

from acestep.streaming.audio_engine import AudioEngine
from acestep.streaming.generator_backend import LeadProfile
from acestep.streaming.pipeline_runner import PipelineRunner, SAMPLE_RATE
from acestep.streaming.session import StreamingSession


class _Backend:
    name = "fake"

    def playable_duration_s(self):
        return 10.0

    def lead_profile(self):
        return LeadProfile()


class _State:
    running = False
    params = {}
    last_activity_ts = 0.0


def _engine(duration_s=10.0):
    return AudioEngine(
        np.zeros((int(duration_s * SAMPLE_RATE), 2), dtype=np.float32),
        SAMPLE_RATE,
    )


def _runner():
    eng = _engine()
    return PipelineRunner(_Backend(), eng, state=_State(), vae_window=0.36), eng


def test_anchor_zero_bypasses_transport_lead(monkeypatch):
    runner, eng = _runner()
    eng.render_anchor_s = 0.0
    monkeypatch.setattr(runner, "_playhead_seconds_now", lambda: 4.0)
    playhead, advance, start, anchored = runner._render_placement(10.0, False)
    assert playhead == 4.0
    assert advance == 0.0
    assert start == 0.0
    assert anchored is True


def test_audio_engine_starts_with_anchor_cleared():
    assert _engine().render_anchor_s is None


def test_anchor_uses_absolute_song_seconds_with_cyclic_wrap(monkeypatch):
    runner, eng = _runner()
    eng.render_anchor_s = 12.25
    monkeypatch.setattr(runner, "_playhead_seconds_now", lambda: 4.0)
    _, advance, start, anchored = runner._render_placement(10.0, False)
    assert advance == 0.0
    assert start == pytest.approx(2.25)
    assert anchored is True


def test_cleared_anchor_resumes_transport_lead(monkeypatch):
    runner, eng = _runner()
    eng.render_anchor_s = None
    monkeypatch.setattr(runner, "_playhead_seconds_now", lambda: 4.0)
    monkeypatch.setattr(runner, "_decode_advance_s", lambda: 0.25)
    _, advance, start, anchored = runner._render_placement(10.0, False)
    assert advance == 0.25
    assert start == pytest.approx(4.25)
    assert anchored is False


def test_anchor_is_ignored_by_walk_mode(monkeypatch):
    runner, eng = _runner()
    eng.render_anchor_s = 2.0
    monkeypatch.setattr(runner, "_playhead_seconds_now", lambda: 4.0)
    monkeypatch.setattr(runner, "_decode_advance_s", lambda: 0.25)
    _, advance, start, anchored = runner._render_placement(10.0, True)
    assert advance == 0.25
    assert start == 4.25
    assert anchored is False


def _session_for_params():
    session = StreamingSession.__new__(StreamingSession)
    session.audio_eng = _engine()
    session.state = SimpleNamespace(
        last_params_raw={}, last_activity_ts=0.0, _lock=threading.RLock(),
    )
    session.virtual_knobs = {}
    session._knob_specs_by_name = {}
    session._report_staleness = SimpleNamespace(staleness_s=lambda *_: 0.0)
    return session


def test_anchor_absent_retains_and_null_clears():
    session = _session_for_params()
    session.set_knobs({}, 1.0, render_anchor_s=0.0)
    assert session.audio_eng.render_anchor_s == 0.0
    session.set_knobs({}, 2.0)
    assert session.audio_eng.render_anchor_s == 0.0
    session.set_knobs({}, 3.0, render_anchor_s=None)
    assert session.audio_eng.render_anchor_s is None


def test_clockless_params_do_not_reset_playhead():
    session = _session_for_params()
    session.audio_eng.position = 7 * SAMPLE_RATE
    session.set_knobs({}, None)
    assert session.audio_eng.position == 7 * SAMPLE_RATE


def test_patch_wrapped_writes_and_emits():
    runner, eng = _runner()
    pcm = np.full((SAMPLE_RATE, 2), 0.5, dtype=np.float32)
    backend = SimpleNamespace(render_window=lambda _: SimpleNamespace(pcm=pcm))
    emitted = []
    runner.on_audio_ready = lambda wav, start, end: emitted.append((start, end))
    runner._patch_wrapped(backend, eng.current, 0.0, 0, SAMPLE_RATE // 2)
    assert eng.current[0, 0] == pytest.approx(0.5)
    assert emitted == [(0, SAMPLE_RATE // 2)]


def test_emit_trim_never_suppresses_stationary_anchor_window():
    runner, _ = _runner()
    runner._emit_trim = True
    assert runner._should_trim_window_emit(None, anchored=False) is True
    assert runner._should_trim_window_emit(None, anchored=True) is False
    assert runner._should_trim_window_emit(48000, anchored=False) is False
