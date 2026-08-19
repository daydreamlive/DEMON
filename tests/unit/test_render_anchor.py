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
    playhead, advance, start, anchored, queue_anchor = (
        runner._render_placement(10.0, False)
    )
    assert playhead == 4.0
    assert advance == 0.0
    assert start == 0.0
    assert anchored is True
    assert queue_anchor is None


def test_audio_engine_starts_with_anchor_cleared():
    eng = _engine()
    assert eng.render_anchor_s is None
    assert eng.peek_render_anchor() is None


def test_anchor_uses_absolute_song_seconds_with_cyclic_wrap(monkeypatch):
    runner, eng = _runner()
    eng.render_anchor_s = 12.25
    monkeypatch.setattr(runner, "_playhead_seconds_now", lambda: 4.0)
    _, advance, start, anchored, _ = runner._render_placement(10.0, False)
    assert advance == 0.0
    assert start == pytest.approx(2.25)
    assert anchored is True


def test_cleared_anchor_resumes_transport_lead(monkeypatch):
    runner, eng = _runner()
    eng.render_anchor_s = None
    monkeypatch.setattr(runner, "_playhead_seconds_now", lambda: 4.0)
    monkeypatch.setattr(runner, "_decode_advance_s", lambda: 0.25)
    _, advance, start, anchored, queue_anchor = (
        runner._render_placement(10.0, False)
    )
    assert advance == 0.25
    assert start == pytest.approx(4.25)
    assert anchored is False
    assert queue_anchor is None


def test_anchor_is_ignored_by_walk_mode(monkeypatch):
    runner, eng = _runner()
    eng.render_anchor_s = 2.0
    monkeypatch.setattr(runner, "_playhead_seconds_now", lambda: 4.0)
    monkeypatch.setattr(runner, "_decode_advance_s", lambda: 0.25)
    _, advance, start, anchored, _ = runner._render_placement(10.0, True)
    assert advance == 0.25
    assert start == 4.25
    assert anchored is False


# ---- anchor queue ----------------------------------------------------------


def test_queue_head_places_render_when_scalar_clear(monkeypatch):
    runner, eng = _runner()
    eng.set_render_anchor_queue([3.5, 7.0])
    monkeypatch.setattr(runner, "_playhead_seconds_now", lambda: 4.0)
    _, advance, start, anchored, queue_anchor = (
        runner._render_placement(10.0, False)
    )
    assert advance == 0.0
    assert start == pytest.approx(3.5)
    assert anchored is True
    assert queue_anchor is not None
    assert queue_anchor[0] == pytest.approx(3.5)


def test_scalar_anchor_preempts_queue(monkeypatch):
    runner, eng = _runner()
    eng.render_anchor_s = 1.0
    eng.set_render_anchor_queue([3.5, 7.0])
    monkeypatch.setattr(runner, "_playhead_seconds_now", lambda: 4.0)
    _, _, start, anchored, queue_anchor = runner._render_placement(10.0, False)
    assert start == pytest.approx(1.0)
    assert anchored is True
    # Scalar placement must NOT consume the queue — it resumes intact.
    assert queue_anchor is None
    assert eng.peek_render_anchor()[0] == pytest.approx(3.5)


def test_queue_is_ignored_by_walk_mode(monkeypatch):
    runner, eng = _runner()
    eng.set_render_anchor_queue([3.5])
    monkeypatch.setattr(runner, "_playhead_seconds_now", lambda: 4.0)
    monkeypatch.setattr(runner, "_decode_advance_s", lambda: 0.25)
    _, _, start, anchored, queue_anchor = runner._render_placement(10.0, True)
    assert start == 4.25
    assert anchored is False
    assert queue_anchor is None


def test_queue_pop_is_guarded_by_expected_head():
    eng = _engine()
    eng.set_render_anchor_queue([3.5, 7.0])
    head, gen = eng.peek_render_anchor()
    # Wrong expected head: the queue is untouched.
    eng.pop_render_anchor(9.9, gen)
    assert eng.peek_render_anchor()[0] == pytest.approx(3.5)
    eng.pop_render_anchor(3.5, gen)
    assert eng.peek_render_anchor()[0] == pytest.approx(7.0)
    eng.pop_render_anchor(7.0, gen)
    assert eng.peek_render_anchor() is None


def test_queue_pop_is_guarded_against_aba_replacement():
    # Replacing [1, 2] with [1, 3] while ``1`` renders must NOT let the old
    # tick pop the NEW queue's identical head — the generation guard, not
    # the value compare, is what makes replace-mid-tick safe.
    eng = _engine()
    eng.set_render_anchor_queue([1.0, 2.0])
    head, gen = eng.peek_render_anchor()
    eng.set_render_anchor_queue([1.0, 3.0])   # client replace mid-tick
    eng.pop_render_anchor(head, gen)          # stale pop: same value, old gen
    assert eng.peek_render_anchor()[0] == pytest.approx(1.0)
    head2, gen2 = eng.peek_render_anchor()
    eng.pop_render_anchor(head2, gen2)        # current-gen pop works
    assert eng.peek_render_anchor()[0] == pytest.approx(3.0)


def test_queue_wraps_cyclically_like_scalar(monkeypatch):
    runner, eng = _runner()
    eng.set_render_anchor_queue([12.25])
    monkeypatch.setattr(runner, "_playhead_seconds_now", lambda: 4.0)
    _, _, start, anchored, queue_anchor = runner._render_placement(10.0, False)
    assert start == pytest.approx(2.25)
    # The pop key is the RAW queued value, not the wrapped placement.
    assert queue_anchor[0] == pytest.approx(12.25)
    assert anchored is True


def test_sticky_params_survive_newest_wins_coalescing():
    from demos.realtime_motion_graph_web.ws_adapter import _fold_sticky_params

    # Superseded snapshot carried the anchor + queue; newer one doesn't.
    newer = {"raw": {}, "playback_pos": 1.0}
    _fold_sticky_params(
        {"render_anchor_s": 2.5, "render_anchor_queue_s": [1.0]}, newer,
    )
    assert newer["render_anchor_s"] == 2.5
    assert newer["render_anchor_queue_s"] == [1.0]

    # A newer explicit value — INCLUDING null (a clear) — wins.
    newer = {"render_anchor_s": None, "render_anchor_queue_s": [9.0]}
    _fold_sticky_params(
        {"render_anchor_s": 2.5, "render_anchor_queue_s": [1.0]}, newer,
    )
    assert newer["render_anchor_s"] is None
    assert newer["render_anchor_queue_s"] == [9.0]


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


def test_queue_absent_retains_list_replaces_null_clears():
    session = _session_for_params()
    eng = session.audio_eng
    session.set_knobs({}, 1.0, render_anchor_queue_s=[3.5, 7.0])
    assert eng.peek_render_anchor()[0] == pytest.approx(3.5)
    session.set_knobs({}, 2.0)
    assert eng.peek_render_anchor()[0] == pytest.approx(3.5)
    session.set_knobs({}, 3.0, render_anchor_queue_s=[8.0])
    assert eng.peek_render_anchor()[0] == pytest.approx(8.0)
    session.set_knobs({}, 4.0, render_anchor_queue_s=None)
    assert eng.peek_render_anchor() is None


def test_nonempty_queue_bumps_activity_clock():
    session = _session_for_params()
    session.state.last_activity_ts = 0.0
    session.set_knobs({}, 1.0, render_anchor_queue_s=[3.5])
    assert session.state.last_activity_ts > 0.0
    # Clearing is not activity — an idle pod must not be woken to do nothing.
    session.state.last_activity_ts = 0.0
    session.set_knobs({}, 2.0, render_anchor_queue_s=[])
    assert session.state.last_activity_ts == 0.0


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
