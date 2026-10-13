"""The live_input family ("Audio input"): a canvas the client feeds.

CPU only, no model: sessions are real StreamingSessions built through the
family's create path, the analysis tap runs against a stub submitter, and
the runner test drives the real PipelineRunner for a fraction of a second.
"""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from acestep.analysis.engine import AnalysisResult
from acestep.analysis.registry import AnalyzerOption, AnalyzerSpec, register
from acestep.analysis.scheduler import Governor
from acestep.analysis.tap import AnalysisTap
from acestep.streaming import live_input_backend as lib
from acestep.streaming.config import SessionConfig
from acestep.streaming.events import (
    AnalysisUpdate,
    AudioInAck,
    AudioReady,
    CommandFailed,
    SwapReady,
)
from acestep.streaming.families import (
    LIVE_INPUT_CAPTURE_DEFAULT_S,
    LIVE_INPUT_CAPTURE_MAX_S,
    LIVE_INPUT_CAPTURE_MIN_S,
    get_family,
    resolve_checkpoint,
)
from acestep.streaming.generator_backend import Capabilities
from acestep.streaming.preflight import PreflightRequest
from acestep.streaming.session import StreamingSession
from acestep.streaming.source import Audio

SR = 48000

STUB = register(AnalyzerSpec(
    name="stub_live_notes",
    display_name="Stub live notes",
    item_kind="note",
    make_engine=lambda size, device: None,
    probe=lambda: None,
    licence="stub licence",
    sizes=("small",),
    default_size="small",
    window_s=4.0,
    context_s=0.5,
    options=(
        AnalyzerOption("size", "str", default=None, bounds=("small",)),
        AnalyzerOption("horizon_s", "float", default=8.0, bounds=(4.0, 30.0)),
    ),
))


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class RecordingBus:
    def __init__(self):
        self.events = []
        self.listeners = []

    def publish(self, event):
        self.events.append(event)

    def subscribe(self, listener, **_kw):
        self.listeners.append(listener)
        return listener

    def unsubscribe(self, sub):
        if sub in self.listeners:
            self.listeners.remove(sub)

    def close(self):
        self.listeners.clear()

    def deliver(self, event):
        for fn in list(self.listeners):
            fn(event)

    def of(self, cls):
        return [e for e in self.events if isinstance(e, cls)]


class StubSubmitter:
    def __init__(self):
        self.submitted = []

    def submit(self, analyzer, size, request, on_result, on_error, cancel_token):
        self.submitted.append((request, on_result, on_error, cancel_token))

    def cancel(self, cancel_token):
        n = sum(1 for s in self.submitted if s[3] is cancel_token)
        self.submitted = [s for s in self.submitted if s[3] is not cancel_token]
        return n

    def engine_info(self, analyzer, size):
        return {"name": "stubengine", "size": size, "loaded": False, "licence": "stub licence"}


def _config(**extra):
    cfg = SessionConfig.from_dict({"backend": "live_input", **extra})
    cfg.backend = "live_input"
    return cfg


def _create(*, clip=None, capture_s=30.0, text_only=None):
    cfg = _config(live_input_capture_s=capture_s)
    if text_only is None:
        text_only = clip is None
    cfg.text_only = text_only
    wf = clip if clip is not None else torch.zeros(2, SR)
    ss = StreamingSession.create(
        audio=Audio(waveform=wf, sample_rate=SR),
        config=cfg,
        checkpoint="live_input/v1",
        session_id="test-live",
    )
    bus = RecordingBus()
    # Swap the real bus for a synchronous recorder; the backend publishes
    # through the session's bus object, so rebind both.
    ss.bus.close()
    ss.bus = bus
    ss.backend.bus = bus
    return ss, bus


def _frame(n=4800, value=0.5, channels=1):
    return np.full((n, channels), value, dtype=np.float32)


def _backend(*, capture_s=30.0, clock=None):
    """A capture backend over a real AudioEngine, with a recording bus."""
    from acestep.streaming.audio_engine import AudioEngine

    n = int(lib.clamp_capture_s(capture_s) * SR)
    eng = AudioEngine(np.zeros((n, 1), dtype=np.float32), SR)
    state = SimpleNamespace(
        _lock=threading.Lock(), source_epoch=0, params={"x": 1},
        playback_samples=n, duration=n / SR, n_channels=1,
    )
    bus = RecordingBus()
    be = lib.LiveInputBackend(
        audio_eng=eng, state=state, bus=bus, mode=lib.MODE_CAPTURE,
        capture_s=capture_s, clock=clock or Clock(),
    )
    return be, eng, state, bus


# ---- family registration and boot ------------------------------------------


def test_family_is_registered_with_alias_cap_and_no_knobs():
    spec = get_family("live_input")
    assert spec.display_name == "Audio input"
    assert resolve_checkpoint("live-input") == ("live_input", "live_input/v1")
    assert spec.session_cap("live_input/v1") == 4
    assert spec.knob_universe() == []
    assert spec.warmup_policy == "none" and spec.preload is None
    assert spec.text_only is not None
    names = [f.name for f in spec.config_fields]
    assert names == ["live_input_capture_s"]


def test_capture_bounds_match_the_backend():
    assert LIVE_INPUT_CAPTURE_DEFAULT_S == lib.DEFAULT_CAPTURE_S
    assert LIVE_INPUT_CAPTURE_MIN_S == lib.MIN_CAPTURE_S
    assert LIVE_INPUT_CAPTURE_MAX_S == lib.MAX_CAPTURE_S
    assert lib.clamp_capture_s(None) == 600.0
    assert lib.clamp_capture_s("junk") == 600.0
    assert lib.clamp_capture_s(5) == 30.0
    assert lib.clamp_capture_s(99999) == 1800.0
    assert lib.clamp_capture_s(120) == 120.0


def test_preflight_passes_without_a_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = get_family("live_input").preflight(PreflightRequest(model_id="live_input/v1"))
    assert result.ok


def test_config_field_parses_into_family_config():
    assert SessionConfig.from_dict({"live_input_capture_s": "45"}).family_config[
        "live_input_capture_s"] == 45.0
    assert SessionConfig.from_dict({}).family_config["live_input_capture_s"] is None


def test_capabilities_are_append_only_client_fed():
    be, *_ = _backend()
    caps = be.capabilities()
    assert caps.refines_audio is False
    assert caps.swap is True and caps.swap_resize is True
    assert caps.generates_audio is False
    assert caps.audio_in is True
    assert not any((caps.timbre, caps.structure, caps.write_audio, caps.lora,
                    caps.loop_band, caps.depth, caps.bounce))
    # Every other family keeps the defaults.
    assert Capabilities().generates_audio is True
    assert Capabilities().audio_in is False


# ---- session create ----------------------------------------------------------


def test_create_from_nothing_is_a_silent_mono_capture():
    ss, _bus = _create(capture_s=45.0)
    try:
        assert ss.audio_eng.current.shape == (45 * SR, 1)
        assert not ss.audio_eng.current.any()
        assert ss.initial_buffer.shape == (45 * SR, 1)
        assert ss.state.n_channels == 1
        assert ss.geometry_payload()["duration_s"] == pytest.approx(45.0)
        assert ss.capabilities_payload()["audio_in"] is True
        assert ss.backend.analysis_frontier_s() == 0.0
        assert not ss.knob_manifest_payload()["knobs"]
    finally:
        ss.close()


def test_an_empty_upload_is_a_capture_too():
    # The SDK always sends a PCM frame; an empty one means "no source".
    ss, _bus = _create(clip=torch.zeros(1, 0), capture_s=30.0, text_only=False)
    try:
        assert ss.audio_eng.current.shape == (30 * SR, 1)
        assert ss.capabilities_payload()["audio_in"] is True
        assert ss.backend.analysis_frontier_s() == 0.0
    finally:
        ss.close()


def test_create_from_nothing_defaults_to_600s():
    ss, _bus = _create(capture_s=None)
    try:
        assert ss.audio_eng.current.shape[0] == 600 * SR
    finally:
        ss.close()


def test_create_from_a_clip_uses_the_clip_as_the_canvas():
    clip = torch.linspace(-0.5, 0.5, 3 * SR).repeat(2, 1)
    ss, bus = _create(clip=clip)
    try:
        assert ss.audio_eng.current.shape == (3 * SR, 2)
        np.testing.assert_allclose(ss.audio_eng.current[:, 0], clip[0].numpy())
        assert ss.backend.analysis_frontier_s() == pytest.approx(3.0)
        # A clip canvas takes no audio_in: it fails loudly.
        ss.audio_in(Audio(waveform=torch.zeros(1, 4800), sample_rate=SR), seq=0)
        assert bus.of(CommandFailed) and bus.of(CommandFailed)[0].command == "audio_in"
        assert not bus.of(AudioReady)
    finally:
        ss.close()


def test_swap_source_makes_the_new_file_the_canvas():
    ss, bus = _create(capture_s=30.0)
    try:
        clip = torch.full((1, 5 * SR), 0.25)
        ss.swap_source(Audio(waveform=clip, sample_rate=SR), duration_s=5.0)
        ss.apply_pending()
        swaps = bus.of(SwapReady)
        assert len(swaps) == 1 and swaps[0].source_epoch == 1
        assert ss.audio_eng.current.shape[0] == 5 * SR
        assert ss.backend.analysis_frontier_s() == pytest.approx(5.0)
        assert ss.state.duration == pytest.approx(5.0)
    finally:
        ss.close()


# ---- audio_in -------------------------------------------------------------------


def test_audio_in_appends_at_the_frontier_and_moves_the_playhead():
    clock = Clock()
    be, eng, _state, bus = _backend(clock=clock)
    be.append_input(_frame(value=0.1), seq=0)
    clock.t += 0.1
    be.append_input(_frame(value=0.2), seq=1)
    ready = bus.of(AudioReady)
    assert [r.start_sample for r in ready] == [0, 4800]
    assert all(r.tick_ms == 0.0 and r.dec_ms == 0.0 and r.num_gens == 0 for r in ready)
    assert ready[0].params == {"x": 1}
    assert np.allclose(eng.current[:4800], 0.1) and np.allclose(eng.current[4800:9600], 0.2)
    # No fade at the frame boundary (patch would dip toward silence).
    assert eng.current[4800, 0] == pytest.approx(0.2)
    assert eng.position == 0                  # frontier still inside 0.25 s
    assert eng.position_report_wall_s == clock.t
    be.append_input(_frame(n=SR), seq=2)
    assert eng.position == 9600 + SR - int(0.25 * SR)
    assert be.analysis_frontier_s() == pytest.approx(1.2)


def test_stereo_frames_downmix_into_a_mono_capture():
    be, eng, _state, _bus = _backend()
    frame = np.stack([np.full(4800, 0.2), np.full(4800, 0.6)], axis=1).astype(np.float32)
    be.append_input(frame, seq=0)
    assert np.allclose(eng.current[:4800, 0], 0.4)


def test_missing_frames_are_filled_with_silence():
    be, eng, _state, bus = _backend()
    be.append_input(_frame(value=0.3), seq=0)
    be.append_input(_frame(value=0.3), seq=3)          # seq 1 and 2 lost
    assert be.gap_fills == 1
    assert not eng.current[4800:3 * 4800].any()
    assert np.allclose(eng.current[3 * 4800:4 * 4800], 0.3)
    assert not bus.of(SwapReady)


def test_start_s_gap_up_to_two_seconds_is_filled():
    be, eng, _state, bus = _backend()
    be.append_input(_frame(), seq=0, start_s=0.0)
    be.append_input(_frame(), seq=1, start_s=1.5)
    assert be.analysis_frontier_s() == pytest.approx(1.6)
    assert not eng.current[4800:int(1.5 * SR)].any()
    assert not bus.of(SwapReady)


def test_long_gap_rewind_and_restart_start_new_takes():
    be, eng, state, bus = _backend()
    be.append_input(_frame(value=0.3), seq=0, start_s=0.0)
    be.append_input(_frame(value=0.4), seq=1, start_s=5.0)     # > 2 s ahead
    assert state.source_epoch == 1 and be.takes == 2
    assert np.allclose(eng.current[:4800], 0.4)
    be.append_input(_frame(), seq=2, start_s=4.0)               # rewind
    assert state.source_epoch == 2
    be.append_input(_frame(), seq=0)                            # stream restart
    assert state.source_epoch == 3
    assert [s.source_epoch for s in bus.of(SwapReady)] == [1, 2, 3]
    assert be.analysis_frontier_s() == pytest.approx(0.1)


def test_full_canvas_rolls_into_a_new_take():
    be, eng, state, bus = _backend(capture_s=30.0)
    n = len(eng.current)
    be.append_input(np.full((n - 1000, 1), 0.1, dtype=np.float32), seq=0)
    be.append_input(_frame(value=0.7), seq=1)
    swaps = bus.of(SwapReady)
    assert len(swaps) == 1 and swaps[0].source_epoch == 1
    assert swaps[0].duration == pytest.approx(30.0) and swaps[0].channels == 1
    # The tail filled the old take; the remainder opens the new one.
    ready = bus.of(AudioReady)
    assert (ready[-2].start_sample, ready[-2].num_samples) == (n - 1000, 1000)
    assert (ready[-1].start_sample, ready[-1].num_samples) == (0, 3800)
    assert np.allclose(eng.current[:3800], 0.7) and not eng.current[3800:].any()
    assert be.analysis_frontier_s() == pytest.approx(3800 / SR)
    assert state.source_epoch == 1


def test_acks_are_coalesced_and_flushed_by_the_runner_loop():
    clock = Clock()
    be, _eng, _state, bus = _backend(clock=clock)
    for seq in range(5):                       # five frames inside 200 ms
        be.append_input(_frame(), seq=seq)
        clock.t += 0.03
    acks = bus.of(AudioInAck)
    assert [a.seq for a in acks] == [0]
    be.produce({}, None, "generate")           # not due yet
    assert len(bus.of(AudioInAck)) == 1
    clock.t += 0.1
    be.produce({}, None, "generate")
    acks = bus.of(AudioInAck)
    assert [a.seq for a in acks] == [0, 4]
    assert acks[-1].frontier_s == pytest.approx(0.5)
    be.produce({}, None, "generate")           # nothing pending
    assert len(bus.of(AudioInAck)) == 2


def test_audio_in_is_capability_gated_on_other_backends():
    bus = RecordingBus()
    called = []
    fake = SimpleNamespace(
        backend=SimpleNamespace(
            name="sa3", capabilities=lambda: Capabilities(),
            append_input=lambda *a, **k: called.append(a),
        ),
        bus=bus,
        state=SimpleNamespace(last_activity_ts=0.0),
    )
    fake._require_capability = lambda cap, cmd: StreamingSession._require_capability(fake, cap, cmd)
    StreamingSession.audio_in(fake, Audio(waveform=torch.zeros(1, 480), sample_rate=SR), seq=0)
    failed = bus.of(CommandFailed)
    assert failed and failed[0].command == "audio_in" and failed[0].requires == "audio_in"
    assert not called


def test_session_audio_in_reaches_the_canvas():
    ss, bus = _create(capture_s=30.0)
    try:
        ss.audio_in(Audio(waveform=torch.full((2, 4800), 0.5), sample_rate=SR), seq=0, start_s=0.0)
        assert np.allclose(ss.audio_eng.current[:4800], 0.5)
        assert bus.of(AudioReady)[0].channels == 1
    finally:
        ss.close()


def test_runner_idles_and_closes_cleanly():
    ss, bus = _create(capture_s=30.0)
    t = threading.Thread(target=ss.run, daemon=True)
    t.start()
    try:
        ss.audio_in(Audio(waveform=torch.zeros(1, 4800), sample_rate=SR), seq=0)
        ss.audio_in(Audio(waveform=torch.zeros(1, 4800), sample_rate=SR), seq=1)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(bus.of(AudioInAck)) < 2:
            time.sleep(0.02)
        # The second ack was coalesced; the runner loop flushed it.
        assert [a.seq for a in bus.of(AudioInAck)] == [0, 1]
    finally:
        ss.state.running = False
        t.join(2.0)
    assert not t.is_alive()
    assert ss.closed.is_set()


# ---- analysis: governor and frontier ------------------------------------------


def test_governor_without_a_generator_is_always_healthy():
    g = Governor(refines_audio=False, generates_audio=False)
    assert g.known() and g.mode(0.0) == "ok"
    g.observe_frontier_lead(0.25, 1.0)                 # below FRONTIER_LEAD_MIN_S
    g.observe_slice_lead(-1.0, 1.0, 1.0)               # negative landing lead
    assert g.healthy(1.0) and g.may_submit(1.0)
    # The same signals back a generator family off.
    h = Governor(refines_audio=False)
    h.observe_frontier_lead(0.25, 1.0)
    assert h.mode(1.0) == "backoff"


def _tap_on(ss, clock, submitter):
    caps = ss.backend.capabilities()
    tap = AnalysisTap(
        spec=STUB, options={}, submitter=submitter,
        audio_eng=ss.audio_eng, bus=ss.bus,
        refines_audio=caps.refines_audio, generates_audio=caps.generates_audio,
        frontier_fn=ss.backend.analysis_frontier_s,
        source_epoch=ss.state.source_epoch, next_seq=iter(range(1, 10**6)).__next__,
        clock=clock, interval_s=3600.0,
    )
    assert tap.start()
    return tap


def test_static_clip_is_analysed_without_any_write():
    clip = torch.full((1, 9 * SR), 0.2)
    ss, bus = _create(clip=clip)
    clock = Clock()
    sub = StubSubmitter()
    tap = _tap_on(ss, clock, sub)
    try:
        assert tap.mode == "confirm"
        tap.step()
        assert len(sub.submitted) == 1
        request, on_result, _err, _tok = sub.submitted.pop()
        assert request.t0_s == 0.0
        on_result(AnalysisResult(
            kind="note", range_start_s=request.t0_s, span_s=request.span_s,
            items=[{"start_s": 0.1, "end_s": 0.3, "pitch": 60, "instrument": "acoustic_piano"}],
            source_epoch=request.source_epoch, revision=request.revision, wall_s=0.01,
        ))
        upd = [u for u in bus.of(AnalysisUpdate) if not u.clear]
        assert upd and upd[0].basis == "settled"
        # Every cell of the clip gets its turn, with no playhead at all.
        for _ in range(2):
            tap.step()
            req, res, _e, _t = sub.submitted.pop()
            res(AnalysisResult(kind="note", range_start_s=req.t0_s, span_s=req.span_s,
                               items=[], source_epoch=req.source_epoch,
                               revision=req.revision, wall_s=0.01))
        assert tap._observer.dirty == [False, False, False]
    finally:
        tap.close()
        ss.close()


def test_capture_is_analysed_behind_the_frontier_and_clears_on_a_new_take():
    ss, bus = _create(capture_s=30.0)
    clock = Clock()
    sub = StubSubmitter()
    tap = _tap_on(ss, clock, sub)
    try:
        tap.step()
        assert not sub.submitted                  # nothing final yet
        for seq in range(46):                     # 4.6 s: cell 0 plus context
            ss.audio_in(Audio(waveform=torch.full((1, 4800), 0.1), sample_rate=SR), seq=seq)
        tap.step()
        assert len(sub.submitted) == 1 and sub.submitted[0][0].t0_s == 0.0
        assert tap.mode == "confirm"              # never backoff: no generator
        epoch_before = ss.state.source_epoch
        ss.audio_in(Audio(waveform=torch.zeros(1, 4800), sample_rate=SR), seq=0)  # restart
        assert ss.state.source_epoch == epoch_before + 1
        swap = bus.of(SwapReady)[-1]
        bus.deliver(swap)
        clears = [u for u in bus.of(AnalysisUpdate) if u.clear]
        assert clears[-1].source_epoch == swap.source_epoch
        assert tap._observer.frontier_s() == pytest.approx(0.1)
    finally:
        tap.close()
        ss.close()
