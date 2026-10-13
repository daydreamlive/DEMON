"""AnalysisTap and the StreamingSession wiring, against a stub host.

No GPU, no model: the submitter is a stub that records requests and lets
the test deliver results (or errors) by hand, and the analyzer is a stub
spec registered here.
"""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from acestep.analysis.engine import AnalysisResult, EngineBusy, EngineUnavailable
from acestep.analysis.registry import AnalyzerOption, AnalyzerSpec, register
from acestep.analysis.tap import AnalysisTap
from acestep.streaming.audio_engine import AudioEngine
from acestep.streaming.events import AnalysisClear, AnalysisState, AnalysisUpdate, SwapReady

SR = 48000

STUB = register(AnalyzerSpec(
    name="stub_notes",
    display_name="Stub notes",
    item_kind="note",
    make_engine=lambda size, device: None,
    probe=lambda: None,
    licence="stub licence",
    sizes=("small", "medium"),
    default_size="small",
    window_s=4.0,
    context_s=0.5,
    options=(
        AnalyzerOption("size", "str", default=None, bounds=("small", "medium")),
        AnalyzerOption("horizon_s", "float", default=8.0, bounds=(4.0, 30.0)),
    ),
))

UNAVAILABLE = register(AnalyzerSpec(
    name="stub_absent",
    display_name="Stub absent",
    item_kind="note",
    make_engine=lambda size, device: None,
    probe=lambda: "stub dependency not installed",
))


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class StubSubmitter:
    def __init__(self, info_error=None):
        self.submitted = []   # (request, on_result, on_error, token)
        self.cancelled = []
        self.info_error = info_error

    def submit(self, analyzer, size, request, on_result, on_error, cancel_token):
        self.submitted.append((request, on_result, on_error, cancel_token))

    def cancel(self, cancel_token):
        self.cancelled.append(cancel_token)
        n = sum(1 for s in self.submitted if s[3] is cancel_token)
        self.submitted = [s for s in self.submitted if s[3] is not cancel_token]
        return n

    def engine_info(self, analyzer, size):
        if self.info_error is not None:
            raise self.info_error
        return {"name": "stubengine", "size": size, "loaded": False, "licence": "stub licence"}

    def answer(self, k=-1, items=None):
        request, on_result, _on_error, _tok = self.submitted.pop(k)
        on_result(_result_for(request, items))
        return request


def _result_for(request, items=None):
    if items is None:
        items = [{"start_s": request.t0_s + 0.1, "end_s": request.t0_s + 0.4,
                  "pitch": 60, "instrument": "piano"}]
    return AnalysisResult(
        kind="note", range_start_s=request.t0_s, span_s=request.span_s,
        items=items, source_epoch=request.source_epoch,
        revision=request.revision, wall_s=0.05,
    )


class RecordingBus:
    """Synchronous stand-in for EventBus: records published events and
    lets the observer subscribe (callbacks are delivered by the test)."""

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

    def deliver(self, event):
        for fn in list(self.listeners):
            fn(event)

    def of(self, cls):
        return [e for e in self.events if isinstance(e, cls)]


def _make_tap(*, refines_audio=True, seconds=16.0, submitter=None, clock=None, options=None,
              generates_audio=True, frontier_fn=None):
    eng = AudioEngine(np.zeros((int(seconds * SR), 2), np.float32), SR)
    bus = RecordingBus()
    submitter = submitter or StubSubmitter()
    clock = clock or Clock()
    seq = iter(range(1, 10 ** 6))
    tap = AnalysisTap(
        spec=STUB, options=STUB.validate_options(options or {}),
        submitter=submitter, audio_eng=eng, bus=bus,
        refines_audio=refines_audio, source_epoch=0,
        next_seq=lambda: next(seq), clock=clock, interval_s=3600.0,
        generates_audio=generates_audio, frontier_fn=frontier_fn,
    )
    return tap, eng, bus, submitter, clock


def _report_playhead(eng, clock, seconds):
    eng.position = int(seconds * SR)
    eng.position_report_wall_s = clock.t


def test_enable_publishes_state_then_clear():
    tap, eng, bus, sub, clock = _make_tap()
    assert tap.start()
    assert [type(e) for e in bus.events] == [AnalysisState, AnalysisClear]
    state, clear = bus.events
    assert state.enabled and state.engine == "stubengine" and state.size == "small"
    assert state.licence == "stub licence" and state.mode == "no_playhead"
    assert state.error is None and state.dropped == 0
    assert clear.clear and clear.items == () and clear.source_epoch == 0
    tap.close()


def test_lookahead_publishes_update_with_seq_and_basis():
    tap, eng, bus, sub, clock = _make_tap()
    tap.start()
    _report_playhead(eng, clock, 1.0)
    tap.step()
    assert tap.mode == "lookahead"
    assert len(sub.submitted) == 1
    req = sub.submitted[0][0]
    assert req.t0_s == 4.0  # nearest cell ahead of a playhead at 1.0
    tap.step()
    assert len(sub.submitted) == 1  # one in flight per tap
    sub.answer()
    upd = bus.of(AnalysisUpdate)[-1]
    assert not upd.clear and upd.basis == "provisional" and not upd.late
    assert upd.range_start_s == 4.0 and upd.span_s == 4.0 and upd.kind == "note"
    assert upd.items[0]["pitch"] == 60 and upd.playhead_s == pytest.approx(1.0)
    assert upd.seq > bus.of(AnalysisUpdate)[0].seq
    tap.close()


def test_stale_result_is_dropped_and_cell_redirtied():
    tap, eng, bus, sub, clock = _make_tap()
    tap.start()
    _report_playhead(eng, clock, 1.0)
    tap.step()
    req = sub.submitted[0][0]
    eng.patch_window(np.ones((100, 2), np.float32), int(5.0 * SR))  # inside the cell
    n_updates = len(bus.of(AnalysisUpdate))
    sub.answer()
    assert len(bus.of(AnalysisUpdate)) == n_updates
    assert tap.dropped == 0  # held until the content check has run
    tap.step()
    assert len(bus.of(AnalysisUpdate)) == n_updates
    assert tap.dropped == 1
    assert tap._observer.dirty[int(req.t0_s // 4)]


def test_engine_busy_counts_dropped_and_error_publishes_state():
    tap, eng, bus, sub, clock = _make_tap()
    tap.start()
    tap.step()
    _req, _ok, on_error, _tok = sub.submitted.pop()
    on_error(EngineBusy("dropped"))
    assert tap.dropped == 1 and tap.enabled
    clock.t += 5.0
    tap.step()
    _req, _ok, on_error, _tok = sub.submitted.pop()
    on_error(RuntimeError("bad window"))
    st = bus.of(AnalysisState)[-1]
    assert st.enabled and "bad window" in st.error
    clock.t += 5.0
    tap.step()
    _req, _ok, on_error, _tok = sub.submitted.pop()
    on_error(EngineUnavailable("not enough free GPU memory for Stub small"))
    st = bus.of(AnalysisState)[-1]
    assert not st.enabled and st.mode == "off"
    assert st.error == "not enough free GPU memory for Stub small"


def test_engine_info_refusal_fails_the_enable():
    sub = StubSubmitter(info_error=EngineUnavailable("weights are gated"))
    tap, eng, bus, sub, clock = _make_tap(submitter=sub)
    assert not tap.start()
    assert [type(e) for e in bus.events] == [AnalysisState]
    assert bus.events[0].error == "weights are gated" and not bus.events[0].enabled


def test_swap_cancels_queued_work_and_clears_with_new_epoch():
    from acestep.streaming.events import SwapReady

    tap, eng, bus, sub, clock = _make_tap()
    tap.start()
    tap.step()
    old_token = sub.submitted[0][3]
    in_flight = sub.submitted[0]
    eng.swap(np.zeros((8 * SR, 2), np.float32))
    bus.deliver(SwapReady(
        duration=8.0, sample_rate=SR, channels=2, bpm=120, key="C",
        time_signature="4", fixture_name=None,
        initial_buffer=np.zeros((1, 2), np.float32), source_epoch=1,
    ))
    assert sub.cancelled == [old_token]
    clear = bus.of(AnalysisUpdate)[-1]
    assert clear.clear and clear.source_epoch == 1
    assert tap.dropped == 1
    # A result for the withdrawn request that was already running is ignored.
    n = len(bus.events)
    in_flight[1](_result_for(in_flight[0]))
    assert len(bus.events) == n
    clock.t += 5.0  # past the unknown-health pacing interval
    tap.step()
    assert sub.submitted[-1][0].source_epoch == 1
    tap.close()


def test_disable_cancels_and_late_results_are_ignored():
    tap, eng, bus, sub, clock = _make_tap()
    tap.start()
    tap.step()
    pending = sub.submitted[0]
    tap.stop()
    assert sub.cancelled and tap.dropped == 1
    st = bus.of(AnalysisState)[-1]
    assert not st.enabled and st.mode == "off" and st.dropped == 1
    n = len(bus.events)
    pending[1](_result_for(pending[0]))
    assert len(bus.events) == n
    tap.stop()  # idempotent
    assert len(bus.events) == n
    assert not eng._write_listeners


def test_ring_cell_settles_once_the_playhead_passes_it():
    tap, eng, bus, sub, clock = _make_tap()
    tap.start()
    _report_playhead(eng, clock, 3.0)
    tap.step()
    sub.answer()  # cell 1 [4, 8), provisional
    assert bus.of(AnalysisUpdate)[-1].basis == "provisional"
    _report_playhead(eng, clock, 7.9)
    tap.step()
    sub.submitted.clear()
    tap._in_flight = None
    _report_playhead(eng, clock, 8.1)
    tap.step()
    settled = [u for u in bus.of(AnalysisUpdate) if u.basis == "settled"]
    assert len(settled) == 1 and settled[0].range_start_s == 4.0
    assert settled[0].items[0]["pitch"] == 60
    tap.close()


def _noise(seconds, seed=0):
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal((int(seconds * SR), 2))).astype(np.float32)


def _settle_cell_1(tap, eng, bus, sub, clock):
    """Analyse cell 1 [4, 8) and walk the playhead past it."""
    _report_playhead(eng, clock, 3.0)
    tap.step()
    sub.answer()
    _report_playhead(eng, clock, 7.9)
    tap.step()
    sub.submitted.clear()
    tap._in_flight = None
    _report_playhead(eng, clock, 8.1)
    tap.step()
    return [u for u in bus.of(AnalysisUpdate) if u.basis == "settled"]


def test_identical_ring_rewrite_keeps_the_settled_result():
    tap, eng, bus, sub, clock = _make_tap()
    src = _noise(16.0)
    eng.patch_window(src, 0)
    tap.start()
    assert len(_settle_cell_1(tap, eng, bus, sub, clock)) == 1
    n = len(bus.events)
    # The generator re-renders the cell with the same music.
    eng.patch_window(src[3 * SR:9 * SR].copy(), 3 * SR)
    sub.submitted.clear()
    tap._in_flight = None
    clock.t += 5.0
    tap.step()
    assert not tap._observer.dirty[1] and not tap._observer.pending[1]
    assert all(r[0].t0_s != 4.0 for r in sub.submitted)
    assert tap._published[1].basis == "settled"
    assert not [e for e in bus.events[n:] if isinstance(e, AnalysisUpdate) and e.range_start_s == 4.0]


def test_changed_ring_rewrite_dirties_the_cell():
    tap, eng, bus, sub, clock = _make_tap()
    eng.patch_window(_noise(16.0), 0)
    tap.start()
    assert len(_settle_cell_1(tap, eng, bus, sub, clock)) == 1
    eng.patch_window(_noise(4.0, seed=5) * 3.0, 4 * SR)
    sub.submitted.clear()
    tap._in_flight = None
    tap.step()
    assert tap._observer.dirty[1]


def test_result_held_over_an_identical_rewrite_is_published():
    tap, eng, bus, sub, clock = _make_tap()
    src = _noise(16.0)
    eng.patch_window(src, 0)
    tap.start()
    _report_playhead(eng, clock, 3.0)
    tap.step()
    req = sub.submitted[0][0]
    eng.patch_window(src[4 * SR:8 * SR].copy(), 4 * SR)
    n = len(bus.of(AnalysisUpdate))
    sub.answer()
    assert len(bus.of(AnalysisUpdate)) == n  # held
    tap.step()
    upd = bus.of(AnalysisUpdate)[-1]
    assert upd.range_start_s == req.t0_s and upd.items and tap.dropped == 0


def test_truncated_result_replaces_the_range_and_counts_dropped():
    tap, eng, bus, sub, clock = _make_tap()
    tap.start()
    _report_playhead(eng, clock, 3.0)
    tap.step()
    request, on_result, _err, _tok = sub.submitted.pop()
    on_result(AnalysisResult(
        kind="note", range_start_s=request.t0_s, span_s=request.span_s,
        items=[], source_epoch=request.source_epoch, revision=request.revision,
        wall_s=6.0, truncated=True,
    ))
    upd = bus.of(AnalysisUpdate)[-1]
    assert upd.range_start_s == 4.0 and upd.items == () and not upd.clear
    assert upd.basis == "provisional" and tap.dropped == 1
    tap._in_flight = None
    _report_playhead(eng, clock, 7.9)
    tap.step()
    sub.submitted.clear()
    tap._in_flight = None
    _report_playhead(eng, clock, 8.1)
    tap.step()
    assert not [u for u in bus.of(AnalysisUpdate) if u.basis == "settled"]
    tap.close()
    assert bus.of(AnalysisState)[-1].dropped == 1


def test_append_only_confirms_behind_the_frontier():
    tap, eng, bus, sub, clock = _make_tap(refines_audio=False)
    tap.start()
    assert bus.of(AnalysisState)[0].mode == "confirm"
    tap.step()
    assert not sub.submitted  # no frontier yet
    eng.patch_window(np.ones((int(4.6 * SR), 2), np.float32), 0)
    clock.t += 5.0
    tap.step()
    req = sub.submitted[0][0]
    assert req.t0_s == 0.0
    sub.answer()
    assert bus.of(AnalysisUpdate)[-1].basis == "settled"


# ---- frontier follow (growing client-fed canvas) ---------------------------


class Capture:
    """Appends to a capture canvas at its frontier, like LiveInputBackend."""

    def __init__(self, eng):
        self.eng = eng
        self.frontier = 0

    def to(self, seconds):
        end = int(round(seconds * SR))
        self.eng.patch_window(
            np.full((end - self.frontier, 2), 0.1, np.float32), self.frontier,
        )
        self.frontier = end


def _follow_tap(seconds=60.0):
    tap, eng, bus, sub, clock = _make_tap(
        refines_audio=False, generates_audio=False, seconds=seconds,
    )
    assert tap.start() and tap._follow
    return tap, eng, bus, sub, clock, Capture(eng)


def test_follow_publishes_each_fresh_stretch_once():
    tap, eng, bus, sub, clock, cap = _follow_tap()
    assert bus.of(AnalysisState)[0].mode == "confirm"
    tap.step()
    assert not sub.submitted  # nothing written yet
    cap.to(1.2)
    tap.step()
    assert not sub.submitted  # under a stride past the guard
    cap.to(1.3)
    tap.step()
    req = sub.submitted[0][0]
    assert req.t0_s == 0.0 and req.span_s == pytest.approx(1.0)
    assert req.context_lead_s == 0.0 and req.context_tail_s == pytest.approx(0.3)
    assert req.audio.shape[0] == int(1.3 * SR)
    cap.to(2.0)
    tap.step()
    assert len(sub.submitted) == 1  # one in flight
    sub.answer()
    first = bus.of(AnalysisUpdate)[-1]
    assert first.basis == "settled" and not first.clear
    assert first.range_start_s == 0.0 and first.span_s == pytest.approx(1.0)
    assert first.items[0]["start_s"] == pytest.approx(0.1)
    tap.step()
    assert not sub.submitted  # 2.0 s: only 0.7 s reportable past 1.0
    cap.to(2.3)
    tap.step()
    req = sub.submitted[0][0]
    assert req.t0_s == pytest.approx(1.0) and req.span_s == pytest.approx(1.0)
    assert req.context_lead_s == pytest.approx(1.0)
    assert req.context_tail_s == pytest.approx(0.3)
    sub.answer()
    second = bus.of(AnalysisUpdate)[-1]
    assert second.range_start_s == pytest.approx(first.range_start_s + first.span_s)
    assert second.span_s == pytest.approx(1.0) and second.basis == "settled"
    # Each stretch is published once: nothing more until new audio.
    n = len(bus.of(AnalysisUpdate))
    tap.step()
    assert not sub.submitted and len(bus.of(AnalysisUpdate)) == n
    assert tap.dropped == 0
    tap.close()


def test_follow_stall_leaves_a_gap_and_slow_engine_widens_the_stretch():
    tap, eng, bus, sub, clock, cap = _follow_tap()
    cap.to(1.3)
    tap.step()
    cap.to(2.6)  # engine still busy: the next window reports more
    sub.answer()
    tap.step()
    req = sub.submitted[0][0]
    assert req.t0_s == pytest.approx(1.0) and req.span_s == pytest.approx(1.3)
    cap.to(12.0)  # a long stall while this one is in flight
    sub.answer()
    tap.step()
    req = sub.submitted[0][0]
    assert req.t0_s == pytest.approx(7.0)  # window start: 2.3 .. 7.0 never analysed
    assert req.context_lead_s == 0.0
    assert req.span_s == pytest.approx(12.0 - 0.3 - 7.0)
    sub.answer()
    upd = bus.of(AnalysisUpdate)[-1]
    assert upd.range_start_s == pytest.approx(7.0)
    tap.close()


def test_follow_drops_a_result_overtaken_by_a_rewrite_and_retries_the_stretch():
    tap, eng, bus, sub, clock, cap = _follow_tap()
    cap.to(1.3)
    tap.step()
    eng.patch_window(np.ones((100, 2), np.float32), int(0.5 * SR))  # behind the frontier
    n = len(bus.of(AnalysisUpdate))
    sub.answer()
    assert len(bus.of(AnalysisUpdate)) == n and tap.dropped == 1
    cap.to(1.4)
    tap.step()
    req = sub.submitted[0][0]
    assert req.t0_s == 0.0 and req.span_s == pytest.approx(1.1)
    tap.close()


def test_follow_engine_busy_keeps_the_stretch():
    tap, eng, bus, sub, clock, cap = _follow_tap()
    cap.to(1.3)
    tap.step()
    _req, _ok, on_error, _tok = sub.submitted.pop()
    on_error(EngineBusy("held"))
    assert tap.dropped == 1 and tap.enabled and tap._emit_hi_s == 0.0
    tap.step()
    assert sub.submitted[0][0].t0_s == 0.0
    tap.close()


def test_follow_new_take_starts_over():
    tap, eng, bus, sub, clock, cap = _follow_tap(seconds=30.0)
    cap.to(5.0)
    tap.step()
    sub.answer()
    assert tap._emit_hi_s == pytest.approx(4.7)
    cap.to(6.0)
    tap.step()
    stale = sub.submitted[0]
    # The backend opens a new take: silent canvas, new epoch.
    eng.swap(np.zeros((int(30.0 * SR), 2), np.float32))
    bus.deliver(SwapReady(
        duration=30.0, sample_rate=SR, channels=2, bpm=None, key=None,
        time_signature=None, fixture_name=None,
        initial_buffer=np.zeros((1, 2), np.float32), source_epoch=1,
    ))
    assert tap._emit_hi_s == 0.0 and tap._follow
    assert bus.of(AnalysisClear)[-1].source_epoch == 1
    n = len(bus.of(AnalysisUpdate))
    stale[1](_result_for(stale[0]))  # the old take's result arrives late
    assert len(bus.of(AnalysisUpdate)) == n
    cap.frontier = 0
    cap.to(1.3)
    tap.step()
    req = sub.submitted[-1][0]
    assert req.source_epoch == 1 and req.t0_s == 0.0
    sub.answer()
    upd = bus.of(AnalysisUpdate)[-1]
    assert upd.source_epoch == 1 and upd.range_start_s == 0.0
    tap.close()


def test_static_clip_on_a_client_fed_canvas_keeps_cells():
    tap, eng, bus, sub, clock = _make_tap(
        refines_audio=False, generates_audio=False, seconds=16.0,
        frontier_fn=lambda: 16.0,
    )
    tap.start()
    assert not tap._follow
    tap.step()
    req = sub.submitted[0][0]
    assert req.t0_s == 0.0 and req.span_s == 4.0
    tap.close()


def test_close_is_bounded_and_worker_runs():
    eng = AudioEngine(np.zeros((8 * SR, 2), np.float32), SR)
    bus = RecordingBus()
    sub = StubSubmitter()
    tap = AnalysisTap(
        spec=STUB, options=STUB.validate_options({}), submitter=sub,
        audio_eng=eng, bus=bus, refines_audio=True, source_epoch=0,
        next_seq=iter(range(1, 1000)).__next__, interval_s=0.01,
    )
    tap.start()
    deadline = time.monotonic() + 2.0
    while not sub.submitted and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sub.submitted
    t0 = time.monotonic()
    tap.close(timeout=1.0)
    assert time.monotonic() - t0 < 1.5
    assert not tap._thread.is_alive()


# ---- session wiring ---------------------------------------------------------


def _bare_session(submitter, refines_audio=True):
    from acestep.streaming.session import StreamingSession

    s = object.__new__(StreamingSession)
    s.bus = RecordingBus()
    s.audio_eng = AudioEngine(np.zeros((8 * SR, 2), np.float32), SR)
    s.state = SimpleNamespace(source_epoch=3)
    s.backend = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(refines_audio=refines_audio),
    )
    s._init_analysis()
    s._analysis_submitter = submitter
    return s


def test_session_enable_disable_close():
    sub = StubSubmitter()
    s = _bare_session(sub)
    s.set_analysis_tap("stub_notes", True, {"horizon_s": 6})
    tap = s._analysis_taps["stub_notes"]
    assert tap.horizon_s == 6.0 and tap.enabled
    clear = s.bus.of(AnalysisUpdate)[0]
    assert clear.clear and clear.source_epoch == 3
    # Re-enable replaces the tap without an intermediate "off".
    s.set_analysis_tap("stub_notes", True, {"size": "medium"})
    assert s._analysis_taps["stub_notes"] is not tap and not tap.enabled
    assert all(st.enabled for st in s.bus.of(AnalysisState))
    s.set_analysis_tap("stub_notes", False)
    assert "stub_notes" not in s._analysis_taps
    assert s.bus.of(AnalysisState)[-1].mode == "off"
    s.set_analysis_tap("stub_notes", True)
    live = s._analysis_taps["stub_notes"]
    s._close_analysis_taps()
    assert not live.enabled and not s._analysis_taps
    n = len(s.bus.events)
    s.set_analysis_tap("stub_notes", True)  # no-op after close
    assert len(s.bus.events) == n


def test_session_disable_unknown_tap_still_acks():
    s = _bare_session(StubSubmitter())
    s.set_analysis_tap("stub_notes", False)
    st = s.bus.of(AnalysisState)[-1]
    assert not st.enabled and st.error is None


@pytest.mark.parametrize("tap,options,needle", [
    ("nope", None, "unknown analyzer"),
    ("stub_absent", None, "stub dependency not installed"),
    ("stub_notes", {"horizon_s": 99}, "horizon_s"),
    ("stub_notes", {"typo": 1}, "unknown option"),
    ("stub_notes", {"size": "huge"}, "size"),
    ("stub_notes", ["not", "a", "dict"], "options must be an object"),
])
def test_session_validation_errors_become_state_errors(tap, options, needle):
    s = _bare_session(StubSubmitter())
    s.set_analysis_tap(tap, True, options)
    st = s.bus.of(AnalysisState)[-1]
    assert not st.enabled and st.mode == "off"
    assert needle in st.error
    assert not s._analysis_taps


def test_session_binds_the_host_lazily():
    import sys

    s = _bare_session(None)
    host_mod = SimpleNamespace(get_host=lambda: StubSubmitter())
    saved = sys.modules.get("acestep.analysis.host")
    sys.modules["acestep.analysis.host"] = host_mod
    try:
        s.set_analysis_tap("stub_notes", True)
    finally:
        if saved is None:
            sys.modules.pop("acestep.analysis.host", None)
        else:
            sys.modules["acestep.analysis.host"] = saved
    assert isinstance(s._analysis_submitter, StubSubmitter)
    assert s._analysis_taps["stub_notes"].enabled
    s._close_analysis_taps()


def test_session_close_stops_taps_before_the_bus():
    from acestep.streaming.session import StreamingSession

    src = __import__("inspect").getsource(StreamingSession.close)
    assert src.index("_close_analysis_taps()") < src.index("self.bus.close()")


def test_concurrent_enable_disable_is_safe():
    s = _bare_session(StubSubmitter())

    def flip(n):
        for k in range(n):
            s.set_analysis_tap("stub_notes", k % 2 == 0)

    threads = [threading.Thread(target=flip, args=(20,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    s._close_analysis_taps()
    assert not s._analysis_taps


# ---- stateful follow (streaming engine) -------------------------------------

STUB_STREAM = register(AnalyzerSpec(
    name="stub_stream",
    display_name="Stub streaming notes",
    item_kind="note",
    make_engine=lambda size, device: None,
    probe=lambda: None,
    sizes=("small",),
    default_size="small",
    stateful=True,
    options=(AnalyzerOption("size", "str", default=None, bounds=("small",)),),
))


class FakeStreamEngine:
    """Scripted stand-in for the notes engine's streaming path: a per-take
    ledger of ``(start, pitch) -> end or None``; each hop applies the next
    scripted step (StepResult-like: notes added, notes closed, truncated)
    and reports the ledger's notes by onset inside the request's cell,
    closed ones in ``items`` and sounding ones in ``open_items`` ending at
    the frontier, like ``NotesEngine._analyze_stream``."""

    def __init__(self, script=()):
        self.script = list(script)
        self.ledger = {}
        self.take = None
        self.resets = 0
        self.requests = []

    def analyze(self, request):
        self.requests.append(request)
        stream = request.options["stream"]
        if stream["take"] != self.take:
            if self.take is not None:
                self.resets += 1
            self.take = stream["take"]
            self.ledger = {}
        ws, frontier = stream["window_start_s"], stream["frontier_s"]
        step = self.script.pop(0) if self.script else {}
        truncated = bool(step.get("truncated"))
        if not truncated:
            for start, pitch, end in step.get("add", ()):
                self.ledger[(start, pitch)] = end
            for start, pitch, end in step.get("close", ()):
                self.ledger[(start, pitch)] = end
        self.ledger = {k: e for k, e in self.ledger.items() if e is None or e > ws}
        lo, hi = request.t0_s, request.t0_s + request.span_s
        picked = sorted((k, e) for k, e in self.ledger.items() if lo - 1e-9 <= k[0] < hi)

        def item(k, end):
            return {"start_s": k[0], "end_s": end, "pitch": k[1], "instrument": "piano"}

        return AnalysisResult(
            kind="note", range_start_s=request.t0_s, span_s=request.span_s,
            items=[item(k, e) for k, e in picked if e is not None],
            open_items=tuple(item(k, frontier) for k, e in picked if e is None),
            source_epoch=request.source_epoch, revision=request.revision,
            wall_s=0.03, truncated=truncated,
        )


class StreamSubmitter(StubSubmitter):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def run(self):
        """Answer the oldest request through the fake engine."""
        request, on_result, _on_error, _tok = self.submitted.pop(0)
        on_result(self.engine.analyze(request))
        return request


class ClientStore:
    """Python port of the client AnalysisStore's apply rules."""

    def __init__(self):
        self.items = []
        self.epoch = None
        self.range_seq = {}

    def apply(self, u):
        if self.epoch is not None and u.source_epoch < self.epoch:
            return
        if u.clear:
            self.items, self.range_seq, self.epoch = [], {}, u.source_epoch
        else:
            key = f"{u.range_start_s:.4f}"
            if key in self.range_seq and u.seq <= self.range_seq[key]:
                return
        t0, t1 = u.range_start_s, u.range_start_s + u.span_s
        kept = [it for it in self.items if it["start_s"] < t0 or it["start_s"] >= t1]
        self.items = sorted(kept + [dict(it, basis=u.basis) for it in u.items],
                            key=lambda it: (it["start_s"], it["pitch"]))
        self.range_seq[f"{t0:.4f}"] = u.seq


def _stream_tap(script=(), seconds=30.0):
    engine = FakeStreamEngine(script)
    sub = StreamSubmitter(engine)
    eng = AudioEngine(np.zeros((int(seconds * SR), 2), np.float32), SR)
    bus = RecordingBus()
    seq = iter(range(1, 10 ** 6))
    tap = AnalysisTap(
        spec=STUB_STREAM, options=STUB_STREAM.validate_options({}),
        submitter=sub, audio_eng=eng, bus=bus, refines_audio=False,
        source_epoch=0, next_seq=lambda: next(seq), clock=Clock(),
        interval_s=3600.0, generates_audio=False,
    )
    assert tap.start() and tap._follow and tap.stateful
    return tap, eng, bus, sub, engine, Capture(eng)


def _store_of(bus):
    store = ClientStore()
    for e in bus.of(AnalysisUpdate):
        store.apply(e)
    return store


def test_stream_stride_gating_request_options_and_one_in_flight():
    from acestep.analysis import scheduler as sch

    assert sch.FOLLOW_STRIDE_STATEFUL_S == 0.15 and sch.FOLLOW_GUARD_STATEFUL_S == 0.15
    tap, eng, bus, sub, engine, cap = _stream_tap()
    cap.to(0.29)
    tap.step()
    assert not sub.submitted  # 0.14 s past the guard: under a stride
    cap.to(0.30)
    tap.step()
    req = sub.submitted[0][0]
    stream = req.options["stream"]
    assert stream["take"] == 0 and stream["id"] == tap._stream_id
    assert stream["id"].startswith("stub_stream-")
    assert stream["window_start_s"] == 0.0 and stream["frontier_s"] == pytest.approx(0.3)
    assert req.t0_s == 0.0 and req.span_s == pytest.approx(0.3)
    assert req.audio.shape[0] == int(round(0.3 * SR))  # no trailing padding
    cap.to(0.9)
    tap.step()
    assert len(sub.submitted) == 1  # one in flight, however far the frontier ran
    sub.run()
    tap.step()  # the engine was slower than the stride: next hop at once
    req = sub.submitted[0][0]
    assert req.options["stream"]["frontier_s"] == pytest.approx(0.9)
    sub.run()
    cap.to(1.04)
    tap.step()
    assert not sub.submitted  # 0.14 s of new audio
    cap.to(1.05)
    tap.step()
    assert len(sub.submitted) == 1
    sub.run()
    cap.to(7.0)
    tap.step()
    req = sub.submitted[0][0]
    assert req.options["stream"]["window_start_s"] == pytest.approx(2.0)
    assert req.audio.shape[0] == int(round(5.0 * SR))
    # A distinct ledger id per tap: two sessions never share a ledger.
    other = _stream_tap()[0]
    assert other._stream_id != tap._stream_id
    tap.close()
    other.close()


def test_stream_open_note_is_provisional_then_settled_by_a_covering_replace():
    script = [
        {"add": [(0.1, 60, None)]},                       # hop 1: 60 starts
        {},                                               # hop 2: still sounding
        {"close": [(0.1, 60, 0.4)], "add": [(0.42, 64, 0.5)]},  # hop 3: closes
    ]
    tap, eng, bus, sub, engine, cap = _stream_tap(script)
    cap.to(0.3)
    tap.step()
    sub.run()
    u = bus.of(AnalysisUpdate)[-1]
    assert u.basis == "provisional" and u.range_start_s == pytest.approx(0.1)
    assert u.range_start_s + u.span_s == pytest.approx(0.3)
    assert [(i["start_s"], i["end_s"]) for i in u.items] == [(0.1, pytest.approx(0.3))]
    cap.to(0.45)
    tap.step()
    sub.run()
    u = bus.of(AnalysisUpdate)[-1]
    assert u.basis == "provisional" and u.range_start_s == pytest.approx(0.1)
    assert u.items[0]["end_s"] == pytest.approx(0.45)  # draws to the live edge
    store = _store_of(bus)
    assert len(store.items) == 1 and store.items[0]["basis"] == "provisional"
    n = len(bus.of(AnalysisUpdate))
    cap.to(0.6)
    tap.step()
    sub.run()
    ups = bus.of(AnalysisUpdate)[n:]
    assert len(ups) == 1  # one covering replace
    u = ups[0]
    assert u.basis == "settled" and u.range_start_s == pytest.approx(0.1)
    assert u.range_start_s + u.span_s == pytest.approx(0.6)
    assert [(i["start_s"], i["end_s"], i["pitch"]) for i in u.items] == [
        (0.1, 0.4, 60), (0.42, 0.5, 64)]
    store = _store_of(bus)
    assert [(i["start_s"], i["end_s"], i["basis"]) for i in store.items] == [
        (0.1, 0.4, "settled"), (0.42, 0.5, "settled")]  # no duplicate
    # Nothing changed: the next replace only covers the fresh stretch.
    cap.to(0.75)
    tap.step()
    sub.run()
    u = bus.of(AnalysisUpdate)[-1]
    assert u.range_start_s == pytest.approx(0.6) and u.items == ()
    assert len(_store_of(bus).items) == 2
    assert tap.dropped == 0
    tap.close()


def test_stream_mixed_hop_splits_settled_before_the_sounding_onset():
    script = [{"add": [(0.05, 60, 0.12), (0.2, 67, None)]}]
    tap, eng, bus, sub, engine, cap = _stream_tap(script)
    cap.to(0.3)
    tap.step()
    sub.run()
    settled, prov = bus.of(AnalysisUpdate)[-2:]
    assert settled.basis == "settled" and settled.range_start_s == 0.0
    assert settled.span_s == pytest.approx(0.2)
    assert [i["pitch"] for i in settled.items] == [60]
    assert prov.basis == "provisional" and prov.range_start_s == pytest.approx(0.2)
    assert prov.range_start_s + prov.span_s == pytest.approx(0.3)
    assert [i["pitch"] for i in prov.items] == [67]
    assert prov.seq > settled.seq
    store = _store_of(bus)
    assert [(i["pitch"], i["basis"]) for i in store.items] == [
        (60, "settled"), (67, "provisional")]
    tap.close()


def test_stream_sounding_note_before_the_window_still_gets_its_end():
    script = [{"add": [(1.0, 48, None)]}] + [{}] * 3 + [{"close": [(1.0, 48, 6.5)]}]
    tap, eng, bus, sub, engine, cap = _stream_tap(script)
    for f in (1.2, 3.0, 5.0, 6.6, 6.8):
        cap.to(f)
        tap.step()
        sub.run()
    req = engine.requests[-1]
    assert req.options["stream"]["window_start_s"] == pytest.approx(1.8)
    assert req.t0_s == pytest.approx(1.0)  # cell reaches back to the onset
    u = bus.of(AnalysisUpdate)[-1]
    assert u.basis == "settled" and u.range_start_s == pytest.approx(1.0)
    store = _store_of(bus)
    assert [(i["start_s"], i["end_s"], i["basis"]) for i in store.items] == [
        (1.0, 6.5, "settled")]
    tap.close()


def test_stream_truncated_hop_publishes_the_notes_it_carries():
    script = [{"add": [(0.1, 60, 0.2)]}, {"truncated": True}]
    tap, eng, bus, sub, engine, cap = _stream_tap(script)
    cap.to(0.3)
    tap.step()
    eng.patch_window(np.ones((100, 2), np.float32), int(0.25 * SR))  # hop 1 goes stale
    sub.run()
    assert tap.dropped == 1 and not _store_of(bus).items
    cap.to(0.45)
    tap.step()
    sub.run()  # truncated: carries the ledger's note, nothing new
    assert tap.dropped == 2
    u = bus.of(AnalysisUpdate)[-1]
    assert u.basis == "settled" and u.range_start_s == 0.0  # nothing published before
    store = _store_of(bus)
    assert [(i["start_s"], i["basis"]) for i in store.items] == [(0.1, "settled")]
    tap.close()


def test_stream_stale_result_is_dropped_and_the_next_hop_publishes_it():
    script = [{"add": [(0.1, 60, 0.2)]}, {}]
    tap, eng, bus, sub, engine, cap = _stream_tap(script)
    cap.to(0.3)
    tap.step()
    eng.patch_window(np.ones((100, 2), np.float32), int(0.1 * SR))  # behind the frontier
    n = len(bus.of(AnalysisUpdate))
    sub.run()
    assert len(bus.of(AnalysisUpdate)) == n and tap.dropped == 1
    cap.to(0.45)
    tap.step()
    sub.run()
    store = _store_of(bus)
    assert [i["start_s"] for i in store.items] == [0.1]
    tap.close()


def test_stream_new_take_resets_ledger_and_clears():
    script = [{"add": [(0.1, 60, None)]}, {}, {"add": [(0.05, 62, 0.1)]}]
    tap, eng, bus, sub, engine, cap = _stream_tap(script)
    cap.to(0.3)
    tap.step()
    sub.run()
    cap.to(0.5)
    tap.step()
    stale = sub.submitted[0]
    eng.swap(np.zeros((int(30.0 * SR), 2), np.float32))
    bus.deliver(SwapReady(
        duration=30.0, sample_rate=SR, channels=2, bpm=None, key=None,
        time_signature=None, fixture_name=None,
        initial_buffer=np.zeros((1, 2), np.float32), source_epoch=1,
    ))
    assert bus.of(AnalysisClear)[-1].source_epoch == 1
    assert tap._mirror == {} and tap._emit_hi_s == 0.0 and tap._pub_hi_s == 0.0
    n = len(bus.of(AnalysisUpdate))
    stale[1](engine.analyze(stale[0]))  # the old take's hop returns late
    assert len(bus.of(AnalysisUpdate)) == n
    sub.submitted.clear()
    cap.frontier = 0
    cap.to(0.3)
    tap.step()
    req = sub.submitted[0][0]
    assert req.source_epoch == 1 and req.options["stream"]["take"] == 1
    sub.run()
    assert engine.resets == 1  # the engine dropped the old take's ledger
    store = _store_of(bus)
    assert store.epoch == 1
    assert [(i["start_s"], i["pitch"]) for i in store.items] == [(0.05, 62)]
    tap.close()


def test_stateless_follow_keeps_the_one_second_policy():
    tap, eng, bus, sub, clock, cap = _follow_tap()
    assert not tap.stateful
    cap.to(1.3)
    tap.step()
    req = sub.submitted[0][0]
    assert "stream" not in req.options and req.span_s == pytest.approx(1.0)
    tap.close()
