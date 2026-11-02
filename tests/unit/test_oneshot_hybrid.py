"""``render_decode`` "hybrid": the two-version one-shot schedule.

CPU-only, fake codecs (the "windows" tests' ramp codec). The load-bearing
properties:

* the schedule: every version's DiT runs before any version's rest, each DiT
  delayed only by its predecessor's first W windows; then each version's rest
  from ONE whole-song decode, in version order; then every version's done;
* the bytes: the first W windows of a version are the "windows" path's slices
  byte for byte; every later window is the whole-song decode (the "song"
  path's own call) cut at the windowed decode's boundaries, byte for byte;
  the sha is the assembled song's;
* without a whole-song engine the rest is windowed (never the eager
  whole-song decode) and the stream is the "windows" stream, byte for byte;
* starvation: on a clock paced at the measured costs, every slice of every
  version lands before playback (from that version's first slice) needs it.
"""

from __future__ import annotations

import time as _real_time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from acestep.streaming import sa3_backend as SB
from acestep.streaming.events import OneshotDone, OneshotSlice
from acestep.streaming.oneshot import KeptLatents, OneshotError, oneshot_pcm_sha256
from acestep.streaming.sa3_backend import (
    DELIVERY_SAMPLE_RATE,
    ONESHOT_HYBRID_FIRST_S,
    ONESHOT_WINDOW_CONTEXT_S,
    SA3_SAMPLES_PER_LATENT,
    hybrid_first_windows,
    oneshot_starvation_slack_ms,
    oneshot_window_plan,
)

from tests.unit import test_sa3_stream_layout as LT
from tests.unit.test_oneshot_windows import _RampWindowCodec, _req


class _WholeSongCodec(_RampWindowCodec):
    """A windowed codec with a whole-song engine (the local box)."""

    def engine_for(self, frames):
        return ("whole", 32, 10**6)


class _WindowOnlyCodec(_RampWindowCodec):
    """A pod's codec: only the window engine (max 96 frames)."""

    def engine_for(self, frames):
        return ("window", 32, 96) if 32 <= int(frames) <= 96 else None


def _backend(render_decode, codec):
    rt = LT._LayoutRuntime({"render_decode": render_decode, "deliver_streams": True})
    b, _ctx, _rt = LT._layout_backend(runtime=rt, codec=codec)
    return b


@pytest.fixture
def small_windows(monkeypatch):
    """The test song is 0.74 s: 0.2 s windows give four, the last clamped."""
    monkeypatch.setattr(SB, "ONESHOT_WINDOW_S", 0.2)
    return round(0.2 * DELIVERY_SAMPLE_RATE)


def _song_of(slices, total, ch):
    song = np.zeros((total, ch), np.float16)
    for s in sorted(slices, key=lambda s: s.index):
        song[s.start_frame:s.start_frame + s.frames] = s.pcm
    return song


def _by_version(events):
    out: dict = {}
    for e in events:
        out.setdefault(e.version, []).append(e)
    return out


# ---- W and the mode ---------------------------------------------------------------


def test_first_windows_cover_the_first_part():
    n6, n3 = 6 * DELIVERY_SAMPLE_RATE, 3 * DELIVERY_SAMPLE_RATE
    assert ONESHOT_HYBRID_FIRST_S == 15.0
    assert hybrid_first_windows(15.0, n6, 20) == 3        # the gen worker's 15 s part
    assert hybrid_first_windows(6.0, n6, 20) == 1
    assert hybrid_first_windows(15.0, n3, 40) == 5        # the runbook's 3 s case
    assert hybrid_first_windows(0.01, n6, 20) == 1        # at least one
    assert hybrid_first_windows(500.0, n6, 20) == 20      # at most the song


def test_mode_parses_seconds_and_refuses_bad_values():
    codec = _WholeSongCodec()
    for value, want in (("hybrid", ("hybrid", ONESHOT_HYBRID_FIRST_S)),
                        (" HYBRID:6 ", ("hybrid", 6.0)),
                        ("windows", ("windows", None)), (None, ("song", None))):
        b = _backend(value, codec)
        assert b._render_decode_spec() == want
        assert b.render_decode_mode() == want[0]
    for bad in ("hybrid:0", "hybrid:-1", "hybrid:x", "hybrid:nan", "hybrids"):
        b = _backend(bad, codec)
        with pytest.raises(OneshotError, match="render_decode"):
            b.render_oneshot(_req(), store=KeptLatents(), emit=lambda e: None)


# ---- the schedule ------------------------------------------------------------------


def _traced(b, codec, log):
    """Log every DiT and every codec call (window = carries the one-shot's
    context; whole = the "song" path's context-free call over the song)."""
    orig_latent = b._oneshot_latent

    def latent(controls):
        log.append(("dit", int(controls["seed"])))
        return orig_latent(controls)
    b._oneshot_latent = latent
    orig_window = codec.decode_window

    def window(latent_bct, start, n, **kw):
        log.append(("win" if "context_sec" in kw else "whole", int(start)))
        return orig_window(latent_bct, start, n, **kw)
    codec.decode_window = window


def test_dits_run_back_to_back_after_the_first_windows(small_windows):
    codec = _WholeSongCodec()
    b = _backend("hybrid:0.3", codec)                     # W = 2 of 4 windows
    log: list = []
    _traced(b, codec, log)
    events: list = []
    res = b.render_oneshot(_req(seeds=(4, 5)), store=KeptLatents(),
                           emit=lambda e: (events.append(e), log.append(("emit", e))))
    assert res.delivered and res.pcm is None and len(res.ids) == 2
    v0, v1 = res.ids
    S = LT.N_STREAMS
    steps = []
    for kind, x in log:
        if kind == "emit":
            steps.append(("slice", x.version, x.index) if isinstance(x, OneshotSlice)
                         else ("done", x.version))
        elif not steps or steps[-1] != (kind,):
            steps.append((kind,))
    assert steps == [
        ("dit",), ("win",), ("slice", v0, 0), ("win",), ("slice", v0, 1),
        ("dit",), ("win",), ("slice", v1, 0), ("win",), ("slice", v1, 1),
        ("whole",), ("slice", v0, 2), ("slice", v0, 3),
        ("whole",), ("slice", v1, 2), ("slice", v1, 3),
        ("done", v0), ("done", v1),
    ]
    assert [x for k, x in log if k == "dit"] == [4, 5]
    # Two windows of S stream calls per version, one whole-song call per stream.
    assert sum(k == "win" for k, _ in log) == 2 * 2 * S
    assert sum(k == "whole" for k, _ in log) == 2 * S
    assert all(x == 0 for k, x in log if k == "whole")
    for done in (e for e in events if isinstance(e, OneshotDone)):
        assert done.ms["first_windows"] == 2.0 and done.ms["whole_song"] == 1.0
        assert done.windows == 4


def test_one_version_takes_its_windows_then_the_whole_song(small_windows):
    codec = _WholeSongCodec()
    b = _backend("hybrid:0.2", codec)                     # W = 1
    log: list = []
    _traced(b, codec, log)
    events: list = []
    b.render_oneshot(_req(seeds=(8,)), store=KeptLatents(), emit=events.append)
    kinds = [k for k, _ in log]
    first_whole = kinds.index("whole")
    assert kinds[0] == "dit" and set(kinds[1:first_whole]) == {"win"}
    assert [type(e).__name__ for e in events] == ["OneshotSlice"] * 4 + ["OneshotDone"]
    assert [e.index for e in events[:4]] == [0, 1, 2, 3]
    assert events[-1].ms["starve_slack_ms"] > 0


# ---- the bytes -----------------------------------------------------------------------


def _whole_song_reference(b, store, vid, seed, total):
    """The "song" path's own call on the kept latent, as the float16 the
    slices carry: [total, 2 * S]."""
    n = LT.N_STREAMS
    lat = store.get(vid).latent.movedim(1, 2)             # engine layout [1, T, C]
    frames = int(lat.shape[1])
    whole44 = (frames // n) * SA3_SAMPLES_PER_LATENT
    audio = b._render_stream_audio(lat, 0, whole44, start48=0, n48=total, decode_seed=seed)
    return audio.permute(1, 0, 2).reshape(total, 2 * n).to(torch.float16).numpy()


def test_first_windows_are_the_windowed_stream_and_the_rest_the_whole_song(small_windows):
    n48 = small_windows
    seeds = (4, 5)
    store = KeptLatents()
    hb = _backend("hybrid:0.2", _WholeSongCodec())        # W = 1
    hev: list = []
    res = hb.render_oneshot(_req(seeds=seeds), store=store, emit=hev.append)
    wev: list = []
    wb = _backend("windows", _RampWindowCodec())
    wres = wb.render_oneshot(_req(seeds=seeds), store=KeptLatents(), emit=wev.append)
    total = round(hb.playable_duration_s() * DELIVERY_SAMPLE_RATE)
    plan = oneshot_window_plan(total, n48)
    ch = 2 * LT.N_STREAMS
    hv, wv = _by_version(hev), _by_version(wev)
    rest_differs = False
    for vid, wvid, seed in zip(res.ids, wres.ids, seeds):
        h_sl = [e for e in hv[vid] if isinstance(e, OneshotSlice)]
        w_sl = [e for e in wv[wvid] if isinstance(e, OneshotSlice)]
        assert [(s.index, s.start_frame, s.frames, s.count) for s in h_sl] == \
            [(s.index, s.start_frame, s.frames, s.count) for s in w_sl] == \
            [(i, s, f, len(plan)) for i, (s, f) in enumerate(plan)]
        assert h_sl[0].pcm.tobytes() == w_sl[0].pcm.tobytes()        # window < W
        ref = _whole_song_reference(hb, store, vid, seed, total)
        for s in h_sl[1:]:                                           # windows >= W
            assert s.pcm.dtype == np.float16 and s.pcm.shape == (s.frames, ch)
            assert s.pcm.tobytes() == ref[s.start_frame:s.start_frame + s.frames].tobytes()
        rest_differs |= any(a.pcm.tobytes() != b.pcm.tobytes() for a, b in zip(h_sl[1:], w_sl[1:]))
        done = hv[vid][-1]
        assert isinstance(done, OneshotDone) and done.frames == total and done.seed == seed
        assert done.sha256 == oneshot_pcm_sha256(_song_of(h_sl, total, ch), LT.N_STREAMS)
        assert store.get(vid).seed == seed
    # The fake's whole-song resample differs from its windows at the guard
    # edges, so the check above is not vacuous.
    assert rest_differs


def test_without_a_whole_song_engine_the_stream_is_the_windows_stream(small_windows):
    for codec in (_RampWindowCodec(), _WindowOnlyCodec()):   # no engine_for; window engine only
        seeds = (4, 5)
        hb = _backend("hybrid:0.2", codec)
        hev: list = []
        res = hb.render_oneshot(_req(seeds=seeds), store=KeptLatents(), emit=hev.append)
        wev: list = []
        wres = _backend("windows", _RampWindowCodec()).render_oneshot(
            _req(seeds=seeds), store=KeptLatents(), emit=wev.append)
        hv, wv = _by_version(hev), _by_version(wev)
        for vid, wvid in zip(res.ids, wres.ids):
            h, w = hv[vid], wv[wvid]
            assert len(h) == len(w)
            for a, b in zip(h[:-1], w[:-1]):
                assert (a.index, a.start_frame, a.frames) == (b.index, b.start_frame, b.frames)
                assert a.pcm.tobytes() == b.pcm.tobytes()
            assert h[-1].sha256 == w[-1].sha256 and h[-1].ms["whole_song"] == 0.0
        # Never a context-free (whole-song, eager on a pod) codec call.
        assert all(kw == {"context_sec": ONESHOT_WINDOW_CONTEXT_S} for _s, _n, kw in codec.calls)


def test_song_and_windows_modes_are_unchanged(small_windows):
    """"windows" still emits per version all its windows then its done, and
    its ms keys are unchanged; "song" still answers with PCM."""
    ev: list = []
    _backend("windows", _WholeSongCodec()).render_oneshot(
        _req(seeds=(4, 5)), store=KeptLatents(), emit=ev.append)
    assert [type(e).__name__ for e in ev] == (["OneshotSlice"] * 4 + ["OneshotDone"]) * 2
    assert set(ev[4].ms) == {"generate", "first_slice", "decode", "pack", "windows_wall"}
    res = _backend("song", _WholeSongCodec()).render_oneshot(
        _req(seeds=(4,)), store=KeptLatents(), emit=ev.append)
    assert res.pcm is not None and not res.delivered


# ---- starvation -------------------------------------------------------------------------


class _Clock:
    """``time`` stand-in for sa3_backend whose perf_counter only moves when
    a modelled cost is paid."""

    def __init__(self):
        self.now = 1000.0

    def perf_counter(self):
        return self.now

    def __getattr__(self, name):
        return getattr(_real_time, name)


def _paced_run(monkeypatch, *, dit_ms, window_ms, whole_ms, first="hybrid:0.2", seeds=(4, 5)):
    """A hybrid render on a clock paced per call: ``dit_ms`` per DiT,
    ``window_ms`` per window (all streams), ``whole_ms`` per whole-song
    decode (all streams)."""
    clock = _Clock()
    monkeypatch.setattr(SB, "time", clock)
    codec = _WholeSongCodec()
    b = _backend(first, codec)
    S = LT.N_STREAMS
    orig_latent = b._oneshot_latent

    def latent(controls):
        out = orig_latent(controls)
        clock.now += dit_ms / 1000.0
        return out
    b._oneshot_latent = latent
    orig_window = codec.decode_window

    def window(latent_bct, start, n, **kw):
        clock.now += (window_ms if "context_sec" in kw else whole_ms) / S / 1000.0
        return orig_window(latent_bct, start, n, **kw)
    codec.decode_window = window
    events: list = []
    b.render_oneshot(_req(seeds=seeds), store=KeptLatents(), emit=events.append)
    return [e for e in events if isinstance(e, OneshotDone)]


def test_no_slice_starves_at_the_measured_costs(monkeypatch, small_windows):
    # The measured costs (bandattr DiT 1.68 s, 6 s window 31 ms, whole-song
    # 0.48 s) scaled by the test's 0.2 s window against the real 6 s one, so
    # the ratio of work to playback time is the real one.
    k = 0.2 / 6.0
    dit, win, whole = 1680 * k, 31 * k, 480 * k
    done = _paced_run(monkeypatch, dit_ms=dit, window_ms=win, whole_ms=whole)
    v0, v1 = (d.ms for d in done)
    # The schedule's arithmetic (W = 1): version 2's first audio waits for
    # version 1's single first window only, not for its whole decode.
    assert v0["first_audio"] == pytest.approx(dit + win, abs=0.1)
    assert v1["first_audio"] == pytest.approx(dit + win + dit + win, abs=0.1)
    assert v0["all_out"] == pytest.approx(2 * dit + 2 * win + whole, abs=0.1)
    assert v1["all_out"] == pytest.approx(2 * dit + 2 * win + 2 * whole, abs=0.1)
    assert v0["starve_slack_ms"] > 0 and v1["starve_slack_ms"] > 0
    # Version 1's window 1 is due 200 ms after its first audio and lands
    # after version 2's DiT + first window + its own whole-song decode.
    assert v0["starve_slack_ms"] == pytest.approx(200 - (dit + win + whole), abs=0.1)


def test_the_starvation_check_has_teeth(monkeypatch, small_windows):
    # A DiT four times longer than the window's playback: version 1's
    # second window would arrive after it was due.
    done = _paced_run(monkeypatch, dit_ms=800.0, window_ms=1.0, whole_ms=16.0)
    assert done[0].ms["starve_slack_ms"] < 0
    assert oneshot_starvation_slack_ms([(0, 10), (48000, 10)], [5.0, 900.0]) == pytest.approx(105.0)
    assert oneshot_starvation_slack_ms([(0, 10), (48000, 10)], [5.0, 1100.0]) == pytest.approx(-95.0)


def test_two_minute_schedule_budget():
    """The runbook's budget on the real geometry (2:00, 6 s windows, W = 3)
    with the gated costs: version 2 first audio and all-on-wire, hybrid vs
    windowed, and the least slack of version 1."""
    total = 120 * DELIVERY_SAMPLE_RATE
    n48 = 6 * DELIVERY_SAMPLE_RATE
    plan = oneshot_window_plan(total, n48)
    w = hybrid_first_windows(ONESHOT_HYBRID_FIRST_S, n48, len(plan))
    dit, win, whole = 1680.0, 31.0, 480.0
    assert (len(plan), w) == (20, 3)
    # Hybrid: DiT1, W windows v1, DiT2, W windows v2, whole v1, whole v2.
    v1_emit = [dit + win * (i + 1) for i in range(w)]
    t = dit + w * win + dit + w * win
    v2_first = dit + w * win + dit + win
    v1_rest = t + whole
    v1_emit += [v1_rest] * (len(plan) - w)
    all_on_wire = v1_rest + whole
    windowed_v2_first = dit + len(plan) * win + dit + win
    windowed_all = 2 * (dit + len(plan) * win)
    assert windowed_v2_first - v2_first == pytest.approx((len(plan) - w) * win)
    assert windowed_all - all_on_wire == pytest.approx(2 * (len(plan) * win) - 2 * w * win - 2 * whole)
    slack = oneshot_starvation_slack_ms(plan, v1_emit)
    assert slack > 3000.0                               # window 3 due at 18 s of playback
