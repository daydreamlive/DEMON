"""The one-shot sha off the critical path: ``OneshotPcmHasher``.

CPU-only. The load-bearing properties:

* the int16 table is the rule (``pcm16_from_f16``) over all 65536 float16
  bit patterns, so the bytes hashed are unchanged;
* the running hash of the slices, in any order and any sizes, is
  ``oneshot_pcm_sha256`` of the assembled song (later windows over earlier);
* a bad feed (overlap, out of range, wrong dtype) or an early close is
  reported, never a wrong digest or a hang;
* the digest is ready within 50 ms of the last slice: on the fake engine
  through ``render_oneshot`` ("windows" and "hybrid"), and on a 2:00 song
  of the real geometry fed at the window decoder's pace.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from acestep.streaming import oneshot as OS
from acestep.streaming.events import OneshotDone, OneshotSlice
from acestep.streaming.oneshot import (
    KeptLatents,
    OneshotPcmHasher,
    oneshot_pcm_sha256,
    pcm16_from_f16,
    pcm16_lut,
)

from tests.unit import test_sa3_stream_layout as LT
from tests.unit.test_oneshot_hybrid import _WholeSongCodec, _backend, _song_of
from tests.unit.test_oneshot_windows import _RampWindowCodec, _req


def _song(frames, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.05, 1.05, (frames, 2 * n)).astype(np.float16)


def test_table_is_the_rule_for_every_float16():
    x = np.arange(65536, dtype=np.uint16).view(np.float16)
    with np.errstate(invalid="ignore"):
        want = pcm16_from_f16(x)
    assert np.array_equal(pcm16_lut(), want)
    assert pcm16_lut().dtype == np.dtype("<i2") and not pcm16_lut().flags.writeable


@pytest.mark.parametrize("order", ["forward", "reverse", "shuffled"])
def test_running_hash_is_the_assembled_songs(monkeypatch, order):
    monkeypatch.setattr(OS, "_PCM16_CHUNK", 7)               # many conversion tasks per slice
    n, total = 3, 1000
    song = _song(total, n)
    cuts = [0, 1, 64, 65, 300, 512, 999, 1000]
    spans = list(zip(cuts[:-1], cuts[1:]))
    if order == "reverse":
        spans = spans[::-1]
    elif order == "shuffled":
        spans = [spans[i] for i in np.random.default_rng(3).permutation(len(spans))]
    h = OneshotPcmHasher(total, n)
    for a, b in spans:
        h.feed(a, song[a:b])
    assert h.digest(5) == oneshot_pcm_sha256(song, n)
    assert h.ready and h.ready_t is not None


def test_bad_feeds_are_reported():
    n, total = 2, 100
    song = _song(total, n)
    h = OneshotPcmHasher(total, n)
    h.feed(0, song[0:60])
    h.feed(50, song[50:100])                                  # overlaps [0, 60)
    with pytest.raises(ValueError, match="overlaps"):
        h.digest(5)
    h = OneshotPcmHasher(total, n)
    h.feed(90, song[0:20])                                    # past the end
    with pytest.raises(ValueError, match="outside"):
        h.digest(5)
    h = OneshotPcmHasher(total, n)
    with pytest.raises(ValueError, match="float16"):
        h.feed(0, song.astype(np.float32))
    h.feed(0, song[:40])
    h.close()                                                  # an aborted render
    with pytest.raises(RuntimeError, match="closed"):
        h.digest(5)
    h = OneshotPcmHasher(total, n)
    with pytest.raises(TimeoutError):
        h.digest(0.01)
    h.close()


@pytest.mark.parametrize("mode,codec", [("windows", _RampWindowCodec), ("hybrid:0.3", _WholeSongCodec)])
def test_done_follows_the_last_slice_within_50_ms(small_windows, mode, codec):
    b = _backend(mode, codec())
    stamps: list = []
    events: list = []

    def emit(e):
        stamps.append(time.perf_counter())
        events.append(e)
    res = b.render_oneshot(_req(seeds=(4, 5)), store=KeptLatents(), emit=emit)
    assert res.delivered
    for vid in res.ids:
        mine = [(e, t) for e, t in zip(events, stamps) if e.version == vid]
        slices = [e for e, _ in mine if isinstance(e, OneshotSlice)]
        last_slice = max(t for e, t in mine if isinstance(e, OneshotSlice))
        done, t_done = next((e, t) for e, t in mine if isinstance(e, OneshotDone))
        song = _song_of(slices, done.frames, done.channels)
        assert done.sha256 == oneshot_pcm_sha256(song, LT.N_STREAMS)
        assert (t_done - last_slice) * 1000.0 < 50.0


@pytest.fixture
def small_windows(monkeypatch):
    from acestep.streaming import sa3_backend as SB
    from acestep.streaming.sa3_backend import DELIVERY_SAMPLE_RATE

    monkeypatch.setattr(SB, "ONESHOT_WINDOW_S", 0.2)
    return round(0.2 * DELIVERY_SAMPLE_RATE)


def test_digest_ready_within_50_ms_of_the_last_slice_at_2_minutes():
    """The real geometry: a 2:00 version is 5,760,000 frames x 10 channels
    (115 MB as int16), twenty 6 s windows, fed at the window decoder's pace
    (about 30 ms per window). Without the running hash the sha took about
    0.6 s after the last slice. The bound is on the median of three runs
    (what is left after the last slice is mostly hashing streams 1.., about
    35 ms of sha256 at 2.6 GB/s; a busy box adds a few ms)."""
    n, w, count = 5, 288_000, 20
    total = w * count
    song = _song(total, n, seed=1)
    want = oneshot_pcm_sha256(song, n)
    gaps = []
    for _ in range(3):
        h = OneshotPcmHasher(total, n)
        for k in range(count):
            time.sleep(0.03)
            h.feed(k * w, song[k * w:(k + 1) * w])
            t_last = time.perf_counter()
        assert h.digest(10) == want
        gaps.append((h.ready_t - t_last) * 1000.0)
    assert sorted(gaps)[1] < 50.0, gaps
