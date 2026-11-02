"""``render_decode`` "windows": the one-shot decoded in the ring's windows.

CPU-only, fake codecs. The load-bearing properties:

* the ring path is unchanged: its window geometry is the same arithmetic
  (checked against the pre-change code kept here as the oracle), and its
  codec calls carry no context argument (the codec's 2 s default);
* a windowed one-shot is the ring's per-stream producer over windows in song
  order, clamped at the song's end exactly as the ring clamps, each window
  emitted as it lands, tagged with the version id and index, closed by an
  ``OneshotDone`` whose sha256 is the assembled song's;
* "song" (and no flag) keeps today's single ``OneshotReady``.
"""

from __future__ import annotations

import hashlib
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zstandard as zstd

from acestep.streaming import sa3_backend as SB
from acestep.streaming.events import OneshotDone, OneshotReady, OneshotSlice
from acestep.streaming.oneshot import (
    KeptLatents,
    OneshotError,
    OneshotResult,
    oneshot_pcm_sha256,
    parse_request,
    pcm16_from_f16,
)
from acestep.streaming.sa3_backend import (
    DELIVERY_SAMPLE_RATE,
    ONESHOT_WINDOW_CONTEXT_S,
    clamp_window_start,
    oneshot_window_plan,
    window_decode_span,
)
from demos.realtime_motion_graph_web import protocol
from demos.realtime_motion_graph_web.audio_codec import (
    send_oneshot_done,
    send_oneshot_slice,
)

from tests.unit import test_sa3_stream_layout as LT
from tests.unit.test_oneshot import _session


# ---- fakes ---------------------------------------------------------------------


class _RampWindowCodec:
    """A windowed codec (SAME-L stand-in) whose audio depends on the
    absolute sample position and on the stream (its latent's mean), so any
    geometry slip shows in the bytes. Records every call with its kwargs."""

    def __init__(self):
        self.calls: list = []

    def decode_window(self, latent_bct, start, n, **kw):
        self.calls.append((int(start), int(n), dict(kw)))
        pos = torch.arange(int(start), int(start) + int(n), dtype=torch.float32)
        level = float(latent_bct.mean()) * 0.1
        ramp = 0.3 * torch.sin(pos / 97.0) + level
        return torch.stack([ramp, -ramp])


class _NoWindowCodec(LT._SliceCodec):
    """SAME-S stand-in: full decode only."""


def _req(seeds=(3,), return_latents=False, redo=None):
    data = {"request_id": "r", "seeds": list(seeds), "return_latents": return_latents}
    if redo is not None:
        data["redo"] = redo
    return parse_request(data)


def _backend(render_decode="windows", codec=None):
    rt = LT._LayoutRuntime({"render_decode": render_decode, "deliver_streams": True})
    codec = codec if codec is not None else _RampWindowCodec()
    b, _ctx, _rt = LT._layout_backend(runtime=rt, codec=codec)
    return b, codec


@pytest.fixture
def small_windows(monkeypatch):
    """The test song is 0.74 s: 0.2 s windows give four, the last clamped."""
    monkeypatch.setattr(SB, "ONESHOT_WINDOW_S", 0.2)
    return round(0.2 * DELIVERY_SAMPLE_RATE)


# ---- geometry: the ring's arithmetic, shared ------------------------------------


def _old_geometry(start48_raw, n48, dur48):
    """The ring's window geometry exactly as 845e5350 wrote it inline in
    ``_render_window_via_codec`` / ``_render_stream_windows``."""
    start48 = max(0, min(start48_raw, max(0, dur48 - n48)))
    m44 = 588
    start44 = (start48 * 44100) // 48000
    n44 = -(-n48 * 147 // 160)
    lo44 = max(0, start44 - m44)
    lead44 = start44 - lo44
    total44 = lead44 + n44 + m44
    lead48 = (lead44 * 48000) // 44100
    return start48, lo44, total44, lead48


@pytest.mark.parametrize("n48", [144000, 4800, 9600])
def test_window_geometry_is_the_rings_arithmetic(n48):
    dur48 = 5_760_000
    for raw in list(range(0, 3000, 7)) + [n48 * k + d for k in range(60) for d in (-3, 0, 5)] + [dur48, dur48 + 9]:
        s = clamp_window_start(raw, n48, dur48)
        assert (s, *window_decode_span(s, n48)) == _old_geometry(raw, n48, dur48)
    # A song shorter than a window starts at 0.
    assert clamp_window_start(500, n48, n48 // 2) == 0


def test_window_plan_tiles_the_song_in_order_and_clamps_the_last():
    n48 = 144000
    total = 5_760_000 + 12_345           # 2:00 and a bit
    plan = oneshot_window_plan(total, n48)
    assert len(plan) == -(-total // n48) == 41
    starts = [s for s, _ in plan]
    assert starts == sorted(starts) and starts[0] == 0
    for i, (s, f) in enumerate(plan[:-1]):
        assert (s, f) == (i * n48, n48)
    last_s, last_f = plan[-1]
    assert last_s == clamp_window_start(40 * n48, n48, total) == total - n48
    assert last_s + last_f == total
    covered = np.zeros(total, bool)
    for s, f in plan:
        covered[s:s + f] = True
    assert covered.all()
    # Exactly divisible: no overlap; shorter than one window: one window.
    assert oneshot_window_plan(3 * n48, n48) == [(0, n48), (n48, n48), (2 * n48, n48)]
    assert oneshot_window_plan(1000, n48) == [(0, 1000)]
    with pytest.raises(ValueError):
        oneshot_window_plan(0, n48)


# ---- the ring path is byte-identical ---------------------------------------------


def _old_ring_render(b, latent_btc, t_start_s):
    """The pre-change ring render under a layout (845e5350
    ``_render_window_via_codec`` + ``_render_stream_windows``)."""
    n48 = int(round(b.vae_window * DELIVERY_SAMPLE_RATE))
    dur48 = int(round((b.playable_duration_s() or 0.0) * DELIVERY_SAMPLE_RATE))
    start48, lo44, total44, lead48 = _old_geometry(
        int(round(t_start_s * DELIVERY_SAMPLE_RATE)), n48, dur48,
    )
    streams = b._render_stream_audio(
        latent_btc, lo44, total44, start48=start48, n48=n48, lead48=lead48,
        decode_seed=b._decode_seed,
    ).cpu().numpy()
    return b._mix_streams(streams.astype(np.float32, copy=False)), start48


def test_ring_slices_are_unchanged_with_the_default_context():
    codec = _RampWindowCodec()
    b, _ctx, _rt = LT._layout_backend(codec=codec)
    latent = LT._run_until_fresh(b, LT._knobs(b, sa3_denoise=0.7))
    src = b._current_result if b._current_result is not None else b._last_result_latent
    for t in (0.0, 0.05, 0.31, 0.6, 0.73, 5.0):
        codec.calls.clear()
        chunk = b.render_window(t)
        new_calls = list(codec.calls)
        codec.calls.clear()
        want, start = _old_ring_render(b, src, t)
        assert chunk.start_sample == start
        assert chunk.pcm.dtype == want.dtype and chunk.pcm.tobytes() == want.tobytes()
        assert new_calls == codec.calls
        assert all(kw == {} for _s, _n, kw in new_calls)   # the codec's own context
    assert latent is not None


class _Helpers:
    """sa3_stream_helpers stand-in for SA3SAMEWindowCodec: records the
    context each decode resolves with."""

    def __init__(self):
        self.contexts: list = []

    def decode_sa3_latent_window(self, sam, latent_bct, *, target_start_sample,
                                 target_num_samples, context_sec, chunked, deterministic):
        self.contexts.append(("eager", context_sec))
        return SimpleNamespace(audio_ct=torch.zeros(2, target_num_samples))

    def resolve_sa3_decode_window(self, latent_bct, **kw):
        self.contexts.append(("trt", kw["context_sec"]))
        from acestep.engine.sa3_stream_helpers import resolve_sa3_decode_window

        return resolve_sa3_decode_window(latent_bct, **kw)


def _window_codec(engines):
    from acestep.engine.sa3_context import SA3SAMEWindowCodec

    c = SA3SAMEWindowCodec.__new__(SA3SAMEWindowCodec)
    c._helpers = _Helpers()
    c._context = SimpleNamespace(sam=None, sample_rate=44100, downsampling_ratio=4096)
    c._scale = 1.0
    c._engines = engines
    return c


def test_window_codec_context_defaults_to_the_rings_and_takes_the_oneshots():
    c = _window_codec([])
    lat = torch.zeros(1, 256, 200)
    c.decode_window(lat, 4096 * 50, 4096 * 33)
    c.decode_window(lat, 4096 * 50, 4096 * 33, context_sec=ONESHOT_WINDOW_CONTEXT_S)
    assert c._helpers.contexts == [("eager", 2.0), ("eager", 0.75)]
    # The TRT path resolves its slice with the same context; with no engine
    # covering it (profile 32..40 here) it decodes eagerly at that context.
    c2 = _window_codec([("x", 32, 40)])
    c2.decode_window(lat, 4096 * 50, 4096 * 33)
    c2.decode_window(lat, 4096 * 50, 4096 * 33, context_sec=0.75)
    assert c2._helpers.contexts == [("trt", 2.0), ("eager", 2.0), ("trt", 0.75), ("eager", 0.75)]


# ---- the windowed one-shot -------------------------------------------------------


def test_windows_emit_slices_in_order_then_done(small_windows):
    n48 = small_windows
    b, codec = _backend("windows")
    events: list = []
    store = KeptLatents()
    res = b.render_oneshot(_req(seeds=(4, 5), return_latents=True), store=store, emit=events.append)
    assert isinstance(res, OneshotResult) and res.delivered and res.pcm is None
    total = round(b.playable_duration_s() * DELIVERY_SAMPLE_RATE)
    plan = oneshot_window_plan(total, n48)
    assert len(plan) == 4 and plan[-1][0] == total - n48     # the ring's clamp
    kinds = [type(e).__name__ for e in events]
    assert kinds == (["OneshotSlice"] * 4 + ["OneshotDone"]) * 2
    for v, vid in enumerate(res.ids):
        part = events[v * 5:(v + 1) * 5]
        slices, done = part[:4], part[4]
        assert [s.index for s in slices] == [0, 1, 2, 3]
        assert all(s.version == vid and s.count == 4 and s.request_id == "r" for s in slices)
        assert [(s.start_frame, s.frames) for s in slices] == plan
        assert all(s.pcm.dtype == np.float16 and s.pcm.shape == (s.frames, 2 * LT.N_STREAMS)
                   and s.channels == 2 * LT.N_STREAMS for s in slices)
        assert isinstance(done, OneshotDone) and done.version == vid and done.frames == total
        assert done.windows == 4 and done.seed == res.seeds[v]
        assert set(done.ms) == {"generate", "first_slice", "decode", "pack", "windows_wall"}
        # The sha is the assembled song's, slices written in index order.
        song = np.zeros((total, 2 * LT.N_STREAMS), np.float16)
        for s in slices:
            song[s.start_frame:s.start_frame + s.frames] = s.pcm
        assert done.sha256 == oneshot_pcm_sha256(song, LT.N_STREAMS)
        assert done.latents.shape == (1, LT.N_STREAMS, LT.C, LT.L)
        assert done.latent_header["frames"] == LT.L
        assert store.get(vid).seed == res.seeds[v]
    # Every window decode carries the one-shot's context; five (here three)
    # stream calls per window, windows in song order.
    assert all(kw == {"context_sec": ONESHOT_WINDOW_CONTEXT_S} for _s, _n, kw in codec.calls)
    assert len(codec.calls) == 2 * 4 * LT.N_STREAMS
    starts = [c[0] for c in codec.calls[:4 * LT.N_STREAMS:LT.N_STREAMS]]
    assert starts == [window_decode_span(s, n48)[0] for s, _ in plan]


def test_windows_audio_matches_the_whole_song_render(small_windows):
    """Same latent, same codec: the windowed song equals the song path's
    within the float16 step and the resampler's window edge."""
    b, _codec = _backend("windows")
    events: list = []
    store = KeptLatents()
    b.render_oneshot(_req(seeds=(9,)), store=store, emit=events.append)
    total = events[-1].frames
    song = np.zeros((total, 2 * LT.N_STREAMS), np.float16)
    for s in events[:-1]:
        song[s.start_frame:s.start_frame + s.frames] = s.pcm
    b2, _ = _backend("song")
    ref = b2.render_oneshot(_req(seeds=(9,)), store=KeptLatents())   # [1, S, N, 2] int16
    got = pcm16_from_f16(song).reshape(total, LT.N_STREAMS, 2).transpose(1, 0, 2)
    diff = np.abs(got.astype(np.int32) - ref.pcm[0].astype(np.int32))
    # float16 at |x| in [0.5, 1) steps 2^-11 (16 LSB, so 8 LSB of rounding);
    # the windowed resample filters the same samples up to its guard margin,
    # a few LSB at a seam. Away from the song edges:
    inner = diff[:, 2000:-2000]
    assert inner.max() <= 24 and float(np.sqrt((inner.astype(float) ** 2).mean())) < 6.0


def test_song_flag_and_no_flag_keep_one_oneshot_ready():
    for flag in ("song", None):
        b, _ = _backend(flag)
        events: list = []
        res = b.render_oneshot(_req(seeds=(1,)), store=KeptLatents(), emit=events.append)
        assert events == [] and not res.delivered and res.pcm is not None
        assert res.pcm.shape[:2] == (1, LT.N_STREAMS)


def test_flag_is_read_per_request_and_refused_when_unknown(small_windows):
    mode = {"v": "song"}
    b, _ = _backend(lambda: mode["v"])
    events: list = []
    assert not b.render_oneshot(_req(), store=KeptLatents(), emit=events.append).delivered
    mode["v"] = "windows"
    assert b.render_oneshot(_req(), store=KeptLatents(), emit=events.append).delivered
    mode["v"] = "chunks"
    with pytest.raises(OneshotError, match="render_decode"):
        b.render_oneshot(_req(), store=KeptLatents(), emit=events.append)


def test_without_emit_the_song_path_runs(small_windows):
    b, _ = _backend("windows")
    res = b.render_oneshot(_req(), store=KeptLatents())             # no emit: an old session
    assert res.pcm is not None and not res.delivered


def test_windows_need_a_windowed_codec():
    b, _ = _backend("windows", codec=_NoWindowCodec())
    with pytest.raises(OneshotError, match="windowed codec"):
        b.render_oneshot(_req(), store=KeptLatents(), emit=lambda e: None)


# ---- the sha rule ----------------------------------------------------------------


def test_pcm16_rule_and_sha_layout():
    x = np.array([0.0, 1.0, -1.0, 0.5, -0.5, 2.0 ** -15, -(2.0 ** -15)], np.float16)
    assert pcm16_from_f16(x).tolist() == [0, 32767, -32767, 16384, -16383, 1, -1]
    song = np.random.default_rng(0).uniform(-1, 1, (37, 6)).astype(np.float16)
    blocks = pcm16_from_f16(song).reshape(37, 3, 2).transpose(1, 0, 2)
    assert oneshot_pcm_sha256(song, 3) == hashlib.sha256(np.ascontiguousarray(blocks).tobytes()).hexdigest()
    with pytest.raises(ValueError):
        oneshot_pcm_sha256(song, 2)


# ---- session and wire ------------------------------------------------------------


class _ProgressiveBackend:
    """Fake pod backend that answers progressively."""

    name = "fake"

    def __init__(self):
        from acestep.streaming.generator_backend import Capabilities

        self._caps = Capabilities(oneshot=True)

    def capabilities(self):
        return self._caps

    def render_oneshot(self, req, *, emit=None):
        pcm = np.zeros((4, 10), np.float16)
        emit(OneshotSlice(req.request_id, "v1", 1, 0, 1, 0, 4, 10, 48000, pcm))
        emit(OneshotDone(req.request_id, "v1", 1, 4, 10, 48000, 1, "ab", {"generate": 1.0}))
        return OneshotResult(None, 48000, ("v1",), (1,), (0, 1, 2, 3, 4),
                             {"generate": 1.0, "decode": 1.0, "pack": 0.0}, 4 / 48000,
                             delivered=True)


def test_session_passes_emit_and_publishes_no_oneshot_ready():
    s, events = _session(_ProgressiveBackend())
    s.oneshot({"request_id": "a", "seeds": [1]})
    s._service_oneshots()
    assert [type(e).__name__ for e in events] == ["OneshotSlice", "OneshotDone"]
    assert not any(isinstance(e, OneshotReady) for e in events)


def _ws():
    sent: list = []
    ws = SimpleNamespace(send=lambda m: sent.append(m if isinstance(m, (str, bytes)) else b"".join(m)))
    return ws, sent


def test_slice_wire_frame_is_a_ring_slice_against_zeros():
    rng = np.random.default_rng(1)
    pcm = rng.uniform(-1, 1, (300, 10)).astype(np.float16)
    ev = OneshotSlice("r", "v7", 3, 2, 5, 288000, 300, 10, 48000, pcm)
    ws, sent = _ws()
    send_oneshot_slice(ws, event=ev, zctx=zstd.ZstdCompressor(level=1))
    head = json.loads(sent[0])
    assert head == {
        "type": "oneshot_slice", "request_id": "r", "version": "v7", "seed": 3,
        "index": 2, "count": 5, "start_frame": 288000, "frames": 300, "channels": 10,
        "sample_rate": 48000, "encoding": "slice_f16_zstd",
    }
    spec = next(e for e in protocol.EVENTS if e.name == "oneshot_slice")
    assert set(head) - {"type"} == {f.name for f in spec.fields} and spec.binary_follow
    frame = sent[1]
    flags, start, frames, ch, tick_ms, dec_ms, num_gens = struct.unpack(
        protocol.SLICE_HDR_FMT, frame[:protocol.SLICE_HDR_SIZE])
    assert (flags, start, frames, ch, num_gens) == (protocol.SLICE_FLAG_DELTA, 288000, 300, 10, 2)
    body = zstd.ZstdDecompressor().decompress(frame[protocol.SLICE_HDR_SIZE:])
    assert body == pcm.astype("<f2").tobytes()


def test_done_wire_frame_with_and_without_latents():
    lat = np.arange(1 * 5 * 2 * 3, dtype=np.float16).reshape(1, 5, 2, 3)
    hdr = {"n_streams": 5, "channels": 2, "frames": 3, "seconds": 1.5}
    ev = OneshotDone("r", "v7", 3, 72000, 10, 48000, 1, "cafe", {"generate": 2.0},
                     latent_header=hdr, latents=lat)
    ws, sent = _ws()
    send_oneshot_done(ws, event=ev)
    head = json.loads(sent[0])
    assert head["sha256"] == "cafe" and head["version"] == "v7" and head["frames"] == 72000
    assert head["latents"] == {"dtype": "float16", "shape": [1, 5, 2, 3], "bytes": lat.nbytes}
    assert head["latent_header"] == hdr
    assert sent[1] == lat.astype("<f2").tobytes() and len(sent) == 2
    spec = next(e for e in protocol.EVENTS if e.name == "oneshot_done")
    assert set(head) - {"type"} == {f.name for f in spec.fields}
    ws2, sent2 = _ws()
    send_oneshot_done(ws2, event=OneshotDone("r", "v", 0, 1, 10, 48000, 1, "00", {}))
    assert len(sent2) == 1 and "latents" not in json.loads(sent2[0])


def test_default_kept_window_is_aligned_and_fits_the_window_engine():
    """The default kept slice keeps the 147:160 bookkeeping exact and every
    window of a 2:00 one-shot within the window engine's profile (96)."""
    import math

    from acestep.engine.sa3_trt import CANONICAL_SAME_L_WINDOW

    n48 = int(round(SB.ONESHOT_WINDOW_S * DELIVERY_SAMPLE_RATE))
    assert n48 % 160 == 0
    frames_total, total48 = 1292, 5760000
    ctx = math.ceil(ONESHOT_WINDOW_CONTEXT_S * 44100 / 4096)
    for start48, _kept in oneshot_window_plan(total48, n48):
        lo44, t44, _lead = window_decode_span(start48, n48)
        span = min(frames_total, math.ceil((lo44 + t44) / 4096) + ctx) - max(0, lo44 // 4096 - ctx)
        assert span <= CANONICAL_SAME_L_WINDOW[2]
