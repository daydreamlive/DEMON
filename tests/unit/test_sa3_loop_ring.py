"""SA3 loop ring (#365): circular denoising so a loop has no seam.

Adapter mechanics (roll/unroll, per-timestep offsets, the batch-1 TRT
loop), the ring geometry (frame snap, source stretch, the env switch,
the swap stretch rule) and the backend's decode-side periodic tail.
CPU + fake DiTs/codecs throughout.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from acestep.engine import sa3_context as ctx_mod
from acestep.engine.sa3_adapter import SA3Adapter, ring_margins, ring_offset
from acestep.streaming.knobs import KnobState
from acestep.streaming.sa3_backend import (
    SA3_SAMPLE_RATE,
    SA3Backend,
    _ring_stretch_swap,
    delivered_samples,
    sa3_knob_specs,
)

SR = 44100
DS = 4096
C = 8          # channel count is irrelevant to the ring; keep it small
T = 20         # window frames
N = 14         # ring frames (loop) inside the window

# A pingpong-ish 8-step schedule at denoise 0.9 (t values only matter
# through ring_offset).
SCHEDULE_8 = [0.9, 0.86, 0.8, 0.71, 0.6, 0.45, 0.28, 0.11]


def _adapter(dit, ring_frames=None):
    return SA3Adapter(
        dit,
        schedule_builder=lambda d: torch.linspace(float(d), 0.0, 9),
        device="cpu",
        dtype=torch.float32,
        ring_frames=ring_frames,
    )


def _aux(b):
    return [{"cross_attn_cond": torch.ones(1, 3, 4),
             "cross_attn_mask": torch.ones(1, 3)} for _ in range(b)]


def _forward(adapter, x_btc, ts):
    b = x_btc.shape[0]
    return adapter.batched_forward(
        x_btc, list(ts), [None] * b, [None] * b, [None] * b, _aux(b),
    )


class _CircularConvDit(torch.nn.Module):
    """Translation-equivariant on a ring of ``period`` frames: a circular
    depthwise conv over the first ``period`` frames, its output tiled over
    the whole window (native [B,C,T]). For a ``period``-periodic input
    that is the same as convolving every frame circularly, so it does not
    care where in the window the ring starts."""

    def __init__(self, period):
        super().__init__()
        self.period = period
        g = torch.Generator().manual_seed(0)
        self.w = torch.randn(C, 1, 5, generator=g)

    def forward(self, x, t, **_):
        ring = x[..., : self.period]
        padded = torch.cat([ring[..., -2:], ring, ring[..., :2]], dim=-1)
        y = torch.nn.functional.conv1d(padded, self.w, groups=C)
        reps = -(-x.shape[-1] // self.period)
        return y.repeat(1, 1, reps)[..., : x.shape[-1]] + t.view(-1, 1, 1)


class _PositionDit(torch.nn.Module):
    """Returns each frame's INPUT channel-0 value broadcast — i.e. the
    velocity at position j is whatever latent sat at position j."""

    def forward(self, x, t, **_):
        return x[:, :1, :].expand_as(x).clone()


class _StepBundleDit:
    """Batch-1 TRT stand-in: ``step_bundle`` writes into a persistent
    output buffer, which the adapter must materialize per slot."""

    trt_batch1 = True

    def __init__(self, inner):
        self.inner = inner
        self.buf = None
        self.calls = 0

    def step_bundle(self, x_1ct, t, bundle):
        self.calls += 1
        v = self.inner(x_1ct, torch.tensor([t]))
        if self.buf is None:
            self.buf = torch.empty_like(v)
        self.buf.copy_(v)
        return self.buf


# ---- adapter ---------------------------------------------------------------


def test_ring_offset_is_deterministic_and_spread():
    for n in (14, 136, 161, 220, 646):
        ks = [ring_offset(t, n) for t in SCHEDULE_8]
        assert ks == [ring_offset(t, n) for t in SCHEDULE_8]
        assert all(0 <= k < n for k in ks)
        assert any(k != 0 for k in ks)
    # Distinct (a hash may collide once in a while; never collapse).
    for n in (136, 161, 220, 646):
        ks = [ring_offset(t, n) for t in SCHEDULE_8]
        assert len(set(ks)) >= 7, (n, ks)
    # The shifted schedule's structural steps sit close together near
    # t = 1; their offsets must still spread over the ring, or the
    # sequence start lands on one fixed bar inside the loop.
    clustered = [1.0, 0.994141, 0.984375, 0.958008, 0.942383, 0.927734]
    for n in (136, 161, 220):
        ks = sorted(ring_offset(t, n) for t in clustered)
        assert ks[-1] - ks[0] > n // 2, (n, ks)


def test_ring_on_matches_ring_off_for_an_equivariant_dit():
    dit = _CircularConvDit(N)
    x = torch.randn(3, T, C)
    ts = SCHEDULE_8[:3]
    v_off = _forward(_adapter(dit), x, ts)
    v_on = _forward(_adapter(dit, ring_frames=N), x, ts)
    assert v_on.shape == x.shape and v_on.dtype == x.dtype
    assert torch.allclose(v_on[:, :N], v_off[:, :N], atol=1e-5)
    # The tail is the periodic continuation of the ring.
    assert torch.equal(v_on[:, N:], v_on[:, : T - N])


def test_roll_unroll_restores_order():
    x = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(2, T, C).clone()
    v = _forward(_adapter(_PositionDit(), ring_frames=N), x, SCHEDULE_8[:2])
    expect = torch.tensor([j % N for j in range(T)], dtype=torch.float32)
    for b in range(2):
        assert torch.equal(v[b, :, 0], expect)


def test_dit_sees_a_rolled_periodic_window():
    seen = {}

    class _Spy(torch.nn.Module):
        def forward(self, x, t, **_):
            seen["x"] = x.clone()
            return torch.zeros_like(x)

    x = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(1, T, C).clone()
    t = SCHEDULE_8[1]
    _forward(_adapter(_Spy(), ring_frames=N), x, [t])
    k = ring_offset(t, N)
    m_l, m_r = ring_margins(N, T)
    assert (m_l, m_r) == (3, 3)
    got = seen["x"][0, 0]  # native [C, T] -> channel 0
    ring = torch.roll(torch.arange(N, dtype=torch.float32), k)
    # The rolled ring occupies [m_l, m_l + N) of the DiT input ...
    assert torch.equal(got[m_l:m_l + N], ring)
    # ... and both margins are its periodic continuation, so neither
    # window edge is a loop position with nothing beyond it.
    assert torch.equal(got[:m_l], ring[N - m_l:])
    assert torch.equal(got[m_l + N:], ring[:m_r])
    assert torch.equal(got, torch.cat([ring[N - m_l:], ring, ring[:m_r]]))


def test_velocity_is_read_from_the_ring_span():
    """A position-marking DiT (velocity = the window index): every output
    frame must come from ``[m_l, m_l + N)``, rolled back by ``-k``."""

    class _Index(torch.nn.Module):
        def forward(self, x, t, **_):
            idx = torch.arange(x.shape[-1], dtype=x.dtype)
            return idx.view(1, 1, -1).expand_as(x).clone()

    ts = SCHEDULE_8[3:5]
    v = _forward(_adapter(_Index(), ring_frames=N), torch.zeros(2, T, C), ts)
    m_l, _ = ring_margins(N, T)
    for b, t in enumerate(ts):
        k = ring_offset(t, N)
        expect = torch.tensor([m_l + (j + k) % N for j in range(T)],
                              dtype=torch.float32)
        assert torch.equal(v[b, :, 0], expect)
        assert int(v[b, :, 0].min()) >= m_l
        assert int(v[b, :, 0].max()) < m_l + N


def test_ring_margins_split():
    assert ring_margins(14, 20) == (3, 3)
    assert ring_margins(14, 21) == (3, 4)   # odd headroom: extra frame right
    assert ring_margins(136, 168) == (16, 16)
    assert ring_margins(14, 15) == (0, 1)   # < 2 frames: phase C layout
    assert ring_margins(14, 14) == (0, 0)


@pytest.mark.parametrize("t_len", [N, N + 1])
def test_small_headroom_falls_back_to_ring_at_window_start(t_len):
    seen = {}

    class _Spy(torch.nn.Module):
        def forward(self, x, t, **_):
            seen["x"] = x.clone()
            return x[:, :1, :].expand_as(x).clone()

    x = torch.arange(t_len, dtype=torch.float32).view(1, t_len, 1)
    x = x.expand(1, t_len, C).clone()
    t = SCHEDULE_8[2]
    v = _forward(_adapter(_Spy(), ring_frames=N), x, [t])
    k = ring_offset(t, N)
    ring = torch.roll(torch.arange(N, dtype=torch.float32), k)
    assert torch.equal(seen["x"][0, 0], torch.cat([ring, ring[: t_len - N]]))
    expect = torch.tensor([j % N for j in range(t_len)], dtype=torch.float32)
    assert torch.equal(v[0, :, 0], expect)


def test_ring_off_is_bit_identical():
    dit = _CircularConvDit(T)
    x = torch.randn(2, T, C)
    a = _forward(_adapter(dit), x, SCHEDULE_8[:2])
    b = _forward(_adapter(dit, ring_frames=None), x, SCHEDULE_8[:2])
    assert torch.equal(a, b)


def test_ring_larger_than_window_fails_loudly():
    with pytest.raises(ValueError, match="ring_frames"):
        _forward(_adapter(_PositionDit(), ring_frames=T + 1),
                 torch.zeros(1, T, C), [0.5])


def test_trt_batch1_path_rings_and_materializes():
    inner = _CircularConvDit(N)
    trt = _StepBundleDit(inner)
    x = torch.randn(3, T, C)
    ts = SCHEDULE_8[2:5]
    v_trt = _forward(_adapter(trt, ring_frames=N), x, ts)
    v_ref = _forward(_adapter(inner), x, ts)
    assert trt.calls == 3
    assert torch.allclose(v_trt[:, :N], v_ref[:, :N], atol=1e-5)
    # Each slot kept its own velocity (the persistent buffer did not
    # leak the last slot into the earlier ones).
    assert not torch.allclose(v_trt[0], v_trt[2])


# ---- geometry --------------------------------------------------------------


def _bare_context(song_seconds=180.0, outro_pad_s=3.0, extension=None):
    c = ctx_mod.SA3Context.__new__(ctx_mod.SA3Context)
    c.model_id = "medium"
    c.downsampling_ratio = DS
    c.sample_rate = SR
    c.song_seconds = song_seconds
    c.outro_pad_s = outro_pad_s
    c.extension = extension

    def _window(duration_s):
        # (seconds + pad) * sr aligned up to 8192, like upstream.
        target = int((duration_s + c.outro_pad_s) * SR)
        return (((target + 8191) // 8192) * 8192) // DS

    c.window_latent_frames = _window
    return c


def test_snap_round_vs_floor():
    # 12.632 s = 136.006 frames -> 136; 12.6 s = 135.66 -> 136 (round up).
    assert ctx_mod.snap_ring_frames(12.632) == 136
    assert ctx_mod.snap_ring_frames(12.6) == 136
    # Rounding up that does not fit falls back to the floor.
    assert ctx_mod.snap_ring_frames(12.6, fits=lambda n: n <= 135) == 135
    assert ctx_mod.snap_ring_frames(12.6, fits=lambda n: False) == 0


def test_snap_floors_at_the_trt_clamp(monkeypatch):
    from acestep.engine import sa3_trt

    c = _bare_context()
    # Engine cap = the window of a 10.0 s loop; 10.04 s rounds up to 108
    # frames (10.03 s), whose window still fits; pick a duration whose
    # round-up crosses an alignment boundary.
    cap = c.window_latent_frames(10.0)
    monkeypatch.setattr(sa3_trt, "max_dit_engine_latents", lambda mid: cap)
    d = None
    for i in range(1, 2000):
        cand = 10.0 + i * 0.001
        frames = cand * SR / DS
        up = round(frames)
        if up > frames and c.window_latent_frames(up * DS / SR) > cap \
                and c.window_latent_frames(math.floor(frames) * DS / SR) <= cap:
            d = cand
            break
    assert d is not None
    assert c.snap_loop_ring_frames(d, backend="tensorrt") == math.floor(d * SR / DS)
    # Eager has no engine cap: plain rounding.
    assert c.snap_loop_ring_frames(d, backend="eager") == round(d * SR / DS)


def test_plan_stretches_source_onto_the_ring_period():
    c = _bare_context()
    sr = 48000
    dur = 12.632
    wav = torch.randn(2, int(dur * sr))
    plan = c.plan_loop_ring(wav, sr, dur, env={})
    assert plan.reason == "on"
    assert plan.frames == 136
    playable_44k = 136 * DS
    assert plan.playable_s == playable_44k / SR
    rate, stretched = plan.source_audio
    assert rate == SR
    assert stretched.shape == (2, playable_44k)
    assert stretched.dtype == torch.float32
    expect_ratio = playable_44k / (wav.shape[-1] * SR / sr)
    assert plan.stretch_ratio == pytest.approx(expect_ratio)
    assert abs(plan.stretch_ratio - 1.0) < 0.0037
    # Playable, ring and delivered buffer agree.
    assert int(round(plan.playable_s * SR)) == playable_44k
    import torchaudio

    buf = torchaudio.functional.resample(stretched, SR, sr)
    assert buf.shape[-1] == delivered_samples(playable_44k)


def test_stretch_preserves_a_tone():
    sr = 48000
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 440 * t)[None].astype(np.float32)
    out, ratio = ctx_mod.stretch_to_length(tone, sr, 44100, 44100)
    assert out.shape == (1, 44100)
    assert ratio == pytest.approx(1.0)
    spec = np.abs(np.fft.rfft(out[0].numpy()))
    assert abs(int(np.argmax(spec)) - 440) <= 1


@pytest.mark.parametrize("env,ctx_kw,src_s,reason", [
    ({"DEMON_SA3_LOOP_RING": "0"}, {}, 12.632, "env_disabled"),
    ({}, {"song_seconds": None, "outro_pad_s": 6.0}, 12.632, "legacy_label"),
    ({}, {"extension": object()}, 12.632, "model_extension"),
    ({}, {}, 8.0, "source_shorter_than_loop"),
])
def test_plan_off_keeps_todays_geometry(env, ctx_kw, src_s, reason):
    c = _bare_context(**ctx_kw)
    sr = 48000
    wav = torch.randn(2, int(src_s * sr))
    plan = c.plan_loop_ring(wav, sr, 12.632, env=env)
    assert plan.frames is None
    assert plan.reason == reason
    assert plan.playable_s == 12.632
    assert plan.stretch_ratio == 1.0
    assert plan.source_audio[0] == sr and plan.source_audio[1] is wav


def test_loop_ring_setting():
    assert ctx_mod.loop_ring_setting({}) is True
    assert ctx_mod.loop_ring_setting({"DEMON_SA3_LOOP_RING": "1"}) is True
    for off in ("0", "off", "false"):
        assert ctx_mod.loop_ring_setting({"DEMON_SA3_LOOP_RING": off}) is False


def test_swap_stretch_rule():
    playable_44k = 136 * DS
    # Within 1%: stretched onto the period exactly, at 44.1 kHz.
    near = torch.randn(2, int(playable_44k * 48000 / SR * 1.004))
    out, rate = _ring_stretch_swap(near, 48000, playable_44k)
    assert rate == SA3_SAMPLE_RATE and out.shape == (2, playable_44k)
    # A full song swapped into a loop: today's truncate/tile path.
    far = torch.randn(2, 48000 * 30)
    out, rate = _ring_stretch_swap(far, 48000, playable_44k)
    assert out is far and rate == 48000


# ---- backend ---------------------------------------------------------------


class _Cond:
    def __init__(self, frames=T):
        self.cond_bundle = {
            "cross_attn_cond": torch.ones(1, 3, 4),
            "cross_attn_mask": torch.ones(1, 3),
            "cfg_scale": 1.0,
        }
        self.latent_frames = frames
        self.audio_sample_size = frames * DS


class _RecordingCodec:
    def __init__(self):
        self.latents = []

    def decode_full(self, latent_bct, *, decode_seed=None):
        self.latents.append(latent_bct.clone())
        return torch.zeros(2, latent_bct.shape[-1] * DS)


class _RecordingWindowCodec(_RecordingCodec):
    def decode_window(self, latent_bct, start, num):
        self.latents.append(latent_bct.clone())
        return torch.zeros(2, int(num))


def _sched_factory(steps):
    return lambda d: torch.linspace(float(d), 0.0, steps + 1)


def _ring_backend(codec, ring_frames=N):
    adapter = SA3Adapter(
        _PositionDit(), schedule_builder=_sched_factory(3),
        device="cpu", dtype=torch.float32, ring_frames=ring_frames,
    )
    return SA3Backend(
        adapter=adapter, codec=codec, cond=_Cond(),
        schedule_builder_factory=_sched_factory,
        knob_state=KnobState(sa3_knob_specs()),
        steps=3, depth=1, vae_window_s=0.1,
        playable_duration_s=(ring_frames or T) * DS / SR,
    )


@pytest.mark.parametrize("codec_cls", [_RecordingCodec, _RecordingWindowCodec])
def test_decode_side_periodic_tail(codec_cls):
    b = _ring_backend(codec_cls())
    latent = torch.randn(1, T, C)
    original = latent.clone()
    b._current_result = latent
    b.render_window(0.0)
    decoded = b.codec.latents[-1].movedim(1, 2)  # back to [1, T, C]
    assert torch.equal(decoded[:, :N], latent[:, :N])
    assert torch.equal(decoded[:, N:], latent[:, : T - N])
    # The pipeline's result tensor is never mutated.
    assert torch.equal(latent, original)


def test_decode_tail_untouched_without_ring():
    b = _ring_backend(_RecordingCodec(), ring_frames=None)
    latent = torch.randn(1, T, C)
    b._current_result = latent
    b.render_window(0.0)
    assert torch.equal(b.codec.latents[-1].movedim(1, 2), latent)


def test_playable_duration_is_the_ring():
    b = _ring_backend(_RecordingCodec())
    assert b.playable_duration_s() == N * DS / SR
    assert b._ring_frames_for(b.playable_duration_s(), T) == N
    assert b._loop_ring is True
    assert _ring_backend(_RecordingCodec(), ring_frames=None)._loop_ring is False


def test_from_context_builds_the_ring_adapter():
    class _Ctx:
        device = torch.device("cpu")
        dtype = torch.float32
        dit = None
        model_id = ""
        diffusion_objective = "rf_denoiser"
        sample_rate = SR
        downsampling_ratio = DS

        def cond_seconds_total(self, d):
            return float(d)

        def make_dit(self, **_):
            return _PositionDit()

        def make_schedule_builder(self, _cond, steps):
            return _sched_factory(steps)

        def make_codec(self, **_):
            return _RecordingCodec()

    kw = dict(prompt="p", duration_s=N * DS / SR,
              knob_state=KnobState(sa3_knob_specs()), cond=_Cond(),
              source_latent_bct=torch.zeros(1, C, T), steps=2, depth=1)
    on = SA3Backend.from_context(_Ctx(), ring_frames=N, **kw)
    assert on.adapter.ring_frames == N
    off = SA3Backend.from_context(_Ctx(), **kw)
    assert off.adapter.ring_frames is None
