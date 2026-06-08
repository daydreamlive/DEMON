"""SA3Adapter through the SHARED StreamPipeline (Tier-2 seam).

The ported spike tests (``test_sa3_stream_pipeline.py``) validate the
standalone ringbuffer; these validate the same SA3 semantics riding
the production :class:`~acestep.engine.stream.StreamPipeline` via
:class:`~acestep.engine.sa3_adapter.SA3Adapter`: native-layout
transpose at the adapter boundary, aux_cond bundle stacking with
cross-attn padding, SA3 schedule wiring, and the audio-to-audio
partial-denoise init that the stream-continuity design depends on.
CPU + mock DiT throughout.
"""

from __future__ import annotations

import pytest
import torch

from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.sa3_adapter import SA3Adapter
from acestep.engine.stream import SlotRequest, StreamPipeline

C = 256  # SA3 latent channels
T = 6    # latent frames


def _cond(length: int) -> dict:
    return {
        "cross_attn_cond": torch.ones(1, length, 4),
        "cross_attn_mask": torch.ones(1, length),
        "global_cond": torch.zeros(1, 2),
        "cfg_scale": 1.0,
        "batch_cfg": True,
        "rescale_cfg": True,
        "padding_mask": None,
        "apg_scale": 1.0,
    }


class _ZeroDit(torch.nn.Module):
    """Records call shapes; returns zero velocity (native [B,C,T])."""

    def __init__(self):
        super().__init__()
        self.calls: list = []

    def forward(self, x, t, **kwargs):
        self.calls.append({
            "batch": x.shape[0],
            "shape": tuple(x.shape),
            "cross_L": kwargs["cross_attn_cond"].shape[1],
            "mask": kwargs["cross_attn_mask"].clone(),
        })
        return torch.zeros_like(x)


class _VelocityDit(torch.nn.Module):
    def forward(self, x, t, **kwargs):
        return 0.125 * x + t.view(-1, 1, 1)


def _schedule_builder(steps: int):
    def _build(denoise: float) -> torch.Tensor:
        # SA3's build_schedule pins t[0] = sigma_max; a linear ramp is
        # shape-faithful enough for the seam mechanics under test.
        return torch.linspace(float(denoise), 0.0, steps + 1)
    return _build


def _pipeline(dit, *, steps=3, depth=2, method="ode"):
    adapter = SA3Adapter(
        dit,
        schedule_builder=_schedule_builder(steps),
        device="cpu",
        dtype=torch.float32,
    )
    config = DiffusionConfig(
        infer_steps=steps,
        infer_method=method,
        noise_on_cpu=True,
        dcw_enabled=False,
    )
    return StreamPipeline(None, config, pipeline_depth=depth, adapter=adapter)


def _request(seed: int, *, L=3, denoise=1.0, source=None) -> SlotRequest:
    return SlotRequest(
        seed=seed,
        denoise=denoise,
        source_latents=source,
        aux_cond=_cond(L),
        latent_frames=T,
    )


def _drain(pipe, requests, ticks):
    out = []
    queue = list(requests)
    for _ in range(ticks):
        if queue:
            pipe.submit(queue.pop(0))
        fin = pipe.tick()
        if fin is not None:
            out.append(fin)
    return out


def _cpu_noise(seed: int) -> torch.Tensor:
    """The shared pipeline's noise convention (noise_on_cpu): seeded
    [1, D, T] then transposed to engine layout [1, T, D]."""
    torch.manual_seed(seed)
    return torch.randn(1, C, T).movedim(-1, -2)


def test_shared_pipeline_emits_engine_layout_sa3_latents():
    dit = _ZeroDit()
    pipe = _pipeline(dit, steps=3, depth=2)

    emitted = _drain(pipe, [_request(s) for s in (11, 22, 33, 44)], ticks=12)

    assert len(emitted) == 4
    assert all(t.shape == (1, T, C) for t in emitted)
    # Slots batched into one forward, native layout at the DiT.
    assert any(c["batch"] == 2 for c in dit.calls)
    assert all(c["shape"][1] == C and c["shape"][2] == T for c in dit.calls)


def test_mixed_cross_attn_lengths_pad_to_batch_max():
    dit = _ZeroDit()
    pipe = _pipeline(dit, steps=3, depth=2)

    _drain(pipe, [_request(1, L=3), _request(2, L=5)], ticks=8)

    batched = [c for c in dit.calls if c["batch"] == 2]
    assert batched, "no batched forward observed"
    for c in batched:
        assert c["cross_L"] == 5
        # The shorter bundle's mask is zero-padded, keeping pad inert.
        assert torch.all(c["mask"][0, 3:] == 0) or torch.all(c["mask"][1, 3:] == 0)


def test_partial_denoise_anchors_to_source():
    # The audio-to-audio continuity math: with a zero-velocity model,
    # ODE leaves the init mix untouched, so the finished latent must be
    # exactly sigma*noise + (1-sigma)*source in the shared pipeline's
    # own noise convention.
    dit = _ZeroDit()
    pipe = _pipeline(dit, steps=2, depth=1)
    source = torch.full((1, T, C), 2.0)
    denoise = 0.6

    emitted = _drain(
        pipe, [_request(123, denoise=denoise, source=source)], ticks=6,
    )

    expected = denoise * _cpu_noise(123) + (1.0 - denoise) * source
    assert len(emitted) == 1
    # atol covers the t_start scalar precision: the pipeline mixes with
    # schedule[0].item() (float32-rounded 0.6), the expectation with
    # python 0.6 — a 2e-7 absolute effect on near-zero elements.
    assert torch.allclose(emitted[0], expected, atol=1e-6)


def test_velocity_dit_matches_hand_rolled_euler():
    pipe = _pipeline(_VelocityDit(), steps=3, depth=1)
    emitted = _drain(pipe, [_request(77)], ticks=6)
    assert len(emitted) == 1

    # Hand-rolled Euler in engine layout with the same mock velocity.
    schedule = _schedule_builder(3)(1.0)
    xt = _cpu_noise(77).clone()
    for i in range(3):
        t_curr, t_next = schedule[i].item(), schedule[i + 1].item()
        vt = 0.125 * xt + t_curr
        xt = xt + (t_next - t_curr) * vt
    assert torch.allclose(emitted[0], xt, atol=1e-6)


def test_sde_pingpong_emits():
    pipe = _pipeline(_ZeroDit(), steps=3, depth=2, method="sde")
    emitted = _drain(pipe, [_request(s) for s in (5, 6, 7)], ticks=10)
    assert len(emitted) == 3
    assert all(t.shape == (1, T, C) for t in emitted)


def test_schedule_length_mismatch_fails_loudly():
    adapter = SA3Adapter(
        _ZeroDit(),
        schedule_builder=_schedule_builder(5),  # wrong: pipeline wants 3
        device="cpu",
        dtype=torch.float32,
    )
    config = DiffusionConfig(
        infer_steps=3, infer_method="ode", noise_on_cpu=True, dcw_enabled=False,
    )
    pipe = StreamPipeline(None, config, pipeline_depth=1, adapter=adapter)
    pipe.submit(_request(9))
    with pytest.raises(ValueError, match="schedule length mismatch"):
        pipe.tick()


def test_missing_aux_cond_fails_loudly():
    pipe = _pipeline(_ZeroDit(), steps=2, depth=1)
    pipe.submit(SlotRequest(seed=1, latent_frames=T))  # no aux_cond
    with pytest.raises(ValueError, match="aux_cond"):
        pipe.tick()


def _adapter(steps=4):
    return SA3Adapter(
        _ZeroDit(),
        schedule_builder=_schedule_builder(steps),
        device="cpu",
        dtype=torch.float32,
    )


def test_shift_alpha_warps_schedule_and_pins_endpoints():
    adapter = _adapter(steps=4)
    config = DiffusionConfig(
        infer_steps=4, infer_method="ode", noise_on_cpu=True, dcw_enabled=False,
    )
    base = adapter.build_schedule(config, 0.8, "cpu", torch.float32)

    adapter.shift_alpha = 2.0
    warped = adapter.build_schedule(config, 0.8, "cpu", torch.float32)

    # t[0] re-pinned to sigma_max exactly (slot init mixes source/noise
    # by it — upstream build_schedule pins the same way post-shift);
    # t[-1]=0 is a fixed point of the Flux map.
    assert torch.equal(warped[0], base[0])
    assert warped[-1].item() == 0.0
    # Interior follows the Flux alpha map, pushed toward noise (a>1).
    expect = 2.0 * base[1:-1] / (1.0 + 1.0 * base[1:-1])
    assert torch.allclose(warped[1:-1], expect, atol=1e-6)
    assert torch.all(warped[1:-1] > base[1:-1])
    # Still a strictly decreasing schedule.
    assert torch.all(warped[:-1] > warped[1:])

    # alpha=1 is exactly the stock schedule (no warp branch entered).
    adapter.shift_alpha = 1.0
    assert torch.equal(
        adapter.build_schedule(config, 0.8, "cpu", torch.float32), base,
    )


def test_shift_alpha_below_one_pulls_toward_refinement():
    adapter = _adapter(steps=4)
    config = DiffusionConfig(
        infer_steps=4, infer_method="ode", noise_on_cpu=True, dcw_enabled=False,
    )
    base = adapter.build_schedule(config, 1.0, "cpu", torch.float32)
    adapter.shift_alpha = 0.5
    warped = adapter.build_schedule(config, 1.0, "cpu", torch.float32)
    assert torch.all(warped[1:-1] < base[1:-1])
    assert torch.all(warped[:-1] > warped[1:])


def test_shift_alpha_invalid_fails_loudly():
    adapter = _adapter(steps=4)
    config = DiffusionConfig(
        infer_steps=4, infer_method="ode", noise_on_cpu=True, dcw_enabled=False,
    )
    adapter.shift_alpha = 0.0
    with pytest.raises(ValueError, match="shift_alpha"):
        adapter.build_schedule(config, 1.0, "cpu", torch.float32)
