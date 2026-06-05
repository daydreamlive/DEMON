from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SA3_DIR = _REPO_ROOT / "scripts" / "sa3"
# Vendored SA3 source: the canonical location is the (untracked)
# notes/SA3/stable-audio-3 tree; DEMON_SA3_SRC overrides for worktrees
# that don't carry it. The vendor-parity tests below skip when the
# package is unavailable; the ringbuffer tests run anywhere.
_SA3_SRC = Path(
    os.environ.get(
        "DEMON_SA3_SRC", _REPO_ROOT / "notes" / "SA3" / "stable-audio-3",
    )
)
if str(_SA3_SRC) not in sys.path:
    sys.path.insert(0, str(_SA3_SRC))
if str(_SA3_DIR) not in sys.path:
    sys.path.insert(0, str(_SA3_DIR))

import pytest

_requires_vendor = pytest.mark.skipif(
    importlib.util.find_spec("stable_audio_3") is None,
    reason="vendored stable_audio_3 source not available "
           "(set DEMON_SA3_SRC or vendor notes/SA3/stable-audio-3)",
)

from sa3_stream_pipeline import SA3Request, SA3StreamPipeline, stack_sa3_cond_bundles


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
    def __init__(self):
        super().__init__()
        self.batch_sizes: list[int] = []
        self.cross_attn_lengths: list[int] = []

    def forward(self, x, t, **kwargs):
        self.batch_sizes.append(x.shape[0])
        self.cross_attn_lengths.append(kwargs["cross_attn_cond"].shape[1])
        return torch.zeros_like(x)


class _VelocityDit(torch.nn.Module):
    def forward(self, x, t, **kwargs):
        return 0.125 * x + t.view(-1, 1, 1)


def _pipeline(dit, *, depth=2, steps=3, schedule=None, sampler="ode"):
    schedule = schedule if schedule is not None else torch.linspace(1.0, 0.0, steps + 1)
    return SA3StreamPipeline(
        dit,
        schedule_builder=lambda _denoise: schedule,
        depth=depth,
        steps=steps,
        device="cpu",
        dtype=torch.float32,
        sampler=sampler,
    )


def test_stack_sa3_cond_bundles_pads_cross_attention():
    out = stack_sa3_cond_bundles([_cond(3), _cond(5)])

    assert out["cross_attn_cond"].shape == (2, 5, 4)
    assert out["cross_attn_mask"].shape == (2, 5)
    assert torch.all(out["cross_attn_mask"][0, 3:] == 0)
    assert out["cfg_scale"] == 1.0


def test_ringbuffer_batches_active_slots_and_emits_after_warmup():
    dit = _ZeroDit()
    pipe = _pipeline(dit, depth=2, steps=3)

    emitted = []
    for tick in range(8):
        if tick < 4:
            pipe.submit(SA3Request(cond_bundle=_cond(3), latent_frames=6, seed=tick))
        out = pipe.tick()
        if out is not None:
            emitted.append(out)

    assert len(emitted) == 4
    assert all(t.shape == (1, 256, 6) for t in emitted)
    assert 2 in dit.batch_sizes
    assert pipe.active_slots == 0


def test_partial_denoise_initializes_from_source_and_seeded_noise():
    dit = _ZeroDit()
    schedule = torch.tensor([0.25, 0.0])
    pipe = _pipeline(dit, depth=1, steps=1, schedule=schedule)
    source = torch.full((1, 256, 4), 2.0)

    req = SA3Request(
        cond_bundle=_cond(2),
        latent_frames=4,
        source_latents=source,
        seed=123,
        denoise=0.25,
    )

    torch.manual_seed(123)
    expected_noise = torch.randn(1, 256, 4)
    expected = 0.25 * expected_noise + 0.75 * source

    out = pipe.drain_one(req)

    assert torch.allclose(out, expected)


def test_new_length_drops_stale_slots_before_batching():
    dit = _ZeroDit()
    pipe = _pipeline(dit, depth=2, steps=2)

    pipe.submit(SA3Request(cond_bundle=_cond(2), latent_frames=4, seed=1))
    pipe.tick()
    assert pipe.active_slots == 1

    pipe.submit(SA3Request(cond_bundle=_cond(2), latent_frames=6, seed=2))
    pipe.tick()

    assert pipe.active_slots == 1
    assert pipe._slots[0] is not None
    assert pipe._slots[0].request.latent_frames == 6


@_requires_vendor
def test_depth1_pingpong_matches_vendor_sampler_from_same_seed():
    import stable_audio_3.inference.sampling as sampling

    seed = 1528
    steps = 3
    schedule = torch.tensor([1.0, 0.6, 0.25, 0.0])
    dit = _VelocityDit()

    torch.manual_seed(seed)
    noise = torch.randn(1, 256, 5)
    expected = sampling.sample_flow_pingpong(
        dit,
        noise.clone(),
        sigmas=schedule,
        disable_tqdm=True,
        **_cond(4),
    )

    pipe = _pipeline(dit, depth=1, steps=steps, schedule=schedule, sampler="pingpong")
    out = pipe.drain_one(SA3Request(cond_bundle=_cond(4), latent_frames=5, seed=seed))

    assert torch.allclose(out, expected)


@_requires_vendor
def test_depth1_ode_source_denoise_matches_vendor_euler_init_mix():
    import stable_audio_3.inference.sampling as sampling

    seed = 1528
    denoise = 0.6
    steps = 3
    schedule = torch.tensor([0.6, 0.35, 0.1, 0.0])
    source = torch.linspace(-1.0, 1.0, 1 * 256 * 5).reshape(1, 256, 5)
    dit = _VelocityDit()

    torch.manual_seed(seed)
    noise = torch.randn(1, 256, 5)
    xt = source * (1.0 - denoise) + noise * denoise
    expected = sampling.sample_discrete_euler(
        dit,
        xt.clone(),
        sigmas=schedule,
        disable_tqdm=True,
        **_cond(4),
    )

    pipe = _pipeline(dit, depth=1, steps=steps, schedule=schedule, sampler="ode")
    out = pipe.drain_one(
        SA3Request(
            cond_bundle=_cond(4),
            latent_frames=5,
            source_latents=source,
            seed=seed,
            denoise=denoise,
        )
    )

    assert torch.allclose(out, expected)


@_requires_vendor
def test_depth2_pingpong_matches_independent_vendor_runs_per_seed():
    import stable_audio_3.inference.sampling as sampling

    seeds = [1528, 1529]
    steps = 3
    schedule = torch.tensor([1.0, 0.6, 0.25, 0.0])
    dit = _VelocityDit()

    expected = []
    for seed in seeds:
        torch.manual_seed(seed)
        noise = torch.randn(1, 256, 5)
        expected.append(
            sampling.sample_flow_pingpong(
                dit,
                noise.clone(),
                sigmas=schedule,
                disable_tqdm=True,
                **_cond(4),
            )
        )

    pipe = _pipeline(dit, depth=2, steps=steps, schedule=schedule, sampler="pingpong")
    for seed in seeds:
        pipe.submit(SA3Request(cond_bundle=_cond(4), latent_frames=5, seed=seed))

    emitted = []
    for _ in range(steps + 4):
        out = pipe.tick()
        if out is not None:
            emitted.append(out)

    assert len(emitted) == len(expected)
    for out, ref in zip(emitted, expected):
        assert torch.allclose(out, ref)
