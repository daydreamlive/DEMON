"""YuE2Adapter through the SHARED StreamPipeline (Tier-2 seam).

A fake velocity ``v(x, s) = a*x + b*s + offset(bundle)`` stands in for
the NAR; everything else is the production pipeline. What is pinned:

* the ring (depth 1 and 4) reproduces upstream's explicit midpoint
  solver exactly (Euler on ``v_eff`` == the midpoint update);
* truncated schedules have ``ceil(steps*d)`` steps and constant ``dt``;
* rows on two bundles run as two grouped velocity calls;
* the final step lands on the midpoint update (the pipeline's x0 path);
* an off-grid timestep raises instead of integrating wrong;
* noise is upstream's frame-major per-seed draw.

CPU, float32, no weights.
"""

from __future__ import annotations

import math

import pytest
import torch

from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.stream import SlotRequest, StreamPipeline
from acestep.engine.yue2_adapter import YuE2Adapter, group_rows_by_bundle
from acestep.engine.yue2_velocity import (
    raw_time,
    truncated_grid,
    upstream_noise,
)

T = 12
STEPS = 32


class _Bundle:
    def __init__(self, offset: float):
        self.offset = offset


class _FakeVelocity:
    """``a*x + b*s + offset``; records ``(bundle id, rows)`` per call."""

    def __init__(self, a: float = -0.5, b: float = 0.25):
        self.a, self.b = a, b
        self.calls: list = []

    def __call__(self, bundle, state, raw_times):
        self.calls.append((id(bundle), state.shape[0]))
        s = torch.sigmoid(torch.tensor(raw_times, dtype=torch.float64))
        s = s.to(state.dtype).view(-1, 1, 1)
        return self.a * state + self.b * s + bundle.offset


def _reference_solve(velocity, bundle, x, schedule, steps=STEPS):
    """Upstream ``CachedNAR.solve`` with a pluggable velocity."""
    h = 1.0 / steps
    for s in schedule[:-1].tolist():
        first = velocity(bundle, x, [raw_time(s)] * x.shape[0])
        mid = x - first * (h / 2)
        second = velocity(bundle, mid, [raw_time(s - h / 2)] * x.shape[0])
        x = x - second * h
    return x


def _pipeline(velocity, depth=1, steps=STEPS):
    adapter = YuE2Adapter(velocity, steps=steps)
    config = DiffusionConfig(infer_steps=steps, dcw_enabled=False, noise_on_cpu=True)
    return StreamPipeline(None, config, pipeline_depth=depth, adapter=adapter, queue_cap=1)


def _run_until_result(pipe, make_request, max_ticks=200):
    for _ in range(max_ticks):
        pipe.submit(make_request())
        out = pipe.tick()
        if out is not None:
            return out, pipe.last_finished_request
    raise AssertionError("no result")


@pytest.mark.parametrize("depth", [1, 4])
def test_ring_matches_the_upstream_midpoint_solve(depth):
    vel = _FakeVelocity()
    bundle = _Bundle(0.1)
    pipe = _pipeline(vel, depth=depth)
    out, _ = _run_until_result(
        pipe, lambda: SlotRequest(seed=7, latent_frames=T, aux_cond=bundle),
    )
    noise = upstream_noise(7, T)[None]
    expected = _reference_solve(vel, bundle, noise, truncated_grid(STEPS, 1.0))
    assert torch.equal(out, expected)


def test_partial_denoise_renoises_the_source_and_matches_the_truncated_solve():
    vel = _FakeVelocity()
    bundle = _Bundle(-0.05)
    source = torch.linspace(-1, 1, T * 64).view(1, T, 64)
    pipe = _pipeline(vel)
    out, _ = _run_until_result(pipe, lambda: SlotRequest(
        seed=3, denoise=0.5, source_latents=source, latent_frames=T, aux_cond=bundle,
    ))
    schedule = truncated_grid(STEPS, 0.5)
    s0 = schedule[0].item()
    x = s0 * upstream_noise(3, T)[None] + (1.0 - s0) * source
    assert torch.equal(out, _reference_solve(vel, bundle, x, schedule))


@pytest.mark.parametrize("d", [1.0, 0.75, 0.5, 0.3, 0.25, 1 / 32, 0.01, 0.0])
def test_truncated_grid_has_ceil_steps_and_constant_dt(d):
    sched = truncated_grid(STEPS, d)
    assert sched[-1].item() == 0.0
    assert len(sched) - 1 == max(1, math.ceil(STEPS * d - 1e-9))
    dt = sched[:-1] - sched[1:]
    assert torch.equal(dt, torch.full_like(dt, 1.0 / STEPS))


def test_two_bundles_run_as_two_grouped_calls():
    vel = _FakeVelocity()
    a, b = _Bundle(0.0), _Bundle(1.0)
    pipe = _pipeline(vel, depth=2)
    pipe.submit(SlotRequest(seed=1, latent_frames=T, aux_cond=a))
    pipe.tick()
    vel.calls.clear()
    pipe.submit(SlotRequest(seed=1, latent_frames=T, aux_cond=b))
    pipe.tick()
    # Two slots on two bundles: one midpoint (2 calls) per bundle, batch 1.
    assert vel.calls == [(id(a), 1), (id(a), 1), (id(b), 1), (id(b), 1)]
    assert [len(rows) for _, rows in group_rows_by_bundle([a, b, a])] == [2, 1]


def test_final_step_lands_on_the_midpoint_update():
    vel = _FakeVelocity()
    bundle = _Bundle(0.2)
    source = torch.full((1, T, 64), 0.5)
    pipe = _pipeline(vel)
    out, _ = _run_until_result(pipe, lambda: SlotRequest(
        seed=5, denoise=1 / 32, source_latents=source, latent_frames=T, aux_cond=bundle,
    ))
    h = 1.0 / STEPS
    x = h * upstream_noise(5, T)[None] + (1.0 - h) * source
    first = vel(bundle, x, [raw_time(h)])
    second = vel(bundle, x - first * (h / 2), [raw_time(h / 2)])
    assert torch.equal(out, x - second * h)


def test_off_grid_timestep_raises():
    adapter = YuE2Adapter(_FakeVelocity(), steps=STEPS)
    x = torch.zeros(1, T, 64)
    with pytest.raises(ValueError, match="grid"):
        adapter.batched_forward(x, [0.33], [None], [None], [None], [_Bundle(0.0)])
    with pytest.raises(ValueError, match="aux_cond"):
        adapter.batched_forward(x, [0.5], [None], [None], [None], [None])


def test_step_count_mismatch_raises():
    adapter = YuE2Adapter(_FakeVelocity(), steps=STEPS)
    with pytest.raises(ValueError, match="rebuild"):
        adapter.build_schedule(DiffusionConfig(infer_steps=16), 1.0, "cpu", torch.float32)


def test_noise_is_the_upstream_frame_major_draw():
    adapter = YuE2Adapter(_FakeVelocity(), steps=STEPS)
    req = SlotRequest(seed=11, latent_frames=T, aux_cond=_Bundle(0.0))
    noise = adapter.make_noise(req)
    generator = torch.Generator(device="cpu").manual_seed(11)
    assert torch.equal(noise[0], torch.randn((T, 64), generator=generator))
    with pytest.raises(ValueError, match="int seed"):
        adapter.make_noise(SlotRequest(seed=None, latent_frames=T))


def test_latent_frames_is_required():
    adapter = YuE2Adapter(_FakeVelocity(), steps=STEPS)
    with pytest.raises(ValueError, match="latent_frames"):
        adapter.request_frames(SlotRequest(seed=1))
