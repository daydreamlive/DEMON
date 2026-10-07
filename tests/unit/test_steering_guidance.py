"""Steering-slot ``guidance`` option (CFG-free amplification of the steer).

guidance 1.0 must be bit-identical to the plain slot; guidance w runs the
positive pass twice (steering zeroed, steered) and returns
v0 + w * (v1 - v0). CPU only, fake adapters from test_steering_seam.
"""

from __future__ import annotations

import torch

from tests.unit.test_steering_seam import (
    _EagerAdapter,
    _EngineAdapter,
    _drain,
    _pipe,
    _vec,
)


def _cfg(guidance=None, alpha=1.5):
    c = {
        "layer": 1, "vector": _vec(4), "magnitude": 2.0, "alpha": alpha,
        "weights": (1.0, 1.0, 1.0),
    }
    if guidance is not None:
        c["guidance"] = guidance
    return c


def _run(adapter_cls, cfg):
    pipe = _pipe(adapter_cls())
    if cfg is not None:
        pipe.set_steering([cfg])
    return _drain(pipe, [3, 4], ticks=8)


def _check_guidance_one_identical(adapter_cls):
    base = _run(adapter_cls, _cfg())
    got = _run(adapter_cls, _cfg(guidance=1.0))
    assert len(base) == len(got) == 2
    for a, b in zip(base, got):
        assert torch.equal(a, b)


def test_guidance_one_bit_identical_eager():
    _check_guidance_one_identical(_EagerAdapter)


def test_guidance_one_bit_identical_engine_input():
    _check_guidance_one_identical(_EngineAdapter)


def _check_extrapolation(adapter_cls):
    """One step from identical inputs: v_g == v0 + g (v1 - v0)."""
    g = 3.0
    pipe = _pipe(adapter_cls(), steps=1)
    pipe.set_steering([_cfg(guidance=g)])
    assert pipe._steering_guidance == g
    # Reference velocities from one-step pipelines without guidance.
    v1 = _drain(_steered(adapter_cls, None), [9], ticks=2)[0]
    v0 = _drain(_pipe(adapter_cls(), steps=1), [9], ticks=2)[0]
    vg = _drain(pipe, [9], ticks=2)[0]
    # One ODE step from the same noise is affine in the velocity, so the
    # final latent obeys the same extrapolation.
    assert torch.allclose(vg, v0 + g * (v1 - v0), atol=1e-5)


def _steered(adapter_cls, guidance):
    p = _pipe(adapter_cls(), steps=1)
    p.set_steering([_cfg(guidance=guidance)])
    return p


def test_guidance_three_extrapolates_eager():
    _check_extrapolation(_EagerAdapter)


def test_guidance_three_extrapolates_engine_input():
    _check_extrapolation(_EngineAdapter)


def test_engine_called_twice_per_step_only_when_guided():
    ad = _EngineAdapter()
    pipe = _pipe(ad, steps=1)
    pipe.set_steering([_cfg(guidance=3.0)])
    _drain(pipe, [1], ticks=2)
    # Steered pass gets the tensor, plain pass gets None.
    assert len(ad.received) == 2
    assert ad.received[0] is not None and ad.received[1] is None
    ad2 = _EngineAdapter()
    p2 = _pipe(ad2, steps=1)
    p2.set_steering([_cfg(guidance=1.0)])
    _drain(p2, [1], ticks=2)
    assert len(ad2.received) == 1


def test_zero_alpha_ignores_guidance():
    base = _run(_EagerAdapter, None)
    got = _run(_EagerAdapter, _cfg(guidance=5.0, alpha=0.0))
    for a, b in zip(base, got):
        assert torch.equal(a, b)
