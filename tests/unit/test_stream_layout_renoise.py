"""A stream layout's per-step re-noise hook and the extension status in the
params telemetry. Tier 1, no GPU.

* ``StreamLayout.make_renoise`` replaces the pingpong sampler's re-noise
  draw at every step it returns a tensor for, and the stock draw runs when
  it returns None or the layout lacks it (bit-identical to no hook).
* ``on_fresh_generation`` publishes the installed extension runtime's
  ``status()`` as ``params["extension"]``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from acestep.engine.sa3_adapter import SA3Adapter
from tests.unit.test_sa3_backend import _schedule_builder_factory
from tests.unit.test_sa3_stream_layout import (
    C, L, N_STREAMS, _LayoutRuntime, _StackDit, _knobs, _layout_backend,
    _run_until_fresh,
)


class _RenoiseRuntime(_LayoutRuntime):
    """Records every re-noise call; draws from its own generator, or
    returns None (stock draw) when ``passthrough`` is set."""

    def __init__(self, passthrough=False):
        self.renoise_calls: list = []
        self.passthrough = passthrough
        self._gen = torch.Generator().manual_seed(123)
        super().__init__({"make_renoise": self._renoise})

    def _renoise(self, request, step):
        self.renoise_calls.append(int(step))
        if self.passthrough:
            return None
        return torch.randn(1, N_STREAMS * L, C, generator=self._gen)


def _final_latent(runtime, seed=7):
    torch.manual_seed(0)
    b, _context, _ = _layout_backend(runtime=runtime)
    return _run_until_fresh(b, _knobs(b, sa3_denoise=0.7, seed=seed)), b


def test_stock_adapter_has_no_renoise_hook():
    a = SA3Adapter(_StackDit(), schedule_builder=_schedule_builder_factory(2),
                   device="cpu", dtype=torch.float32)
    assert getattr(a, "make_renoise", None) is None


def test_renoise_shape_is_checked():
    a = SA3Adapter(_StackDit(), schedule_builder=_schedule_builder_factory(2),
                   device="cpu", dtype=torch.float32,
                   make_renoise=lambda r, s: torch.ones(1, L, C))
    req = SimpleNamespace(latent_frames=N_STREAMS * L)
    with pytest.raises(ValueError):
        a.make_renoise(req, 0)
    none = SA3Adapter(_StackDit(), schedule_builder=_schedule_builder_factory(2),
                      device="cpu", dtype=torch.float32,
                      make_renoise=lambda r, s: None)
    assert none.make_renoise(req, 0) is None


def test_layout_renoise_replaces_the_draw_and_none_keeps_stock():
    stock, _ = _final_latent(_LayoutRuntime())
    passthrough_rt = _RenoiseRuntime(passthrough=True)
    passthrough, _ = _final_latent(passthrough_rt)
    assert passthrough_rt.renoise_calls        # consulted ...
    assert torch.equal(passthrough, stock)     # ... and None = stock, bit-exact
    hooked_rt = _RenoiseRuntime()
    hooked, _ = _final_latent(hooked_rt)
    assert hooked_rt.renoise_calls and set(hooked_rt.renoise_calls) <= {0}
    assert not torch.equal(hooked, stock)


def test_params_carry_the_extension_status():
    rt = _LayoutRuntime()
    rt.status = lambda: {"engine": "eager", "seeds": [1, 2, 3]}
    b, _context, _ = _layout_backend(runtime=rt)
    b.state = SimpleNamespace(params={}, prompt_text="p")
    b.on_fresh_generation({})
    assert b.state.params["extension"] == {"engine": "eager", "seeds": [1, 2, 3]}

    def boom():
        raise RuntimeError("x")

    rt.status = boom
    b.state.params.pop("extension")
    b.on_fresh_generation({})
    assert "extension" not in b.state.params
    assert np.isscalar(b.state.params["num_gens"])
