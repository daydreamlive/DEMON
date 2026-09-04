"""`MiniMaxContext.make_dit` parks the eager DiT when the engine serves.

The context loads the eager bf16 DiT at construction. When a TensorRT
engine takes over rendering, those 4.9 GB must leave the card: with them
resident a streaming session sat at 32.1 GB of a 32.6 GB card and WDDM
paged both stages into the ground (see `docs/MINIMAX.md` §3). This pins
the placement seam without loading a checkpoint.
"""

import torch
import torch.nn as nn

from acestep.engine.minimax_context import MiniMaxContext


def _bare_context(device: str = "cpu") -> MiniMaxContext:
    ctx = object.__new__(MiniMaxContext)
    ctx.device = torch.device(device)
    ctx._dit = nn.Linear(4, 4)
    return ctx


def test_eager_backend_keeps_dit_on_context_device():
    ctx = _bare_context()
    dit = ctx.make_dit(latent_frames=689, backend="eager")
    assert dit is ctx._dit
    assert next(dit.parameters()).device.type == "cpu"


def test_engine_parks_eager_dit_on_host(monkeypatch):
    ctx = _bare_context()
    parked = []
    monkeypatch.setattr(
        ctx, "_place_eager_dit", lambda device: parked.append(device.type),
    )

    class _Engine:
        pass

    import acestep.engine.minimax_trt as trt

    monkeypatch.setattr(trt, "find_dit_engine", lambda frames: _Engine())
    dit = ctx.make_dit(latent_frames=689, backend="tensorrt")
    assert isinstance(dit, _Engine)
    assert parked == ["cpu"]


def test_no_engine_falls_back_to_resident_eager(monkeypatch):
    ctx = _bare_context()
    parked = []
    monkeypatch.setattr(
        ctx, "_place_eager_dit", lambda device: parked.append(device.type),
    )
    import acestep.engine.minimax_trt as trt

    monkeypatch.setattr(trt, "find_dit_engine", lambda frames: None)
    dit = ctx.make_dit(latent_frames=689, backend="tensorrt")
    assert dit is ctx._dit
    assert parked == ["cpu"]


def test_place_is_a_noop_when_already_there():
    ctx = _bare_context()
    ctx._place_eager_dit(torch.device("cpu"))
    assert next(ctx._dit.parameters()).device.type == "cpu"
