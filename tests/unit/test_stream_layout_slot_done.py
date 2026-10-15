"""A stream layout's finished-slot observer. Tier 1, no GPU.

* ``StreamLayout.on_slot_done`` is called exactly once for every slot
  whose denoise completes, with the slot's request and the very latent
  the tick delivers, on both the same-tick and the leftover path.
* Absent, delivery is unchanged (bit-identical to a layout without it).
* A raising observer is logged and ignored: the latent is still delivered.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from acestep.engine.sa3_adapter import SA3Adapter
from tests.unit.test_sa3_backend import _schedule_builder_factory
from tests.unit.test_sa3_stream_layout import (
    C, L, N_STREAMS, _LayoutRuntime, _StackDit, _knobs, _layout_backend,
    _run_until_fresh,
)


class _DoneRuntime(_LayoutRuntime):
    """Records every finished-slot call; raises when ``boom`` is set."""

    def __init__(self, boom=False):
        self.done_calls: list = []
        self.boom = boom
        super().__init__({"on_slot_done": self._done})

    def _done(self, request, latent):
        self.done_calls.append((request, latent))
        if self.boom:
            raise RuntimeError("observer failed")


def _record_ticks(monkeypatch, pipeline):
    """Wrap the pipeline class's tick to record (latent, request) per
    delivery, so the test survives the backend rebuilding its pipeline."""
    delivered: list = []
    cls = type(pipeline)
    orig = cls.tick

    def tick(self):
        out = orig(self)
        if out is not None:
            delivered.append((out, self.last_finished_request))
        return out

    monkeypatch.setattr(cls, "tick", tick)
    return delivered


def _run(monkeypatch, runtime, seed=7):
    torch.manual_seed(0)
    b, _context, _ = _layout_backend(runtime=runtime)
    delivered = _record_ticks(monkeypatch, b.pipeline)
    result = _run_until_fresh(b, _knobs(b, sa3_denoise=0.7, seed=seed))
    return b, delivered, result


def test_stock_adapter_has_no_slot_done_hook():
    a = SA3Adapter(_StackDit(), schedule_builder=_schedule_builder_factory(2),
                   device="cpu", dtype=torch.float32)
    assert getattr(a, "on_slot_done", None) is None
    fn = lambda r, x: None  # noqa: E731
    hooked = SA3Adapter(_StackDit(), schedule_builder=_schedule_builder_factory(2),
                        device="cpu", dtype=torch.float32, on_slot_done=fn)
    assert hooked.on_slot_done is fn


def test_called_once_per_finished_slot_with_the_delivered_latent(monkeypatch):
    rt = _DoneRuntime()
    _b, delivered, _ = _run(monkeypatch, rt)
    assert delivered
    assert len(rt.done_calls) == len(delivered)
    for (req, seen), (out, out_req) in zip(rt.done_calls, delivered):
        assert seen is out
        assert torch.equal(seen, out)
        assert req is out_req
        assert tuple(seen.shape) == (1, N_STREAMS * L, C)


def test_leftover_path_calls_once_with_the_delivered_latent():
    rt = _DoneRuntime()
    b, _context, _ = _layout_backend(runtime=rt)
    p = b.pipeline
    p._queue = []
    latent = torch.randn(1, N_STREAMS * L, C)
    req = SimpleNamespace(latent_frames=N_STREAMS * L, seed=3)
    slot = SimpleNamespace(step_idx=2, t_schedule=[1.0, 0.5, 0.0],
                           xt=latent, request=req)
    p._slots = [slot] + [None] * (len(p._slots) - 1)
    out = p.tick()
    assert out is latent
    assert len(rt.done_calls) == 1
    assert rt.done_calls[0][0] is req
    assert torch.equal(rt.done_calls[0][1], out)
    assert p.tick() is None
    assert len(rt.done_calls) == 1


def test_absent_hook_leaves_delivery_unchanged(monkeypatch):
    _b, stock_delivered, stock = _run(monkeypatch, _LayoutRuntime())
    monkeypatch.undo()
    rt = _DoneRuntime()
    _b2, hooked_delivered, hooked = _run(monkeypatch, rt)
    assert rt.done_calls
    assert len(stock_delivered) == len(hooked_delivered)
    for (a, _), (h, _) in zip(stock_delivered, hooked_delivered):
        assert torch.equal(a, h)


def test_raising_hook_does_not_break_delivery(monkeypatch):
    _b, stock_delivered, _ = _run(monkeypatch, _LayoutRuntime())
    monkeypatch.undo()
    rt = _DoneRuntime(boom=True)
    _b2, delivered, _ = _run(monkeypatch, rt)
    assert len(delivered) == len(stock_delivered)
    assert len(rt.done_calls) == len(delivered)
    for (a, _), (h, _) in zip(stock_delivered, delivered):
        assert torch.equal(a, h)
