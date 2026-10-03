"""SA3 TADA offline hooks (acestep.engine.sa3_tada) on a tiny fake trunk.

The fake mirrors the vendored layout the accessors walk
(``sam.model.model.model.transformer.layers``) and the call shape of the
vendored block (``self.cross_attn(norm(x), context=context)``), so the
capture, K/V patch and steering hooks run exactly as on SA3.
"""

from __future__ import annotations

import torch

from acestep.engine import sa3_tada

H = 4
NB = 3


class _XAttn(torch.nn.Module):
    """Output depends on the context (stands in for K = cW_K, V = cW_V)."""

    def forward(self, x, context=None):
        return x * 0.5 + context.mean(dim=1, keepdim=True)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = _XAttn()

    def forward(self, x, context=None):
        x = x + self.cross_attn(x, context=context)
        return x + torch.tanh(x)


class _NS(torch.nn.Module):
    pass


def _sam():
    sam = _NS()
    sam.model = _NS()
    sam.model.model = _NS()
    sam.model.model.model = _NS()
    sam.model.model.model.transformer = _NS()
    sam.model.model.model.transformer.layers = torch.nn.ModuleList(
        [_Block() for _ in range(NB)]
    )
    return sam


def _run(sam, x, ctx, steps=2):
    outs = []
    for _ in range(steps):
        h = x
        for blk in sam.model.model.model.transformer.layers:
            h = blk(h, context=ctx)
        outs.append(h)
    return outs


def test_target_records_cross_attn_means_per_step_and_layer():
    from acestep.tada import ActivationRecorder, forward_counter

    sam = _sam()
    target = sa3_tada.sa3_target(sam)
    assert target.num_blocks == NB
    x = torch.randn(1, 5, H)
    ctx = torch.randn(1, 7, H)
    with ActivationRecorder(target, blocks=[0, 2], context=forward_counter(1)) as rec:
        _run(sam, x, ctx, steps=2)
    store = rec.steps()
    assert sorted(store) == [0, 1] and sorted(store[0]) == [0, 2]
    want = (x * 0.5 + ctx.mean(dim=1, keepdim=True)).mean(dim=(0, 1))
    assert torch.allclose(store[0][0][0], want)


def test_target_patch_is_the_clean_context_at_that_layer_only():
    from acestep.tada.patching import run_patched

    sam = _sam()
    target = sa3_tada.sa3_target(sam)
    x = torch.randn(1, 5, H)
    clean, corrupt = torch.randn(1, 7, H), torch.randn(1, 7, H)
    ref_clean = _run(sam, x, clean, steps=1)[0]
    ref_corrupt = _run(sam, x, corrupt, steps=1)[0]
    _, all_p = run_patched(target, range(NB), lambda: _run(sam, x, clean, 1),
                           lambda: _run(sam, x, corrupt, 1))
    assert torch.allclose(all_p[0], ref_clean)
    _, one = run_patched(target, [1], lambda: _run(sam, x, clean, 1),
                         lambda: _run(sam, x, corrupt, 1))
    assert not torch.allclose(one[0], ref_clean) and not torch.allclose(one[0], ref_corrupt)
    assert torch.equal(_run(sam, x, corrupt, steps=1)[0], ref_corrupt)


def test_steer_offline_adds_alpha_v_per_step():
    sam = _sam()
    x = torch.randn(1, 5, H)
    ctx = torch.randn(1, 7, H)
    v0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    base = _run(sam, x, ctx, steps=2)
    with sa3_tada.steer_offline(sam, {0: {1: v0}, 1: {1: v1}}, alpha=0.0):
        zero = _run(sam, x, ctx, steps=2)
    assert all(torch.equal(a, b) for a, b in zip(zero, base))
    with sa3_tada.steer_offline(sam, {0: {1: v0}, 1: {1: v1}}, alpha=2.0):
        got = _run(sam, x, ctx, steps=2)
    for step, v in ((0, v0), (1, v1)):
        h = x
        for b, blk in enumerate(sam.model.model.model.transformer.layers):
            xa = h * 0.5 + ctx.mean(dim=1, keepdim=True)
            if b == 1:
                xa = xa + 2.0 * v
            h = h + xa
            h = h + torch.tanh(h)
        assert torch.allclose(got[step], h, atol=1e-6)
