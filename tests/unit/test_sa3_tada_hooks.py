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


def test_capture_means_per_step_and_layer():
    sam = _sam()
    x = torch.randn(1, 5, H)
    ctx = torch.randn(1, 7, H)
    with sa3_tada.capture_cross_attn_means(sam, layers=[0, 2]) as store:
        _run(sam, x, ctx, steps=2)
    assert sorted(store) == [0, 1]
    assert sorted(store[0]) == [0, 2]
    want = (x * 0.5 + ctx.mean(dim=1, keepdim=True)).mean(dim=(0, 1))
    assert torch.allclose(store[0][0], want)


def test_patch_layer_context_equals_clean_at_that_layer_only():
    sam = _sam()
    x = torch.randn(1, 5, H)
    clean, corrupt = torch.randn(1, 7, H), torch.randn(1, 7, H)
    with sa3_tada.capture_context(sam) as box:
        ref_clean = _run(sam, x, clean, steps=1)[0]
    assert torch.equal(box[0], clean)
    ref_corrupt = _run(sam, x, corrupt, steps=1)[0]
    with sa3_tada.patch_context(sam, range(NB), box[0]):
        all_patched = _run(sam, x, corrupt, steps=1)[0]
    assert torch.allclose(all_patched, ref_clean)
    with sa3_tada.patch_context(sam, [1], box[0]):
        one = _run(sam, x, corrupt, steps=1)[0]
    assert not torch.allclose(one, ref_clean) and not torch.allclose(one, ref_corrupt)
    # Hooks are gone afterwards.
    assert torch.equal(_run(sam, x, corrupt, steps=1)[0], ref_corrupt)


def test_steer_adds_alpha_v_on_the_cross_attn_output_per_step():
    sam = _sam()
    x = torch.randn(1, 5, H)
    ctx = torch.randn(1, 7, H)
    v0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    base = _run(sam, x, ctx, steps=2)
    with sa3_tada.steer_cross_attn(sam, {0: {1: v0}, 1: {1: v1}}, alpha=0.0):
        zero = _run(sam, x, ctx, steps=2)
    assert all(torch.equal(a, b) for a, b in zip(zero, base))
    with sa3_tada.steer_cross_attn(sam, {0: {1: v0}, 1: {1: v1}}, alpha=2.0):
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
