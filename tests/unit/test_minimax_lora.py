"""Mapping a ComfyUI-layout MiniMax-Music3 LoRA onto our DiT.

Two things are being defended here, and both are failures that look like
success.

The first is the fused-``to_qkv`` split. The native checkpoint packs
attention as one ``[3*dim, dim]`` matrix where we keep three; splitting
it in the wrong order produces a model that runs, sounds like music, and
is wrong. The packing was verified empirically against
``Comfy-Org/MiniMax-Music-3`` (contiguous ``[q; k; v]`` matched to
~1.9e-3, head-interleaved to ~6.4e-1), and these tests pin that result
so it cannot drift.

The second is an adapter that does nothing. The community "8-step turbo"
LoRA for this model ships with every ``lora_up`` at zero, so it merges
cleanly and changes nothing at any strength. Loud beats silent.

CPU-only, no weights, no network.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from acestep.engine.minimax_lora import (
    NATIVE_PREFIX,
    MiniMaxLoRAError,
    apply_native_lora,
    load_native_lora,
)

DIM, FF, RANK, BLOCKS = 2048, 8192, 4, 2


def _lora(tmp_path, *, zero_up=False, blocks=BLOCKS, alpha=RANK, drop=None,
          extra=None):
    """A synthetic native-layout LoRA with known contents."""
    sd = {}
    shapes = {
        "self_attn.to_qkv": (3 * DIM, DIM),
        "self_attn.to_out": (DIM, DIM),
        "ff.ff.0.proj": (2 * FF, DIM),
        "ff.ff.2": (DIM, FF),
    }
    g = torch.Generator().manual_seed(0)
    for b in range(blocks):
        for target, (out_dim, in_dim) in shapes.items():
            if drop and (b, target) == drop[0]:
                sd[f"{NATIVE_PREFIX}{b}.{target}.{drop[1]}.weight"] = torch.zeros(1)
                continue
            up = torch.zeros(out_dim, RANK) if zero_up else torch.randn(
                out_dim, RANK, generator=g)
            sd[f"{NATIVE_PREFIX}{b}.{target}.lora_up.weight"] = up
            sd[f"{NATIVE_PREFIX}{b}.{target}.lora_down.weight"] = torch.randn(
                RANK, in_dim, generator=g)
            sd[f"{NATIVE_PREFIX}{b}.{target}.alpha"] = torch.tensor(float(alpha))
    if extra:
        sd.update(extra)
    p = tmp_path / "lora.safetensors"
    save_file(sd, str(p))
    return p


# ---- the mapping -----------------------------------------------------------


def test_maps_every_target_to_our_names(tmp_path):
    d = load_native_lora(_lora(tmp_path))
    assert len(d) == BLOCKS * 6           # qkv splits into three
    for b in range(BLOCKS):
        for leaf in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out",
                     "ff_in", "ff_out"):
            assert f"transformer_blocks.{b}.{leaf}.weight" in d


def test_shapes_match_the_model(tmp_path):
    d = load_native_lora(_lora(tmp_path))
    assert d["transformer_blocks.0.attn.to_q.weight"].shape == (DIM, DIM)
    assert d["transformer_blocks.0.attn.to_out.weight"].shape == (DIM, DIM)
    assert d["transformer_blocks.0.ff_in.weight"].shape == (2 * FF, DIM)
    assert d["transformer_blocks.0.ff_out.weight"].shape == (DIM, FF)


def test_qkv_splits_contiguously_not_interleaved(tmp_path):
    """Pins the empirically verified packing.

    Both hypotheses produce correctly shaped tensors, so shape checks
    cannot tell them apart — only the values can.
    """
    p = _lora(tmp_path)
    from safetensors.torch import load_file
    raw = load_file(str(p))
    up = raw[f"{NATIVE_PREFIX}0.self_attn.to_qkv.lora_up.weight"]
    down = raw[f"{NATIVE_PREFIX}0.self_attn.to_qkv.lora_down.weight"]
    full = (up @ down) * (RANK / RANK)

    d = load_native_lora(p)
    assert torch.allclose(d["transformer_blocks.0.attn.to_q.weight"], full[:DIM])
    assert torch.allclose(d["transformer_blocks.0.attn.to_k.weight"],
                          full[DIM:2 * DIM])
    assert torch.allclose(d["transformer_blocks.0.attn.to_v.weight"],
                          full[2 * DIM:])
    # And is genuinely distinguishable from the interleaved reading.
    il = full.view(32, 3, DIM // 32, DIM)[:, 0].reshape(DIM, DIM)
    assert not torch.allclose(d["transformer_blocks.0.attn.to_q.weight"], il)


# ---- scaling ---------------------------------------------------------------


def test_alpha_over_rank_scales_the_delta(tmp_path):
    a = load_native_lora(_lora(tmp_path, alpha=RANK))
    b = load_native_lora(_lora(tmp_path, alpha=2 * RANK))
    k = "transformer_blocks.0.attn.to_out.weight"
    assert torch.allclose(b[k], a[k] * 2.0, atol=1e-5)


def test_strength_scales_linearly(tmp_path):
    p = _lora(tmp_path)
    full = load_native_lora(p, strength=1.0)
    half = load_native_lora(p, strength=0.5)
    k = "transformer_blocks.1.ff_in.weight"
    assert torch.allclose(half[k], full[k] * 0.5, atol=1e-6)


# ---- failing loud ----------------------------------------------------------


def test_an_untrained_adapter_is_refused(tmp_path):
    """The real-world case. `guillaume127/MiniMax-Music-3-Turbo-FP8` and
    its byte-identical `modulsx` duplicate both ship with all 144
    `lora_up` tensors zero, so they merge perfectly and do nothing."""
    with pytest.raises(MiniMaxLoRAError, match="no-op"):
        load_native_lora(_lora(tmp_path, zero_up=True))


def test_a_null_control_can_be_asked_for_explicitly(tmp_path):
    d = load_native_lora(_lora(tmp_path, zero_up=True), allow_noop=True)
    assert d and all(not v.any() for v in d.values())


def test_unknown_key_layout_is_refused(tmp_path):
    p = _lora(tmp_path, extra={"some.other.format.lora_A": torch.zeros(1)})
    with pytest.raises(MiniMaxLoRAError, match="unrecognized"):
        load_native_lora(p)


def test_unknown_target_is_refused(tmp_path):
    p = _lora(tmp_path, extra={
        f"{NATIVE_PREFIX}0.self_attn.to_nowhere.lora_up.weight": torch.zeros(4, 4),
    })
    with pytest.raises(MiniMaxLoRAError, match="no counterpart"):
        load_native_lora(p)


def test_half_a_pair_is_refused(tmp_path):
    """A LoRA missing one side of one projection would otherwise apply to
    143 of 144 and be a differently-broken model, silently."""
    p = _lora(tmp_path, drop=((0, "ff.ff.2"), "lora_up"))
    with pytest.raises(MiniMaxLoRAError, match="missing"):
        load_native_lora(p)


# ---- merging ---------------------------------------------------------------


class _TinyBlockModel(torch.nn.Module):
    """Just enough parameter names to exercise the merge.

    Names are carried explicitly rather than reconstructed from mangled
    attribute names -- a fake that has to be clever about its own naming
    ends up testing the fake.
    """

    LEAVES = (
        ("attn.to_q", (DIM, DIM)), ("attn.to_k", (DIM, DIM)),
        ("attn.to_v", (DIM, DIM)), ("attn.to_out", (DIM, DIM)),
        ("ff_in", (2 * FF, DIM)), ("ff_out", (DIM, FF)),
    )

    def __init__(self):
        super().__init__()
        self._named = {}
        store = torch.nn.ParameterList()
        for b in range(BLOCKS):
            for leaf, shape in self.LEAVES:
                p = torch.nn.Parameter(torch.zeros(shape))
                store.append(p)
                self._named[f"transformer_blocks.{b}.{leaf}.weight"] = p
        self.store = store

    def named_parameters(self, *a, **k):
        return iter(self._named.items())


def test_merge_lands_on_every_parameter(tmp_path):
    m = _TinyBlockModel()
    n = apply_native_lora(m, _lora(tmp_path))
    assert n == BLOCKS * 6
    assert all(p.any() for p in m._named.values()), (
        "a merged LoRA left some parameter untouched"
    )


def test_merge_is_additive_and_strength_scaled(tmp_path):
    p = _lora(tmp_path)
    a, b = _TinyBlockModel(), _TinyBlockModel()
    apply_native_lora(a, p, strength=1.0)
    apply_native_lora(b, p, strength=0.5)
    k = "transformer_blocks.0.attn.to_v.weight"
    assert torch.allclose(b._named[k], a._named[k] * 0.5, atol=1e-6)


def test_merge_refuses_a_model_it_does_not_fit(tmp_path):
    with pytest.raises(MiniMaxLoRAError, match="does not have"):
        apply_native_lora(torch.nn.Linear(2, 2), _lora(tmp_path))
