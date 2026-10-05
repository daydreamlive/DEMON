"""The graphed AR session's pieces that do not need the model.

The decode attention is the part that carries numerics: it replaces
HF's attention function for the LM's single-token step over a static
cache. It is checked here against the reference formulation (expand the
KV heads, causal mask over the filled slots, `scaled_dot_product_attention`)
on random tensors, for the single-token decode shape and for a
multi-token prefill shape, with stale garbage in the slots past the
current position that the mask must hide.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from acestep.engine.minimax_ar_graph import (
    DECODE_ATTENTION,
    POS_KWARG,
    decode_attention,
)


def _reference(query, key, value, pos, q_len):
    """`repeat_kv` + causal-over-filled-slots mask + sdpa."""
    batch, q_heads, _, dim = query.shape
    kv_heads, slots = key.shape[1], key.shape[2]
    group = q_heads // kv_heads
    key = key[:, :, None].expand(batch, kv_heads, group, slots, dim).reshape(batch, q_heads, slots, dim)
    value = value[:, :, None].expand(batch, kv_heads, group, slots, dim).reshape(batch, q_heads, slots, dim)
    q_pos = pos + torch.arange(q_len)
    mask = torch.arange(slots)[None, :] <= q_pos[:, None]
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask[None, None])
    return out.transpose(1, 2).contiguous()


@pytest.mark.parametrize("q_len,pos", [(1, 0), (1, 37), (1, 63), (5, 20), (8, 0)])
def test_decode_attention_matches_reference(q_len, pos):
    torch.manual_seed(0)
    batch, q_heads, kv_heads, slots, dim = 2, 8, 2, 64, 16
    query = torch.randn(batch, q_heads, q_len, dim)
    key = torch.randn(batch, kv_heads, slots, dim)
    value = torch.randn(batch, kv_heads, slots, dim)
    # Slots the mask must hide carry garbage of a scale that would
    # swamp the answer if it leaked.
    key[:, :, pos + q_len:] = 100.0
    value[:, :, pos + q_len:] = -100.0

    got, weights = decode_attention(
        None, query, key, value, None, scaling=dim**-0.5,
        **{POS_KWARG: torch.tensor([pos])},
    )
    want = _reference(query, key, value, pos, q_len)
    assert weights is None
    assert got.shape == (batch, q_len, q_heads, dim)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


def test_decode_attention_head_grouping_is_repeat_kv_order():
    """q head h must read kv head h // group, which is `repeat_kv`'s
    layout. Give each kv head a distinct value and check the routing."""
    batch, q_heads, kv_heads, slots, dim = 1, 4, 2, 4, 2
    query = torch.zeros(batch, q_heads, 1, dim)
    key = torch.zeros(batch, kv_heads, slots, dim)
    value = torch.zeros(batch, kv_heads, slots, dim)
    value[0, 0] = 1.0
    value[0, 1] = 2.0
    out, _ = decode_attention(
        None, query, key, value, None, **{POS_KWARG: torch.tensor([slots - 1])},
    )
    assert torch.equal(out[0, 0, :, 0], torch.tensor([1.0, 1.0, 2.0, 2.0]))


def test_decode_attention_default_scaling_is_inverse_sqrt_dim():
    torch.manual_seed(1)
    query = torch.randn(1, 2, 1, 16)
    key = torch.randn(1, 1, 8, 16)
    value = torch.randn(1, 1, 8, 16)
    kw = {POS_KWARG: torch.tensor([7])}
    a, _ = decode_attention(None, query, key, value, None, **kw)
    b, _ = decode_attention(None, query, key, value, None, scaling=1 / math.sqrt(16), **kw)
    assert torch.allclose(a, b)


def test_attention_is_registered_with_transformers():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    assert ALL_ATTENTION_FUNCTIONS[DECODE_ATTENTION] is decode_attention
    # No HF-built mask: the function builds its own from the position.
    assert ALL_MASK_ATTENTION_FUNCTIONS[DECODE_ATTENTION](
        batch_size=1, cache_position=torch.zeros(1, dtype=torch.long),
        kv_length=4, kv_offset=0,
    ) is None


def test_graphed_stream_refuses_cpu_and_cpu_sampling():
    from acestep.engine.minimax_ar_graph import GraphedARStream

    class _AR:
        device = torch.device("cpu")
        sample_on_cpu = False

    with pytest.raises(ValueError, match="CUDA"):
        GraphedARStream(_AR(), prompt="x", lyrics="y")

    if torch.cuda.is_available():
        class _CudaAR(_AR):
            device = torch.device("cuda", 0)
            sample_on_cpu = True

        with pytest.raises(ValueError, match="sample_on_cpu"):
            GraphedARStream(_CudaAR(), prompt="x", lyrics="y")
