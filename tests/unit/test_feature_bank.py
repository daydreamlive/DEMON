#!/usr/bin/env python3
"""Unit tests for the StreamV2V-style feature bank patch.

These tests run on CPU and don't load checkpoint weights -- they
construct a tiny ``AceStepAttention`` from a small synthetic config,
patch it via ``feature_bank``, and verify the read/write contract:

1. ``enable`` then forward with empty bank should produce output
   numerically equal to the un-patched forward (banked half is fully
   masked out, contributing zero to softmax).
2. After the first forward, the bank holds entries keyed by
   ``(layer_idx, step_idx)`` for every row in the batch.
3. A second forward with a populated bank should produce output that
   differs measurably from the un-patched forward (banked K/V now
   contributes to the attention sum).
4. ``write_enabled = False`` should leave the bank unchanged across
   a forward.
5. ``disable`` should restore the original forward behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture(scope="module")
def turbo_modules():
    """Return the vendored AceStepAttention / AceStepConfig + RoPE module."""
    from acestep.models.modeling_acestep_v15_turbo import AceStepAttention
    from acestep.models.configuration_acestep_v15 import AceStepConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    return AceStepAttention, AceStepConfig, Qwen3RotaryEmbedding


@pytest.fixture
def tiny_config(turbo_modules):
    _, AceStepConfig, _ = turbo_modules
    cfg = AceStepConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rope_theta=1000000,
        use_sliding_window=True,
        sliding_window=64,
        # Mirror the v15-turbo default pattern: full_attention on odd
        # indices, sliding_attention on even.
        layer_types=[
            "full_attention" if (i % 2 == 1) else "sliding_attention"
            for i in range(4)
        ],
    )
    cfg._attn_implementation = "sdpa"
    return cfg


@pytest.fixture
def attn_pair(turbo_modules, tiny_config):
    """Build a single full_attention layer's self_attn and a RoPE module."""
    AceStepAttention, _, Qwen3RotaryEmbedding = turbo_modules
    torch.manual_seed(0)
    attn = AceStepAttention(
        tiny_config, layer_idx=1, is_cross_attention=False,
    ).eval()
    rope = Qwen3RotaryEmbedding(tiny_config)
    return attn, rope


def _make_inputs(tiny_config, B=3, T=16, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(B, T, tiny_config.hidden_size)
    pos_ids = torch.arange(T).unsqueeze(0)
    return x, pos_ids


def _patch(attn, bank):
    """Apply the feature-bank patch to a single AceStepAttention module."""
    import types
    from acestep.engine.feature_bank import _patched_self_attn_forward

    attn._feature_bank = bank
    attn._unpatched_forward = attn.forward
    attn.forward = types.MethodType(_patched_self_attn_forward, attn)


def _unpatch(attn):
    if hasattr(attn, "_unpatched_forward"):
        attn.forward = attn._unpatched_forward
        del attn._unpatched_forward
        del attn._feature_bank


@torch.no_grad()
def test_empty_bank_matches_unpatched(attn_pair, tiny_config):
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    x, pos_ids = _make_inputs(tiny_config)
    pos_emb = rope(x, pos_ids)

    y_unpatched, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 1, 2]
    _patch(attn, bank)

    y_patched_empty, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )

    # With no bank entries, the banked half is fully masked. Output
    # must match the un-patched forward to within SDPA numeric noise.
    torch.testing.assert_close(
        y_patched_empty, y_unpatched, atol=1e-5, rtol=1e-4,
    )

    _unpatch(attn)


@torch.no_grad()
def test_first_forward_populates_bank(attn_pair, tiny_config):
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    x, pos_ids = _make_inputs(tiny_config)
    pos_emb = rope(x, pos_ids)

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 3, 5]
    _patch(attn, bank)

    assert bank.num_entries() == 0
    attn(x, attention_mask=None, position_embeddings=pos_emb)

    # Bank stores per (layer, step); writeback scatters batch rows into
    # the step axis at step_indices = [0, 3, 5].
    assert 1 in bank.layer_banks
    assert bank.is_step_valid(0)
    assert bank.is_step_valid(3)
    assert bank.is_step_valid(5)
    assert not bank.is_step_valid(1)
    assert bank.num_entries() == 3

    layer_bank = bank.layer_banks[1]  # [2, num_steps, kv_heads, T, head_dim]
    K0, V0 = layer_bank[0, 0], layer_bank[1, 0]
    assert K0.shape == (tiny_config.num_key_value_heads, x.shape[1], tiny_config.head_dim)
    assert V0.shape == K0.shape

    _unpatch(attn)


@torch.no_grad()
def test_bank_with_identical_input_is_a_noop(attn_pair, tiny_config):
    """When banked K/V equals current K/V, the math degenerates.

    Doubling identical K positions splits softmax weight evenly across
    each pair (w/2 + w/2 = w), and identical V at both positions sums
    back to the same output as the un-patched forward. So the right
    way to verify the bank is *active* is the cross-song scenario
    (different inputs across writes/reads); see
    ``test_cross_song_handoff_roundtrip``. This test pins the
    degenerate-input invariant so we notice if it ever drifts.
    """
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    x, pos_ids = _make_inputs(tiny_config)
    pos_emb = rope(x, pos_ids)

    y_unpatched, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 1, 2]
    _patch(attn, bank)

    attn(x, attention_mask=None, position_embeddings=pos_emb)
    y_with_bank, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )
    torch.testing.assert_close(y_with_bank, y_unpatched, atol=1e-5, rtol=1e-4)

    _unpatch(attn)


@torch.no_grad()
def test_write_disabled_leaves_bank_unchanged(attn_pair, tiny_config):
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    x, pos_ids = _make_inputs(tiny_config)
    pos_emb = rope(x, pos_ids)

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 1, 2]
    _patch(attn, bank)

    # Seed the bank with one forward.
    attn(x, attention_mask=None, position_embeddings=pos_emb)
    K0_before = bank.layer_banks[1][0, 0].clone()

    # Run again with writes disabled.
    bank.write_enabled = False
    x2 = x + 0.5
    attn(x2, attention_mask=None, position_embeddings=pos_emb)

    K0_after = bank.layer_banks[1][0, 0]
    torch.testing.assert_close(K0_after, K0_before)

    _unpatch(attn)


@torch.no_grad()
def test_disable_restores_original_forward(attn_pair, tiny_config):
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    x, pos_ids = _make_inputs(tiny_config)
    pos_emb = rope(x, pos_ids)

    y_baseline, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 1, 2]
    _patch(attn, bank)

    _unpatch(attn)
    assert not hasattr(attn, "_feature_bank")
    assert not hasattr(attn, "_unpatched_forward")

    y_after_unpatch, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )
    torch.testing.assert_close(y_after_unpatch, y_baseline)


@torch.no_grad()
def test_skips_non_banked_layers(turbo_modules, tiny_config):
    """Layer not in bank.banked must fall through to the original forward."""
    AceStepAttention, _, Qwen3RotaryEmbedding = turbo_modules
    from acestep.engine.feature_bank import FeatureBank

    torch.manual_seed(0)
    # layer_idx=0 is sliding_attention; we mark only layer 1 as banked.
    attn = AceStepAttention(
        tiny_config, layer_idx=0, is_cross_attention=False,
    ).eval()
    rope = Qwen3RotaryEmbedding(tiny_config)
    x, pos_ids = _make_inputs(tiny_config)
    pos_emb = rope(x, pos_ids)

    y_unpatched, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 1, 2]
    _patch(attn, bank)

    y_patched, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )

    # Layer 0 isn't in bank.banked -- patched forward must fall
    # through to the original. Output should match exactly (same
    # code path).
    torch.testing.assert_close(y_patched, y_unpatched)
    # And the bank must remain empty.
    assert bank.num_entries() == 0

    _unpatch(attn)


@torch.no_grad()
def test_step_indices_length_mismatch_raises(attn_pair, tiny_config):
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    x, pos_ids = _make_inputs(tiny_config, B=3)
    pos_emb = rope(x, pos_ids)

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 1]  # length 2, but B=3 -- mismatch.
    _patch(attn, bank)

    with pytest.raises(RuntimeError, match="step_indices length"):
        attn(x, attention_mask=None, position_embeddings=pos_emb)

    _unpatch(attn)


@torch.no_grad()
def test_cross_song_handoff_roundtrip(attn_pair, tiny_config):
    """Simulate the StreamPipeline handoff: song A writes, song B reads.

    Pretend we have two consecutive 'songs' going through the
    pipeline. Song A occupies one batch row at step 3 on tick T;
    song B occupies a different batch row at step 3 on tick T+1.
    Song B's forward should see song A's K/V via the bank entry at
    ``(layer_idx=1, step=3)``.
    """
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    pos_ids = torch.arange(16).unsqueeze(0)

    bank = FeatureBank(banked_layers=(1,))
    _patch(attn, bank)

    # Tick T: song A only, single row at step 3.
    torch.manual_seed(101)
    x_A = torch.randn(1, 16, tiny_config.hidden_size)
    pos_emb_A = rope(x_A, pos_ids)
    bank.step_indices = [3]
    attn(x_A, attention_mask=None, position_embeddings=pos_emb_A)
    assert 1 in bank.layer_banks
    assert bank.is_step_valid(3)
    KA = bank.layer_banks[1][0, 3].clone()

    # Tick T+1: song B (different inputs), also at step 3, in row 0.
    torch.manual_seed(202)
    x_B = torch.randn(1, 16, tiny_config.hidden_size)
    pos_emb_B = rope(x_B, pos_ids)
    bank.step_indices = [3]

    # Run with bank temporarily disabled to capture B's "no bank"
    # baseline.
    bank.enabled = False
    y_B_no_bank, _ = attn(
        x_B, attention_mask=None, position_embeddings=pos_emb_B,
    )
    bank.enabled = True

    # The 'no bank' run still wrote nothing because the patched
    # forward fell through to the un-patched path. KA should be
    # untouched.
    torch.testing.assert_close(bank.layer_banks[1][0, 3], KA)

    # Real banked run: song B reads KA, then overwrites with KB.
    y_B_with_bank, _ = attn(
        x_B, attention_mask=None, position_embeddings=pos_emb_B,
    )
    diff = (y_B_with_bank - y_B_no_bank).abs().max().item()
    assert diff > 1e-3, (
        f"Cross-song handoff produced no detectable change "
        f"(max-abs diff = {diff:.3e})."
    )

    # Bank now holds KB, not KA.
    KB = bank.layer_banks[1][0, 3]
    assert (KB - KA).abs().max().item() > 1e-3

    _unpatch(attn)
