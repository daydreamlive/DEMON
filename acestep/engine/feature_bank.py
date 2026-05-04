"""Feature-bank port of StreamV2V's K/V re-injection for ACE-Step.

V1 scope: bank only the 12 ``full_attention`` layers (odd indices in
the default v1.5-turbo ``layer_types``). At every tick, each active
slot's self-attention reads the (K, V) cached at its current denoise
step from the prior tick's slot-at-the-same-step, concatenates along
the seq dim, and runs SDPA. After the forward, the slot writes its
own (K, V) back to the bank for the next tick to consume.

Per-layer storage: a single tensor of shape
``[2, num_steps, num_kv_heads, T, head_dim]``. Dim 0 = K vs V, dim 1
= denoise step slot, the rest are the standard attention shape. This
layout matches the daydreamlive/StreamDiffusion ``kvo_cache`` design
and keeps the patched attention free of Python control flow on bank
state -- gather/scatter via ``index_select`` / ``index_copy_`` so the
function can be torch.compile'd or ONNX-exported.

Hard requirements for V1:

- Single condition per slot, no CFG. Multi-cond / CFG produces
  multiple rows per slot at the same step_idx; ``index_copy_`` handles
  duplicate indices (last write wins) but bank semantics get muddier.
  V1 mitigates by only writing on the positive pass.
- PyTorch decoder path only (eager or compiled). The TRT decoder path
  needs the bank exposed as engine I/O; that lives in a separate
  re-export pass.
- Decoder must NOT be ``torch.compile``'d at install time.
  ``enable_feature_bank_on_decoder`` refuses ``OptimizedModule``; the
  caller can compile *after* the patch is installed.

Open empirical questions (require listening to answer): whether the
banked K/V transfers musical identity at all under ACE-Step's
RoPE-positioned attention; whether 12-layer coverage (skipping all
SWA layers) is enough; whether the model's softmax actually picks
banked tokens or always favors current K at the same RoPE position.
"""

from __future__ import annotations

import math
import types
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from loguru import logger

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


DEFAULT_BANKED_LAYERS: Tuple[int, ...] = tuple(range(1, 24, 2))


class FeatureBank:
    """Holds per-layer K/V tensors plus shared step-indexing metadata.

    Storage:
      - ``layer_banks[layer_idx]``: tensor ``[2, num_steps, kv_heads, T,
        head_dim]``. Allocated lazily on first forward through that
        layer (so callers don't need to know K/V dimensions upfront).
      - ``valid_steps``: ``[num_steps]`` bool. ``valid_steps[k]`` is
        True once *any* layer has written to step ``k``. Shared
        because every banked layer goes through the same step on the
        same row, so a step is "valid" for all layers simultaneously.
      - ``step_indices``: ``[B]`` long, set per-tick by
        ``StreamPipeline._tick_pt`` before each decoder forward. Tells
        the patched attention which row maps to which step slot.

    Hot-updatable knobs:
      - ``strength``: softmax-mass scalar on banked tokens. 1.0 =
        equal weighting, 0.0 = bank fully masked out.
      - ``write_enabled``: if False, reads pass through but no
        scatter happens. Used to gate the CFG negative pass off.
      - ``enabled``: if False, the patched forward falls through to
        the un-patched attention.
    """

    def __init__(
        self,
        banked_layers: Sequence[int] = DEFAULT_BANKED_LAYERS,
        num_steps: int = 8,
        strength: float = 1.0,
    ):
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        self.banked: frozenset[int] = frozenset(banked_layers)
        self.num_steps: int = int(num_steps)
        self.strength: float = float(strength)
        self.enabled: bool = True
        self.write_enabled: bool = True

        # Per-layer K/V tensors, lazy-allocated on first forward.
        self.layer_banks: Dict[int, torch.Tensor] = {}
        # [num_steps] bool, shared. Lazy-allocated once we know device.
        self.valid_steps: Optional[torch.Tensor] = None
        # [B] long, set per-tick.
        self.step_indices: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop all cached entries. Call between unrelated streams."""
        for t in self.layer_banks.values():
            t.zero_()
        if self.valid_steps is not None:
            self.valid_steps.zero_()

    def num_entries(self) -> int:
        """Diagnostic: how many ``(layer, step)`` slots are populated.

        At saturation: ``len(banked_layers) * num_steps``. Matches the
        old dict-based ``len(bank.bank)`` semantic so existing logging
        continues to read sensibly.
        """
        if self.valid_steps is None:
            return 0
        return int(self.valid_steps.sum().item()) * len(self.layer_banks)

    def is_step_valid(self, step_idx: int) -> bool:
        """True iff some prior forward has written to step ``step_idx``."""
        if self.valid_steps is None:
            return False
        return bool(self.valid_steps[step_idx].item())

    # ------------------------------------------------------------------
    # Per-tick state
    # ------------------------------------------------------------------

    def set_step_indices(
        self,
        idxs: Union[Sequence[int], torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> None:
        """Set the per-row step indices for the next forward.

        Accepts either a Python sequence (auto-promoted to a long
        tensor) or an existing tensor. The tensor lives on the same
        device as the bank's other state; ``device`` overrides on
        first call.
        """
        if isinstance(idxs, torch.Tensor):
            self.step_indices = idxs.to(dtype=torch.long)
            if device is not None:
                self.step_indices = self.step_indices.to(device=device)
        else:
            target_device = device
            if target_device is None and self.valid_steps is not None:
                target_device = self.valid_steps.device
            self.step_indices = torch.tensor(
                list(idxs), dtype=torch.long, device=target_device,
            )

    # ------------------------------------------------------------------
    # Lazy allocation
    # ------------------------------------------------------------------

    def get_or_alloc_layer(
        self,
        layer_idx: int,
        num_kv_heads: int,
        T: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the ``[2, num_steps, kv_heads, T, head_dim]`` tensor.

        Allocates on first call for this layer with the given shape /
        device / dtype. If a tensor already exists with a different
        ``(kv_heads, T, head_dim)``, raises -- the caller has changed
        T (e.g. switched durations) and should ``reset()`` first or
        rebuild the bank.
        """
        existing = self.layer_banks.get(layer_idx)
        target_shape = (2, self.num_steps, num_kv_heads, T, head_dim)
        if existing is not None:
            if existing.shape != target_shape:
                raise RuntimeError(
                    f"FeatureBank layer {layer_idx} was allocated with "
                    f"shape {tuple(existing.shape)} but current forward "
                    f"requires {target_shape}. Call bank.reset() and "
                    f"reallocate, or rebuild the bank for the new T."
                )
            return existing

        new = torch.zeros(*target_shape, device=device, dtype=dtype)
        self.layer_banks[layer_idx] = new

        if self.valid_steps is None:
            self.valid_steps = torch.zeros(
                self.num_steps, dtype=torch.bool, device=device,
            )
        return new


# ----------------------------------------------------------------------
# Patched self-attention forward (eager + compile-friendly)
# ----------------------------------------------------------------------

def _patched_self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value=None,
    cache_position=None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    **kwargs,
):
    """Drop-in replacement for ``AceStepAttention.forward`` (self-attn).

    Mirrors the un-patched projection / norm / RoPE pipeline, then
    reads the bank tensor at ``[layer_idx, step_indices]``, concats
    along seq dim, runs SDPA with a strength-aware additive mask, and
    scatters the current K/V back into the bank at the same step.

    Falls through to the un-patched forward when banking is disabled
    or the layer isn't in ``bank.banked``. Cross-attn never reaches
    here -- the layer's cross-attention is a separate module -- but
    the early-return on ``encoder_hidden_states`` is defended for
    safety.
    """
    bank: FeatureBank = self._feature_bank

    if encoder_hidden_states is not None:
        return self._unpatched_forward(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )

    if not bank.enabled or self.layer_idx not in bank.banked:
        return self._unpatched_forward(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]  # [B, T]
    hidden_shape = (*input_shape, -1, self.head_dim)

    Q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    K = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    V = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

    B, kv_heads, T, head_dim = K.shape

    # Lazy-allocate this layer's bank tensor matching K/V shape.
    layer_bank = bank.get_or_alloc_layer(
        self.layer_idx, kv_heads, T, head_dim, K.device, K.dtype,
    )  # [2, num_steps, kv_heads, T, head_dim]

    # Promote Python-list step_indices to tensor on first use.
    if bank.step_indices is None:
        raise RuntimeError(
            "FeatureBank.step_indices not set. StreamPipeline must call "
            "bank.set_step_indices(...) before each decoder forward."
        )
    step_idx_t = bank.step_indices
    if not isinstance(step_idx_t, torch.Tensor):
        step_idx_t = torch.tensor(
            step_idx_t, dtype=torch.long, device=K.device,
        )
    elif step_idx_t.device != K.device:
        step_idx_t = step_idx_t.to(K.device)

    if step_idx_t.shape[0] != B:
        raise RuntimeError(
            f"FeatureBank.step_indices length {step_idx_t.shape[0]} does "
            f"not match batch size {B}."
        )

    # Read: gather along step axis. Shape: [2, B, kv_heads, T, head_dim]
    gathered = layer_bank.index_select(1, step_idx_t)
    K_bank = gathered[0]  # [B, kv_heads, T, head_dim]
    V_bank = gathered[1]

    # Per-row validity (step k valid iff any prior write at step k).
    valid_b = bank.valid_steps[step_idx_t]  # [B] bool

    K_full = torch.cat([K, K_bank], dim=2)  # [B, kv_heads, 2T, head_dim]
    V_full = torch.cat([V, V_bank], dim=2)

    # Build attn_bias [B, 1, T, 2T] with strength + per-row validity.
    NEG_INF = torch.finfo(Q.dtype).min
    s = bank.strength
    if s <= 0.0:
        log_s_scalar = NEG_INF
    elif s == 1.0:
        log_s_scalar = 0.0
    else:
        log_s_scalar = math.log(s)

    bank_col_mask = valid_b.view(B, 1, 1, 1).expand(B, 1, T, T)
    log_s_t = torch.tensor(log_s_scalar, device=Q.device, dtype=Q.dtype)
    neg_inf_t = torch.tensor(NEG_INF, device=Q.device, dtype=Q.dtype)
    bank_cols = torch.where(bank_col_mask, log_s_t, neg_inf_t)
    cur_cols = torch.zeros(B, 1, T, T, device=Q.device, dtype=Q.dtype)
    attn_bias = torch.cat([cur_cols, bank_cols], dim=-1)  # [B, 1, T, 2T]

    if self.num_key_value_groups > 1:
        K_full = K_full.repeat_interleave(self.num_key_value_groups, dim=1)
        V_full = V_full.repeat_interleave(self.num_key_value_groups, dim=1)

    attn_out = F.scaled_dot_product_attention(
        Q, K_full, V_full,
        attn_mask=attn_bias,
        dropout_p=0.0,
        scale=self.scaling,
    )  # [B, q_heads, T, head_dim]

    # Write: scatter current K/V into the step axis (in-place).
    if bank.write_enabled:
        K_det = K.detach()
        V_det = V.detach()
        layer_bank[0].index_copy_(0, step_idx_t, K_det)
        layer_bank[1].index_copy_(0, step_idx_t, V_det)
        bank.valid_steps[step_idx_t] = True

    attn_out = attn_out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    attn_out = self.o_proj(attn_out)
    return attn_out, None


# ----------------------------------------------------------------------
# Install / uninstall
# ----------------------------------------------------------------------

def _is_compiled(module: torch.nn.Module) -> bool:
    """True when ``module`` is wrapped by ``torch.compile``."""
    OptimizedModule = getattr(
        getattr(torch, "_dynamo", None), "eval_frame", None
    )
    if OptimizedModule is None:
        return False
    OptimizedModule = getattr(OptimizedModule, "OptimizedModule", None)
    if OptimizedModule is None:
        return False
    return isinstance(module, OptimizedModule)


def enable_feature_bank_on_decoder(
    decoder: torch.nn.Module,
    bank: FeatureBank,
) -> None:
    """Patch the self-attn forward on every banked layer in ``decoder``.

    Idempotent. Refuses ``OptimizedModule`` so the caller doesn't
    silently install an invisible patch behind a compiled trace -- the
    caller can compile *after* this returns.
    """
    if _is_compiled(decoder):
        raise RuntimeError(
            "Feature bank cannot be installed on a torch.compile'd decoder. "
            "Install the patch first, then call torch.compile."
        )

    if not hasattr(decoder, "layers"):
        raise RuntimeError(
            "decoder has no .layers attribute -- not an AceStepDiTModel?"
        )

    patched = 0
    for idx, layer in enumerate(decoder.layers):
        if idx not in bank.banked:
            continue
        attn = layer.self_attn
        if hasattr(attn, "_unpatched_forward"):
            attn._feature_bank = bank
            continue
        attn._feature_bank = bank
        attn._unpatched_forward = attn.forward
        attn.forward = types.MethodType(_patched_self_attn_forward, attn)
        patched += 1

    logger.info(
        "Feature bank enabled on %d layers: %s",
        patched, sorted(bank.banked),
    )


def disable_feature_bank_on_decoder(decoder: torch.nn.Module) -> None:
    """Restore the original forward on every patched self-attn module."""
    if not hasattr(decoder, "layers"):
        return
    restored = 0
    for layer in decoder.layers:
        attn = layer.self_attn
        if hasattr(attn, "_unpatched_forward"):
            attn.forward = attn._unpatched_forward
            del attn._unpatched_forward
            del attn._feature_bank
            restored += 1
    logger.info("Feature bank disabled on %d layers", restored)
