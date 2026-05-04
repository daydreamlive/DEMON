"""ONNX export wrapper for the feature-bank-enabled DiT decoder.

This builds on ``DecoderForExport`` (which already monkey-patches the
stock ``AceStepDiTModel.forward`` with a trace-friendly version)
and adds bank K/V as engine I/O. The signature follows the
daydreamlive/StreamDiffusion ``kvo_cache`` pattern, adapted for
ACE-Step's step-indexed bank:

    forward(
        hidden_states, timestep, encoder_hidden_states, context_latents,
        bank_k, bank_v, valid_steps, step_indices,
    ) -> (velocity, k_out, v_out)

Where:
  - ``bank_k`` / ``bank_v``: ``[num_banked_layers, num_steps, kv_heads,
    T_lat, head_dim]`` -- full history fed in by the host.
  - ``valid_steps``: ``[num_steps]`` boolean mask. Banked tokens at
    step k are visible to attention only when ``valid_steps[k]`` is
    True.
  - ``step_indices``: ``[B]`` long. For each batch row, which step
    slot in the bank to read / write.
  - ``k_out`` / ``v_out``: ``[num_banked_layers, B, kv_heads, T_lat,
    head_dim]`` -- per-row current K/V the host scatters into the bank
    ring buffer for next tick.

The host owns the ring rotation. Engine is stateless.

This module re-implements only the layer-call portion of the decoder
forward (the surrounding timestep / patch / mask / proj_out logic is
inherited from ``DecoderForExport._patch_decoder_for_trace``). For
banked layers we bypass the upstream ``AceStepDiTLayer.forward`` and
inline the AdaLN / self-attn / cross-attn / MLP blocks so we can
splice in a bank-aware self-attn that has bank K/V as explicit
tensor inputs (not Python state). Non-banked layers go through the
stock layer call unchanged.
"""

from __future__ import annotations

import types
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .export import DecoderForExport
from ..feature_bank import DEFAULT_BANKED_LAYERS


def _exportable_bank_self_attn(
    attn_mod: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    bank_k: torch.Tensor,        # [num_steps, kv_heads, T, head_dim]
    bank_v: torch.Tensor,        # [num_steps, kv_heads, T, head_dim]
    bank_bias: torch.Tensor,     # [num_steps] additive mask in stream dtype
    step_indices: torch.Tensor,  # [B] long
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bank-aware self-attention as a pure tensor function.

    Returns ``(attn_out, k_curr, v_curr)``: current K/V are returned
    so the host can scatter them into the bank ring at
    ``step_indices`` after the forward.

    ``bank_bias[k]`` is the host-precomputed additive bias on banked
    tokens whose step_idx is ``k``. Encoding:

      - 0.0           -> banked tokens get full softmax weight
      - log(strength) -> bank softmax mass scaled by ``strength`` < 1
      - -large        -> step ``k`` not yet written; mask out

    Folding both validity AND strength into a single per-step bias
    avoids ``bool`` casts and ``torch.where(bool, scalar, scalar)``
    inside the graph -- both patterns triggered TRT segfaults under
    strongly_typed when the original split form was used.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn_mod.head_dim)

    Q = attn_mod.q_norm(attn_mod.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    K = attn_mod.k_norm(attn_mod.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    V = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

    B, kv_heads, T, head_dim = K.shape

    K_bank = bank_k.index_select(0, step_indices)
    V_bank = bank_v.index_select(0, step_indices)
    bias_b = bank_bias.index_select(0, step_indices)  # [B]

    K_full = torch.cat([K, K_bank], dim=2)
    V_full = torch.cat([V, V_bank], dim=2)

    # Per-row bias broadcast to [B, 1, T, T]; current half is zero.
    bank_cols = bias_b.view(B, 1, 1, 1).expand(B, 1, T, T)
    cur_cols = torch.zeros(B, 1, T, T, device=Q.device, dtype=Q.dtype)
    attn_bias = torch.cat([cur_cols, bank_cols], dim=-1)

    if attn_mod.num_key_value_groups > 1:
        K_full = K_full.repeat_interleave(attn_mod.num_key_value_groups, dim=1)
        V_full = V_full.repeat_interleave(attn_mod.num_key_value_groups, dim=1)

    out = F.scaled_dot_product_attention(
        Q, K_full, V_full,
        attn_mask=attn_bias,
        dropout_p=0.0,
        scale=attn_mod.scaling,
    )
    out = out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    out = attn_mod.o_proj(out)
    return out, K, V


def _exportable_banked_layer_forward(
    layer_mod: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    temb: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor],
    bank_k: torch.Tensor,
    bank_v: torch.Tensor,
    bank_bias: torch.Tensor,
    step_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One banked DiT layer forward, inlined for trace.

    Mirrors the AdaLN -> self-attn -> cross-attn -> MLP structure of
    the upstream ``AceStepDiTLayer.forward`` but routes the self-attn
    through ``_exportable_bank_self_attn`` so bank K/V are explicit
    tensor inputs / outputs of the traced graph.
    """
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        layer_mod.scale_shift_table + temb
    ).chunk(6, dim=1)

    norm_hidden = (
        layer_mod.self_attn_norm(hidden_states) * (1 + scale_msa) + shift_msa
    ).type_as(hidden_states)

    attn_out, k_curr, v_curr = _exportable_bank_self_attn(
        layer_mod.self_attn,
        norm_hidden,
        position_embeddings,
        bank_k, bank_v,
        bank_bias, step_indices,
    )
    hidden_states = (hidden_states + attn_out * gate_msa).type_as(hidden_states)

    if layer_mod.use_cross_attention:
        norm_hidden = layer_mod.cross_attn_norm(hidden_states).type_as(hidden_states)
        cross_out, _ = layer_mod.cross_attn(
            hidden_states=norm_hidden,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states = hidden_states + cross_out

    norm_hidden = (
        layer_mod.mlp_norm(hidden_states) * (1 + c_scale_msa) + c_shift_msa
    ).type_as(hidden_states)
    ff_output = layer_mod.mlp(norm_hidden)
    hidden_states = (hidden_states + ff_output * c_gate_msa).type_as(hidden_states)

    return hidden_states, k_curr, v_curr


def _patch_decoder_for_bank_aware_trace(
    decoder: nn.Module,
    banked_layers: Sequence[int],
) -> None:
    """Replace decoder.forward with a bank-aware trace-friendly version.

    Keeps the timestep / patch / mask / norm-out / proj-out scaffolding
    from the upstream layer's ``_export_forward``, but swaps the
    layer-iteration block for one that:

      - calls non-banked layers via the stock ``layer_module(...)``
      - calls banked layers via ``_exportable_banked_layer_forward``,
        slicing per-layer bank K/V from the stacked ``bank_k`` /
        ``bank_v`` inputs.

    Returns: the new forward returns a 3-tuple
    ``(velocity, k_stack, v_stack)`` where the stacks are
    ``[num_banked_layers, B, kv_heads, T_lat, head_dim]``.
    """
    # Disable enable_gqa in SDPA for ONNX traceability (matches
    # DecoderForExport._patch_decoder_for_trace).
    import transformers.integrations.sdpa_attention as _sdpa_mod
    _sdpa_mod.use_gqa_in_sdpa = lambda *args, **kwargs: False

    sliding_window = decoder.config.sliding_window
    layer_types = decoder.config.layer_types
    banked_set = set(int(i) for i in banked_layers)
    banked_order: List[int] = sorted(banked_set)

    def _bank_aware_forward(
        self_dec,
        hidden_states,
        timestep,
        timestep_r,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents,
        bank_k,
        bank_v,
        bank_bias,
        step_indices,
        use_cache=None,
        past_key_values=None,
        cache_position=None,
        position_ids=None,
        output_attentions=False,
        return_hidden_states=None,
        custom_layers_config=None,
        enable_early_exit=False,
        **flash_attn_kwargs,
    ):
        # --- Timestep + patch (unchanged from _export_forward) ---
        temb_t, timestep_proj_t = self_dec.time_embed(timestep)
        temb_r, timestep_proj_r = self_dec.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
        hidden_states = self_dec.proj_in(hidden_states)
        encoder_hidden_states = self_dec.condition_embedder(encoder_hidden_states)

        seq_len_pat = hidden_states.shape[1]
        cache_position = torch.arange(seq_len_pat, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self_dec.rotary_emb(hidden_states, position_ids)

        # --- Sliding-window mask (unchanged from _export_forward) ---
        indices = cache_position
        diff = indices.unsqueeze(0) - indices.unsqueeze(1)
        sw_mask = torch.where(
            torch.abs(diff) <= sliding_window,
            torch.zeros(1, device=hidden_states.device, dtype=hidden_states.dtype),
            torch.full(
                (1,), torch.finfo(hidden_states.dtype).min,
                device=hidden_states.device, dtype=hidden_states.dtype,
            ),
        )
        sw_mask = sw_mask.unsqueeze(0).unsqueeze(0)

        # --- Layer loop with bank routing ---
        # Position in `bank_k` / `bank_v` for this banked layer:
        # banked layers in iteration order map to indices in
        # `banked_order` (sorted).
        bank_idx_map = {layer_idx: i for i, layer_idx in enumerate(banked_order)}
        k_outs: List[torch.Tensor] = []
        v_outs: List[torch.Tensor] = []

        for i, layer_module in enumerate(self_dec.layers):
            attn_mask = sw_mask if layer_types[i] == "sliding_attention" else None

            if i in banked_set:
                bi = bank_idx_map[i]
                # bank_k / bank_v are rank-5 tensors of shape
                # [N_banked, num_steps, kv_heads, T_lat, head_dim].
                # Index dim 0 to slice this layer's window. The earlier
                # rank-4 collapse (via narrow on a stacked [N*S, ...]
                # input) was thought to work around a Myelin rank-5
                # issue, but the rank-4 form turned out to ALSO crash
                # the autotuner at all seq lengths -- whereas a working
                # 10s engine artifact exists from the rank-5 form.
                layer_bk = bank_k[bi]
                layer_bv = bank_v[bi]
                hidden_states, k_curr, v_curr = _exportable_banked_layer_forward(
                    layer_module,
                    hidden_states,
                    position_embeddings,
                    timestep_proj,
                    encoder_hidden_states,
                    layer_bk,
                    layer_bv,
                    bank_bias,
                    step_indices,
                )
                k_outs.append(k_curr)
                v_outs.append(v_curr)
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    position_embeddings,
                    timestep_proj,
                    attn_mask,
                    position_ids,
                    None,    # past_key_values
                    False,   # output_attentions
                    False,   # use_cache
                    cache_position,
                    encoder_hidden_states,
                    None,    # encoder_attention_mask
                )
                hidden_states = layer_outputs[0]

        # --- norm_out + proj_out (unchanged) ---
        shift, scale = (
            self_dec.scale_shift_table + temb.unsqueeze(1)
        ).chunk(2, dim=1)
        hidden_states = (
            self_dec.norm_out(hidden_states) * (1 + scale) + shift
        ).type_as(hidden_states)
        velocity = self_dec.proj_out(hidden_states)

        # Stack along a new leading dim -> rank-5 [N_banked, B, kv_heads,
        # T, head_dim]. Restored to the rank-5 layout that produced a
        # working 10s engine artifact; the earlier rank-4 collapse was
        # a misdiagnosed workaround that broke the build at every seq.
        k_stack = torch.stack(k_outs, dim=0)
        v_stack = torch.stack(v_outs, dim=0)
        return velocity, k_stack, v_stack

    decoder.forward = types.MethodType(_bank_aware_forward, decoder)


class BankAwareDecoderForExport(DecoderForExport):
    """``DecoderForExport`` extended with bank K/V as engine I/O.

    Construction-wise, this is just ``DecoderForExport`` plus a second
    decoder-forward swap: the parent first installs its own
    trace-friendly forward, then ``__init__`` here replaces it with
    the bank-aware variant. The precision plumbing (Lambdas replaced,
    SDPA forced, mixed-precision recipes, etc.) is inherited
    unchanged.

    The ``forward`` signature is what gets exported to ONNX:

        forward(hidden_states, timestep, encoder_hidden_states,
                context_latents, bank_k, bank_v, valid_steps,
                step_indices)
            -> (velocity, k_stack, v_stack)
    """

    def __init__(
        self,
        decoder: nn.Module,
        banked_layers: Sequence[int] = DEFAULT_BANKED_LAYERS,
        mixed_precision: bool = False,
        precision: str = "fp32",
    ):
        super().__init__(
            decoder=decoder,
            mixed_precision=mixed_precision,
            precision=precision,
        )
        self.banked_layers: Tuple[int, ...] = tuple(int(i) for i in banked_layers)
        # Replace parent's _export_forward with bank-aware variant.
        _patch_decoder_for_bank_aware_trace(self.decoder, self.banked_layers)
        logger.info(
            "BankAwareDecoderForExport: bank-aware trace forward installed "
            "for layers %s",
            list(self.banked_layers),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,         # [B, T, 64]
        timestep: torch.Tensor,              # [B]
        encoder_hidden_states: torch.Tensor, # [B, L_enc, 2048]
        context_latents: torch.Tensor,       # [B, T, 128]
        bank_k: torch.Tensor,                # [num_banked, num_steps, kv_heads, T_lat, head_dim]
        bank_v: torch.Tensor,                # same shape
        bank_bias: torch.Tensor,             # [num_steps] additive bias on banked tokens
        step_indices: torch.Tensor,          # [B] long
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decoder(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            context_latents=context_latents,
            bank_k=bank_k,
            bank_v=bank_v,
            bank_bias=bank_bias,
            step_indices=step_indices,
            use_cache=False,
            past_key_values=None,
            output_attentions=False,
        )


# ----------------------------------------------------------------------
# ONNX export entry point
# ----------------------------------------------------------------------

def export_bank_aware_decoder_to_onnx(
    decoder: nn.Module,
    onnx_path,
    *,
    batch_size: int = 1,
    seq_len: int = 250,           # 10 s at 25 Hz, must be even
    enc_len: int = 64,
    num_steps: int = 8,
    banked_layers: Sequence[int] = DEFAULT_BANKED_LAYERS,
    precision: str = "bf16",
    device: str = "cuda",
    do_constant_folding: bool = True,
) -> None:
    """Export the bank-aware decoder to ONNX.

    Mirrors the legacy / dynamo selection in ``trt/export.py`` -- bf16
    paths must use the dynamo exporter due to a known bug in the
    torchscript exporter's complex-tensor handling under bf16.

    The ONNX graph has 8 inputs and 3 outputs:
      INPUTS:
        hidden_states           [B, T, 64]
        timestep                [B]
        encoder_hidden_states   [B, L_enc, 2048]
        context_latents         [B, T, 128]
        bank_k                  [N_banked, num_steps, kv_heads, T_lat, head_dim]
        bank_v                  [N_banked, num_steps, kv_heads, T_lat, head_dim]
        valid_steps             [num_steps]    (float; 1.0 means valid)
        step_indices            [B]            (int64)
      OUTPUTS:
        velocity                [B, T, 64]
        k_stack                 [N_banked, B, kv_heads, T_lat, head_dim]
        v_stack                 [N_banked, B, kv_heads, T_lat, head_dim]

    ``T_lat == T // 2`` because of the patch_size=2 patch embedding.

    Caller is responsible for sequencing this with the ``trt/build.py``
    engine build (which needs to be told about the new I/O profiles).
    """
    from pathlib import Path
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap decoder for export. Pure bf16 by default (matches v15-turbo
    # native dtype); bf16 builds use the dynamo exporter.
    wrapper = BankAwareDecoderForExport(
        decoder=decoder,
        banked_layers=banked_layers,
        mixed_precision=False,
        precision=precision,
    ).eval()

    if precision in ("bf16", "bf16_mixed"):
        wrapper = wrapper.to(device)
        trace_dtype = torch.bfloat16
        ts_dtype = torch.bfloat16
    elif precision == "fp32":
        wrapper = wrapper.float().to(device)
        trace_dtype = torch.float32
        ts_dtype = torch.float32
    else:
        raise ValueError(
            f"precision={precision!r} not supported for bank-aware export "
            f"(yet). Use 'bf16', 'bf16_mixed', or 'fp32'."
        )

    # Inferred latent T after patch embedding (Conv1d stride=2).
    config = decoder.config
    head_dim = config.head_dim
    kv_heads = config.num_key_value_heads
    T_lat = seq_len // 2
    n_banked = len(banked_layers)

    example_inputs = (
        torch.randn(batch_size, seq_len, 64, device=device, dtype=trace_dtype),
        torch.full((batch_size,), 0.5, device=device, dtype=ts_dtype),
        torch.randn(batch_size, enc_len, 2048, device=device, dtype=trace_dtype),
        torch.randn(batch_size, seq_len, 128, device=device, dtype=trace_dtype),
        # rank-5 [N_banked, num_steps, kv_heads, T_lat, head_dim]
        torch.zeros(n_banked, num_steps, kv_heads, T_lat, head_dim, device=device, dtype=trace_dtype),
        torch.zeros(n_banked, num_steps, kv_heads, T_lat, head_dim, device=device, dtype=trace_dtype),
        torch.zeros(num_steps, device=device, dtype=trace_dtype),  # bank_bias
        torch.zeros(batch_size, device=device, dtype=torch.long),
    )

    input_names = [
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "context_latents",
        "bank_k",
        "bank_v",
        "bank_bias",
        "step_indices",
    ]
    output_names = ["velocity", "k_stack", "v_stack"]

    use_dynamo = precision in ("bf16", "bf16_mixed")

    logger.info(
        "Tracing bank-aware decoder for ONNX (T=%d, L=%d, num_steps=%d, "
        "banked=%d, exporter=%s) ...",
        seq_len, enc_len, num_steps, n_banked,
        "dynamo" if use_dynamo else "torchscript",
    )

    with torch.no_grad():
        if use_dynamo:
            from torch.export import Dim
            # seq / num_steps / T_lat STATIC: dynamo surfaces
            # divisibility constraints once bank tensors enter the
            # graph. Multi-duration support comes from per-duration
            # engines (matching the existing ACE-Step build pipeline).
            # batch and enc are dynamic.
            batch = Dim("batch", min=1, max=8)
            enc = Dim("enc", min=32, max=512)
            dynamic_shapes = {
                "hidden_states":         {0: batch},
                "timestep":              {0: batch},
                "encoder_hidden_states": {0: batch, 1: enc},
                "context_latents":       {0: batch},
                "bank_k":                {},
                "bank_v":                {},
                "bank_bias":             {},
                "step_indices":          {0: batch},
            }
            torch.onnx.export(
                wrapper,
                example_inputs,
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
            )
        else:
            dynamic_axes = {
                "hidden_states":         {0: "batch", 1: "seq_len"},
                "timestep":              {0: "batch"},
                "encoder_hidden_states": {0: "batch", 1: "enc_len"},
                "context_latents":       {0: "batch", 1: "seq_len"},
                "bank_k":                {3: "seq_len_lat"},
                "bank_v":                {3: "seq_len_lat"},
                "step_indices":          {0: "batch"},
                "velocity":              {0: "batch", 1: "seq_len"},
                "k_stack":               {1: "batch", 3: "seq_len_lat"},
                "v_stack":               {1: "batch", 3: "seq_len_lat"},
            }
            torch.onnx.export(
                wrapper,
                example_inputs,
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=18,
                do_constant_folding=do_constant_folding,
                dynamo=False,
            )

    size_mb = onnx_path.stat().st_size / (1 << 20)
    logger.info("Bank-aware ONNX saved to %s (%.1f MB)", onnx_path, size_mb)
