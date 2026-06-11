"""ONNX export and TensorRT engine build for the ACE-Step decoder.

Export flow:
  1. Wrap decoder in DecoderForExport (fixes Lambda, forces SDPA, no cache)
  2. Export to ONNX with dynamic B / T / L_enc axes
  3. Build TRT engine with FP16 and optimization profiles

Precision strategy:
  - Export weights in fp32 (preserves full precision in ONNX graph)
  - TRT builder converts to fp16 internally with its own kernel selection
  - This avoids the bf16-to-fp16 silent truncation that causes wrong output
    when exporting directly in half precision
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Union

from loguru import logger
import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Traceable replacement for the Lambda(transpose) modules
# ------------------------------------------------------------------

class _Transpose12(nn.Module):
    """Transpose dims 1 and 2.  Drop-in for Lambda(lambda x: x.transpose(1, 2))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)


class _Fp32CastWrapper(nn.Module):
    """Run an inner module in fp32, casting around it.

    Used for the XL bf16_mixed recipe where TensorRT has no bf16 kernel for
    the proj_out ConvTranspose1d shape.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        inner.float()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        return self.inner(x.float()).to(out_dtype)


class _LinearFp16Wrapper(nn.Module):
    """Wrap a single nn.Linear so it computes in fp16 with bf16 I/O.

    Used by the bf16-hybrid decoder recipe: the residual stream stays in
    bf16 (which holds the peak activations that overflow fp16), but each
    matmul is cast tight to fp16 so TRT picks fp16 tensor-core kernels
    for the layer body. The trace records:
        Cast(bf16 -> fp16) -> matmul (fp16 weights, fp16 inputs) -> Cast(fp16 -> bf16)
    Per-Linear granularity keeps TRT's fusion radius to a single matmul
    plus its two casts, which empirically survives strongly_typed
    compilation. The inner Linear's input is post-RMSNorm * AdaLN scale,
    bounded inside fp16 range.

    Note: dynamo exports lose the FQN for these wrapped Linears (every
    wrapped weight becomes an anonymous ``val_*`` SSA value in the
    ONNX initializer list). The build pipeline patches that back via
    a fingerprint pass — see :func:`rename_val_initializers_to_fqn`.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        inner.half()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        return self.inner(x.to(torch.float16)).to(out_dtype)


# ------------------------------------------------------------------
# Export wrapper
# ------------------------------------------------------------------

class DecoderForExport(nn.Module):
    """Thin wrapper that makes AceStepDiTModel safe for ONNX tracing.

    Changes vs. the raw decoder forward():
      - Lambda modules replaced with _Transpose12 for traceability
      - Attention implementation forced to SDPA (no flash_attn CUDA kernels)
      - KV cache disabled (use_cache=False, past_key_values=None)
      - output_attentions=False (no extra tuple elements)
      - timestep_r set equal to timestep (inference convention)
      - Returns the velocity tensor directly, not a tuple

    The input attention_mask and encoder_attention_mask parameters are
    intentionally set to None because the decoder's forward() shadows
    them immediately with local None assignments (lines 1378-1382 in
    the modeling file) and constructs full bidirectional masks from
    scratch via create_4d_mask().  Passing None here is therefore
    identical to passing torch.ones and avoids two unnecessary dynamic
    inputs in the TRT engine.
    """

    def __init__(
        self,
        decoder: nn.Module,
        mixed_precision: bool = False,
        precision: str = "fp32",
    ):
        """
        Args:
            decoder: AceStepDiTModel instance to wrap.
            mixed_precision: Bf16-hybrid recipe for the 2B turbo decoder.
                When True, ``precision`` is ignored. Bf16 trunk + fp32
                islands (timestep, AdaLN tables, RMSNorms, norm_out) +
                per-Linear fp16 cast wrappers + fp32 wrapper around
                proj_out's ConvTranspose1d. Strongly-typed builds. See
                :meth:`_setup_mixed_precision` for the rationale.
            precision: Used when ``mixed_precision`` is False. One of:
                - "fp32": leave dtypes as-is for tracing in fp32 (default)
                - "bf16_mixed": bf16 bulk with an fp32 proj_out deconv island
                  for XL turbo. Requires the dynamo exporter.
        """
        super().__init__()
        self.decoder = decoder
        if mixed_precision:
            self.precision = "bf16_hybrid"
        else:
            self.precision = precision

        # Replace Lambda with traceable transpose
        self._replace_lambdas()

        # Force SDPA so the graph contains only standard ops
        self.decoder.config._attn_implementation = "sdpa"

        # Patch the decoder forward to be ONNX-trace-safe
        self._patch_decoder_for_trace()

        if mixed_precision:
            self._setup_mixed_precision()
        elif precision == "bf16_mixed":
            self._setup_bf16_mixed()

    # List of (parent_module_attr, linear_attr_name) pairs that are safe
    # to wrap in fp16 inside each AceStepDiTLayer. Inputs to these Linears
    # are bounded by post-RMSNorm output * AdaLN scale, which fits inside
    # fp16 dynamic range. Outputs feed into residual adds that stay in
    # bf16 outside the wrapper.
    LINEAR_WRAP_TARGETS = (
        ("self_attn", "q_proj"),
        ("self_attn", "k_proj"),
        ("self_attn", "v_proj"),
        ("self_attn", "o_proj"),
        ("cross_attn", "q_proj"),
        ("cross_attn", "k_proj"),
        ("cross_attn", "v_proj"),
        ("cross_attn", "o_proj"),
        ("mlp", "gate_proj"),
        ("mlp", "up_proj"),
        ("mlp", "down_proj"),
    )

    # ---- internal helpers ----

    def _replace_lambdas(self) -> None:
        for seq in (self.decoder.proj_in, self.decoder.proj_out):
            for i, mod in enumerate(seq):
                if type(mod).__name__ == "Lambda":
                    seq[i] = _Transpose12()

    def _setup_mixed_precision(self) -> None:
        """Bf16-hybrid recipe for the 2B turbo decoder.

        Bf16 trunk for the residual stream + fp32 islands for the
        precision-critical ops (timestep embedding, AdaLN tables,
        RMSNorms, norm_out) + per-Linear fp16 cast wrappers around
        every matmul in each DiT layer + fp32 wrapper around proj_out's
        ConvTranspose1d (TRT has no bf16 deconv kernel for the
        [hidden, 64, 2, 1] shape).

        The inner fp16 matmul casts are safe because each Linear's
        input is bounded by RMSNorm * AdaLN scale, which fits inside
        fp16 dynamic range. The bf16 residual stream handles peaks
        that overflow fp16, fixing the high-t NaN that the old pure-fp16
        trunk produced on the production decoder.

        Note on the fp32 islands: a three-way profile against an
        islands-free reference engine showed the same output cosine
        similarity (~0.99988) as a same-ONNX rebuild against itself,
        which suggested the islands were dead code. They are not:
        removing them produces an ONNX that segfaults TRT's
        strongly_typed builder when combined with the time_embed
        reshape patch in _patch_decoder_for_trace. Keep the islands.
        """
        decoder = self.decoder

        # Bf16 trunk
        decoder.to(torch.bfloat16)

        # fp32 islands: timestep, output AdaLN, output norm
        decoder.time_embed.float()
        decoder.time_embed_r.float()
        decoder.scale_shift_table = nn.Parameter(
            decoder.scale_shift_table.data.float()
        )
        decoder.norm_out.float()

        # Per-layer fp32 islands: AdaLN table + all RMSNorms
        for layer in decoder.layers:
            layer.scale_shift_table = nn.Parameter(
                layer.scale_shift_table.data.float()
            )
            layer.self_attn_norm.float()
            layer.mlp_norm.float()
            if hasattr(layer, "cross_attn_norm"):
                layer.cross_attn_norm.float()

        # fp32 wrapper around proj_out's ConvTranspose1d. _replace_lambdas
        # already swapped the surrounding Lambdas for _Transpose12, so we
        # locate the deconv by type.
        deconv_idx = None
        for i, mod in enumerate(decoder.proj_out):
            if isinstance(mod, nn.ConvTranspose1d):
                deconv_idx = i
                break
        if deconv_idx is None:
            raise RuntimeError(
                "mixed_precision: could not find ConvTranspose1d inside proj_out"
            )
        decoder.proj_out[deconv_idx] = _Fp32CastWrapper(decoder.proj_out[deconv_idx])

        # Per-Linear fp16 wrappers across every DiT layer.
        wrapped = 0
        for layer in decoder.layers:
            for parent_attr, lin_attr in self.LINEAR_WRAP_TARGETS:
                parent = getattr(layer, parent_attr, None)
                if parent is None:
                    continue
                lin = getattr(parent, lin_attr, None)
                if lin is None:
                    continue
                setattr(parent, lin_attr, _LinearFp16Wrapper(lin))
                wrapped += 1
        logger.info(
            "mixed_precision (bf16 hybrid): wrapped {} Linears across {} layers",
            wrapped, len(decoder.layers),
        )

    def _setup_bf16_mixed(self) -> None:
        """bf16 bulk + fp32 island for XL turbo's unsupported deconv op."""
        decoder = self.decoder
        decoder.to(torch.bfloat16)

        deconv_idx = None
        for i, mod in enumerate(decoder.proj_out):
            if isinstance(mod, nn.ConvTranspose1d):
                deconv_idx = i
                break
        if deconv_idx is None:
            raise RuntimeError(
                "bf16_mixed: could not find ConvTranspose1d inside proj_out"
            )
        decoder.proj_out[deconv_idx] = _Fp32CastWrapper(decoder.proj_out[deconv_idx])

    def _patch_decoder_for_trace(self) -> None:
        """Monkey-patch the decoder forward to be ONNX-trace-safe.

        Fixes three trace-hostile patterns in the stock forward():

          1. GQA in SDPA: transformers passes ``enable_gqa=True`` to
             ``F.scaled_dot_product_attention`` when num_key_value_groups > 1
             and attention_mask is None.  The ONNX exporter cannot convert
             this.  We monkey-patch ``use_gqa_in_sdpa`` to return False so
             the SDPA path falls back to ``repeat_kv`` (head expansion via
             ``repeat_interleave``), which is fully traceable.

          2. Shape-dependent Python branches: the original forward captures
             ``original_seq_len = shape[1]`` as a Python int (baked constant
             in ONNX) and uses ``if pad_length > 0`` (baked branch).  We
             remove padding/cropping entirely; the caller must ensure
             seq_len is a multiple of patch_size (=2, i.e. even).

          3. ``create_4d_mask()`` builds shape-dependent masks that bake
             traced dimensions.  Replaced with inline tensor ops for the
             sliding window mask (bidirectional, ``|i-j| <= window``).
             Full attention layers get ``None`` (is_causal=False on the
             module means SDPA treats None as bidirectional).
        """
        import types

        # --- Fix GQA: disable enable_gqa in SDPA for ONNX traceability ---
        # When use_gqa_in_sdpa returns False, the transformers SDPA function
        # manually expands K/V heads via repeat_kv (repeat_interleave) instead
        # of passing enable_gqa=True.  repeat_interleave traces cleanly.
        import transformers.integrations.sdpa_attention as _sdpa_mod
        _sdpa_mod.use_gqa_in_sdpa = lambda *args, **kwargs: False

        decoder = self.decoder
        sliding_window = decoder.config.sliding_window  # 128
        layer_types = decoder.config.layer_types  # list of "full_attention"/"sliding_attention"

        # Dynamo bakes the trace-time batch size into ``unflatten(1, (6, -1))``
        # in TimestepEmbedding.forward, which corrupts dynamic-batch
        # execution. Replace the unflatten with an explicit reshape so the
        # batch dim stays symbolic. Harmless for the legacy torchscript path.
        time_embed_dim = decoder.time_embed.time_proj.out_features // 6

        def _patched_time_embed_forward(self_te, t):
            t_freq = self_te.timestep_embedding(t, self_te.in_channels)
            temb = self_te.linear_1(t_freq.to(t.dtype))
            temb = self_te.act1(temb)
            temb = self_te.linear_2(temb)
            timestep_proj = self_te.time_proj(self_te.act2(temb)).reshape(-1, 6, time_embed_dim)
            return temb, timestep_proj

        decoder.time_embed.forward = types.MethodType(
            _patched_time_embed_forward, decoder.time_embed,
        )
        decoder.time_embed_r.forward = types.MethodType(
            _patched_time_embed_forward, decoder.time_embed_r,
        )

        def _export_forward(
            self_dec,
            hidden_states,
            timestep,
            timestep_r,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            context_latents,
            steering,
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
            # Timestep embeddings
            temb_t, timestep_proj_t = self_dec.time_embed(timestep)
            temb_r, timestep_proj_r = self_dec.time_embed_r(timestep - timestep_r)
            temb = temb_t + temb_r
            timestep_proj = timestep_proj_t + timestep_proj_r

            # Concatenate context
            hidden_states = torch.cat([context_latents, hidden_states], dim=-1)

            # No padding or cropping.  seq_len must be a multiple of
            # patch_size (=2).  This avoids shape-dependent Python branches
            # that bake constants into the ONNX graph.

            # proj_in (patch embedding: Conv1d stride=2 halves seq_len)
            hidden_states = self_dec.proj_in(hidden_states)
            encoder_hidden_states = self_dec.condition_embedder(encoder_hidden_states)

            # Position IDs / embeddings
            seq_len_pat = hidden_states.shape[1]
            cache_position = torch.arange(seq_len_pat, device=hidden_states.device)
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = self_dec.rotary_emb(hidden_states, position_ids)

            # Sliding window mask: bidirectional, |i-j| <= window.
            # Uses tensor ops (arange, abs, where) so ONNX can trace them.
            # Full attention layers get None (is_causal=False on the module
            # means SDPA treats None as fully bidirectional).
            indices = cache_position  # [seq_len_pat]
            diff = indices.unsqueeze(0) - indices.unsqueeze(1)  # [S, S]
            sw_mask = torch.where(
                torch.abs(diff) <= sliding_window,
                torch.zeros(1, device=hidden_states.device, dtype=hidden_states.dtype),
                torch.full((1,), torch.finfo(hidden_states.dtype).min, device=hidden_states.device, dtype=hidden_states.dtype),
            )
            sw_mask = sw_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

            # Layer loop: static branching on layer_types. The added
            # ``steering[:, i, :]`` shift is the in-engine equivalent of
            # StreamPipeline._install_steering_hooks; the host zeros rows
            # that aren't at an active step so they get a no-op add.
            for i, layer_module in enumerate(self_dec.layers):
                attn_mask = sw_mask if layer_types[i] == "sliding_attention" else None
                layer_outputs = layer_module(
                    hidden_states,
                    position_embeddings,
                    timestep_proj,
                    attn_mask,
                    position_ids,
                    None,   # past_key_values
                    False,  # output_attentions
                    False,  # use_cache
                    cache_position,
                    encoder_hidden_states,
                    None,   # encoder_attention_mask
                )
                hidden_states = layer_outputs[0] + steering[:, i, :].unsqueeze(1).type_as(layer_outputs[0])

            # Output AdaLN + proj_out (ConvTranspose1d stride=2 doubles seq_len)
            shift, scale = (self_dec.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            hidden_states = (self_dec.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
            hidden_states = self_dec.proj_out(hidden_states)

            return (hidden_states, None)

        decoder.forward = types.MethodType(_export_forward, decoder)

    # ---- forward ----

    def forward(
        self,
        hidden_states: torch.Tensor,       # [B, T, 64]
        timestep: torch.Tensor,            # [B]
        encoder_hidden_states: torch.Tensor,  # [B, L_enc, 2048]
        context_latents: torch.Tensor,     # [B, T, 128]
        steering: torch.Tensor,            # [B, num_layers, hidden_size]
    ) -> torch.Tensor:
        outputs = self.decoder(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            context_latents=context_latents,
            steering=steering,
            use_cache=False,
            past_key_values=None,
            output_attentions=False,
        )
        return outputs[0]  # velocity [B, T, 64]


# ------------------------------------------------------------------
# ONNX export
# ------------------------------------------------------------------

@dataclass
class OnnxExportConfig:
    """Configuration for ONNX export."""

    # Trace input sizes (should be "typical" values)
    batch_size: int = 1
    seq_len: int = 750       # 30s at 25 Hz, must be even
    enc_len: int = 200       # typical encoder seq len

    opset_version: int = 17
    do_constant_folding: bool = True

    # Mixed precision: export the bf16-hybrid recipe for the 2B turbo
    # decoder (bf16 trunk + fp32 islands for timestep / AdaLN / norms +
    # per-Linear fp16 wrappers + fp32 proj_out deconv island). Use with
    # TRTBuildConfig.strongly_typed=True so the engine respects the dtype
    # assignments. Ignored when ``precision`` is set.
    mixed_precision: bool = False

    # Trace dtype for the wrapper. Used when ``mixed_precision`` is False.
    # Supported values: "fp32" and "bf16_mixed".
    precision: str = "fp32"

    # When True, disables ONNX constant folding to preserve PyTorch
    # parameter names as ONNX initializer names.  Required for TRT
    # REFIT so the refitter can address weights by their original names.
    # Without this, nn.Linear weights get auto-generated names like
    # "onnx__MatMul_12882" that can't be mapped back to LoRA targets.
    for_refit: bool = False


def export_decoder_onnx(
    model,
    onnx_path: Union[str, Path],
    device: str = "cuda",
    config: Optional[OnnxExportConfig] = None,
) -> Path:
    """Export the decoder to ONNX with dynamic shapes.

    Args:
        model: AceStepConditionGenerationModel (the full model, we extract .decoder).
        onnx_path: Where to write the .onnx file.
        device: Device for tracing ("cuda" or "cpu").
        config: Export configuration.  Defaults are fine for most cases.

    Returns:
        Path to the written ONNX file.
    """
    if config is None:
        config = OnnxExportConfig()

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    decoder = model.decoder
    wrapper = DecoderForExport(
        decoder,
        mixed_precision=config.mixed_precision,
        precision=config.precision,
    ).eval()

    if config.mixed_precision:
        # Mixed precision: bf16 trunk + fp32 islands + per-Linear fp16
        # wrappers + fp32 proj_out deconv island. Trace inputs are bf16
        # to match the residual stream; timestep stays fp32 for the
        # fp32 time_embed island.
        wrapper = wrapper.to(device)
        trace_dtype = torch.bfloat16
        ts_dtype = torch.float32
        logger.info("Exporting with mixed precision (bf16 trunk + fp32 islands + per-Linear fp16)")
    elif config.precision == "bf16_mixed":
        wrapper = wrapper.to(device)
        trace_dtype = torch.bfloat16
        ts_dtype = torch.bfloat16
        logger.info("Exporting bf16 mixed (bf16 bulk + fp32 deconv island)")
    else:
        # Full fp32 export
        wrapper = wrapper.float().to(device)
        trace_dtype = torch.float32
        ts_dtype = torch.float32

    B = config.batch_size
    T = config.seq_len
    L = config.enc_len
    # Steering: one additive vector per (batch row, DiT layer). Sized
    # off the live decoder so XL reuses this path without a config knob.
    num_layers = decoder.config.num_hidden_layers
    hidden_size = decoder.config.hidden_size

    example_inputs = (
        torch.randn(B, T, 64, device=device, dtype=trace_dtype),
        torch.full((B,), 0.5, device=device, dtype=ts_dtype),
        torch.randn(B, L, 2048, device=device, dtype=trace_dtype),
        torch.randn(B, T, 128, device=device, dtype=trace_dtype),
        torch.zeros(B, num_layers, hidden_size, device=device, dtype=trace_dtype),
    )

    input_names = [
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "context_latents",
        "steering",
    ]
    output_names = ["velocity"]

    dynamic_axes = {
        "hidden_states":          {0: "batch", 1: "seq_len"},
        "timestep":               {0: "batch"},
        "encoder_hidden_states":  {0: "batch", 1: "enc_len"},
        "context_latents":        {0: "batch", 1: "seq_len"},
        # num_layers / hidden_size are static within a checkpoint family.
        "steering":               {0: "batch"},
        "velocity":               {0: "batch", 1: "seq_len"},
    }

    # For refit-enabled builds on the torchscript exporter, disable
    # constant folding to preserve weight names as ONNX initializer names.
    # TRT does its own constant folding internally, so this has no effect
    # on engine quality. The dynamo exporter ignores this flag (it doesn't
    # accept ``do_constant_folding``) and preserves weight names natively.
    do_constant_folding = config.do_constant_folding
    if config.for_refit:
        do_constant_folding = False
        logger.info("REFIT mode: constant folding disabled to preserve weight names")

    # The legacy torchscript-based ONNX exporter (dynamo=False) has a bug
    # in its shape-type inference pass when tracing bf16 graphs: it produces
    # complex tensors during constant folding, then fails with
    # "ScalarType ComplexDouble is an unexpected tensor scalar type". The
    # new dynamo-based exporter (torch.export) doesn't have this bug.
    # Use dynamo for any bf16-containing trace (mixed_precision is now
    # bf16-trunk hybrid); keep legacy for fp32.
    use_dynamo = (
        config.mixed_precision
        or config.precision == "bf16_mixed"
    )

    logger.info(
        "Tracing decoder for ONNX export (T={}, L={}, exporter={}) ...",
        T, L, "dynamo" if use_dynamo else "torchscript",
    )

    with torch.no_grad():
        if use_dynamo:
            # torch.onnx's dynamo path prints Unicode status markers. On
            # Windows cp1252 consoles those can raise UnicodeEncodeError
            # after graph capture succeeds, aborting an otherwise valid export.
            for stream in (sys.stdout, sys.stderr):
                if hasattr(stream, "reconfigure"):
                    stream.reconfigure(encoding="utf-8", errors="replace")

            from torch.export import Dim

            batch = Dim("batch", min=1, max=8)
            # 6000 = 240s at 25 Hz, covering the canonical engine matrix.
            seq = Dim("seq", min=126, max=6000)
            enc = Dim("enc", min=32, max=512)
            dynamic_shapes = {
                "hidden_states":         {0: batch, 1: seq},
                "timestep":              {0: batch},
                "encoder_hidden_states": {0: batch, 1: enc},
                "context_latents":       {0: batch, 1: seq},
                "steering":              {0: batch},
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
            torch.onnx.export(
                wrapper,
                example_inputs,
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=config.opset_version,
                do_constant_folding=do_constant_folding,
                dynamo=False,
            )

    # XL dynamo exports may use external data for the large weight payload.
    # Keep patched ONNX protobufs next to the source file so those relative
    # external_data references continue to resolve during TRT parsing.

    # Dynamo lifts every wrapped-Linear weight as an anonymous ``val_*``
    # initializer, which breaks TRT refit (LoRA targets resolve to
    # nothing). Re-attach the FQN by fingerprinting each ``val_*``
    # against the live decoder's 2D parameters. Cheap (~few seconds)
    # and only runs once per ONNX export.
    if config.for_refit and use_dynamo:
        rename_val_initializers_to_fqn(
            onnx_path=onnx_path,
            decoder=wrapper.decoder,
        )

    size_mb = onnx_path.stat().st_size / (1 << 20)
    logger.info("ONNX saved to {} ({:.1f} MB)", onnx_path, size_mb)
    return onnx_path


def rename_val_initializers_to_fqn(
    onnx_path: Union[str, Path],
    decoder: nn.Module,
) -> int:
    """Replace ``val_*`` initializer names with their torch FQNs (proto-only).

    The bf16-hybrid recipe wraps Linears in ``_LinearFp16Wrapper``; the
    dynamo exporter lifts each wrapped weight as an anonymous SSA
    ``val_*`` initializer instead of preserving the FQN. TRT refit
    addresses weights by their ONNX name, so anonymous slots can't
    receive LoRA deltas keyed by ``layers.X.self_attn.q_proj.weight``.

    This pass matches each ``val_*`` initializer to a torch parameter
    by **exact byte hash** at the parameter's native dtype, and renames
    it to the FQN that the runtime decoder (unwrapped) uses — i.e.
    ``.inner.`` is stripped from the wrapped path so LoRA delta keys
    resolve directly. ONNX MatMul stores Linear weights transposed
    relative to ``nn.Linear.weight``, so both orientations are hashed.

    Byte-hash, not L2 fingerprint: an earlier ``(shape, L2)`` cohort
    match with a 1e-3 tolerance silently misrouted 96 of 264 LoRA
    target weights in the 2B bf16-hybrid recipe — many fp16 Linears
    of the same shape have L2 norms close enough to collide, and the
    rename happily wrote each ``val_*`` to whichever same-shape FQN
    happened to be unclaimed. LoRA refit then materialized 192 deltas
    into wrong matmul slots, producing a ~20× larger output delta on
    NEW vs the pre-bf16 reference engine. Exact bytes can't collide
    unless two parameters genuinely contain the same numbers.

    The rename operates **only on the proto file**: external-data
    files emitted by dynamo are not re-read or re-written, so weight
    bytes are byte-for-byte preserved on disk. An earlier version
    used ``onnx.load(load_external_data=True) -> onnx.save(...)`` and
    silently re-encoded the dynamo graph (the consolidated .data file
    doubled from 3.15 GB to 6.5 GB, with 376 BFLOAT16 initializers
    flipped to FLOAT16/FLOAT), destroying the strongly-typed dtype
    layout the bf16-hybrid recipe relies on. The proto-only path
    here cannot mutate weight bytes or dtypes by construction.

    Returns the number of renamed initializers.
    """
    import hashlib

    import onnx
    from onnx import TensorProto
    import numpy as np

    onnx_path = Path(onnx_path)
    # IMPORTANT: load proto only. Loading external data would populate
    # raw_data fields; the matching save call would then re-emit them
    # via onnx's external-data writer, which has been observed to
    # drift initializer dtype/encoding for this graph. We need name
    # changes only — raw bytes must stay verbatim on disk.
    model = onnx.load(str(onnx_path), load_external_data=False)
    base_dir = onnx_path.parent

    def _sha(b: bytes) -> bytes:
        return hashlib.sha256(b).digest()

    # ------------------------------------------------------------------
    # Build the torch-side hash index. For each 2D parameter, we record
    # its raw bytes at the parameter's native dtype in two orientations:
    # the torch storage [out, in] and the transposed [in, out] that
    # ONNX MatMul uses. Dynamo writes weights into the ONNX graph in
    # whichever orientation the traced matmul consumes, so we must
    # match either layout. We do NOT cast across dtypes (fp16 <-> bf16
    # bytes are not equal even for the same numerical value); the
    # 2B bf16-hybrid recipe keeps each param in its declared dtype
    # (fp16 for wrapped Linears, fp32 for islands, bf16 for trunk).
    # ------------------------------------------------------------------
    _TORCH_TO_ONNX_DT = {
        torch.float32: TensorProto.FLOAT,
        torch.float16: TensorProto.FLOAT16,
        torch.bfloat16: TensorProto.BFLOAT16,
    }

    def _bytes_for(p: torch.Tensor) -> Optional[bytes]:
        """Raw bytes of a 2D torch tensor in its native dtype.

        bf16 has no numpy dtype; reinterpret as uint16 first, which
        matches ONNX's raw_data layout for BFLOAT16 initializers.
        """
        p_cpu = p.detach().cpu().contiguous()
        if p_cpu.dtype == torch.bfloat16:
            return p_cpu.view(torch.uint16).numpy().tobytes()
        if p_cpu.dtype in (torch.float16, torch.float32):
            return p_cpu.numpy().tobytes()
        return None

    # key: (onnx_dtype_enum, shape_tuple, sha256_digest) -> (canonical FQN,
    # transposed_flag). The transposed flag captures whether the ONNX
    # initializer (when this key matches) stored the weight in
    # ``[in_dim, out_dim]`` (ONNX MatMul convention) vs ``[out_dim, in_dim]``
    # (torch nn.Linear convention). The rename pass records this per
    # renamed FQN in a sidecar manifest so the runtime LoRA refit can
    # write deltas in the orientation TRT actually stored — dynamo's
    # MatMul keeps the transposed layout, torchscript's Gemm keeps torch.
    torch_hash_index: dict[tuple, tuple[str, bool]] = {}
    for name, p in decoder.named_parameters():
        if p.dim() != 2:
            continue
        canon = "decoder." + name.replace(".inner.weight", ".weight")
        onnx_dt = _TORCH_TO_ONNX_DT.get(p.dtype)
        if onnx_dt is None:
            continue
        # Original orientation [out, in] — matches torch nn.Linear.
        b_orig = _bytes_for(p)
        if b_orig is None:
            continue
        shape_orig = tuple(p.shape)
        torch_hash_index.setdefault(
            (onnx_dt, shape_orig, _sha(b_orig)), (canon, False),
        )
        # Transposed orientation [in, out] — matches ONNX MatMul layout.
        p_t = p.transpose(0, 1)
        b_trans = _bytes_for(p_t)
        if b_trans is not None:
            shape_trans = (shape_orig[1], shape_orig[0])
            torch_hash_index.setdefault(
                (onnx_dt, shape_trans, _sha(b_trans)), (canon, True),
            )

    def _read_external_bytes(init) -> Optional[bytes]:
        """Pull raw bytes for one initializer from its external_data file.

        Returns None for non-external initializers (no external_data
        records). Caller falls back to ``init.raw_data`` for those.
        """
        loc = None
        offset = 0
        length: Optional[int] = None
        for ed in init.external_data:
            if ed.key == "location":
                loc = ed.value
            elif ed.key == "offset":
                offset = int(ed.value)
            elif ed.key == "length":
                length = int(ed.value)
        if loc is None:
            return None
        ext_path = base_dir / loc
        with open(ext_path, "rb") as f:
            f.seek(offset)
            return f.read(length) if length is not None else f.read()

    used_names: set[str] = {init.name for init in model.graph.initializer}
    renamed = 0
    skipped_no_match = 0
    skipped_already_claimed = 0
    val_seen = 0
    val_inits_changed: dict[str, str] = {}
    # Per-FQN orientation: True if ONNX stored the matched weight in
    # ``[in_dim, out_dim]`` (transposed relative to torch nn.Linear),
    # which is what dynamo's MatMul writer does. False means orig
    # ``[out_dim, in_dim]`` (torchscript Gemm convention).
    transposed_fqns: list[str] = []
    claimed_torch: set[str] = set()
    float_dtypes = (TensorProto.BFLOAT16, TensorProto.FLOAT16, TensorProto.FLOAT)

    for init in model.graph.initializer:
        if not init.name.startswith("val_"):
            continue
        val_seen += 1
        dims = tuple(init.dims)
        if len(dims) != 2:
            continue
        nelem = int(np.prod(dims))
        if nelem < 16:
            continue
        if init.data_type not in float_dtypes:
            continue

        raw = _read_external_bytes(init)
        if raw is None:
            # Inline raw_data on a small init; bytes are already in proto.
            raw = bytes(init.raw_data) if init.raw_data else None
        if raw is None:
            continue

        expected_bytes = nelem * (4 if init.data_type == TensorProto.FLOAT else 2)
        if len(raw) != expected_bytes:
            skipped_no_match += 1
            continue

        key = (init.data_type, dims, _sha(raw))
        result = torch_hash_index.get(key)
        if result is None:
            skipped_no_match += 1
            continue
        canon, is_transposed = result
        if canon in claimed_torch or canon in used_names:
            skipped_already_claimed += 1
            continue
        val_inits_changed[init.name] = canon
        claimed_torch.add(canon)
        used_names.add(canon)
        if is_transposed:
            transposed_fqns.append(canon)
        renamed += 1

    # Apply the rename map. Touch initializer names and every node input
    # that referenced the old SSA name. Do NOT rewrite node.output[i]:
    # an initializer name is not a node output, and a blind rewrite
    # there could mutate an unrelated value if dynamo ever reused a
    # ``val_*`` name across the initializer pool and an intermediate
    # value pool.
    for init in model.graph.initializer:
        if init.name in val_inits_changed:
            init.name = val_inits_changed[init.name]
    for node in model.graph.node:
        for i, ref in enumerate(node.input):
            if ref in val_inits_changed:
                node.input[i] = val_inits_changed[ref]
    # Some dynamo-exported graphs also declare initializers as graph
    # value_info / graph.input entries (typed shape decls). Keep those
    # in sync with the rename.
    for vi in list(model.graph.input) + list(model.graph.value_info):
        if vi.name in val_inits_changed:
            vi.name = val_inits_changed[vi.name]

    # Save proto only — external-data files keep their original bytes.
    # We deliberately do NOT pass save_as_external_data here: that would
    # rebuild the .data file from raw_data fields (which are empty for
    # external initializers in our in-memory model anyway), and onnx's
    # writer has been observed to drift the per-initializer data_type
    # for this graph during the round trip. Proto-only save preserves
    # every external_data reference verbatim.
    onnx.save(model, str(onnx_path))

    # Emit the refit manifest next to the ONNX. TRTLoRAManager reads this
    # at runtime so it can transpose deltas before writing them into the
    # engine slot, which dynamo's MatMul builder stored in the transposed
    # ``[in_dim, out_dim]`` layout (vs torch nn.Linear's ``[out, in]``).
    import json
    manifest = {
        "version": 1,
        "onnx_path": str(onnx_path.name),
        "weights_transposed": sorted(transposed_fqns),
    }
    manifest_path = onnx_path.with_suffix(onnx_path.suffix + ".refit_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "ONNX val_* rename (proto-only, byte-hash): {} renamed, {} no_match, "
        "{} already_claimed (val_total={}). {} weights are transposed-layout "
        "(manifest: {}).",
        renamed, skipped_no_match, skipped_already_claimed, val_seen,
        len(transposed_fqns), manifest_path.name,
    )
    return renamed


def patch_decoder_onnx_dynamic_batch_reshapes(
    onnx_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    force: bool = False,
) -> Path:
    """Patch dynamo-exported reshape constants that accidentally bake B=1.

    The XL bf16 dynamo exporter can emit `Reshape` shape constants like
    ``[1, 6, 2560]`` even when dynamic batch was requested. TensorRT then
    builds an engine with a dynamic input profile, but `infer_shapes` fails
    at runtime for B>1. Rewriting the first dimension to ``-1`` preserves the
    non-batch shape and lets TRT infer the active batch from the input tensor.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path.with_name(f"{onnx_path.stem}_dynbatch{onnx_path.suffix}")
    output_path = Path(output_path)
    if output_path.parent != onnx_path.parent:
        raise ValueError(
            "Dynamic-batch patched ONNX must live next to the source ONNX "
            "so relative external_data references continue to resolve."
        )

    def _carry_refit_manifest() -> None:
        """Mirror the source's LoRA refit manifest next to the patched ONNX.

        Always runs (both cache-hit and rebuild branches) so the engine
        builder can find the manifest sidecar regardless of whether the
        dynbatch patch ran this invocation.
        """
        src_manifest = Path(str(onnx_path) + ".refit_manifest.json")
        if not src_manifest.is_file():
            return
        dst_manifest = Path(str(output_path) + ".refit_manifest.json")
        if (
            dst_manifest.is_file()
            and dst_manifest.stat().st_mtime >= src_manifest.stat().st_mtime
        ):
            return
        dst_manifest.write_text(
            src_manifest.read_text(encoding="utf-8"), encoding="utf-8",
        )
        logger.info("Carried refit manifest forward: {}", dst_manifest.name)

    if (
        output_path.exists()
        and not force
        and output_path.stat().st_mtime >= onnx_path.stat().st_mtime
    ):
        logger.info("Reusing dynamic-batch patched decoder ONNX: {}", output_path)
        _carry_refit_manifest()
        return output_path

    logger.info(
        "Patching decoder ONNX Reshape batch constants: {} -> {}",
        onnx_path, output_path,
    )
    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph

    def get_const_shape(name: str) -> list[int] | None:
        for node in graph.node:
            if node.op_type == "Constant" and name in node.output:
                for attr in node.attribute:
                    if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                        return numpy_helper.to_array(attr.t).flatten().tolist()
        for init in graph.initializer:
            if init.name == name:
                return numpy_helper.to_array(init).flatten().tolist()
        return None

    def set_const_shape(name: str, new_shape: list[int]) -> bool:
        new_arr = np.asarray(new_shape, dtype=np.int64)
        for node in graph.node:
            if node.op_type == "Constant" and name in node.output:
                for attr in node.attribute:
                    if attr.name == "value":
                        attr.t.CopyFrom(numpy_helper.from_array(new_arr, name=name))
                        return True
        for init in graph.initializer:
            if init.name == name:
                init.CopyFrom(numpy_helper.from_array(new_arr, name=name))
                return True
        return False

    seen: set[str] = set()
    matching = 0
    patched = 0
    examples: list[tuple[str, str, list[int]]] = []
    for node in graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shape_name = node.input[1]
        shape = get_const_shape(shape_name)
        if not shape or len(shape) < 2 or shape[0] != 1:
            continue
        matching += 1
        if len(examples) < 8:
            examples.append((node.name or "(no-name)", shape_name, shape))
        if -1 in shape[1:] or shape_name in seen:
            continue
        if set_const_shape(shape_name, [-1] + list(shape[1:])):
            patched += 1
            seen.add(shape_name)

    onnx.save(model, str(output_path))
    logger.info(
        "Patched {} unique Reshape constants ({} B=1 Reshapes found) in {}",
        patched, matching, output_path,
    )
    for node_name, shape_name, shape in examples:
        logger.info("  Reshape {} const {}: {}", node_name, shape_name, shape)
    _carry_refit_manifest()
    return output_path


# ------------------------------------------------------------------
# TensorRT engine build
# ------------------------------------------------------------------

@dataclass
class TRTBuildConfig:
    """Configuration for TensorRT engine build."""

    fp16: bool = True
    bf16: bool = False          # TRT 9.0+ on Ampere/Hopper
    tf32: bool = True           # TF32 for fp32 accumulation kernels

    workspace_gb: float = 4.0

    # Dynamic shape profiles: (min, optimal, max) per axis
    batch_min: int = 1
    batch_opt: int = 1
    batch_max: int = 4

    seq_min: int = 126          # ~5s, even
    seq_opt: int = 750          # 30s
    seq_max: int = 1500         # 60s

    enc_min: int = 32
    enc_opt: int = 200
    enc_max: int = 512

    # Builder optimization level (0-5, higher = slower build, faster engine)
    builder_optimization_level: int = 3

    # When True, TRT respects the dtypes in the ONNX graph exactly.
    # Use with mixed-precision ONNX export to ensure fp32 regions
    # (timestep embedding, AdaLN, norms) stay in fp32 while
    # attention/MLP run in fp16.
    strongly_typed: bool = False

    # Enable weight refitting.  Allows updating engine weights at runtime
    # via trt.Refitter without rebuilding.  Required for dynamic LoRA.
    # Slight engine size increase; negligible performance impact.
    refit: bool = False

    # DiT variant name, included in engine filename when not "turbo"
    # so engines from different checkpoints coexist in the same directory.
    variant: str = "turbo"

    # ONNX export precision recipe used to produce the parsed graph.
    # This is metadata-only for engine selection/rebuild decisions; TensorRT
    # precision flags are still governed by fp16/bf16/strongly_typed above.
    onnx_precision: str = "fp32"

    @property
    def max_duration_s(self) -> int:
        """Max duration in seconds, derived from seq_max at 25Hz."""
        return self.seq_max // 25

    def engine_filename(self) -> str:
        """Standardized engine filename.

        Format: ``spectral_decoder_{variant}_{prec}[_refit]_b{batch_max}_{dur}s.engine``;
        variant tag omitted for "turbo". The ``spectral_`` prefix marks
        engines that carry the steering input; pre-steering engines used
        ``decoder_*`` and the runtime rejects them.
        """
        if self.strongly_typed:
            # fp8_mixed gets its own tag so FP8 engines never collide
            # with the bf16/fp16-mixed engines built from the same
            # checkpoint. Everything else uses the legacy "mixed" tag
            # for backward compat with existing on-disk engines.
            if self.onnx_precision == "fp8_mixed":
                prec = "fp8"
            else:
                prec = "mixed"
        elif self.bf16:
            prec = "bf16"
        elif self.fp16:
            prec = "fp16"
        else:
            prec = "fp32"
        refit_tag = "_refit" if self.refit else ""
        dur = self.max_duration_s
        # Include variant in name for non-turbo models
        variant_tag = f"_{self.variant}" if self.variant != "turbo" else ""
        return f"spectral_decoder{variant_tag}_{prec}{refit_tag}_b{self.batch_max}_{dur}s.engine"


def build_trt_engine(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path],
    config: Optional[TRTBuildConfig] = None,
) -> Path:
    """Parse ONNX and build a TensorRT engine with dynamic shapes.

    Args:
        onnx_path: Path to the ONNX model.
        engine_path: Where to write the serialized TRT engine.
        config: Build configuration.

    Returns:
        Path to the written engine file.
    """
    import tensorrt as trt

    if config is None:
        config = TRTBuildConfig()

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)

    # TensorRT 10 networks are always explicit-batch; only opt into strong typing.
    net_flags = 0
    if config.strongly_typed and hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        net_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        logger.info("Using STRONGLY_TYPED network (precision from ONNX graph)")

    network = builder.create_network(net_flags)
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Parsing ONNX from {} ...", onnx_path)
    # Use parse_from_file so TRT resolves external data relative to the ONNX path
    onnx_abs = str(onnx_path.resolve())
    if not parser.parse_from_file(onnx_abs):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: {}", parser.get_error(i))
        raise RuntimeError("ONNX parsing failed")

    logger.info(
        "Network: {} inputs, {} outputs, {} layers",
        network.num_inputs, network.num_outputs, network.num_layers,
    )

    # Builder config
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(config.workspace_gb * (1 << 30)),
    )

    # Precision flags. STRONGLY_TYPED mode forbids FP16/BF16 flags
    # (TRT enforces this with an API error: kBF16 must not be set when
    # strongly_typed). The dtypes are baked into the ONNX graph instead.
    # TF32 is still allowed under strongly_typed.
    if not config.strongly_typed:
        if config.fp16:
            build_config.set_flag(trt.BuilderFlag.FP16)
        if config.bf16 and hasattr(trt.BuilderFlag, "BF16"):
            build_config.set_flag(trt.BuilderFlag.BF16)

    if config.tf32:
        build_config.set_flag(trt.BuilderFlag.TF32)

    if config.refit:
        build_config.set_flag(trt.BuilderFlag.REFIT)
        logger.info("REFIT enabled: engine weights can be updated at runtime")

    if hasattr(build_config, "builder_optimization_level"):
        build_config.builder_optimization_level = config.builder_optimization_level

    # Optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()

    Bmin, Bopt, Bmax = config.batch_min, config.batch_opt, config.batch_max
    Smin, Sopt, Smax = config.seq_min, config.seq_opt, config.seq_max
    Emin, Eopt, Emax = config.enc_min, config.enc_opt, config.enc_max

    profile.set_shape(
        "hidden_states",
        min=(Bmin, Smin, 64), opt=(Bopt, Sopt, 64), max=(Bmax, Smax, 64),
    )
    profile.set_shape(
        "timestep",
        min=(Bmin,), opt=(Bopt,), max=(Bmax,),
    )
    profile.set_shape(
        "encoder_hidden_states",
        min=(Bmin, Emin, 2048), opt=(Bopt, Eopt, 2048), max=(Bmax, Emax, 2048),
    )
    profile.set_shape(
        "context_latents",
        min=(Bmin, Smin, 128), opt=(Bopt, Sopt, 128), max=(Bmax, Smax, 128),
    )

    # Read L, D off the parsed network so this stays variant-agnostic.
    steering_idx = None
    for ti in range(network.num_inputs):
        if network.get_input(ti).name == "steering":
            steering_idx = ti
            break
    if steering_idx is None:
        raise RuntimeError(
            f"Decoder ONNX at {onnx_path} is missing the 'steering' input "
            "(it predates spectral steering). Delete it and re-run the "
            "build to fetch/export a current one, or re-run with "
            "--export-locally to export from your local checkpoint."
        )
    steering_shape = tuple(network.get_input(steering_idx).shape)
    if len(steering_shape) != 3:
        raise RuntimeError(f"Unexpected steering input rank: {steering_shape}")
    _, steer_L, steer_D = steering_shape
    profile.set_shape(
        "steering",
        min=(Bmin, steer_L, steer_D),
        opt=(Bopt, steer_L, steer_D),
        max=(Bmax, steer_L, steer_D),
    )

    profile_idx = build_config.add_optimization_profile(profile)
    if profile_idx < 0:
        raise RuntimeError("Failed to add TensorRT optimization profile")

    logger.info(
        "Building TRT engine (fp16={}, bf16={}, opt_level={}) ...",
        config.fp16, config.bf16, config.builder_optimization_level,
    )
    logger.info(
        "  Profiles: B=[{},{},{}]  T=[{},{},{}]  L_enc=[{},{},{}]",
        Bmin, Bopt, Bmax, Smin, Sopt, Smax, Emin, Eopt, Emax,
    )

    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = engine_path.stat().st_size / (1 << 20)
    logger.info("Engine saved to {} ({:.1f} MB)", engine_path, size_mb)
    return engine_path


# ------------------------------------------------------------------
# Validation helper
# ------------------------------------------------------------------

@torch.no_grad()
def validate_trt_vs_pytorch(
    model,
    engine_path: Union[str, Path],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seq_len: int = 750,
    enc_len: int = 200,
    seed: int = 42,
) -> dict:
    """Compare TRT decoder output against PyTorch decoder output.

    Returns a dict with per-element statistics so you can gauge accuracy.
    """
    from .runtime import TRTDecoder

    torch.manual_seed(seed)
    B = 1

    hidden_states = torch.randn(B, seq_len, 64, device=device, dtype=dtype)
    timestep = torch.tensor([0.75], device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(B, enc_len, 2048, device=device, dtype=dtype)
    context_latents = torch.randn(B, seq_len, 128, device=device, dtype=dtype)

    # PyTorch reference
    model.decoder.eval()
    with torch.no_grad():
        pt_out = model.decoder(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            context_latents=context_latents,
            use_cache=False,
        )[0]

    # TRT
    trt_decoder = TRTDecoder(engine_path)
    trt_out = trt_decoder(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        context_latents=context_latents,
    )

    # Compare
    diff = (pt_out.float() - trt_out.float()).abs()
    rel_diff = diff / (pt_out.float().abs() + 1e-8)

    results = {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "pt_mean": pt_out.float().mean().item(),
        "trt_mean": trt_out.float().mean().item(),
        "pt_std": pt_out.float().std().item(),
        "trt_std": trt_out.float().std().item(),
        "cosine_sim": torch.nn.functional.cosine_similarity(
            pt_out.float().flatten().unsqueeze(0),
            trt_out.float().flatten().unsqueeze(0),
        ).item(),
    }

    logger.info("Validation results:")
    for k, v in results.items():
        logger.info("  {:<20}: {:.6f}", k, v)

    return results
