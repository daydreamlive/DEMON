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

            # Layer loop: static branching on layer_types (config, not runtime)
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
                hidden_states = layer_outputs[0]

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
    ) -> torch.Tensor:
        outputs = self.decoder(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            context_latents=context_latents,
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

    example_inputs = (
        torch.randn(B, T, 64, device=device, dtype=trace_dtype),
        torch.full((B,), 0.5, device=device, dtype=ts_dtype),
        torch.randn(B, L, 2048, device=device, dtype=trace_dtype),
        torch.randn(B, T, 128, device=device, dtype=trace_dtype),
    )

    input_names = [
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "context_latents",
    ]
    output_names = ["velocity"]

    dynamic_axes = {
        "hidden_states":          {0: "batch", 1: "seq_len"},
        "timestep":               {0: "batch"},
        "encoder_hidden_states":  {0: "batch", 1: "enc_len"},
        "context_latents":        {0: "batch", 1: "seq_len"},
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

    size_mb = onnx_path.stat().st_size / (1 << 20)
    logger.info("ONNX saved to {} ({:.1f} MB)", onnx_path, size_mb)
    return onnx_path


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
    if (
        output_path.exists()
        and not force
        and output_path.stat().st_mtime >= onnx_path.stat().st_mtime
    ):
        logger.info("Reusing dynamic-batch patched decoder ONNX: {}", output_path)
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
        """Generate a standardized engine filename from build config.

        Format: decoder_{variant}_{precision}[_refit]_b{batch_max}_{duration}s.engine
        The variant tag is omitted for "turbo" (backward compat).
        Uses seconds so naming is stable across frame rates.
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
        return f"decoder{variant_tag}_{prec}{refit_tag}_b{self.batch_max}_{dur}s.engine"


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
