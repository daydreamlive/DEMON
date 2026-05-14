"""Insert FP8 QDQ into an existing bf16 decoder ONNX (hand-rolled).

Why we don't use ModelOpt
-------------------------
We previously routed through ``modelopt.onnx.quantization.quantize`` with
``dq_only=True``. On the XL DiT graph it silently emitted only 4
``DequantizeLinear`` nodes and 2 FP8 weight initializers — leaving 351
of the 353 MatMul-with-initializer-weight candidates entirely untouched.
The resulting "FP8 engine" was effectively fp16 (because of the
bf16 -> fp16 cast we had to do for ORT's CUDA EP) plus two FP8
stragglers, and produced NaN at runtime from fp16 overflow in the
38-layer DiT residual stream. Diagnostics: ``benchmarks-pr17/
fp8_clamp_diagnostic.json`` and ``fp8_scales_diagnostic.json``.

What this module does instead
-----------------------------
Walks the bf16 ONNX directly. For every ``MatMul`` node whose
``input[1]`` is a bf16 initializer (the standard ``nn.Linear`` weight
pattern) and whose name doesn't match an exclusion pattern
(time_embed):

  1. Decode the bf16 weight to fp32 in PyTorch.
  2. Compute a per-output-channel absmax scale
     ``scale = max(|w|, axis=all-but-last) / FP8_E4M3_MAX``.
  3. Quantize via ``tensor.to(torch.float8_e4m3fn)`` (PyTorch handles
     saturating round-to-nearest-even for E4M3FN).
  4. Replace the initializer's ``data_type``/``raw_data`` with FP8.
  5. Add a sibling per-output-channel bf16 scale initializer.
  6. Add a scalar FP8 zero_point (value 0).
  7. Insert ``DequantizeLinear(weight, scale, zero_point)`` with
     ``axis=-1`` and rewire the MatMul's ``input[1]`` to its output.

Bulk activations stay bf16. The DiT MatMuls run as
``bf16 activation @ DQ(fp8 weight) -> bf16``, which TRT 10.16's
strongly-typed parser maps onto Blackwell FP8 GEMM tactics.

Refit interaction
-----------------
Each rewritten Linear now exposes two named tensors: the original
initializer name (now FP8 bytes) and a new ``<name>_fp8_scale``
initializer. The LoRA refit path in ``lora_refit.py`` needs both to
reconstruct fp32 deltas from a LoRA adapter. The mapping is recorded
in the FP8 manifest written next to the patched ONNX.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from loguru import logger


# FP8 E4M3FN: max finite magnitude is 448.0. There is no Inf; NaN is a
# single bit pattern (0x7F / 0xFF). PyTorch's
# ``torch.float8_e4m3fn`` cast handles saturating round-to-nearest-even.
FP8_E4M3_MAX = 448.0

# Floor for per-output-channel absmax. A literal zero-row in the weight
# would produce scale=0 and turn DequantizeLinear into 0/0. Clamping the
# absmax to a tiny positive value keeps the scale finite; the row is
# already zero in fp8 so dequant gives 0 regardless of scale.
_ABSMAX_FLOOR = 1e-12


@dataclass
class FP8OnnxConfig:
    """Configuration for FP8 QDQ insertion.

    Most fields are vestigial from the ModelOpt-driven implementation
    that this module replaces. They're kept for API compatibility so
    callers in ``build.py`` don't need to change.
    """
    op_types_to_quantize: tuple[str, ...] = ("MatMul",)
    high_precision_dtype: str = "bf16"
    mha_accumulation_dtype: str = "bf16"
    dq_only: bool = True
    # Unused: weight-only FP8 needs no calibration.
    calibration_eps: tuple[str, ...] = ()
    calibration_batch: int = 4
    use_calibration_shapes: bool = True
    enc_len_for_calibration: int = 200
    # FP8 needs opset >= 19. The XL bf16 export already targets 20.
    opset: int = 20
    log_level: str = "INFO"


# ------------------------------------------------------------------
# Exclusion patterns: weight initializers we keep at bf16.
# ------------------------------------------------------------------

# The dynamo exporter preserves these names. All other Linear weights
# in the XL DiT graph are anonymous ``val_NNN`` initializers (the
# exporter constant-folded them), so name-based exclusion only matches
# the time-conditioning Linears here. That is the intended scope: we
# want every DiT bulk MatMul quantized.
_EXCLUDE_INITIALIZER_PATTERNS = (
    "decoder.time_embed",       # time_embed.linear_1/2/time_proj
    "decoder.time_embed_r",     # time_embed_r.linear_1/2/time_proj
)


def _is_excluded_init_name(name: str) -> bool:
    return any(pat in name for pat in _EXCLUDE_INITIALIZER_PATTERNS)


# ------------------------------------------------------------------
# Core quantization
# ------------------------------------------------------------------

def _quantize_weight_e4m3fn(w_fp32, *, eps: float = _ABSMAX_FLOOR):
    """Per-output-channel symmetric FP8 E4M3FN quantization.

    Input: fp32 tensor with output channels along the LAST axis.
    Output: (fp8_bytes, scale_fp32_tensor) where fp8_bytes is the
    little-endian FP8 storage in row-major order and scale_fp32_tensor
    is a 1D tensor of length ``w_fp32.shape[-1]`` holding per-channel
    scales (to be saved as bf16 next to the FP8 initializer).
    """
    import torch

    if w_fp32.ndim < 2:
        # Defensive: 1D weights don't appear in the XL DiT, but if
        # they ever do, treat the single axis as "the output channel"
        # and use a scalar scale equal to the global absmax.
        absmax = w_fp32.abs().amax()
        scale = (absmax.clamp(min=eps) / FP8_E4M3_MAX).reshape(1)
        scaled = w_fp32 / scale
    else:
        # Output channels = last axis. Reduce absmax across every other
        # axis so the scale shape is exactly the last dim.
        reduce_axes = tuple(range(w_fp32.ndim - 1))
        absmax = w_fp32.abs().amax(dim=reduce_axes)
        scale = absmax.clamp(min=eps) / FP8_E4M3_MAX
        # Broadcast for division: scale gets a leading-1 on every
        # non-output axis.
        bcast_shape = (1,) * (w_fp32.ndim - 1) + (scale.shape[0],)
        scaled = w_fp32 / scale.view(bcast_shape)

    # torch.float8_e4m3fn cast saturates at +/-448 and rounds to nearest
    # even. The NaN bit pattern only appears for inputs that are already
    # NaN, which a properly-trained weight tensor never has.
    w_fp8 = scaled.to(torch.float8_e4m3fn).contiguous()
    fp8_bytes = w_fp8.view(torch.uint8).numpy().tobytes()
    return fp8_bytes, scale.contiguous()


def _weight_l2_bf16(init) -> float:
    """L2 norm of a bf16 initializer's values, after upcast to fp32.

    Matches the signature computed by collect_activation_absmax.py on
    the PyTorch side (bf16 cast then fp32 upcast). Used as a unique
    identifier for matching ONNX weights back to PyTorch Linears.
    """
    import torch

    if not init.raw_data:
        raise NotImplementedError(f"bf16 init {init.name} has no raw_data")
    t = torch.frombuffer(bytearray(init.raw_data), dtype=torch.bfloat16)
    return float(t.to(torch.float32).pow(2).sum().sqrt().item())


def _make_scalar_scale_initializer(name: str, scale_value: float) -> "onnx.TensorProto":
    """Build a scalar bf16 scale initializer (0-D)."""
    import onnx
    import torch

    t = torch.tensor(scale_value, dtype=torch.bfloat16)
    raw = t.view(torch.uint16).numpy().tobytes()
    init = onnx.TensorProto()
    init.name = name
    init.data_type = int(onnx.TensorProto.BFLOAT16)
    # Empty dims = 0-D scalar.
    init.raw_data = raw
    return init


def _make_scalar_fp8_zero_point_initializer(name: str) -> "onnx.TensorProto":
    """Scalar (0-D) FP8 E4M3FN zero_point with value 0."""
    import onnx

    init = onnx.TensorProto()
    init.name = name
    init.data_type = int(onnx.TensorProto.FLOAT8E4M3FN)
    init.raw_data = bytes([0])
    return init


def _make_q_node(
    input_name: str,
    scale_name: str,
    zp_name: str,
    output_name: str,
    *,
    node_name: str,
) -> "onnx.NodeProto":
    import onnx
    from onnx import helper

    return helper.make_node(
        "QuantizeLinear",
        inputs=[input_name, scale_name, zp_name],
        outputs=[output_name],
        name=node_name,
    )


def _make_inv_s_initializer(name: str, inv_s_fp32) -> "onnx.TensorProto":
    """bf16 initializer for ``1/s`` used in the SmoothQuant Mul.

    Shape: ``[in_features]``. The Mul broadcasts this over the batch
    and sequence dims of the activation feeding the consumer Linear.
    """
    import onnx
    import torch

    inv_s_bf16 = inv_s_fp32.to(torch.bfloat16).contiguous()
    raw = inv_s_bf16.view(torch.uint16).numpy().tobytes()
    init = onnx.TensorProto()
    init.name = name
    init.data_type = int(onnx.TensorProto.BFLOAT16)
    init.dims.extend(list(inv_s_bf16.shape))
    init.raw_data = raw
    return init


def _make_mul_node(
    input_a: str,
    input_b: str,
    output: str,
    *,
    node_name: str,
) -> "onnx.NodeProto":
    from onnx import helper

    return helper.make_node(
        "Mul",
        inputs=[input_a, input_b],
        outputs=[output],
        name=node_name,
    )


def _make_scale_initializer(name: str, scale_fp32) -> "onnx.TensorProto":
    """Build a bf16 scale initializer from a 1D fp32 scale tensor.

    bf16 is chosen over fp32 because the surrounding graph is bf16 — a
    bf16 scale lets DequantizeLinear's output dtype match the MatMul's
    activation input without an extra Cast.
    """
    import onnx
    import torch

    scale_bf16 = scale_fp32.to(torch.bfloat16).contiguous()
    raw = scale_bf16.view(torch.uint16).numpy().tobytes()
    init = onnx.TensorProto()
    init.name = name
    init.data_type = int(onnx.TensorProto.BFLOAT16)
    init.dims.extend(list(scale_fp32.shape))
    init.raw_data = raw
    return init


def _make_fp8_zero_point_initializer(name: str) -> "onnx.TensorProto":
    """Scalar FP8 E4M3FN zero_point initializer with value 0.

    DequantizeLinear's zero_point input must match the input's dtype
    (FLOAT8E4M3FN here). Using a scalar zero_point shared by every
    channel is correct for symmetric quantization.
    """
    import onnx

    init = onnx.TensorProto()
    init.name = name
    init.data_type = int(onnx.TensorProto.FLOAT8E4M3FN)
    # Scalar: empty dims, single zero byte.
    init.raw_data = bytes([0])
    return init


def _make_dq_node(
    weight_name: str,
    scale_name: str,
    zp_name: str,
    output_name: str,
    *,
    axis: int,
    node_name: str,
) -> "onnx.NodeProto":
    import onnx
    from onnx import helper

    return helper.make_node(
        "DequantizeLinear",
        inputs=[weight_name, scale_name, zp_name],
        outputs=[output_name],
        name=node_name,
        axis=axis,
    )


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

_VALID_PERCENTILE_FIELDS = ("absmax", "p99", "p99_9", "p99_99")


def _smoothquant_factor(act_per_chan, weight_per_in_chan, *, alpha: float,
                        clamp_min: float = 1.0, clamp_max: float = 1e3):
    """Compute SmoothQuant per-channel factor ``s``.

    ``s[c] = (max(|X[..., c]|))^alpha / (max(|W[c, :]|))^(1-alpha)``

    where X is the activation feeding the Linear (in_features in the
    last dim) and W is the weight in ONNX layout ``[in, out]`` (so
    ``max(|W[c, :]|)`` is the absmax over the c-th input row of W).

    ``clamp_min`` defaults to 1.0 so that smoothing only ever DIVIDES
    the activation (s >= 1 ⇒ act/s <= act, weight*s >= weight). Without
    this clamp, channels where the weight magnitude exceeds the
    activation magnitude would produce ``s < 1``, which would
    AMPLIFY the activation — defeating the entire point of SmoothQuant
    and degrading numerics on those channels. ``clamp_max`` bounds the
    weight inflation from extreme outlier activations.
    """
    import torch

    a = torch.as_tensor(act_per_chan, dtype=torch.float32).clamp(min=1e-6)
    w = torch.as_tensor(weight_per_in_chan, dtype=torch.float32).clamp(min=1e-6)
    s = (a ** alpha) / (w ** (1.0 - alpha))
    s = s.clamp(min=clamp_min, max=clamp_max)
    return s


def _load_activation_absmax(
    json_path: Path,
    *,
    percentile_field: str = "absmax",
    outlier_skip_ratio: float = 0.0,
) -> tuple[dict[tuple, list[dict]], dict]:
    """Build a lookup from (transposed_shape, l2_bf16_rounded) -> [linear_record, ...].

    ONNX MatMul weight initializers have the PyTorch ``nn.Linear.weight``
    *transposed* (PyTorch stores [out, in]; ONNX MatMul reads
    [in, out]). The L2 norm is invariant under transpose so we use
    ``(transposed_shape, l2)`` as the lookup key. The L2 is rounded to a
    handful of significant figures because bf16 quantization noise can
    perturb the last few bits.

    ``percentile_field`` chooses which stored statistic drives the
    scale. "absmax" is the literal maximum (sensitive to outliers),
    "p99"/"p99_9"/"p99_99" are tail quantiles — recommended for
    DiT/transformer activations that have outlier-heavy distributions.
    """
    if percentile_field not in _VALID_PERCENTILE_FIELDS:
        raise ValueError(
            f"percentile_field must be one of {_VALID_PERCENTILE_FIELDS}; "
            f"got {percentile_field!r}"
        )
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    linears = raw["linears"]
    lookup: dict[tuple, list[dict]] = {}
    for path, rec in linears.items():
        torch_shape = tuple(rec["weight_shape"])      # PyTorch [out, in]
        if len(torch_shape) != 2:
            continue
        onnx_shape = (torch_shape[1], torch_shape[0])  # ONNX [in, out]
        # Round L2 to 3 decimal places. We previously rounded to 4, but
        # PyTorch's sum-of-squares and ONNX's bytes-loaded-and-summed
        # produce results that diverge by ~1 ULP at the 4th decimal —
        # e.g. PyTorch 285.4444 vs ONNX 285.4443 for the same weight.
        # That mismatch silently routed 23 weights (including layer 16
        # mlp.down_proj, the worst outlier in the whole model) to the
        # W8A16 fallback path, completely bypassing both the W8A8 quant
        # AND any SmoothQuant treatment. 3 decimals is well within the
        # L2 separation between distinct Linears (~0.01) so collisions
        # remain effectively impossible.
        l2_key = round(rec["weight_l2_bf16"], 3)
        key = (onnx_shape, l2_key)
        amax_for_scale = rec.get(percentile_field, rec["absmax"])
        # Older JSONs may not have percentile fields; fall back if so.
        if amax_for_scale is None or amax_for_scale <= 0:
            amax_for_scale = rec["absmax"]
        # Outlier ratio uses p99.9 as a stable denominator. Layers with
        # large ratios have a "massive activation" pattern (a tiny number
        # of huge values dominating absmax) and benefit from staying on
        # the bf16 activation path — clipping their outliers destroys
        # load-bearing signal.
        p999 = rec.get("p99_9", rec["absmax"])
        outlier_ratio = (rec["absmax"] / p999) if p999 > 0 else 1.0
        skip_activation_quant = (
            outlier_skip_ratio > 0.0 and outlier_ratio > outlier_skip_ratio
        )
        lookup.setdefault(key, []).append({
            "linear_path": path,
            "absmax": rec["absmax"],
            "p99_9": p999,
            "outlier_ratio": outlier_ratio,
            "skip_activation_quant": skip_activation_quant,
            "scale_amax": amax_for_scale,  # what to use for the FP8 scale
            "per_channel_absmax": rec.get("per_channel_absmax"),
            "weight_l2_bf16": rec["weight_l2_bf16"],
        })
    return lookup, raw


def patch_bf16_onnx_to_fp8(
    bf16_onnx_path: Union[str, Path],
    calibration_npz_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    *,
    config: Optional[FP8OnnxConfig] = None,
    force: bool = False,
    activation_absmax_json_path: Optional[Union[str, Path]] = None,
    activation_percentile: str = "absmax",
    activation_outlier_skip_ratio: float = 0.0,
    smoothquant_alpha: float = 0.0,
    quantize_attention: bool = False,
    attention_softmax_max: float = 1.05,
    attention_generic_max: float = 10.0,
) -> Path:
    """Patch a bf16 decoder ONNX with FP8 (E4M3FN).

    When ``activation_absmax_json_path`` is None, runs in weight-only
    mode (W8A16): per-output-channel symmetric quant on each MatMul's
    weight initializer, activations stay bf16.

    When ``activation_absmax_json_path`` points at a JSON written by
    ``scripts/collect_activation_absmax.py``, runs in W8A8 mode: every
    MatMul activation input gets a per-tensor symmetric Q->DQ chain
    scaled by ``activation_absmax / FP8_E4M3_MAX``. This is required
    for TRT 10.x to pick FP8 GEMM tactics; weight-only by itself
    leaves the GEMM as bf16 with a free dequant.

    ``calibration_npz_path`` is accepted for backwards compatibility
    with the build wiring but is unused.
    """
    if config is None:
        config = FP8OnnxConfig()

    src = Path(bf16_onnx_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"bf16 ONNX not found: {src}")

    if output_path is None:
        output_path = src.with_name(src.stem + "_fp8" + src.suffix)
    output_path = Path(output_path).resolve()
    if output_path.parent != src.parent:
        raise ValueError(
            "FP8 patched ONNX must be a sibling of the source "
            "(external_data references are relative paths)."
        )

    # Build a name->output_absmax map keyed by (transposed_shape, L2)
    # for the attention quantization path. Populated from the same JSON
    # as activation_lookup (each Linear record has output_absmax).
    output_amax_lookup: dict[tuple, list[float]] = {}

    activation_lookup: dict[tuple, list[dict]] | None = None
    activation_meta: dict | None = None
    if activation_absmax_json_path is not None:
        amax_path = Path(activation_absmax_json_path).resolve()
        if not amax_path.exists():
            raise FileNotFoundError(f"Activation absmax JSON not found: {amax_path}")
        activation_lookup, activation_meta = _load_activation_absmax(
            amax_path,
            percentile_field=activation_percentile,
            outlier_skip_ratio=activation_outlier_skip_ratio,
        )
        # Index Linear OUTPUT absmax by the same (onnx_shape, L2) key used
        # for matching weights. Used by the attention quantization path.
        for path, rec in activation_meta.get("linears", {}).items():
            torch_shape = tuple(rec.get("weight_shape") or ())
            if len(torch_shape) != 2:
                continue
            onnx_shape = (torch_shape[1], torch_shape[0])
            l2_key = round(rec.get("weight_l2_bf16", 0.0), 3)
            out_amax = rec.get("output_absmax", 0.0)
            output_amax_lookup.setdefault((onnx_shape, l2_key), []).append(out_amax)
        n_skip = sum(
            1 for recs in activation_lookup.values()
            for r in recs if r["skip_activation_quant"]
        )
        logger.info(
            "Loaded activation absmax JSON ({} linears, {} unique shape+L2 keys, "
            "scale field={!r}, outlier_skip_ratio={}, fallback-to-W8A16={})",
            len(activation_meta.get("linears", {})), len(activation_lookup),
            activation_percentile, activation_outlier_skip_ratio, n_skip,
        )
        if (
            output_path.exists()
            and not force
            and output_path.stat().st_mtime >= src.stat().st_mtime
            and output_path.stat().st_mtime >= amax_path.stat().st_mtime
        ):
            logger.info("Reusing FP8 ONNX (newer than source + absmax JSON): {}", output_path)
            return output_path
    else:
        if (
            output_path.exists()
            and not force
            and output_path.stat().st_mtime >= src.stat().st_mtime
        ):
            logger.info("Reusing FP8 ONNX (newer than source): {}", output_path)
            return output_path

    if calibration_npz_path:
        logger.info(
            "FP8 patch ignores --calibration-npz; use --activation-absmax-json "
            "for W8A8 or omit it for weight-only mode."
        )

    import onnx
    import torch

    mode_label = "W8A8 (full)" if activation_lookup is not None else "W8A16 (weight-only)"
    logger.info("=" * 60)
    logger.info("FP8 QDQ INSERTION ({}, hand-rolled)", mode_label)
    logger.info("=" * 60)
    logger.info("  source: {}", src)
    logger.info("  output: {}", output_path)
    logger.info("  weight scheme: per-output-channel symmetric E4M3FN, scale=bf16")
    if activation_lookup is not None:
        logger.info(
            "  activation scheme: per-tensor symmetric E4M3FN (scale=bf16, "
            "amax field={!r}, outlier_skip_ratio={}, smoothquant_alpha={})",
            activation_percentile, activation_outlier_skip_ratio,
            smoothquant_alpha,
        )

    logger.info("Loading bf16 ONNX (with external data) ...")
    model = onnx.load(str(src), load_external_data=True)
    g = model.graph

    inits = {i.name: i for i in g.initializer}
    BF16 = int(onnx.TensorProto.BFLOAT16)

    # Index MatMul nodes by their weight-init input.
    matmul_consumers: dict[str, list] = {}
    for node in g.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue
        w = node.input[1]
        if w in inits:
            matmul_consumers.setdefault(w, []).append(node)

    candidates_total = len(matmul_consumers)
    excluded_by_name = []
    bad_dtype = []
    bad_ndim = []
    to_quantize: list[tuple[str, list]] = []  # [(weight_name, [matmul_nodes])]
    for weight_name, nodes in matmul_consumers.items():
        init = inits[weight_name]
        if _is_excluded_init_name(weight_name):
            excluded_by_name.append(weight_name)
            continue
        if init.data_type != BF16:
            bad_dtype.append((weight_name, init.data_type))
            continue
        if len(init.dims) != 2:
            bad_ndim.append((weight_name, list(init.dims)))
            continue
        to_quantize.append((weight_name, nodes))

    logger.info(
        "Candidates: total={} excluded_by_name={} non_bf16={} non_2d={} "
        "to_quantize={}",
        candidates_total, len(excluded_by_name),
        len(bad_dtype), len(bad_ndim), len(to_quantize),
    )

    if not to_quantize:
        raise RuntimeError(
            "No MatMul-with-initializer-weight nodes matched the FP8 patch "
            "criteria. Check the source ONNX and exclusion patterns."
        )

    new_inits: list["onnx.TensorProto"] = []
    new_nodes: list["onnx.NodeProto"] = []
    quantized_log: list[dict] = []

    total_weight_bytes_in = 0
    total_weight_bytes_out = 0

    # For W8A8 we record each MatMul's activation amax so we can run
    # the activation Q->DQ pass after the weight loop. The amax is
    # looked up from the activation JSON by (transposed_shape, L2).
    matmul_activation_amax: dict[str, float] = {}  # node name -> amax
    matmul_to_linear_path: dict[str, str] = {}     # node name -> source linear path
    unmatched_in_lookup: list[str] = []
    smoothquant_log: list[dict] = []
    smoothquant_skipped_no_perchan: list[str] = []

    for weight_name, consumer_nodes in to_quantize:
        init = inits[weight_name]
        # Decode bf16 raw_data -> fp32 (lossless: bf16 is fp32-with-truncated-mantissa).
        raw = init.raw_data
        if not raw:
            raise NotImplementedError(
                f"bf16 init {weight_name} has no raw_data"
            )
        t_bf16 = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        w_fp32 = t_bf16.to(torch.float32).reshape(tuple(init.dims))

        # If we're in W8A8 mode, look up the matching PyTorch Linear
        # for this weight init so we can find its activation amax.
        # The lookup key is (onnx_shape, L2_bf16). Collisions are
        # disambiguated by greedy first-match within the bucket and
        # remove (each Linear matches at most one ONNX init).
        # ``scale_amax`` is already the chosen percentile (or absmax).
        sq_applied = False  # was SmoothQuant applied to this Linear's weight?
        if activation_lookup is not None:
            onnx_shape = tuple(init.dims)
            l2 = round(_weight_l2_bf16(init), 3)
            bucket = activation_lookup.get((onnx_shape, l2))
            if not bucket:
                unmatched_in_lookup.append(weight_name)
                amax = None
                linear_path = None
            else:
                rec = bucket.pop(0)
                linear_path = rec["linear_path"]
                if rec["skip_activation_quant"]:
                    # Outlier-heavy layer: skip activation Q-DQ. The
                    # MatMul stays at bf16 activation × FP8 weight DQ
                    # (W8A16). TRT won't pick FP8 GEMM for this one,
                    # but bulk numerics on the outlier-dominated layer
                    # are preserved.
                    amax = None
                else:
                    amax = rec["scale_amax"]

                # SmoothQuant: move outlier magnitude from per-tensor
                # activation to per-output-channel weight by scaling
                # weight rows by ``s[c]`` and inserting a Mul(act, 1/s)
                # before the MatMul. The Q-DQ activation chain then
                # operates on the SMOOTHED activation (well-behaved
                # per-tensor distribution), unlocking real precision
                # on the bulk while preserving outlier signal in the
                # per-channel-quantized weight.
                #
                # Only applied when:
                #   - alpha > 0
                #   - activation Q-DQ is enabled for this Linear (i.e.
                #     not on the outlier-skip W8A16 fallback path)
                #   - per_channel_absmax exists and matches in_features
                if (
                    smoothquant_alpha > 0.0
                    and amax is not None
                    and rec.get("per_channel_absmax") is not None
                    and len(rec["per_channel_absmax"]) == init.dims[0]
                ):
                    per_chan_act = rec["per_channel_absmax"]
                    # Per-input-channel weight absmax (max over output dim).
                    weight_per_in_chan = w_fp32.abs().amax(dim=1)  # [in]
                    s = _smoothquant_factor(
                        per_chan_act, weight_per_in_chan,
                        alpha=smoothquant_alpha,
                    )
                    # Apply smoothing to the weight (broadcast [in,1] over [in,out]).
                    w_fp32 = w_fp32 * s.unsqueeze(1)
                    # Insert Mul(act, 1/s) before each consumer MatMul.
                    inv_s = (1.0 / s).contiguous()
                    inv_s_name = f"{weight_name}_sq_inv_s"
                    new_inits.append(_make_inv_s_initializer(inv_s_name, inv_s))
                    for mm_idx, mm in enumerate(consumer_nodes):
                        act_tensor = mm.input[0]
                        tag = act_tensor.replace("/", "_").replace(":", "_")
                        mul_out = f"{weight_name}_sq_act_{mm_idx}"
                        mul_name = f"{weight_name}_sq_Mul_{mm_idx}"
                        new_nodes.append(_make_mul_node(
                            act_tensor, inv_s_name, mul_out,
                            node_name=mul_name,
                        ))
                        # Rewire the MatMul to read the SMOOTHED activation.
                        mm.input[0] = mul_out
                    # The smoothed activation's per-tensor amax is the
                    # max across channels of (a_max[c] / s[c]).
                    smoothed_per_chan = (
                        torch.as_tensor(per_chan_act, dtype=torch.float32) / s
                    )
                    amax = float(smoothed_per_chan.max().item())
                    sq_applied = True
                    smoothquant_log.append({
                        "weight": weight_name,
                        "linear_path": linear_path,
                        "shape": list(init.dims),
                        "alpha": smoothquant_alpha,
                        "s_min": float(s.min().item()),
                        "s_max": float(s.max().item()),
                        "s_mean": float(s.mean().item()),
                        "original_act_amax": rec.get("absmax"),
                        "smoothed_act_amax": amax,
                        "act_amax_reduction": (
                            rec["absmax"] / amax if amax > 0 else None
                        ),
                    })
                elif (
                    smoothquant_alpha > 0.0
                    and amax is not None
                    and rec.get("per_channel_absmax") is None
                ):
                    smoothquant_skipped_no_perchan.append(weight_name)

            for mm in consumer_nodes:
                if amax is not None:
                    matmul_activation_amax[mm.name] = amax
                    matmul_to_linear_path[mm.name] = linear_path

        fp8_bytes, scale_fp32 = _quantize_weight_e4m3fn(w_fp32)

        # Rewrite the initializer in place to FP8.
        total_weight_bytes_in += len(raw)
        init.data_type = int(onnx.TensorProto.FLOAT8E4M3FN)
        init.raw_data = fp8_bytes
        total_weight_bytes_out += len(fp8_bytes)

        # Build sibling scale + zero_point initializers.
        scale_name = f"{weight_name}_fp8_scale"
        zp_name = f"{weight_name}_fp8_zp"
        scale_init = _make_scale_initializer(scale_name, scale_fp32)
        zp_init = _make_fp8_zero_point_initializer(zp_name)
        new_inits.append(scale_init)
        new_inits.append(zp_init)

        # Build the DequantizeLinear node and rewire every consumer.
        dq_out_name = f"{weight_name}_fp8_dq"
        dq_node_name = f"{weight_name}_DequantizeLinear"
        dq_node = _make_dq_node(
            weight_name, scale_name, zp_name, dq_out_name,
            axis=len(init.dims) - 1,
            node_name=dq_node_name,
        )
        new_nodes.append(dq_node)

        for mm in consumer_nodes:
            for idx, inp in enumerate(mm.input):
                if inp == weight_name:
                    mm.input[idx] = dq_out_name

        quantized_log.append({
            "weight": weight_name,
            "shape": list(init.dims),
            "scale_init": scale_name,
            "zp_init": zp_name,
            "dq_node": dq_node_name,
            "dq_output": dq_out_name,
            "consumer_count": len(consumer_nodes),
            "consumers": [n.name for n in consumer_nodes],
        })

    # ----------------------------------------------------------------
    # W8A8: insert per-activation Q->DQ chains.
    # ----------------------------------------------------------------
    activation_log: list[dict] = []
    if activation_lookup is not None:
        # Group MatMul nodes by their CURRENT input[0] tensor name (we
        # haven't touched input[0] yet — it's still the original
        # activation source). Shared inputs (Q/K/V projections, etc.)
        # produce one bucket per source tensor; the bucket's effective
        # amax is the max across consumers so no consumer clips.
        activation_bucket: dict[str, dict] = {}
        # Rebuild MatMul lookup since some have had input[1] rewritten
        # to a DQ output already (which we don't care about here).
        node_by_name = {n.name: n for n in g.node}
        for mm_name, amax in matmul_activation_amax.items():
            mm = node_by_name[mm_name]
            act_tensor = mm.input[0]
            b = activation_bucket.setdefault(act_tensor, {
                "amax": 0.0,
                "consumers": [],
                "linear_paths": [],
            })
            if amax > b["amax"]:
                b["amax"] = amax
            b["consumers"].append(mm_name)
            b["linear_paths"].append(matmul_to_linear_path[mm_name])

        logger.info(
            "W8A8 activation buckets: {} unique input tensors across {} MatMuls",
            len(activation_bucket),
            sum(len(v["consumers"]) for v in activation_bucket.values()),
        )

        # Sanitize a tensor name into something safe for new node/init
        # names. Some ONNX intermediate tensors contain characters like
        # ``/`` (e.g. ``/layers.0/Add_output_0``). Replace with ``_``.
        def _safe(name: str) -> str:
            return name.replace("/", "_").replace(":", "_")

        for act_tensor, bucket in activation_bucket.items():
            amax = bucket["amax"]
            if amax <= 0:
                # Defensive: a zero amax would make scale 0 and divide
                # by zero in Q. Fall back to a unit scale; the FP8 cast
                # will saturate harmlessly.
                amax = FP8_E4M3_MAX
            scale_val = amax / FP8_E4M3_MAX
            tag = _safe(act_tensor)

            scale_name = f"{tag}_act_fp8_scale"
            zp_name = f"{tag}_act_fp8_zp"
            q_out = f"{tag}_act_fp8_q"
            dq_out = f"{tag}_act_fp8_dq"
            q_node_name = f"{tag}_act_QuantizeLinear"
            dq_node_name = f"{tag}_act_DequantizeLinear"

            new_inits.append(_make_scalar_scale_initializer(scale_name, scale_val))
            new_inits.append(_make_scalar_fp8_zero_point_initializer(zp_name))
            new_nodes.append(_make_q_node(
                act_tensor, scale_name, zp_name, q_out,
                node_name=q_node_name,
            ))
            new_nodes.append(_make_dq_node(
                q_out, scale_name, zp_name, dq_out,
                axis=-1,
                node_name=dq_node_name,
            ))

            for mm_name in bucket["consumers"]:
                mm = node_by_name[mm_name]
                # Replace EVERY occurrence of act_tensor in mm.input[0]
                # only — we don't want to touch input[1] which is the
                # weight DQ output now.
                if mm.input[0] == act_tensor:
                    mm.input[0] = dq_out

            activation_log.append({
                "act_tensor": act_tensor,
                "scale_init": scale_name,
                "zp_init": zp_name,
                "q_node": q_node_name,
                "dq_node": dq_node_name,
                "q_output": q_out,
                "dq_output": dq_out,
                "amax": amax,
                "scale": scale_val,
                "consumer_count": len(bucket["consumers"]),
                "consumers": bucket["consumers"],
                "linear_paths": bucket["linear_paths"],
            })

        if unmatched_in_lookup:
            logger.warning(
                "W8A8: {} weight initializer(s) had no matching PyTorch Linear; "
                "their activations stay bf16 and TRT will keep those MatMuls on "
                "the bf16 path. Examples: {}",
                len(unmatched_in_lookup),
                unmatched_in_lookup[:5],
            )

    # ----------------------------------------------------------------
    # Attention MatMul Q-DQ (dynamic-input matmuls: Q×K^T and attn×V).
    # ----------------------------------------------------------------
    attention_log: list[dict] = []
    if quantize_attention:
        # Build per-PyTorch-module output_absmax lookup from JSON.
        linear_output_amax: dict[str, float] = {}
        if activation_meta is not None:
            for path, rec in activation_meta.get("linears", {}).items():
                linear_output_amax[path] = rec.get("output_absmax", 0.0)

        node_by_name = {n.name: n for n in g.node}
        # Producer map: tensor name -> producer node (after our edits;
        # DQ-rewritten inputs of Linear MatMuls show DequantizeLinear).
        tensor_producer: dict[str, "onnx.NodeProto"] = {}
        for nd in g.node:
            for out in nd.output:
                tensor_producer[out] = nd

        # Collect dynamic-dynamic MatMul nodes (un-touched by the Linear
        # weight DQ pass — both inputs come from the graph).
        dyn_matmuls = []
        for nd in g.node:
            if nd.op_type != "MatMul" or len(nd.input) < 2:
                continue
            a, b = nd.input[0], nd.input[1]
            ap = tensor_producer.get(a)
            bp = tensor_producer.get(b)
            if (ap is not None and ap.op_type == "DequantizeLinear") or \
               (bp is not None and bp.op_type == "DequantizeLinear"):
                continue
            if a in inits or b in inits:
                continue
            dyn_matmuls.append(nd)

        logger.info(
            "Attention quantization: {} dynamic-input MatMuls to quantize",
            len(dyn_matmuls),
        )

        # Trace producer chain backward to find a source Linear MatMul.
        # Returns ``linear_path`` (PyTorch module path) and the chain
        # ops we passed through (used to detect 1/sqrt(d_k) scaling and
        # softmax-bounded outputs).
        def _trace_to_source(tensor_name: str, max_hops: int = 6) -> dict:
            """BFS up the producer graph from tensor_name."""
            result = {"linear_path": None, "via_softmax": False, "scale_factor": 1.0,
                      "chain_ops": []}
            cur = tensor_name
            for _hop in range(max_hops):
                prod = tensor_producer.get(cur)
                if prod is None:
                    return result
                result["chain_ops"].append(prod.op_type)
                if prod.op_type == "Softmax":
                    result["via_softmax"] = True
                    return result
                if prod.op_type == "MatMul":
                    # Is this a Linear MatMul (input[1] was an init,
                    # now a DQ output)?
                    src_lin = matmul_to_linear_path.get(prod.name)
                    if src_lin is not None:
                        result["linear_path"] = src_lin
                        return result
                    # Otherwise it's another dynamic MatMul (rare in
                    # this pattern); fall through.
                if prod.op_type == "Mul":
                    # Try to read the scalar multiplier (the OTHER input
                    # that ISN'T the tensor we're tracing). Only apply
                    # if it's a single-element constant. Otherwise just
                    # walk through.
                    for ip in prod.input:
                        if ip == cur:
                            continue
                        if ip in inits:
                            init = inits[ip]
                            # Number of elements from the dims attribute.
                            n_elem = 1
                            for d in init.dims:
                                n_elem *= d
                            if n_elem != 1:
                                # Per-channel or per-head scale — don't
                                # apply globally; this case is rare for
                                # the 1/sqrt(d_k) scaling pattern.
                                continue
                            import numpy as _np
                            if init.data_type == int(onnx.TensorProto.BFLOAT16):
                                v = torch.frombuffer(
                                    bytearray(init.raw_data),
                                    dtype=torch.bfloat16,
                                ).to(torch.float32).item()
                                result["scale_factor"] *= abs(v)
                            elif init.data_type == int(onnx.TensorProto.FLOAT):
                                v = float(_np.frombuffer(init.raw_data, dtype=_np.float32)[0])
                                result["scale_factor"] *= abs(v)
                # Walk to input[0] of the producer.
                if not prod.input:
                    return result
                cur = prod.input[0]
            return result

        # Sanitize for new node/init names.
        def _safe_attn(name: str) -> str:
            return name.replace("/", "_").replace(":", "_").replace(".", "_")

        # Group by tensor name and compute per-tensor amax via tracing.
        attn_inputs_to_quantize: dict[str, dict] = {}
        for nd in dyn_matmuls:
            for slot, tensor_name in enumerate(nd.input[:2]):
                trace = _trace_to_source(tensor_name)
                if trace["via_softmax"]:
                    amax = attention_softmax_max
                    kind = "softmax"
                elif trace["linear_path"] and trace["linear_path"] in linear_output_amax:
                    out_amax = linear_output_amax[trace["linear_path"]]
                    amax = out_amax * trace["scale_factor"]
                    if amax <= 0.0:
                        amax = attention_generic_max
                        kind = "fallback_zero_amax"
                    else:
                        kind = f"traced({trace['linear_path']})"
                else:
                    amax = attention_generic_max
                    kind = "fallback_no_trace"
                entry = attn_inputs_to_quantize.setdefault(tensor_name, {
                    "amax": 0.0,
                    "kind": kind,
                    "consumers": [],
                    "chain": trace["chain_ops"],
                })
                if amax > entry["amax"]:
                    entry["amax"] = amax
                entry["consumers"].append((nd.name, slot))

        for tensor_name, entry in attn_inputs_to_quantize.items():
            amax = entry["amax"]
            scale_val = amax / FP8_E4M3_MAX
            tag = _safe_attn(tensor_name) + "_attn"

            scale_name = f"{tag}_fp8_scale"
            zp_name = f"{tag}_fp8_zp"
            q_out = f"{tag}_fp8_q"
            dq_out = f"{tag}_fp8_dq"
            q_node_name = f"{tag}_QuantizeLinear"
            dq_node_name = f"{tag}_DequantizeLinear"

            new_inits.append(_make_scalar_scale_initializer(scale_name, scale_val))
            new_inits.append(_make_scalar_fp8_zero_point_initializer(zp_name))
            new_nodes.append(_make_q_node(
                tensor_name, scale_name, zp_name, q_out,
                node_name=q_node_name,
            ))
            new_nodes.append(_make_dq_node(
                q_out, scale_name, zp_name, dq_out,
                axis=-1,
                node_name=dq_node_name,
            ))

            # Rewire each (matmul, slot) consumer.
            for mm_name, slot in entry["consumers"]:
                mm = node_by_name[mm_name]
                if mm.input[slot] == tensor_name:
                    mm.input[slot] = dq_out

            attention_log.append({
                "tensor": tensor_name,
                "kind": entry["kind"],
                "amax": amax,
                "scale": scale_val,
                "consumers": entry["consumers"],
            })

        logger.info(
            "Attention Q-DQ pairs inserted: {} (softmax: {}, generic: {})",
            len(attention_log),
            sum(1 for r in attention_log if r["kind"] == "softmax"),
            sum(1 for r in attention_log if r["kind"] == "generic"),
        )

    # Extend the graph. Initializers can go in any order; node order
    # technically should be topo-sorted, but TRT 10.16's parser handles
    # any order so long as the DAG is well-formed. We prepend DQ nodes
    # because their inputs are all initializers (no upstream deps).
    g.initializer.extend(new_inits)
    # Prepend new DQ nodes so they topologically precede their consumers.
    existing_nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + existing_nodes)

    # External-data location: standard onnx convention is "<stem>.data".
    ext_data_name = output_path.stem + ".data"
    ext_data_path = output_path.with_name(ext_data_name)
    if ext_data_path.exists():
        # Avoid the user's NEVER-rm rule: rename existing file aside.
        import time
        backup = ext_data_path.with_suffix(ext_data_path.suffix + f".bak-{int(time.time())}")
        ext_data_path.rename(backup)
        logger.info("Moved existing external data aside: {} -> {}", ext_data_path, backup)

    # Reset external-data location markers so onnx.save rewrites them.
    for init in g.initializer:
        init.ClearField("external_data")
        init.data_location = onnx.TensorProto.DEFAULT

    logger.info(
        "Saving FP8 ONNX with external data: {} (data: {})",
        output_path, ext_data_path.name,
    )
    onnx.save(
        model, str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=ext_data_path.name,
    )

    if not output_path.exists():
        raise RuntimeError(f"onnx.save returned without writing {output_path}")

    # Verify the result.
    written = onnx.load(str(output_path), load_external_data=False)
    op_counts = Counter(n.op_type for n in written.graph.node)
    fp8_init_count = sum(
        1 for i in written.graph.initializer
        if i.data_type == int(onnx.TensorProto.FLOAT8E4M3FN)
    )

    logger.info("=" * 60)
    logger.info("FP8 PATCH SUMMARY ({})", mode_label)
    logger.info("=" * 60)
    logger.info("  MatMul-with-init candidates total: {}", candidates_total)
    logger.info("  excluded by name (time_embed):     {}", len(excluded_by_name))
    logger.info("  non-bf16 init (skipped):           {}", len(bad_dtype))
    logger.info("  non-2d init (skipped):             {}", len(bad_ndim))
    logger.info("  weights quantized to FP8 E4M3FN:   {}", len(quantized_log))
    logger.info("  weight bytes in (bf16):  {:.1f} MB", total_weight_bytes_in / 1e6)
    logger.info("  weight bytes out (fp8):  {:.1f} MB", total_weight_bytes_out / 1e6)
    if activation_lookup is not None:
        logger.info("  activation Q->DQ pairs (unique inputs): {}", len(activation_log))
        logger.info("  weight-init lookup misses (no PyTorch match): {}",
                    len(unmatched_in_lookup))
        if smoothquant_alpha > 0.0:
            logger.info("  SmoothQuant alpha:                 {}", smoothquant_alpha)
            logger.info("  SmoothQuant'd weights:             {}", len(smoothquant_log))
            logger.info("  SmoothQuant skipped (no per-chan): {}",
                        len(smoothquant_skipped_no_perchan))
            if smoothquant_log:
                reductions = sorted(
                    (r["act_amax_reduction"] for r in smoothquant_log
                     if r["act_amax_reduction"] is not None),
                    reverse=True,
                )
                if reductions:
                    logger.info(
                        "  SmoothQuant act-amax reduction: max={:.1f}x  "
                        "median={:.1f}x  min={:.1f}x",
                        reductions[0],
                        reductions[len(reductions) // 2],
                        reductions[-1],
                    )
    logger.info("  DequantizeLinear nodes in output:  {}", op_counts.get("DequantizeLinear", 0))
    logger.info("  QuantizeLinear nodes in output:    {}", op_counts.get("QuantizeLinear", 0))
    logger.info("  MatMul nodes in output:            {}", op_counts.get("MatMul", 0))
    logger.info("  FP8 E4M3FN initializers in output: {}", fp8_init_count)
    size_mb = output_path.stat().st_size / 1e6
    data_mb = ext_data_path.stat().st_size / 1e6 if ext_data_path.exists() else 0.0
    logger.info("  written: {} ({:.1f} MB .onnx + {:.1f} MB external data)",
                output_path.name, size_mb, data_mb)

    _write_fp8_manifest(
        output_path=output_path,
        src=src,
        cal=None,
        config=config,
        excluded_by_name=excluded_by_name,
        bad_dtype=bad_dtype,
        bad_ndim=bad_ndim,
        quantized_log=quantized_log,
        activation_log=activation_log,
        activation_absmax_json=(
            str(Path(activation_absmax_json_path).resolve())
            if activation_absmax_json_path is not None else None
        ),
        activation_percentile=(
            activation_percentile if activation_lookup is not None else None
        ),
        unmatched_in_lookup=unmatched_in_lookup,
        smoothquant_alpha=smoothquant_alpha,
        smoothquant_log=smoothquant_log,
        smoothquant_skipped=smoothquant_skipped_no_perchan,
        attention_log=attention_log,
        attention_softmax_max=attention_softmax_max if quantize_attention else None,
        attention_generic_max=attention_generic_max if quantize_attention else None,
    )

    return output_path


def _write_fp8_manifest(
    *,
    output_path: Path,
    src: Path,
    cal: Optional[Path],
    config: FP8OnnxConfig,
    excluded_by_name: list[str],
    bad_dtype: list,
    bad_ndim: list,
    quantized_log: list[dict],
    activation_log: list[dict] | None = None,
    activation_absmax_json: str | None = None,
    activation_percentile: str | None = None,
    unmatched_in_lookup: list[str] | None = None,
    smoothquant_alpha: float = 0.0,
    smoothquant_log: list[dict] | None = None,
    smoothquant_skipped: list[str] | None = None,
    attention_log: list[dict] | None = None,
    attention_softmax_max: float | None = None,
    attention_generic_max: float | None = None,
) -> None:
    """Persist FP8-build metadata for downstream tools (engine builder,
    LoRA refit). The manifest sits next to the patched ONNX.
    """
    mode = "W8A8" if activation_log else "W8A16"
    manifest = {
        "schema_version": 3,
        "patcher": f"demon.fp8_onnx.patch_bf16_onnx_to_fp8 (hand-rolled, {mode})",
        "mode": mode,
        "source_onnx": str(src),
        "patched_onnx": str(output_path),
        "calibration_npz": str(cal) if cal is not None else None,
        "activation_absmax_json": activation_absmax_json,
        "activation_percentile": activation_percentile,
        "config": {
            "op_types_to_quantize": list(config.op_types_to_quantize),
            "high_precision_dtype": config.high_precision_dtype,
            "scale_dtype": "bfloat16",
            "weight_dtype": "float8_e4m3fn",
            "weight_scheme": "per_output_channel_symmetric",
            "activation_scheme": "per_tensor_symmetric" if activation_log else "bf16_passthrough",
            "activation_amax_field": activation_percentile if activation_log else None,
            "fp8_max": FP8_E4M3_MAX,
            "absmax_floor": _ABSMAX_FLOOR,
            "opset": config.opset,
        },
        "excluded_by_name": excluded_by_name,
        "skipped_non_bf16": [
            {"name": n, "data_type": dt} for n, dt in bad_dtype
        ],
        "skipped_non_2d": [
            {"name": n, "dims": d} for n, d in bad_ndim
        ],
        "quantized_count": len(quantized_log),
        "quantized": quantized_log,
        "activation_log_count": len(activation_log) if activation_log else 0,
        "activation_log": activation_log or [],
        "unmatched_weight_inits": unmatched_in_lookup or [],
        "smoothquant_alpha": smoothquant_alpha,
        "smoothquant_applied_count": len(smoothquant_log) if smoothquant_log else 0,
        "smoothquant_skipped_count": len(smoothquant_skipped) if smoothquant_skipped else 0,
        "smoothquant_log": smoothquant_log or [],
        "smoothquant_skipped": smoothquant_skipped or [],
        "attention_quantized": bool(attention_log),
        "attention_softmax_max": attention_softmax_max,
        "attention_generic_max": attention_generic_max,
        "attention_log_count": len(attention_log) if attention_log else 0,
        "attention_log": attention_log or [],
    }
    manifest_path = output_path.with_name(output_path.stem + "_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    logger.info("FP8 manifest written: {}", manifest_path)
