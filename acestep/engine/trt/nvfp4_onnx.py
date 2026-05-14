"""Replace bf16 Linear MatMuls in a decoder ONNX with our NVFP4Linear TRT plugin.

This is the NVFP4 counterpart to `fp8_onnx.py`. It does NOT modify, replace,
or share code with the FP8 path. The two patchers coexist:

  - `fp8_onnx.py` -> inserts Q-DQ chains that TRT compiles to FP8 W8A8 GEMM.
  - `nvfp4_onnx.py` (this file) -> swaps each Linear MatMul node for a custom
    NVFP4Linear plugin op. The plugin (`acestep/engine/trt/plugins/nvfp4_linear/`)
    handles bf16 -> NVFP4 quantization at runtime and calls cuBLASLt NVFP4 GEMM.

The patcher walks the bf16 ONNX and for every `MatMul` node whose `input[1]`
is a bf16 2D initializer:

  1. Decode the bf16 weight to fp32.
  2. NVFP4-quantize: packed FP4 (N, K/2) + CUTLASS-swizzled FP8 E4M3 block
     scales + per-tensor FP32 global scale. (See benchmarks-pr17/cublaslt_nvfp4_spike/
     nvfp4_e2e.py for the reference quant scheme.)
  3. Look up the per-Linear activation absmax in cal2's activation_absmax.json
     and bake a static `act_global_scale = absmax / (FP4_MAX * FP8_E4M3_MAX)`.
  4. Replace the MatMul node with an NVFP4Linear plugin node carrying all of
     the above as attributes.
  5. Outlier-skip layers per the same ratio mechanism as fp8_onnx.py: layers
     above the threshold keep their bf16 MatMul untouched (no NVFP4 speedup,
     but quality is preserved).

The plugin DLL (`nvfp4_linear_plugin.dll`) must be loaded into TRT's plugin
registry before parsing the patched ONNX. The build flow handles that.

Weight initializers are rewritten in-place to a small placeholder (the plugin
no longer reads them - the FP4 bytes live in plugin attributes); we keep the
initializer name on the graph as an orphan to avoid a partial rewrite landing
mid-edit.
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

# Re-use the FP8 patcher's helpers for the absmax-JSON lookup and L2 matching.
# These functions are pure utilities and don't touch any FP8-specific logic.
from .fp8_onnx import (  # noqa: E402
    _ABSMAX_FLOOR,
    _is_excluded_init_name,
    _load_activation_absmax,
    _weight_l2_bf16,
)

# NVFP4 constants (must match the plugin's compile-time constants).
FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_GLOBAL_DIVISOR = FP4_MAX * FP8_E4M3_MAX  # 2688

# Plugin ONNX op identity (must match the C++ plugin's name/version/namespace).
NVFP4_PLUGIN_OP_TYPE = "NVFP4Linear"
NVFP4_PLUGIN_DOMAIN = "trt.plugins"
NVFP4_PLUGIN_VERSION = 1


@dataclass
class NVFP4OnnxConfig:
    """NVFP4 patch config. Mirrors FP8OnnxConfig where it makes sense."""
    op_types_to_quantize: tuple[str, ...] = ("MatMul",)
    high_precision_dtype: str = "bf16"
    opset: int = 20
    block_size: int = 16  # NVFP4 per-block scale group size (fixed by cuBLASLt API)


# ------------------------------------------------------------------
# Quant helpers (Python reference; mirrors the plugin's enqueue path)
# ------------------------------------------------------------------

def _quantize_weight_nvfp4(w_fp32):
    """Quantize a 2D fp32 weight (ONNX layout: [in, out]) to NVFP4.

    Returns:
        packed_fp4_bytes:    flat bytes, (N, K/2) row-major where N=out, K=in
        swz_fp8_scale_bytes: flat bytes, CUTLASS-swizzled FP8 E4M3 scales
        global_scale_fp32:   scalar fp32 (per-tensor)
    """
    import torch
    from torchao.prototype.mx_formats.utils import to_blocked

    # Bring in the e2e Python reference. We import lazily to keep this module
    # importable on stripped-down environments that don't ship torchao.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] /
                          "benchmarks-pr17" / "cublaslt_nvfp4_spike"))
    try:
        from nvfp4_e2e import quantize_to_nvfp4  # noqa: E402
    finally:
        # Don't pollute sys.path after import.
        sys.path = [p for p in sys.path if "cublaslt_nvfp4_spike" not in p]

    # ONNX MatMul weight is [in, out]. The plugin / cuBLASLt expects B in
    # "(N, K)" row-major (N=out, K=in), which is the TRANSPOSE of the ONNX
    # layout. So transpose before quantizing.
    if w_fp32.ndim != 2:
        raise NotImplementedError(f"NVFP4 patcher only handles 2D weights, got {w_fp32.shape}")
    K_dim, N_dim = w_fp32.shape  # ONNX [in, out]
    w_T = w_fp32.t().contiguous()  # [out, in] = (N, K) for the plugin

    w_dev = w_T.to("cuda")
    packed, blk_scale, global_scale = quantize_to_nvfp4(w_dev, block=16)
    swz = to_blocked(blk_scale).contiguous()
    return (
        packed.cpu().numpy().tobytes(),
        swz.cpu().numpy().tobytes(),
        float(global_scale.item()),
    )


def _make_uint8_initializer(name: str, raw_bytes: bytes, dims: list[int]):
    """Create an ONNX INT8 initializer with the given raw bytes.

    These initializers carry the FP4 weight bytes and the swizzled FP8 scale
    bytes. We use INT8 storage (not UINT8) because TRT plugin inputs do not
    accept the UINT8 type. The plugin reinterprets the bytes as raw uint8
    anyway, so the signed-vs-unsigned label is purely a labeling concern.

    These initializers get external_data treatment on save, which is what
    keeps the protobuf under its 2 GiB hard limit.
    """
    import onnx

    init = onnx.TensorProto()
    init.name = name
    init.data_type = int(onnx.TensorProto.INT8)
    init.dims.extend(dims)
    init.raw_data = raw_bytes
    return init


def _make_nvfp4_plugin_node(
    *,
    node_name: str,
    activation_input: str,
    weight_fp4_input: str,
    weight_scale_input: str,
    output: str,
    K: int,
    N: int,
    weight_global_scale: float,
    act_global_scale: float,
):
    """Build an ONNX NodeProto for the NVFP4Linear plugin.

    The plugin takes THREE input tensors:
        inputs[0] = bf16 activation, shape (..., K)
        inputs[1] = packed FP4 weight bytes, uint8 shape (N, K/2)
        inputs[2] = swizzled FP8 scale bytes, uint8 shape (n_swz_bytes,)
    and four scalar attributes (K, N, weight_global_scale, act_global_scale).
    The weight tensors are stored as ONNX initializers (external_data eligible)
    so the protobuf size stays bounded regardless of how many Linears we patch.
    """
    from onnx import helper

    attrs = [
        helper.make_attribute("K", K),
        helper.make_attribute("N", N),
        helper.make_attribute("weight_global_scale", float(weight_global_scale)),
        helper.make_attribute("act_global_scale", float(act_global_scale)),
        # Identifies the plugin to TRT's parser.
        helper.make_attribute("plugin_namespace", ""),
        helper.make_attribute("plugin_version", "1"),
    ]
    n = helper.make_node(
        op_type=NVFP4_PLUGIN_OP_TYPE,
        inputs=[activation_input, weight_fp4_input, weight_scale_input],
        outputs=[output],
        name=node_name,
        domain=NVFP4_PLUGIN_DOMAIN,
    )
    n.attribute.extend(attrs)
    return n


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def patch_bf16_onnx_to_nvfp4(
    bf16_onnx_path: Union[str, Path],
    activation_absmax_json_path: Union[str, Path],
    *,
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[NVFP4OnnxConfig] = None,
    force: bool = False,
    activation_percentile: str = "absmax",
    activation_outlier_skip_ratio: float = 0.0,
) -> Path:
    """Patch a bf16 decoder ONNX with NVFP4Linear plugin nodes.

    Args:
        bf16_onnx_path: source bf16 ONNX path.
        activation_absmax_json_path: cal2 activation_absmax.json (per-Linear).
            Required - the plugin uses a static cal-baked global scale per call,
            so we need an activation absmax for every Linear we replace.
        output_path: target patched ONNX. Defaults to <stem>_nvfp4<suffix>.
        force: regenerate even if output is newer than inputs.
        activation_percentile: which absmax statistic to use (absmax | p99 | ...).
        activation_outlier_skip_ratio: layers with absmax/p99_9 above this ratio
            keep their bf16 MatMul (no plugin replacement, no NVFP4 speedup).

    Returns the path to the patched ONNX.
    """
    if config is None:
        config = NVFP4OnnxConfig()

    src = Path(bf16_onnx_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"bf16 ONNX not found: {src}")
    amax_path = Path(activation_absmax_json_path).resolve()
    if not amax_path.exists():
        raise FileNotFoundError(f"Activation absmax JSON not found: {amax_path}")

    if output_path is None:
        output_path = src.with_name(src.stem + "_nvfp4" + src.suffix)
    output_path = Path(output_path).resolve()
    if output_path.parent != src.parent:
        raise ValueError(
            "Patched ONNX must be a sibling of the source "
            "(external_data references are relative)."
        )

    # Cache freshness check (don't rebuild if newer than both inputs).
    if (
        output_path.exists()
        and not force
        and output_path.stat().st_mtime >= src.stat().st_mtime
        and output_path.stat().st_mtime >= amax_path.stat().st_mtime
    ):
        logger.info("Reusing NVFP4 ONNX (newer than source + absmax JSON): {}", output_path)
        return output_path

    activation_lookup, activation_meta = _load_activation_absmax(
        amax_path,
        percentile_field=activation_percentile,
        outlier_skip_ratio=activation_outlier_skip_ratio,
    )
    n_skip = sum(
        1 for recs in activation_lookup.values()
        for r in recs if r["skip_activation_quant"]
    )
    logger.info(
        "Loaded activation absmax: {} linears, {} keys; outlier_skip_ratio={} "
        "-> {} layers will keep their bf16 MatMul",
        len(activation_meta.get("linears", {})), len(activation_lookup),
        activation_outlier_skip_ratio, n_skip,
    )

    import onnx
    import torch

    logger.info("=" * 60)
    logger.info("NVFP4 PLUGIN INSERTION (replaces bf16 Linear MatMuls)")
    logger.info("=" * 60)
    logger.info("  source: {}", src)
    logger.info("  output: {}", output_path)
    logger.info("  plugin op: domain={!r} op_type={!r} version={}",
                NVFP4_PLUGIN_DOMAIN, NVFP4_PLUGIN_OP_TYPE, NVFP4_PLUGIN_VERSION)

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
    excluded_by_name: list[str] = []
    bad_dtype: list = []
    bad_ndim: list = []
    to_quantize: list[tuple[str, list]] = []
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
        "Candidates: total={} excluded_by_name={} non_bf16={} non_2d={} to_replace={}",
        candidates_total, len(excluded_by_name),
        len(bad_dtype), len(bad_ndim), len(to_quantize),
    )
    if not to_quantize:
        raise RuntimeError("No Linear MatMul candidates matched. Check the source ONNX.")

    new_nodes: list = []                 # nvfp4 plugin replacements
    new_inits: list = []                 # uint8 weight + scale initializers
    replaced_log: list[dict] = []
    skipped_outlier: list[dict] = []
    unmatched_in_lookup: list[str] = []

    # MatMul names that we replace; the original node is dropped from g.node.
    matmuls_to_drop: set[str] = set()
    # Weight initializers to drop (their data is now in plugin attributes).
    inits_to_drop: set[str] = set()

    total_bf16_weight_bytes = 0
    total_fp4_weight_bytes = 0
    total_swz_scale_bytes = 0

    for weight_name, consumer_nodes in to_quantize:
        init = inits[weight_name]
        raw = init.raw_data
        if not raw:
            raise NotImplementedError(f"bf16 init {weight_name} has no raw_data")

        t_bf16 = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        w_fp32 = t_bf16.to(torch.float32).reshape(tuple(init.dims))

        # Look up per-Linear activation absmax.
        onnx_shape = tuple(init.dims)
        l2 = round(_weight_l2_bf16(init), 3)
        bucket = activation_lookup.get((onnx_shape, l2))
        if not bucket:
            unmatched_in_lookup.append(weight_name)
            # No activation absmax available -> can't bake a static scale.
            # Keep this MatMul on the bf16 path. (Rare; logged.)
            continue
        rec = bucket.pop(0)

        if rec["skip_activation_quant"]:
            # Outlier-heavy layer: leave the bf16 MatMul untouched.
            skipped_outlier.append({
                "weight": weight_name,
                "linear_path": rec["linear_path"],
                "absmax": rec["absmax"],
                "outlier_ratio": rec["outlier_ratio"],
            })
            continue

        # Static global activation scale: max(|x|) / (FP4_MAX * FP8_E4M3_MAX).
        act_amax = rec["scale_amax"]
        if act_amax <= 0.0:
            act_amax = max(rec["absmax"], _ABSMAX_FLOOR)
        act_global_scale = act_amax / NVFP4_GLOBAL_DIVISOR
        if act_global_scale < 1e-30:
            act_global_scale = 1e-30

        # Quantize the weight.
        packed_bytes, swz_bytes, w_global_scale = _quantize_weight_nvfp4(w_fp32)
        N_dim, K_dim = (init.dims[1], init.dims[0])  # ONNX [in=K, out=N]
        total_bf16_weight_bytes += len(raw)
        total_fp4_weight_bytes += len(packed_bytes)
        total_swz_scale_bytes += len(swz_bytes)

        # Add the FP4 + swizzled-scale uint8 initializers shared by every
        # consumer MatMul of this original weight. They get external_data
        # treatment on save.
        wfp4_init_name = f"{weight_name}_nvfp4_w"
        wscale_init_name = f"{weight_name}_nvfp4_wscale"
        new_inits.append(_make_uint8_initializer(
            wfp4_init_name, packed_bytes, dims=[N_dim, K_dim // 2],
        ))
        new_inits.append(_make_uint8_initializer(
            wscale_init_name, swz_bytes, dims=[len(swz_bytes)],
        ))

        for mm_idx, mm in enumerate(consumer_nodes):
            act_input = mm.input[0]
            output_name = mm.output[0]
            plug_name = f"{weight_name}_nvfp4plugin_{mm_idx}"
            plug = _make_nvfp4_plugin_node(
                node_name=plug_name,
                activation_input=act_input,
                weight_fp4_input=wfp4_init_name,
                weight_scale_input=wscale_init_name,
                output=output_name,
                K=K_dim, N=N_dim,
                weight_global_scale=w_global_scale,
                act_global_scale=act_global_scale,
            )
            new_nodes.append(plug)
            matmuls_to_drop.add(mm.name)
            replaced_log.append({
                "matmul_node": mm.name,
                "weight": weight_name,
                "linear_path": rec["linear_path"],
                "K": K_dim, "N": N_dim,
                "weight_global_scale": w_global_scale,
                "act_global_scale": act_global_scale,
                "act_amax": act_amax,
                "weight_fp4_init": wfp4_init_name,
                "weight_scale_init": wscale_init_name,
            })

        # The ORIGINAL bf16 weight initializer is replaced by the new
        # uint8 initializers; drop it to avoid carrying duplicate weight
        # bytes in external data.
        inits_to_drop.add(weight_name)

    # ----------------------------------------------------------------
    # Apply the graph edits.
    # ----------------------------------------------------------------
    # Drop original MatMul nodes that we replaced.
    keep_nodes = [n for n in g.node if n.name not in matmuls_to_drop]
    # Prepend the plugin nodes (their only input is the activation, which
    # comes from an existing upstream node, so ordering is fine).
    del g.node[:]
    g.node.extend(new_nodes + keep_nodes)

    # Note: we DON'T drop the original bf16 weight initializers here. They
    # remain in the graph as orphans (no consumers) since their MatMul
    # nodes were removed. The TRT parser strips unreferenced initializers
    # at engine build time. Dropping them explicitly was breaking a Myelin
    # fusion pattern - the bf16 weight's role in the bf16 conv1d-and-friends
    # subgraph appears to depend on the initializer existing.
    g.initializer.extend(new_inits)

    # Declare the plugin's custom op set so ONNX validators accept the
    # opdomain. TRT's parser uses domain+op_type to look up the registered
    # plugin creator.
    has_domain = any(o.domain == NVFP4_PLUGIN_DOMAIN for o in model.opset_import)
    if not has_domain:
        oi = model.opset_import.add()
        oi.domain = NVFP4_PLUGIN_DOMAIN
        oi.version = NVFP4_PLUGIN_VERSION

    # ----------------------------------------------------------------
    # External-data write.
    # ----------------------------------------------------------------
    ext_data_name = output_path.stem + ".data"
    ext_data_path = output_path.with_name(ext_data_name)
    if ext_data_path.exists():
        import time
        backup = ext_data_path.with_suffix(
            ext_data_path.suffix + f".bak-{int(time.time())}"
        )
        ext_data_path.rename(backup)
        logger.info("Moved existing external data aside: {} -> {}", ext_data_path, backup)

    for init in g.initializer:
        init.ClearField("external_data")
        init.data_location = onnx.TensorProto.DEFAULT

    logger.info(
        "Saving NVFP4 ONNX: {} (external data: {})",
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

    # Verify.
    written = onnx.load(str(output_path), load_external_data=False)
    op_counts = Counter(n.op_type for n in written.graph.node)
    plugin_node_count = sum(
        1 for n in written.graph.node
        if n.op_type == NVFP4_PLUGIN_OP_TYPE and n.domain == NVFP4_PLUGIN_DOMAIN
    )

    logger.info("=" * 60)
    logger.info("NVFP4 PATCH SUMMARY")
    logger.info("=" * 60)
    logger.info("  MatMul candidates total:          {}", candidates_total)
    logger.info("  excluded by name (time_embed):    {}", len(excluded_by_name))
    logger.info("  non-bf16 init (skipped):          {}", len(bad_dtype))
    logger.info("  non-2d init (skipped):            {}", len(bad_ndim))
    logger.info("  weights matched in cal lookup:    {}", len(to_quantize) - len(unmatched_in_lookup))
    logger.info("  outlier-skipped (kept bf16):      {}", len(skipped_outlier))
    logger.info("  NVFP4 plugin nodes inserted:      {}", plugin_node_count)
    logger.info("  bf16 weight bytes in:             {:.1f} MB", total_bf16_weight_bytes / 1e6)
    logger.info("  fp4 weight bytes out:             {:.1f} MB", total_fp4_weight_bytes / 1e6)
    logger.info("  swizzled scale bytes:             {:.1f} MB", total_swz_scale_bytes / 1e6)
    logger.info("  unmatched in lookup:              {}", len(unmatched_in_lookup))
    size_mb = output_path.stat().st_size / 1e6
    data_mb = ext_data_path.stat().st_size / 1e6 if ext_data_path.exists() else 0.0
    logger.info("  written: {} ({:.1f} MB .onnx + {:.1f} MB external)",
                output_path.name, size_mb, data_mb)

    _write_nvfp4_manifest(
        output_path=output_path,
        src=src,
        config=config,
        excluded_by_name=excluded_by_name,
        bad_dtype=bad_dtype,
        bad_ndim=bad_ndim,
        replaced_log=replaced_log,
        skipped_outlier=skipped_outlier,
        unmatched_in_lookup=unmatched_in_lookup,
        activation_absmax_json=str(amax_path),
        activation_percentile=activation_percentile,
        activation_outlier_skip_ratio=activation_outlier_skip_ratio,
    )

    return output_path


def _write_nvfp4_manifest(
    *,
    output_path: Path,
    src: Path,
    config: NVFP4OnnxConfig,
    excluded_by_name: list[str],
    bad_dtype: list,
    bad_ndim: list,
    replaced_log: list[dict],
    skipped_outlier: list[dict],
    unmatched_in_lookup: list[str],
    activation_absmax_json: str,
    activation_percentile: str,
    activation_outlier_skip_ratio: float,
) -> None:
    manifest = {
        "schema_version": 1,
        "patcher": "demon.nvfp4_onnx.patch_bf16_onnx_to_nvfp4",
        "mode": "NVFP4_PLUGIN",
        "plugin": {
            "domain": NVFP4_PLUGIN_DOMAIN,
            "op_type": NVFP4_PLUGIN_OP_TYPE,
            "version": NVFP4_PLUGIN_VERSION,
        },
        "source_onnx": str(src),
        "patched_onnx": str(output_path),
        "activation_absmax_json": activation_absmax_json,
        "activation_percentile": activation_percentile,
        "activation_outlier_skip_ratio": activation_outlier_skip_ratio,
        "config": {
            "op_types_to_quantize": list(config.op_types_to_quantize),
            "block_size": config.block_size,
            "fp4_max": FP4_MAX,
            "fp8_e4m3_max": FP8_E4M3_MAX,
            "opset": config.opset,
        },
        "excluded_by_name": excluded_by_name,
        "skipped_non_bf16": [{"name": n, "data_type": dt} for n, dt in bad_dtype],
        "skipped_non_2d": [{"name": n, "dims": d} for n, d in bad_ndim],
        "replaced_count": len(replaced_log),
        "replaced": replaced_log,
        "outlier_skipped_count": len(skipped_outlier),
        "outlier_skipped": skipped_outlier,
        "unmatched_in_lookup": unmatched_in_lookup,
    }
    manifest_path = output_path.with_suffix(".nvfp4_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote NVFP4 manifest: {}", manifest_path)
