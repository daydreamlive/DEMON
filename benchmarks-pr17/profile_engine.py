"""Per-layer timing profile for the XL decoder TRT engines.

Uses TRT's IProfiler hook to measure per-layer wallclock during N
batch=4 forwards. Aggregates by op category (GEMM/MatMul, norm,
elementwise, attention, etc.) so we can see where the tick time
actually goes. Also estimates effective weight bandwidth (total weight
bytes per tick / tick time) vs the RTX 5090's HBM3X peak (~1.79 TB/s)
to answer "memory-bound or compute-bound."

The MatMul fraction matters for deciding whether a precision change
(NVFP4, INT4, etc.) can deliver a step-function speedup. If MatMul is
80%+ of the tick, a 2x faster GEMM gets ~2x overall; if MatMul is 50%,
2x GEMM gets ~1.3x by Amdahl.

Usage::

    python benchmarks-pr17/profile_engine.py
    python benchmarks-pr17/profile_engine.py --engine w8a8_absmax_cal2
    python benchmarks-pr17/profile_engine.py --engine bf16 --iters 100
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
torch.set_grad_enabled(False)
import tensorrt as trt


# RTX 5090 specs (Blackwell, GB203):
#   - HBM3X memory bandwidth: ~1792 GB/s (advertised)
#   - bf16 tensor core throughput: ~165 TFLOPS dense
#   - FP8 tensor core throughput: ~330 TFLOPS dense
#   - NVFP4 tensor core throughput: ~660 TFLOPS dense
RTX_5090_HBM_GBPS = 1792.0
RTX_5090_BF16_TFLOPS = 165.0
RTX_5090_FP8_TFLOPS = 330.0
RTX_5090_NVFP4_TFLOPS = 660.0


VARIANTS_ROOT = Path(__file__).parent / "variants"
INPUT_NAMES = ("hidden_states", "timestep", "encoder_hidden_states", "context_latents")
OUTPUT_NAME = "velocity"

CAL_NPZ = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/calibration/decoder_xl_fp8/calibration.npz"
))

_TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}
if hasattr(trt, "bfloat16"):
    _TRT_TO_TORCH[trt.bfloat16] = torch.bfloat16


# ----------------------------------------------------------------
# Layer categorization (regex on TRT layer names)
# ----------------------------------------------------------------

# TRT layer names typically encode op type + source. After our patching
# they include things like "..._MatMul", "..._fp8_DequantizeLinear",
# "/layers.0/self_attn/q_proj/MatMul", "LayerNormalization", etc.

CATEGORY_PATTERNS = [
    # GEMM / MatMul (the layers our quantization touches)
    ("matmul",       re.compile(r"matmul|gemm|linear", re.I)),
    # Attention math (Q @ K^T, softmax, attn @ V)
    ("attention",    re.compile(r"attn|softmax|baddbmm|bmm|attention", re.I)),
    # Activation reshaping / normalization
    ("norm",         re.compile(r"norm|rmsnorm|layernorm|groupnorm", re.I)),
    # Q/DQ chains we inserted
    ("quant_dq",     re.compile(r"quantize|dequantize|\bq\b|\bdq\b", re.I)),
    # Elementwise ops (Add, Mul, Cast, etc.)
    ("elementwise",  re.compile(r"add|mul|cast|sub|div|tanh|gelu|silu|relu|sigmoid|swiglu|where|select|reshape|transpose|slice|concat|gather", re.I)),
    # Misc / unknown
]


def categorize(layer_name: str) -> str:
    name_lower = layer_name.lower()
    for cat, pat in CATEGORY_PATTERNS:
        if pat.search(name_lower):
            return cat
    return "other"


# ----------------------------------------------------------------
# Profiler hook
# ----------------------------------------------------------------

class LayerProfiler(trt.IProfiler):
    """Accumulates layer-level ms across multiple inferences."""

    def __init__(self):
        super().__init__()
        self.totals: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        self.totals[layer_name] += ms
        self.calls[layer_name] += 1

    def reset(self) -> None:
        self.totals.clear()
        self.calls.clear()


# ----------------------------------------------------------------
# Engine loading
# ----------------------------------------------------------------

class EngineRunner:
    def __init__(self, path: Path):
        self.path = path
        self.size_mb = path.stat().st_size / 1e6
        logger = trt.Logger(trt.Logger.WARNING)
        rt = trt.Runtime(logger)
        with open(path, "rb") as f:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load {path}")
        self.ctx = self.engine.create_execution_context()
        self.profiler = LayerProfiler()
        self.ctx.profiler = self.profiler  # enables IProfiler hooks
        self.in_dtypes = {
            n: _TRT_TO_TORCH.get(self.engine.get_tensor_dtype(n), torch.float32)
            for n in INPUT_NAMES
        }
        self.out_dtype = _TRT_TO_TORCH.get(
            self.engine.get_tensor_dtype(OUTPUT_NAME), torch.float32,
        )
        self.stream = torch.cuda.Stream()

    def run(self, inputs: dict, sync: bool = True):
        dev = torch.device("cuda")
        bufs = {}
        for n in INPUT_NAMES:
            t = inputs[n].to(device=dev, dtype=self.in_dtypes[n]).contiguous()
            bufs[n] = t
            if not self.ctx.set_input_shape(n, tuple(t.shape)):
                raise RuntimeError(f"rejected shape for {n}: {t.shape}")
            if not self.ctx.set_tensor_address(n, t.data_ptr()):
                raise RuntimeError(f"rejected address for {n}")
        miss = self.ctx.infer_shapes()
        if miss:
            raise RuntimeError(f"shapes underspecified: {miss}")
        out_shape = tuple(self.ctx.get_tensor_shape(OUTPUT_NAME))
        out = torch.empty(out_shape, dtype=self.out_dtype, device=dev)
        if not self.ctx.set_tensor_address(OUTPUT_NAME, out.data_ptr()):
            raise RuntimeError("rejected output address")
        # execute_async_v3 with IProfiler set on ctx uses synchronous
        # internal layer instrumentation; the cost is real but layer
        # ordering and individual timings are reliable.
        if not self.ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("execute_async_v3 failed")
        if sync:
            self.stream.synchronize()
        return out, bufs


# ----------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------

def _resolve_engine(name: str) -> Path:
    """Allow either a variant tag (resolved to variants/<tag>/engine.engine)
    or an explicit path.
    """
    p = Path(name)
    if p.exists():
        return p
    candidate = VARIANTS_ROOT / name / "engine.engine"
    if candidate.exists():
        return candidate
    # bf16 special case: live engine path
    if name == "bf16":
        bf16_live = Path(os.path.expanduser(
            "~/.daydream-scope/models/demon/trt_engines/"
            "decoder_xl-turbo_mixed_refit_b4_60s/"
            "decoder_xl-turbo_mixed_refit_b4_60s.engine"
        ))
        if bf16_live.exists():
            return bf16_live
    raise FileNotFoundError(
        f"Could not resolve engine '{name}'. Tried variants/{name}/engine.engine"
    )


def profile_engine(engine_name: str, *, warmup: int = 5, iters: int = 30) -> dict:
    path = _resolve_engine(engine_name)
    print(f"[{engine_name}] engine: {path}")
    print(f"[{engine_name}] size: {path.stat().st_size / 1e6:.1f} MB")
    runner = EngineRunner(path)
    print(f"[{engine_name}] dtypes: {{ {', '.join(f'{n}: {v}' for n,v in runner.in_dtypes.items())} }}  out: {runner.out_dtype}")

    npz = np.load(str(CAL_NPZ))
    s = slice(0, 4)
    inputs = {
        "hidden_states": torch.from_numpy(npz["hidden_states"][s]),
        "timestep": torch.from_numpy(npz["timestep"][s]),
        "encoder_hidden_states": torch.from_numpy(npz["encoder_hidden_states"][s]),
        "context_latents": torch.from_numpy(npz["context_latents"][s]),
    }

    print(f"[{engine_name}] warmup x{warmup} ...")
    for _ in range(warmup):
        runner.run(inputs)
    runner.profiler.reset()

    print(f"[{engine_name}] profiling x{iters} ...")
    t0 = time.perf_counter()
    for _ in range(iters):
        runner.run(inputs)
    elapsed_s = time.perf_counter() - t0
    mean_tick_ms = (elapsed_s / iters) * 1000.0

    # Sum of per-layer ms across all iters, divided by iters.
    per_layer_total_ms = {
        name: ms / iters for name, ms in runner.profiler.totals.items()
    }
    # Group by category.
    by_cat: dict[str, float] = defaultdict(float)
    for name, ms in per_layer_total_ms.items():
        by_cat[categorize(name)] += ms
    cat_total = sum(by_cat.values())

    return {
        "engine_name": engine_name,
        "engine_path": str(path),
        "engine_size_mb": path.stat().st_size / 1e6,
        "mean_tick_ms": mean_tick_ms,
        "iters": iters,
        "per_layer_total_ms": per_layer_total_ms,
        "by_cat_ms": dict(by_cat),
        "cat_total_ms": cat_total,
    }


def analyze(result: dict) -> None:
    name = result["engine_name"]
    tick_ms = result["mean_tick_ms"]
    by_cat = result["by_cat_ms"]
    cat_total = result["cat_total_ms"]
    per_layer = result["per_layer_total_ms"]
    n_layers = len(per_layer)

    print()
    print("=" * 72)
    print(f"PROFILE: {name}  (tick = {tick_ms:.2f} ms wall, {cat_total:.2f} ms per-layer sum, {n_layers} layers)")
    print("=" * 72)
    overhead_ms = tick_ms - cat_total
    print(f"  Per-layer instrumentation sum: {cat_total:.2f} ms ({cat_total/tick_ms*100:.0f}% of tick)")
    print(f"  Stream / launch overhead:      {overhead_ms:.2f} ms ({overhead_ms/tick_ms*100:.0f}% of tick)")
    print()
    print(f"  Category breakdown (of {cat_total:.2f} ms layer time):")
    for cat in sorted(by_cat, key=lambda c: -by_cat[c]):
        ms = by_cat[cat]
        print(f"    {cat:<14s} {ms:7.2f} ms  ({ms/cat_total*100:5.1f}% of layer time, {ms/tick_ms*100:5.1f}% of tick)")
    print()
    # Top 15 slowest layers.
    top = sorted(per_layer.items(), key=lambda kv: -kv[1])[:15]
    print(f"  Top 15 slowest layers:")
    for name, ms in top:
        cat = categorize(name)
        # Truncate name for display.
        disp = name if len(name) <= 70 else name[:67] + "..."
        print(f"    {ms:6.3f} ms  [{cat:<10s}] {disp}")
    print()

    # Bandwidth proxy: weights bytes / tick ms = effective bandwidth.
    # We don't know weight bytes from the profiler alone, but the engine
    # file size minus overhead is a useful upper bound.
    engine_mb = result["engine_size_mb"]
    # Most engine bytes are weights for our DiT (~95%); the rest is
    # graph metadata + scratch.
    weight_mb_est = engine_mb * 0.95
    eff_bw_gbps = (weight_mb_est / 1024.0) / (tick_ms / 1000.0)  # GB/s
    print(f"  Bandwidth analysis (rough):")
    print(f"    Engine size:           {engine_mb:7.1f} MB")
    print(f"    Weight size estimate:  {weight_mb_est:7.1f} MB ({weight_mb_est/1024:.2f} GB)")
    print(f"    Tick time:             {tick_ms:.2f} ms")
    print(f"    Effective wt bw:       {eff_bw_gbps:7.1f} GB/s")
    print(f"    RTX 5090 HBM3X peak:   {RTX_5090_HBM_GBPS:7.1f} GB/s")
    print(f"    Utilization:           {eff_bw_gbps/RTX_5090_HBM_GBPS*100:5.1f}% of HBM peak")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--engine", action="append",
        help="Engine variant tag or path. Can pass multiple. Default: w8a8_absmax_cal2 + bf16.",
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    engines = args.engine or ["w8a8_absmax_cal2", "bf16"]
    results = []
    for name in engines:
        result = profile_engine(name, warmup=args.warmup, iters=args.iters)
        results.append(result)
        analyze(result)

    if len(results) >= 2:
        print("=" * 72)
        print("CROSS-ENGINE COMPARISON")
        print("=" * 72)
        a, b = results[0], results[1]
        print(f"  {a['engine_name']:<25s}: {a['mean_tick_ms']:7.2f} ms  ({a['engine_size_mb']:.0f} MB)")
        print(f"  {b['engine_name']:<25s}: {b['mean_tick_ms']:7.2f} ms  ({b['engine_size_mb']:.0f} MB)")
        speedup = b["mean_tick_ms"] / a["mean_tick_ms"]
        print(f"  {a['engine_name']} speedup vs {b['engine_name']}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
