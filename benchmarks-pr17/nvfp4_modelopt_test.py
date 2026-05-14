"""Run NVIDIA ModelOpt's nvfp4 quantize on a stub MatMul ONNX and check
whether the resulting TRT engine actually picks NVFP4 GEMM tactics.

This is the definitive test for "does the canonical NVFP4 toolchain
work on RTX 5090 + TRT 10.16" — if ModelOpt's own pipeline doesn't
trigger NVFP4 tactics, the issue is fundamentally a TRT 10.16 capability
gap, not a pattern/api issue we can solve from our side.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import onnx
import torch
import tensorrt as trt
from onnx import TensorProto, helper

OUT_DIR = Path(__file__).parent / "nvfp4_stub_out"
OUT_DIR.mkdir(exist_ok=True)

# DiT-like shape for the stub.
M = 6144      # M aligned to 128 (in case NVFP4 tactics need it)
K = 3072
N = 3072


def write_bf16_matmul(path: Path) -> Path:
    """Plain bf16 MatMul — input to ModelOpt's quantize pipeline."""
    inp = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [M, K])
    out = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [M, N])
    rng = np.random.default_rng(0)
    w_bf16 = torch.from_numpy(rng.standard_normal((K, N)).astype(np.float32) * 0.05).to(torch.bfloat16).contiguous()
    raw = w_bf16.view(torch.uint16).numpy().tobytes()
    w_init = TensorProto()
    w_init.name = "weight"
    w_init.data_type = int(TensorProto.BFLOAT16)
    w_init.dims.extend([K, N])
    w_init.raw_data = raw
    mm = helper.make_node("MatMul", inputs=["input", "weight"], outputs=["output"], name="MatMul")
    g = helper.make_graph([mm], "bf16_in", [inp], [out], [w_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 23)])
    m.ir_version = 10
    onnx.save(m, str(path))
    return path


def build_and_bench(onnx_path: Path, *, tag: str) -> dict:
    """Build the given ONNX with TRT and benchmark M×K×N forward."""
    print(f"\n[{tag}] building from {onnx_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f"[{tag}] parse error: {parser.get_error(i)}")
        return {"ok": False}
    cfg = builder.create_builder_config()
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 ** 3)
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        return {"ok": False, "msg": "build returned None"}
    raw = bytes(serialized)
    print(f"[{tag}] built, {len(raw)/1024:.1f} KB")

    rt = trt.Runtime(logger)
    engine = rt.deserialize_cuda_engine(raw)
    inspector = engine.create_engine_inspector()
    info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    # Save info JSON for inspection.
    info_path = OUT_DIR / f"{tag}_engine_info.json"
    info_path.write_text(info, encoding="utf-8")
    print(f"[{tag}] engine info saved -> {info_path}")
    # Look for NVFP4-related strings.
    info_lower = info.lower()
    flags_found = {
        "fp4": info_lower.count("fp4"),
        "nvfp4": info_lower.count("nvfp4"),
        "e2m1": info_lower.count("e2m1"),
        "fp8": info_lower.count("fp8"),
        "sm120": info_lower.count("sm120") + info_lower.count("sm_120"),
        "sm90": info_lower.count("sm90") + info_lower.count("sm_90"),
        "sm80": info_lower.count("sm80") + info_lower.count("sm_80"),
        "bf16": info_lower.count("bf16") + info_lower.count("bfloat16"),
        "tactic": info_lower.count("tactic"),
    }
    print(f"[{tag}] engine info keywords: {flags_found}")

    # Run benchmark
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()
    dev = torch.device("cuda")
    inp = torch.randn(M, K, device=dev, dtype=torch.bfloat16).contiguous()
    ctx.set_input_shape("input", (M, K))
    ctx.set_tensor_address("input", inp.data_ptr())
    miss = ctx.infer_shapes()
    if miss:
        print(f"[{tag}] shapes underspecified: {miss}")
        return {"ok": False}
    out_shape = tuple(ctx.get_tensor_shape("output"))
    out = torch.empty(out_shape, dtype=torch.bfloat16, device=dev)
    ctx.set_tensor_address("output", out.data_ptr())
    for _ in range(10):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    iters = 200
    t0 = time.perf_counter()
    for _ in range(iters):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1000
    tflops = 2 * M * K * N / (dt / 1000) / 1e12
    print(f"[{tag}] tick: {dt:.3f} ms, TFLOPS: {tflops:.1f}")
    return {"ok": True, "tick_ms": dt, "tflops": tflops, "info_path": str(info_path), "keywords": flags_found}


def main():
    print("=" * 70)
    print("NVFP4 ModelOpt + TRT 10.16 feasibility test")
    print(f"Shape: M={M} K={K} N={N}")
    print("=" * 70)

    bf16_path = OUT_DIR / "modelopt_input_bf16.onnx"
    write_bf16_matmul(bf16_path)
    print(f"\nWrote bf16 input ONNX: {bf16_path}")

    # Run ModelOpt's quantize with mode='nvfp4'.
    nvfp4_path = OUT_DIR / "modelopt_output_nvfp4.onnx"
    print(f"\n[modelopt] calling quantize(mode='nvfp4')...")
    # Calibration data: shape must be (n_itr * M, K). One iter = (M, K).
    rng = np.random.default_rng(1)
    cal_data = {"input": rng.standard_normal((M, K)).astype(np.float32)}
    try:
        from modelopt.onnx.quantization import quantize
        quantize(
            onnx_path=str(bf16_path),
            quantize_mode="nvfp4",
            calibration_data=cal_data,
            output_path=str(nvfp4_path),
            high_precision_dtype="bf16",
            log_level="INFO",
        )
        print(f"[modelopt] wrote nvfp4 quantized ONNX: {nvfp4_path}")
    except Exception as e:
        print(f"[modelopt] FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return

    # Compare bf16 baseline vs modelopt-nvfp4 build.
    bf16_results = build_and_bench(bf16_path, tag="bf16_baseline")
    nvfp4_results = build_and_bench(nvfp4_path, tag="modelopt_nvfp4")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for tag, r in [("bf16_baseline", bf16_results), ("modelopt_nvfp4", nvfp4_results)]:
        if r.get("ok"):
            print(f"  {tag:<25s}  {r['tick_ms']:.3f} ms  {r['tflops']:.1f} TFLOPS")
        else:
            print(f"  {tag:<25s}  FAILED: {r.get('msg', '?')}")


if __name__ == "__main__":
    main()
