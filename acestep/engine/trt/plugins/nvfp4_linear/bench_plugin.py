"""Headline speed benchmark: NVFP4 plugin (in TRT engine) vs FP8 (TRT W8A8 baseline).

Production shape: M = 6144 (batch=4 * seq=1536), N = 3072, K = 3072.
This matches benchmarks-pr17/fp8_stub.py which measured TRT's FP8 W8A8 GEMM at
414 TFLOPS (the established baseline).

What we time:
  1. NVFP4 plugin inside a tiny TRT engine, end-to-end per-call time
  2. cuBLASLt NVFP4 GEMM directly (Python ctypes wrapper) — upper bound
  3. FP8 W8A8 cuBLASLt-direct equivalent (the 422 TFLOPS / 414 TFLOPS reference)

Headline: plugin TFLOPS, plugin speedup over FP8 baseline at the same shape.
"""

import ctypes
import os
import sys
import time

import numpy as np

os.add_dll_directory(r"C:\_dev\projects\DEMON\.venv\Lib\site-packages\tensorrt_libs")
os.add_dll_directory(r"C:\_dev\projects\DEMON\.venv\Lib\site-packages\torch\lib")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")

import torch
torch.cuda.init()
_ = torch.empty(1, device="cuda")

import tensorrt as trt

PLUGIN_DLL = r"C:\_dev\projects\DEMON\acestep\engine\trt\plugins\nvfp4_linear\nvfp4_linear_plugin.dll"
_dll = ctypes.CDLL(PLUGIN_DLL)

sys.path.insert(0, r"C:\_dev\projects\DEMON\benchmarks-pr17\cublaslt_nvfp4_spike")
from nvfp4_e2e import quantize_to_nvfp4, cublaslt_nvfp4_matmul  # noqa: E402
from nvfp4_gemm import bench_nvfp4  # noqa: E402
from torchao.prototype.mx_formats.utils import to_blocked  # noqa: E402


def build_plugin_engine(M, N, K):
    device = torch.device("cuda")
    torch.manual_seed(0)
    w = (torch.randn((N, K), dtype=torch.float32, device=device) / (K ** 0.5))
    w_pack, w_blk, w_global = quantize_to_nvfp4(w, block=16)
    w_blk_swz = to_blocked(w_blk).contiguous()

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    registry = trt.get_plugin_registry()
    creator = registry.get_creator("NVFP4Linear", "1", "")

    # Estimate the activation global scale from a typical bf16 absmax.
    # For an IID Gaussian at M=6144, K=3072, max(|x|) ~= 5.5 (sqrt(2*log(M*K))).
    # In production this comes from cal2 activation_absmax.json per-Linear.
    act_global = 5.5 / (6.0 * 448.0)
    K_arr = np.array([K], dtype=np.int32)
    N_arr = np.array([N], dtype=np.int32)
    wg_arr = np.array([float(w_global)], dtype=np.float32)
    ag_arr = np.array([act_global], dtype=np.float32)
    w_fp4_np = w_pack.cpu().numpy().flatten().astype(np.uint8)
    w_scale_np = w_blk_swz.cpu().numpy().flatten().astype(np.uint8)
    pfc = trt.PluginFieldCollection()
    pfc.append(trt.PluginField("K", K_arr, trt.PluginFieldType.INT32))
    pfc.append(trt.PluginField("N", N_arr, trt.PluginFieldType.INT32))
    pfc.append(trt.PluginField("weight_global_scale", wg_arr, trt.PluginFieldType.FLOAT32))
    pfc.append(trt.PluginField("act_global_scale", ag_arr, trt.PluginFieldType.FLOAT32))
    pfc.append(trt.PluginField("weight_fp4", w_fp4_np, trt.PluginFieldType.UNKNOWN))
    pfc.append(trt.PluginField("weight_scale", w_scale_np, trt.PluginFieldType.UNKNOWN))
    plugin = creator.create_plugin("nvfp4_linear_bench", pfc, trt.TensorRTPhase.BUILD)

    builder = trt.Builder(logger)
    network = builder.create_network(0)
    in_tensor = network.add_input("x", trt.DataType.BF16, trt.Dims2(-1, K))
    plugin_layer = network.add_plugin_v3([in_tensor], [], plugin)
    out_tensor = plugin_layer.get_output(0)
    out_tensor.name = "y"
    out_tensor.dtype = trt.DataType.BF16
    network.mark_output(out_tensor)

    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("x", (1, K), (M, K), (M, K))
    config.add_optimization_profile(profile)
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)

    serialized = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    return engine, context, w_pack, w_blk_swz, float(w_global)


def bench_plugin(M, N, K, iters=100, warmup=20):
    device = torch.device("cuda")
    engine, context, _, _, _ = build_plugin_engine(M, N, K)

    # Random bf16 input
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    y = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    context.set_input_shape("x", (M, K))
    context.set_tensor_address("x", x.data_ptr())
    context.set_tensor_address("y", y.data_ptr())

    stream = torch.cuda.Stream()
    stream_h = stream.cuda_stream
    for _ in range(warmup):
        context.execute_async_v3(stream_h)
    stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(iters):
        context.execute_async_v3(stream_h)
    end.record(stream)
    stream.synchronize()
    elapsed_ms = start.elapsed_time(end)
    per_iter_ms = elapsed_ms / iters
    tflops = (2.0 * M * N * K) / (per_iter_ms * 1e-3) / 1e12
    return per_iter_ms, tflops


def bench_torch_fp8(M, N, K, iters=100, warmup=20):
    """FP8 baseline via torch._scaled_mm (the FP8 dense peak we measured earlier)."""
    device = torch.device("cuda")
    a = torch.randn((M, K), dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    b = torch.randn((K, N), dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn).t().contiguous().t()
    scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
    scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
    for _ in range(warmup):
        _ = torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    per_iter_ms = elapsed_ms / iters
    tflops = (2.0 * M * N * K) / (per_iter_ms * 1e-3) / 1e12
    return per_iter_ms, tflops


def main():
    M, N, K = 6144, 3072, 3072
    print(f"=== Headline benchmark: M={M} N={N} K={K} (production shape) ===")
    print()
    print("torch FP8 e4m3 _scaled_mm (the FP8 baseline)...")
    fp8_ms, fp8_tf = bench_torch_fp8(M, N, K)
    print(f"  {fp8_ms:.3f} ms/iter  {fp8_tf:.0f} TFLOPS")
    print()
    print("cuBLASLt NVFP4 GEMM direct (ctypes spike, theoretical upper bound)...")
    direct_tf = bench_nvfp4(M, N, K, iters=100, warmup=20, verbose=False)
    direct_ms = (2.0 * M * N * K) / (direct_tf * 1e12) * 1e3
    print(f"  {direct_ms:.3f} ms/iter  {direct_tf:.0f} TFLOPS")
    print()
    print("NVFP4 TRT plugin (this work)...")
    plug_ms, plug_tf = bench_plugin(M, N, K)
    print(f"  {plug_ms:.3f} ms/iter  {plug_tf:.0f} TFLOPS")
    print()
    print("=" * 60)
    print(f"NVFP4 plugin speedup over FP8 baseline:   {fp8_ms / plug_ms:.2f}x")
    print(f"NVFP4 plugin TFLOPS / cuBLASLt-direct:    {plug_tf / direct_tf * 100:.1f}%")
    print(f"FP8 baseline:                              {fp8_tf:.0f} TFLOPS, {fp8_ms:.3f} ms")
    print(f"NVFP4 plugin:                              {plug_tf:.0f} TFLOPS, {plug_ms:.3f} ms")
    print(f"NVFP4 ceiling (direct cuBLASLt):           {direct_tf:.0f} TFLOPS, {direct_ms:.3f} ms")


if __name__ == "__main__":
    main()
