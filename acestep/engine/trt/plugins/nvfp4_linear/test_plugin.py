"""Numerical test: build a tiny TRT engine with one NVFP4Linear plugin node,
run it, compare against the Python ctypes NVFP4 GEMM reference.

If the plugin output matches the reference (cos sim ~1.0), the C++ plugin
is functionally equivalent to the Python wrapper.
"""

import ctypes
import os
import sys
import numpy as np

# DLL search dirs (must register before importing tensorrt / loading plugin)
os.add_dll_directory(r"C:\_dev\projects\DEMON\.venv\Lib\site-packages\tensorrt_libs")
os.add_dll_directory(r"C:\_dev\projects\DEMON\.venv\Lib\site-packages\torch\lib")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")

import torch
torch.cuda.init()
_ = torch.empty(1, device="cuda")

import tensorrt as trt

# Load plugin DLL (this triggers static REGISTER_TENSORRT_PLUGIN)
PLUGIN_DLL = r"C:\_dev\projects\DEMON\acestep\engine\trt\plugins\nvfp4_linear\nvfp4_linear_plugin.dll"
_dll = ctypes.CDLL(PLUGIN_DLL)

# Bring in the ctypes reference for ground-truth quantization
sys.path.insert(0, r"C:\_dev\projects\DEMON\benchmarks-pr17\cublaslt_nvfp4_spike")
from nvfp4_e2e import quantize_to_nvfp4, cublaslt_nvfp4_matmul  # noqa: E402
from torchao.prototype.mx_formats.utils import to_blocked  # noqa: E402


def main():
    M, N, K = 128, 64, 128
    print(f"Test shape: M={M} N={N} K={K}")
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Generate weight and activation
    x = torch.randn((M, K), dtype=torch.float32, device=device)
    w = torch.randn((N, K), dtype=torch.float32, device=device) / (K ** 0.5)
    print(f"REF: x[0, :16] (fp32):", x[0, :16].cpu().tolist())
    print(f"REF: x[0, :16] (bf16):", x.to(torch.bfloat16)[0, :16].float().cpu().tolist())

    # bf16 reference
    ref = (x.to(torch.bfloat16) @ w.to(torch.bfloat16).T).float()

    # Quantize weight (this is what we pass as plugin attributes)
    w_pack, w_blk_scale, w_global = quantize_to_nvfp4(w, block=16)
    w_blk_swz = to_blocked(w_blk_scale).contiguous()

    # Ground-truth: also quantize activation and run the ctypes wrapper
    x_pack, x_blk_scale, x_global = quantize_to_nvfp4(x, block=16)
    x_blk_swz = to_blocked(x_blk_scale).contiguous()
    alpha_py = float(x_global) * float(w_global)
    print(f"REF: x_global={float(x_global):.6g} alpha={alpha_py:.6g}")
    print(f"REF: x_pack[0:8]: {x_pack[0,:8].cpu().numpy().tolist()}")
    print(f"REF: x_blk_swz flat[0:8]: {x_blk_swz.flatten()[:8].cpu().numpy().tolist()}")
    ref_nvfp4 = cublaslt_nvfp4_matmul(
        x_pack, w_pack, x_blk_swz, w_blk_swz, alpha_py, M, N, K
    ).float()
    cos_ref = torch.cosine_similarity(ref.flatten(), ref_nvfp4.flatten(), dim=0).item()
    print(f"ctypes NVFP4 vs bf16 ref: cos sim = {cos_ref:.6f}")

    # --- Now build a TRT engine with the plugin ---
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    registry = trt.get_plugin_registry()

    creator = registry.get_creator("NVFP4Linear", "1", "")
    if creator is None:
        print("ERROR: NVFP4Linear creator not found in registry")
        return 1
    print(f"Got creator: {creator.name} v{creator.plugin_version}")

    # Build plugin field collection. Keep all numpy arrays alive so TRT can
    # read their buffers (PluginField stores a pointer; if the array goes out
    # of scope, the pointer dangles).
    # Static cal-baked activation global scale: max(|x|) / (FP4_MAX * FP8_E4M3_MAX)
    act_global = float(x.abs().max().item() / (6.0 * 448.0))
    print(f"act_global_scale (test-derived): {act_global:.6g}")
    K_arr = np.array([K], dtype=np.int32)
    N_arr = np.array([N], dtype=np.int32)
    wg_arr = np.array([float(w_global)], dtype=np.float32)
    ag_arr = np.array([act_global], dtype=np.float32)
    w_fp4_np = w_pack.cpu().numpy().flatten().astype(np.uint8)
    w_scale_np = w_blk_swz.cpu().numpy().flatten().astype(np.uint8)
    print(f"weight_fp4 buffer: {w_fp4_np.nbytes} bytes")
    print(f"weight_scale buffer: {w_scale_np.nbytes} bytes")

    pfc = trt.PluginFieldCollection()
    pfc.append(trt.PluginField("K", K_arr, trt.PluginFieldType.INT32))
    pfc.append(trt.PluginField("N", N_arr, trt.PluginFieldType.INT32))
    pfc.append(trt.PluginField("weight_global_scale", wg_arr, trt.PluginFieldType.FLOAT32))
    pfc.append(trt.PluginField("act_global_scale", ag_arr, trt.PluginFieldType.FLOAT32))
    pfc.append(trt.PluginField("weight_fp4", w_fp4_np, trt.PluginFieldType.UNKNOWN))
    pfc.append(trt.PluginField("weight_scale", w_scale_np, trt.PluginFieldType.UNKNOWN))

    plugin = creator.create_plugin("nvfp4_linear_test", pfc, trt.TensorRTPhase.BUILD)
    if plugin is None:
        print("ERROR: create_plugin returned None")
        return 1
    print(f"Plugin created: {plugin}")

    # Network
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch is default in TRT 10
    in_tensor = network.add_input("x", trt.DataType.BF16, trt.Dims2(-1, K))
    plugin_layer = network.add_plugin_v3([in_tensor], [], plugin)
    out_tensor = plugin_layer.get_output(0)
    out_tensor.name = "y"
    out_tensor.dtype = trt.DataType.BF16  # force bf16 binding (no implicit cast)
    network.mark_output(out_tensor)

    # Build config
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("x", (1, K), (M, K), (M, K))
    config.add_optimization_profile(profile)
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)

    print("Building engine...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: build_serialized_network returned None")
        return 1
    print(f"Engine built: {serialized.nbytes} bytes")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()

    # Inspect engine layers / IO
    print(f"\nEngine info:")
    print(f"  num_io_tensors: {engine.num_io_tensors}")
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(nm)
        shape = engine.get_tensor_shape(nm)
        mode = engine.get_tensor_mode(nm)
        print(f"  io[{i}] name='{nm}' mode={mode} dtype={dtype} shape={shape}")
    print(f"  num_layers: {engine.num_layers}")
    inspector = engine.create_engine_inspector()
    print(inspector.get_engine_information(trt.LayerInformationFormat.ONELINE))

    # Allocate input/output (bf16). Zero the output to detect what cuBLASLt writes.
    x_bf16 = x.to(torch.bfloat16).contiguous()
    out_bf16 = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    context.set_input_shape("x", (M, K))
    context.set_tensor_address("x", x_bf16.data_ptr())
    context.set_tensor_address("y", out_bf16.data_ptr())
    print(f"Test x_bf16.data_ptr() = 0x{x_bf16.data_ptr():016x}")
    print(f"Test out_bf16.data_ptr() = 0x{out_bf16.data_ptr():016x}")

    stream = torch.cuda.current_stream().cuda_stream
    if not context.execute_async_v3(stream):
        print("ERROR: execute_async_v3 failed")
        return 1
    torch.cuda.synchronize()

    plugin_out = out_bf16.float()
    print(f"\nplugin output (bf16 view) row 0: {plugin_out[0].tolist()[:32]}")
    print(f"plugin output (bf16 view) row 0 [last 16]: {plugin_out[0, -16:].tolist()}")
    print(f"plugin output (bf16 view) row 1 [first 16]: {plugin_out[1, :16].tolist()}")
    # FP32 reinterpretation of entire output
    fp32_view = out_bf16.view(torch.uint16).cpu().numpy().tobytes()
    fp32_arr = np.frombuffer(fp32_view, dtype=np.float32).reshape(M, N // 2)
    print(f"FP32 reinterpret (M, N//2) shape: {fp32_arr.shape}")
    print(f"  fp32[0, :4]: {fp32_arr[0, :4].tolist()}")
    print(f"  fp32[0, -4:]: {fp32_arr[0, -4:].tolist()}")
    print(f"  fp32[-1, :4]: {fp32_arr[-1, :4].tolist()}")
    cos_fp32_repro = float(np.dot(fp32_arr.flatten(), ref_nvfp4[:, :N//2].cpu().numpy().flatten()) /
                           (np.linalg.norm(fp32_arr) * np.linalg.norm(ref_nvfp4[:, :N//2].cpu().numpy())))
    print(f"  cos(fp32_arr, ref_nvfp4[:, :N//2]): {cos_fp32_repro:.6f}")

    # HYPOTHESIS: output may actually be FP32, not bf16. Reinterpret to check.
    out_view_as_fp32 = out_bf16.view(torch.uint16).cpu().numpy().tobytes()
    # Try first half as FP32 (M*N/2 elements, but we want to see if pattern fits)
    out_fp32 = np.frombuffer(out_view_as_fp32, dtype=np.float32)
    print(f"\nReinterpret output buffer as FP32 (first 4 of row 0):")
    print(f"  {out_fp32[:4].tolist()}")

    # Compare dumped plugin-side packed_act / swz_scales vs Python reference
    if os.path.exists("plugin_packed_act.bin"):
        py_fp4 = x_pack.cpu().numpy().flatten().astype(np.uint8)
        py_swz = x_blk_swz.cpu().numpy().flatten().astype(np.uint8)
        plug_fp4 = np.fromfile("plugin_packed_act.bin", dtype=np.uint8)
        plug_swz = np.fromfile("plugin_swz_scales.bin", dtype=np.uint8)
        print(f"\nFP4 packed: plugin shape={plug_fp4.shape} python shape={py_fp4.shape}")
        print(f"  bytes equal: {(plug_fp4 == py_fp4).sum()} / {plug_fp4.size}")
        diff_fp4 = np.where(plug_fp4 != py_fp4)[0]
        print(f"  first 10 diff positions: {diff_fp4[:10].tolist()}")
        if len(diff_fp4) > 0:
            for i in diff_fp4[:5]:
                print(f"    byte[{i}] plugin=0x{plug_fp4[i]:02x} python=0x{py_fp4[i]:02x}")
        print(f"\nSwizzled scales: plugin shape={plug_swz.shape} python shape={py_swz.shape}")
        print(f"  bytes equal: {(plug_swz == py_swz).sum()} / {plug_swz.size}")
        diff_swz = np.where(plug_swz != py_swz)[0]
        print(f"  first 10 diff positions: {diff_swz[:10].tolist()}")

    # Run cublasLt with the plugin's EXACT dumped data to see what cuBLASLt
    # produces given the plugin-side quantized inputs.
    if os.path.exists("plugin_packed_act.bin"):
        plug_fp4_t = torch.from_numpy(np.fromfile("plugin_packed_act.bin", dtype=np.uint8)).view(M, K // 2).to(device)
        plug_swz_t = torch.from_numpy(np.fromfile("plugin_swz_scales.bin", dtype=np.uint8)).to(device)
        # Recompute alpha as the plugin would (act_global * weight_global). We
        # need the plugin's act_global; read it from the kernel debug print
        # or recompute as max(|x_bf16|) / 2688.
        x_bf16_view = x.to(torch.bfloat16).contiguous().float()
        plug_act_g = x_bf16_view.abs().max().item() / (6.0 * 448.0)
        plug_alpha = plug_act_g * float(w_global)
        print(f"\nPython-side rerun of cuBLASLt with plugin-quantized data:")
        print(f"  plug_act_g = {plug_act_g:.6g}  plug_alpha = {plug_alpha:.6g}")
        plug_repro = cublaslt_nvfp4_matmul(
            plug_fp4_t, w_pack, plug_swz_t, w_blk_swz, plug_alpha, M, N, K,
        ).float()
        cos_repro = torch.cosine_similarity(ref.flatten(), plug_repro.flatten(), dim=0).item()
        cos_match = torch.cosine_similarity(plug_repro.flatten(), plugin_out.flatten(), dim=0).item()
        print(f"  cublasLt-repro vs bf16 ref:  cos = {cos_repro:.6f}")
        print(f"  cublasLt-repro vs plugin:    cos = {cos_match:.6f}")
        print(f"  repro  [0,:4]: {plug_repro[0,:4].tolist()}")

    cos_plugin_vs_ref = torch.cosine_similarity(ref.flatten(), plugin_out.flatten(), dim=0).item()
    cos_plugin_vs_ctypes = torch.cosine_similarity(ref_nvfp4.flatten(), plugin_out.flatten(), dim=0).item()
    rel_plugin = ((plugin_out - ref).norm() / ref.norm()).item()
    print()
    print(f"plugin vs bf16 ref:      cos = {cos_plugin_vs_ref:.6f}   rel_l2 = {rel_plugin:.4f}")
    print(f"plugin vs ctypes NVFP4:  cos = {cos_plugin_vs_ctypes:.6f}")
    print()
    print(f"ref       [0,:4]: {ref[0, :4].tolist()}")
    print(f"ref_nvfp4 [0,:4]: {ref_nvfp4[0, :4].tolist()}")
    print(f"plugin    [0,:4]: {plugin_out[0, :4].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
