"""Smoke test for the NVFP4 ONNX patcher + TRT parser + plugin chain.

Builds a tiny bf16 ONNX with one MatMul, patches it through `nvfp4_onnx`,
parses the result via TRT's ONNX parser (with the plugin DLL preloaded),
builds an engine, runs inference, compares against bf16 reference.

If this works end-to-end, we're safe to push the patcher onto the full XL
bf16 ONNX (8 GB) and build the production engine.
"""

import ctypes
import json
import os
import sys
import tempfile

import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as nph

os.add_dll_directory(r"C:\_dev\projects\DEMON\.venv\Lib\site-packages\tensorrt_libs")
os.add_dll_directory(r"C:\_dev\projects\DEMON\.venv\Lib\site-packages\torch\lib")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")

import torch
torch.cuda.init()
_ = torch.empty(1, device="cuda")

import tensorrt as trt

PLUGIN_DLL = r"C:\_dev\projects\DEMON\acestep\engine\trt\plugins\nvfp4_linear\nvfp4_linear_plugin.dll"
_dll = ctypes.CDLL(PLUGIN_DLL)
print(f"plugin DLL loaded: {_dll}")

# Patcher import (after the DLL is loaded just to make sure)
sys.path.insert(0, r"C:\_dev\projects\DEMON")
from acestep.engine.trt.nvfp4_onnx import patch_bf16_onnx_to_nvfp4  # noqa: E402


def make_tiny_bf16_onnx(out_path: str, M: int, K: int, N: int, seed: int = 0):
    """Build a tiny bf16 ONNX: input x (M, K), weight (K, N) initializer, output (M, N)."""
    torch.manual_seed(seed)
    w_fp32 = (torch.randn((K, N), dtype=torch.float32) / (K ** 0.5)).contiguous()
    w_bf16 = w_fp32.to(torch.bfloat16).contiguous()
    w_init = onnx.TensorProto()
    w_init.name = "W"
    w_init.data_type = int(onnx.TensorProto.BFLOAT16)
    w_init.dims.extend([K, N])
    w_init.raw_data = w_bf16.view(torch.uint16).numpy().tobytes()

    x_vi = helper.make_tensor_value_info("x", onnx.TensorProto.BFLOAT16, [None, K])
    y_vi = helper.make_tensor_value_info("y", onnx.TensorProto.BFLOAT16, [None, N])

    matmul = helper.make_node(
        "MatMul",
        inputs=["x", "W"],
        outputs=["y"],
        name="my_matmul",
    )

    graph = helper.make_graph(
        nodes=[matmul],
        name="tiny",
        inputs=[x_vi],
        outputs=[y_vi],
        initializer=[w_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 20)],
        ir_version=10,
    )
    onnx.save(
        model, out_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(out_path) + ".data",
    )
    return w_fp32


def write_absmax_json(out_path: str, K: int, N: int, w_l2: float, x_absmax: float):
    """Build a minimal activation_absmax.json that the patcher can use."""
    # The patcher matches (onnx_shape, l2_bf16_rounded). Our ONNX weight has
    # shape (K, N) and L2 = w_l2; we provide a single Linear entry with the
    # corresponding torch [out, in] = (N, K) shape so the (transposed) lookup
    # matches.
    payload = {
        "linears": {
            "my_linear": {
                "absmax": x_absmax,
                "p99": x_absmax * 0.5,
                "p99_9": x_absmax * 0.8,
                "p99_99": x_absmax,
                "per_channel_absmax": [x_absmax] * K,
                "in_features": K,
                "weight_shape": [N, K],   # torch convention [out, in]
                "weight_l2_bf16": w_l2,
                "weight_head4_bf16": [0.0] * 4,
                "output_absmax": x_absmax * 2.0,
            }
        }
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)


def main():
    M, N, K = 128, 64, 128
    device = torch.device("cuda")
    with tempfile.TemporaryDirectory() as td:
        bf16_path = os.path.join(td, "tiny_bf16.onnx")
        absmax_json = os.path.join(td, "absmax.json")

        # Build tiny bf16 ONNX.
        w_fp32 = make_tiny_bf16_onnx(bf16_path, M=M, K=K, N=N)
        # Compute L2 the same way fp8_onnx._weight_l2_bf16 does (bf16 -> fp32 -> L2)
        w_bf16 = w_fp32.to(torch.bfloat16)
        w_l2 = float(w_bf16.to(torch.float32).pow(2).sum().sqrt().item())

        # Synthetic activation and its absmax.
        torch.manual_seed(1)
        x_fp32 = torch.randn((M, K), dtype=torch.float32, device=device)
        x_bf16 = x_fp32.to(torch.bfloat16)
        x_absmax = float(x_bf16.float().abs().max().item())
        write_absmax_json(absmax_json, K=K, N=N, w_l2=w_l2, x_absmax=x_absmax)

        # Patch.
        nvfp4_path = os.path.join(td, "tiny_nvfp4.onnx")
        result = patch_bf16_onnx_to_nvfp4(
            bf16_onnx_path=bf16_path,
            activation_absmax_json_path=absmax_json,
            output_path=nvfp4_path,
            force=True,
        )
        print(f"\npatched: {result}")
        assert os.path.exists(nvfp4_path)

        # Parse with TRT (plugin DLL was already loaded above).
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, "")
        builder = trt.Builder(logger)
        # NetworkDefinitionCreationFlag.STRONGLY_TYPED keeps bf16 throughout
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        parser = trt.OnnxParser(network, logger)
        with open(nvfp4_path, "rb") as f:
            ok = parser.parse(f.read(), nvfp4_path)
        if not ok:
            print(f"ONNX parse FAILED:")
            for i in range(parser.num_errors):
                print("  ", parser.get_error(i))
            return 1
        print(f"ONNX parsed OK: {network.num_layers} layers")
        for i in range(network.num_layers):
            lyr = network.get_layer(i)
            print(f"  layer[{i}]: {lyr.name} type={lyr.type}")

        # Make output bf16 binding explicit (we hit this bug in test_plugin.py)
        for oi in range(network.num_outputs):
            out = network.get_output(oi)
            out.dtype = trt.DataType.BF16
            print(f"  output[{oi}]: name={out.name} dtype={out.dtype} shape={out.shape}")

        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        profile.set_shape("x", (1, K), (M, K), (M, K))
        config.add_optimization_profile(profile)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)

        print("Building engine...")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            print("ERROR: build_serialized_network returned None")
            return 1
        print(f"Engine built: {serialized.nbytes} bytes")

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        context = engine.create_execution_context()

        y_bf16 = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
        context.set_input_shape("x", (M, K))
        context.set_tensor_address("x", x_bf16.contiguous().data_ptr())
        context.set_tensor_address("y", y_bf16.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream
        ok = context.execute_async_v3(stream)
        torch.cuda.synchronize()
        if not ok:
            print("ERROR: execute_async_v3 returned False")
            return 1

        # Reference
        ref = (x_bf16.float() @ w_fp32.to(device)).float()
        out = y_bf16.float()
        cos = torch.cosine_similarity(ref.flatten(), out.flatten(), dim=0).item()
        rel = ((out - ref).norm() / ref.norm()).item()
        print(f"\nplugin output (via ONNX parser) vs bf16 ref:")
        print(f"  cos sim: {cos:.6f}")
        print(f"  rel L2:  {rel:.6f}")
        print(f"  ref [0,:4]: {ref[0, :4].tolist()}")
        print(f"  out [0,:4]: {out[0, :4].tolist()}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
