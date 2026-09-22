"""Single-profile YuE2 VAE TensorRT probe, split into CPU export and GPU build.

Use --stage export while weights are downloading/loading; run --stage benchmark
only when the GPU is otherwise free. The initial profile is FP32 without TF32.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.spikes.yue2_feasibility import GraphCall, bench, errors, load_verified_model, timed, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["export", "benchmark"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=37)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = args.artifacts / f"vae_fp32_t{args.frames}.onnx"
    engine_path = args.artifacts / f"vae_fp32_t{args.frames}.trt"
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    from torch.nn.utils import remove_weight_norm
    started = time.perf_counter()
    model, identity = load_verified_model(args.checkpoint, vae=True, device="cpu")
    for module in model.decoder.modules():
        if hasattr(module, "weight_g"):
            remove_weight_norm(module)
    print(f"CPU decoder load: {time.perf_counter()-started:.2f}s", flush=True)
    if args.stage == "export":
        with torch.inference_mode():
            torch.onnx.export(model.decoder, (torch.zeros(1, 64, args.frames),), onnx_path,
                              input_names=["latent"], output_names=["audio"],
                              opset_version=17, dynamo=False, do_constant_folding=True)
        print(f"Exported {onnx_path}", flush=True)
        return
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    build_seconds = 0.0
    if not engine_path.exists():
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path.resolve())):
            raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        config.builder_optimization_level = 2
        config.avg_timing_iterations = 1
        config.max_aux_streams = 0
        started = time.perf_counter()
        print("Building one FP32 profile (TF32 disabled)", flush=True)
        serialized = builder.build_serialized_network(network, config)
        build_seconds = time.perf_counter() - started
        if serialized is None:
            raise RuntimeError("TensorRT build failed")
        engine_path.write_bytes(bytes(serialized))
        print(f"Build completed in {build_seconds:.2f}s", flush=True)
        del serialized, config, parser, network, builder
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    assert tuple(context.get_tensor_shape("latent")) == (1, 64, args.frames)
    latent = torch.randn(1, 64, args.frames, device="cuda", dtype=torch.float32)
    output = torch.empty(tuple(context.get_tensor_shape("audio")), device="cuda", dtype=torch.float32)
    context.set_tensor_address("latent", latent.data_ptr())
    context.set_tensor_address("audio", output.data_ptr())

    def infer(value):
        latent.copy_(value)
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return output

    with torch.inference_mode():
        model.cuda()
        reference = model.decoder(latent).clone()
        result = {"torch": torch.__version__, "tensorrt": trt.__version__, "weight_identity": identity,
                  "gpu": torch.cuda.get_device_name(), "frames": args.frames,
                  "precision": "FP32; TF32 disabled", "build_seconds": build_seconds,
                  "engine_bytes": engine_path.stat().st_size,
                  "eager": bench(lambda: model.decoder(latent)),
                  "trt": bench(lambda: infer(latent)),
                  "parity": errors(infer(latent), reference)}
        graph, capture_ms = timed(lambda: GraphCall(infer, latent))
        result["trt_graph"] = bench(lambda: graph(latent))
        result["capture_ms"] = capture_ms
        result["graph_parity"] = errors(graph(latent), reference)
        write_json(args.output, result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
