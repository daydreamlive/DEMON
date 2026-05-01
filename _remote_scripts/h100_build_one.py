#!/usr/bin/env python3
"""Single TRT build attempt: takes config from argv, builds, exits.

Run as a subprocess so a segfault doesn't kill the orchestrator.
Exit codes: 0 = success, 1 = clean failure, 139 = SIGSEGV (caught by parent).
"""
import argparse
import os
import sys
import time

os.environ.setdefault("ACESTEP_MODELS_DIR", "/root/.daydream-scope/models/rtmg")
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
sys.path.insert(0, "/workspace/acestep")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--strongly-typed", type=int, required=True)
    p.add_argument("--opt", type=int, required=True)
    p.add_argument("--batch-max", type=int, required=True)
    p.add_argument("--set-bf16", type=int, default=0)
    p.add_argument("--set-fp16", type=int, default=0)
    p.add_argument("--workspace-gb", type=float, default=16.0)
    args = p.parse_args()

    import tensorrt as trt
    print(f"[child] TRT version: {trt.__version__}", flush=True)
    print(f"[child] strongly_typed={args.strongly_typed} opt={args.opt} bmax={args.batch_max} bf16={args.set_bf16} fp16={args.set_fp16}", flush=True)

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)

    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if args.strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(args.onnx):
        for i in range(parser.num_errors):
            print(f"[child] parse err: {parser.get_error(i)}", flush=True)
        sys.exit(2)
    print(f"[child] parsed: {network.num_layers} layers", flush=True)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    cfg.set_flag(trt.BuilderFlag.TF32)
    if not args.strongly_typed:
        if args.set_bf16:
            cfg.set_flag(trt.BuilderFlag.BF16)
        if args.set_fp16:
            cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.builder_optimization_level = args.opt

    profile = builder.create_optimization_profile()
    bmax = args.batch_max
    profile.set_shape("hidden_states", min=(1, 126, 64), opt=(bmax, 750, 64), max=(bmax, 1500, 64))
    profile.set_shape("timestep", min=(1,), opt=(bmax,), max=(bmax,))
    profile.set_shape("encoder_hidden_states", min=(1, 32, 2048), opt=(bmax, 200, 2048), max=(bmax, 512, 2048))
    profile.set_shape("context_latents", min=(1, 126, 128), opt=(bmax, 750, 128), max=(bmax, 1500, 128))
    cfg.add_optimization_profile(profile)

    print("[child] calling build_serialized_network...", flush=True)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, cfg)
    elapsed = time.time() - t0
    if serialized is None:
        print(f"[child] BUILD RETURNED None in {elapsed:.1f}s", flush=True)
        sys.exit(3)

    os.makedirs(os.path.dirname(args.engine), exist_ok=True)
    with open(args.engine, "wb") as f:
        f.write(serialized)
    sz = os.path.getsize(args.engine) / 1e9
    print(f"[child] BUILD OK in {elapsed:.1f}s, {sz:.2f} GB -> {args.engine}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
