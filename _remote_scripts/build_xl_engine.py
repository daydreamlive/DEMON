#!/usr/bin/env python3
"""H100: export XL turbo decoder to ONNX (bf16_mixed) → patch dynbatch → build TRT engine.

Single-file pipeline so we don't need to scp multiple debug scripts. Cleans up
the intermediate ONNX after the engine is built.
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

# Repo root
sys.path.insert(0, "/workspace/acestep")
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("ACESTEP_MODELS_DIR", "/root/.daydream-scope/models/rtmg")

from acestep.engine.model_context import ModelContext
from acestep.engine.trt.export import (
    OnnxExportConfig,
    TRTBuildConfig,
    export_decoder_onnx,
    build_trt_engine,
)


def patch_dynbatch(in_onnx: Path, out_onnx: Path) -> None:
    """Replace any reshape that hardcodes batch=1 with [-1, ...]."""
    print(f"\n=== dynbatch patch: {in_onnx.name} -> {out_onnx.name} ===")
    model = onnx.load(str(in_onnx), load_external_data=False)
    graph = model.graph

    def get_const_shape(name):
        for node in graph.node:
            if node.op_type == "Constant" and name in node.output:
                for attr in node.attribute:
                    if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                        return numpy_helper.to_array(attr.t).flatten().tolist()
        for init in graph.initializer:
            if init.name == name:
                return numpy_helper.to_array(init).flatten().tolist()
        return None

    def set_const_shape(name, new_shape):
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

    fixes = 0
    seen = set()
    interesting = []
    for node in graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shape_name = node.input[1]
        shape_arr = get_const_shape(shape_name)
        if not shape_arr:
            continue
        if shape_arr[0] == 1 and len(shape_arr) >= 2:
            interesting.append((node.name or "(no-name)", shape_name, shape_arr))
            if -1 not in shape_arr[1:] and shape_name not in seen:
                if set_const_shape(shape_name, [-1] + list(shape_arr[1:])):
                    fixes += 1
                    seen.add(shape_name)

    print(f"  reshapes with batch=1 first dim: {len(interesting)}")
    for n, sn, s in interesting[:8]:
        print(f"    {n} (const {sn}): {s}")
    print(f"  patched {fixes} unique constants")

    # Save protobuf only (no external data rewrite). Caller MUST place
    # out_onnx in the same directory as in_onnx so the protobuf's
    # relative external_data references still resolve.
    assert out_onnx.parent == in_onnx.parent, (
        f"out_onnx ({out_onnx.parent}) must be in same dir as in_onnx ({in_onnx.parent}) "
        f"so external data references resolve"
    )
    onnx.save(model, str(out_onnx))
    print(f"  saved -> {out_onnx} ({out_onnx.stat().st_size/1e6:.1f} MB protobuf)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="acestep-v15-xl-turbo")
    p.add_argument("--variant", default="xl-turbo")
    p.add_argument("--batch-max", type=int, default=8)
    p.add_argument("--seq-max", type=int, default=1500)
    p.add_argument("--seq-opt", type=int, default=750)
    p.add_argument("--opt-level", type=int, default=3)
    p.add_argument("--workdir", default="/workspace/build")
    args = p.parse_args()

    work = Path(args.workdir)
    onnx_dir = work / "onnx"
    engine_dir = Path("/root/.daydream-scope/models/rtmg/trt_engines") / f"decoder_xl-turbo_bf16mix_dynbatch_b8_240s"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = onnx_dir / "decoder_bf16_mixed.onnx"
    # Patched protobuf MUST live in same dir as original so external_data
    # references resolve; use a different filename
    onnx_dynbatch_path = onnx_dir / "decoder_bf16_mixed_dynbatch.onnx"
    engine_path = engine_dir / f"decoder_xl-turbo_bf16mix_dynbatch_b8_240s.engine"

    print("=" * 70)
    print(f"XL turbo TRT build pipeline (H100)")
    print("=" * 70)
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  workdir:    {work}")
    print(f"  engine out: {engine_path}")
    print(f"  free disk:  ", end=""); os.system("df -h / | tail -1")
    print()

    if onnx_path.exists():
        ext_files = list(onnx_dir.glob("*"))
        ext_total = sum(f.stat().st_size for f in ext_files) / 1e9
        print(f"=== [1+2] ONNX already exists, skipping load+export ===")
        print(f"  {onnx_path}")
        print(f"  protobuf: {onnx_path.stat().st_size/1e6:.1f} MB")
        print(f"  total dir: {len(ext_files)} files, {ext_total:.2f} GB")
    else:
        print("=== [1] Loading PT model for ONNX export ===")
        t0 = time.time()
        ctx = ModelContext(
            config_path=args.checkpoint,
            device="cuda",
            compile_decoder=False,
            compile_vae=False,
            skip_vae=True,
        )
        print(f"  loaded in {time.time()-t0:.1f}s")

        print("\n=== [2] Exporting decoder to ONNX (bf16_mixed) ===")
        cfg = OnnxExportConfig(
            precision="bf16_mixed",
            batch_size=1,
            seq_len=1500,
            enc_len=200,
        )
        t0 = time.time()
        export_decoder_onnx(ctx.model, onnx_path, device="cuda", config=cfg)
        print(f"  exported in {time.time()-t0:.1f}s")
        print(f"  onnx size:  {onnx_path.stat().st_size / 1e9:.2f} GB")
        os.system("df -h / | tail -1")

        # Free PT model BEFORE building TRT (frees ~8 GB GPU memory and host RAM)
        del ctx
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== [3] Patching ONNX for dynamic batch ===")
    patch_dynbatch(onnx_path, onnx_dynbatch_path)
    os.system("df -h / | tail -1")

    print("\n=== [4] Building TRT engine ===")
    build_cfg = TRTBuildConfig(
        fp16=False,
        bf16=False,
        tf32=True,
        workspace_gb=16.0,
        batch_min=1, batch_opt=args.batch_max, batch_max=args.batch_max,
        seq_min=126, seq_opt=args.seq_opt, seq_max=args.seq_max,
        enc_min=32, enc_opt=200, enc_max=512,
        builder_optimization_level=args.opt_level,
        strongly_typed=True,
        refit=False,
        variant=args.variant,
    )
    t0 = time.time()
    build_trt_engine(onnx_dynbatch_path, engine_path, config=build_cfg)
    print(f"  built in {time.time()-t0:.1f}s")
    print(f"  engine size: {engine_path.stat().st_size / 1e9:.2f} GB")

    print("\n=== [5] Cleanup intermediate ONNX (~8 GB) ===")
    if onnx_dir.exists():
        for f in onnx_dir.glob("*"):
            f.unlink()
        onnx_dir.rmdir()
    print("  done")
    os.system("df -h / | tail -1")

    print("\n=== ENGINE READY ===")
    print(engine_path)


if __name__ == "__main__":
    main()
