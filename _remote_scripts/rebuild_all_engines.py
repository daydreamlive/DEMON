#!/usr/bin/env python3
"""Rebuild all decoder TRT engines after upgrading TensorRT.

For each (checkpoint, refit, max_seconds) combination:
  1. Load PT model
  2. Export ONNX (mixed_precision for 2B, bf16_mixed for XL turbo)
  3. Build strongly_typed TRT engine at b=8
  4. Save engine to canonical path
  5. Free model state, move on

Skips an engine if its target file already exists (resume-friendly).
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_MODULES_CACHE", "C:/Users/ryanf/.cache/huggingface_modules")
os.environ.setdefault("HF_HOME", "C:/Users/ryanf/.cache/huggingface")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.set_grad_enabled(False)

from acestep.engine.model_context import ModelContext
from acestep.engine.trt.export import (
    OnnxExportConfig,
    TRTBuildConfig,
    export_decoder_onnx,
    build_trt_engine,
)
from acestep.paths import trt_engines_dir


# (checkpoint, refit, max_seconds)
TARGETS = [
    ("acestep-v15-turbo", True,  240),
    ("acestep-v15-base",  True,  240),
]


def engine_filename(checkpoint, refit, max_seconds):
    if checkpoint == "acestep-v15-turbo":
        prefix = "decoder_mixed"
    elif checkpoint == "acestep-v15-base":
        prefix = "decoder_base_mixed"
    elif checkpoint == "acestep-v15-xl-turbo":
        prefix = "decoder_xl-turbo_bf16mix"
    else:
        raise ValueError(f"unknown checkpoint {checkpoint}")
    refit_tag = "_refit" if refit else ""
    return f"{prefix}{refit_tag}_b8_{max_seconds}s"


def build_one(checkpoint, refit, max_seconds, force=False):
    name = engine_filename(checkpoint, refit, max_seconds)
    out_dir = trt_engines_dir() / name
    engine_path = out_dir / f"{name}.engine"
    if engine_path.exists() and not force:
        sz = engine_path.stat().st_size / 1e9
        print(f"  SKIP (exists, {sz:.2f} GB): {engine_path}")
        return engine_path, "skipped"

    print(f"\n=== {name} ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ONNX cache path: under the existing _onnx_<checkpoint> tree so we don't
    # pollute trt_engines/_onnx_<checkpoint>
    onnx_dir = trt_engines_dir() / f"_onnx_{checkpoint}" / "decoder_mixed_v2"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "decoder_mixed_v2.onnx"

    # 1. Load model
    print(f"[1] loading {checkpoint} ...")
    t0 = time.time()
    ctx = ModelContext(
        config_path=checkpoint,
        device="cuda",
        compile_decoder=False,
        compile_vae=False,
        skip_vae=True,
    )
    print(f"    loaded in {time.time()-t0:.1f}s")

    # 2. Export ONNX (always re-export to keep ONNX consistent with current code)
    print(f"[2] exporting ONNX (mixed_precision=True) ...")
    cfg = OnnxExportConfig(
        mixed_precision=True,
        batch_size=1,
        seq_len=1500,
        enc_len=200,
        for_refit=refit,
    )
    t0 = time.time()
    export_decoder_onnx(ctx.model, onnx_path, device="cuda", config=cfg)
    print(f"    exported in {time.time()-t0:.1f}s")

    # Free PT model before TRT build
    del ctx
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Build TRT engine
    print(f"[3] building TRT engine ...")
    seq_max = max_seconds * 25
    seq_opt = min(seq_max, 750)
    config = TRTBuildConfig(
        fp16=True,
        strongly_typed=True,
        workspace_gb=16.0,
        batch_min=1, batch_opt=8, batch_max=8,
        seq_min=126, seq_opt=seq_opt, seq_max=seq_max,
        enc_min=32, enc_opt=200, enc_max=512,
        builder_optimization_level=3,
        refit=refit,
        variant="turbo" if "turbo" in checkpoint else "base",
    )
    t0 = time.time()
    build_trt_engine(onnx_path, engine_path, config=config)
    elapsed = time.time() - t0
    sz = engine_path.stat().st_size / 1e9
    print(f"    built in {elapsed:.1f}s, {sz:.2f} GB -> {engine_path}")
    return engine_path, f"built_{elapsed:.0f}s_{sz:.2f}GB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="overwrite existing engines")
    p.add_argument("--only", default=None, help="comma-separated checkpoint list to limit")
    args = p.parse_args()

    only = set(args.only.split(",")) if args.only else None

    results = []
    for checkpoint, refit, max_seconds in TARGETS:
        if only and checkpoint not in only:
            continue
        try:
            ep, status = build_one(checkpoint, refit, max_seconds, force=args.force)
            results.append((engine_filename(checkpoint, refit, max_seconds), status))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((engine_filename(checkpoint, refit, max_seconds), f"FAIL: {e}"))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, status in results:
        print(f"  {name:<50} {status}")


if __name__ == "__main__":
    main()
