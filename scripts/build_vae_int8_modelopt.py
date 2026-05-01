#!/usr/bin/env python3
"""Build INT8 VAE decode engine via NVIDIA Model Optimizer (`modelopt`).

Pipeline:
  1. Load .pt calibration latents.
  2. Run modelopt's ONNX quantizer with the chosen calibration method
     (Percentile / Entropy / MinMax / Distribution). It runs the ONNX in
     onnxruntime, hooks activations, computes per-tensor scales, and emits
     a QDQ-annotated ONNX.
  3. Build a TRT engine from the QDQ ONNX — TRT picks INT8 tactics where
     QDQ pairs are present and FP16 elsewhere. No TRT calibrator needed.

Usage:
    uv run python scripts/build_vae_int8_modelopt.py \\
        --engine-name vae_decode_int8_60s_modelopt_pct \\
        --calib-dir <latents> --calibration-method Percentile

Notes:
  * `Percentile` uses the 99.999th percentile by default, configurable via
    --percentile.
  * `Entropy` is ORT's MinMax+Entropy combo (similar to TRT's
    EntropyCalibrator2 but with ORT's tensor naming, so the resulting
    scales differ slightly).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from acestep.paths import models_dir, trt_engines_dir


def _load_latents(calib_dir: Path, max_count: Optional[int] = None) -> list[np.ndarray]:
    paths = sorted(calib_dir.glob("latent_*.pt"))
    if max_count is not None:
        paths = paths[:max_count]
    if not paths:
        raise FileNotFoundError(f"No latents in {calib_dir}")
    out = []
    for p in paths:
        t = torch.load(p, weights_only=True, map_location="cpu")
        out.append(t.float().numpy().astype(np.float32))
    return out


def _make_calib_dict(latents: list[np.ndarray], input_name: str = "latents") -> dict:
    """Stack [1, 64, T] latents along batch dim to {input_name: [N, 64, T]}.

    modelopt's CalibrationDataProvider expects a dict mapping each ONNX
    input to a numpy tensor whose first dim is total calibration samples;
    it slices internally one batch at a time.
    """
    arr = np.concatenate([x.reshape(-1, *x.shape[1:]) for x in latents], axis=0)
    return {input_name: arr}


def _build_trt_from_qdq(qdq_onnx: Path, engine_path: Path, *,
                       min_frames: int, opt_frames: int, max_frames: int,
                       workspace_gb: float = 8.0):
    """Build a TRT engine from a QDQ-quantized ONNX.

    TRT auto-detects INT8 from the embedded QDQ pairs; no calibrator is
    needed. We still set INT8 + FP16 builder flags so non-QDQ ops can use
    fp16 fallback.
    """
    import tensorrt as trt
    from loguru import logger

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(qdq_onnx.resolve())):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: %s", parser.get_error(i))
        raise RuntimeError(f"ONNX parsing failed: {qdq_onnx}")

    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )
    build_config.set_flag(trt.BuilderFlag.INT8)
    build_config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "latents",
        min=(1, 64, min_frames),
        opt=(1, 64, opt_frames),
        max=(1, 64, max_frames),
    )
    build_config.add_optimization_profile(profile)

    logger.info(f"Building TRT engine from QDQ ONNX: {qdq_onnx}")
    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed (see TRT logger output)")
    engine_path.write_bytes(serialized)

    size_mb = engine_path.stat().st_size / (1 << 20)
    logger.info(f"Engine written: {engine_path} ({size_mb:.1f} MB)")
    return engine_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-name", required=True)
    ap.add_argument("--onnx", default=None,
                    help="Default: <trt_engines>/_onnx_vae/vae_decode/vae_decode.onnx")
    ap.add_argument("--calib-dir", default=None,
                    help="Default: <models>/calibration/vae_latents_60s")
    ap.add_argument("--num-calib", type=int, default=128,
                    help="Max calibration latents to use (default: 128)")
    ap.add_argument("--calibration-method",
                    choices=["Entropy", "MinMax", "Percentile", "Distribution"],
                    default="Percentile")
    ap.add_argument("--percentile", type=float, default=99.999,
                    help="Percentile for Percentile calibrator (default: 99.999)")
    ap.add_argument("--workspace-gb", type=float, default=8.0)
    ap.add_argument("--min-frames", type=int, default=125)
    ap.add_argument("--opt-frames", type=int, default=1500)
    ap.add_argument("--max-frames", type=int, default=1500)
    ap.add_argument("--keep-qdq-onnx", action="store_true",
                    help="Keep the intermediate QDQ ONNX file alongside the engine")
    ap.add_argument("--autotune", action="store_true",
                    help="Use modelopt autotune: per-region quantization "
                         "scheme search via trtexec timing. Slow (30-90 min) "
                         "but can find Pareto-better configs than the default "
                         "all-or-nothing modelopt build.")
    ap.add_argument("--autotune-baseline", default=None,
                    help="Path to an existing QDQ ONNX to seed the autotune "
                         "search. Default: G2's QDQ ONNX "
                         "(vae_decode_int8_60s_modelopt_ent32.qdq.onnx) if it "
                         "exists, otherwise unseeded.")
    ap.add_argument("--autotune-num-schemes", type=int, default=50,
                    help="Schemes to search per region (default: 50)")
    args = ap.parse_args()

    onnx_path = (
        Path(args.onnx) if args.onnx
        else trt_engines_dir() / "_onnx_vae" / "vae_decode" / "vae_decode.onnx"
    )
    calib_dir = (
        Path(args.calib_dir) if args.calib_dir
        else models_dir() / "calibration" / "vae_latents_60s"
    )
    engine_dir = trt_engines_dir() / args.engine_name
    engine_path = engine_dir / f"{args.engine_name}.engine"
    qdq_onnx_path = engine_dir / f"{args.engine_name}.qdq.onnx"
    engine_dir.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    latents = _load_latents(calib_dir, max_count=args.num_calib)
    print(f"ONNX:        {onnx_path}")
    print(f"Calibration: {calib_dir}  ({len(latents)} latents)")
    print(f"Method:      {args.calibration_method}"
          + (f"  (percentile={args.percentile})"
             if args.calibration_method == "Percentile" else ""))
    print(f"Engine:      {engine_path}")
    print(f"Profile:     min={args.min_frames}  opt={args.opt_frames}  max={args.max_frames}")
    print()

    # ------ modelopt ONNX quantization ------
    print("Running modelopt ONNX quantization...")
    t0 = time.time()
    import modelopt.onnx.quantization as moq
    calib = _make_calib_dict(latents, input_name="latents")
    print(f"  calibration tensor shape: {calib['latents'].shape}")
    quantize_kwargs = dict(
        onnx_path=str(onnx_path),
        quantize_mode="int8",
        calibration_data=calib,
        calibration_method=args.calibration_method,
        output_path=str(qdq_onnx_path),
        high_precision_dtype="fp16",
        simplify=True,
    )
    if args.autotune:
        baseline = args.autotune_baseline
        if baseline is None:
            default_baseline = (
                trt_engines_dir() / "vae_decode_int8_60s_modelopt_ent32"
                / "vae_decode_int8_60s_modelopt_ent32.qdq.onnx"
            )
            baseline = str(default_baseline) if default_baseline.exists() else None
        autotune_dir = engine_dir / "autotune_state"
        autotune_dir.mkdir(parents=True, exist_ok=True)
        print(f"  autotune ENABLED")
        print(f"    baseline:        {baseline or '(none — will start from scratch)'}")
        print(f"    schemes/region:  {args.autotune_num_schemes}")
        print(f"    state dir:       {autotune_dir}")
        quantize_kwargs.update(
            autotune=True,
            autotune_use_trtexec=True,
            autotune_qdq_baseline=baseline,
            autotune_output_dir=str(autotune_dir),
            autotune_num_schemes_per_region=args.autotune_num_schemes,
            autotune_state_file=str(autotune_dir / "state.json"),
            autotune_pattern_cache_file=str(autotune_dir / "pattern_cache.json"),
            autotune_timing_cache=str(autotune_dir / "timing.cache"),
            autotune_verbose=True,
        )
    moq.quantize(**quantize_kwargs)
    print(f"  modelopt done in {time.time() - t0:.0f}s -> {qdq_onnx_path}")

    # ------ TRT engine build from QDQ ONNX ------
    print("Building TRT engine from QDQ ONNX...")
    t0 = time.time()
    _build_trt_from_qdq(
        qdq_onnx_path, engine_path,
        min_frames=args.min_frames, opt_frames=args.opt_frames,
        max_frames=args.max_frames, workspace_gb=args.workspace_gb,
    )
    print(f"  TRT done in {time.time() - t0:.0f}s")

    # Always keep the QDQ ONNX when autotune is set — the autotuned QDQ
    # is the artifact we want to inspect and possibly reuse.
    if not args.keep_qdq_onnx and not args.autotune:
        try:
            qdq_onnx_path.unlink()
        except FileNotFoundError:
            pass

    size_mb = engine_path.stat().st_size / 1e6
    print(f"\nDone: {engine_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
