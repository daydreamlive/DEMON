#!/usr/bin/env python3
"""Build the INT8 VAE decode TRT engine.

Reads the existing VAE decode ONNX export and the calibration latents
dumped by ``scripts/collect_vae_calibration.py``, runs PTQ via TRT's
calibrator, and writes the engine to
``<trt_engines>/<engine-name>/<engine-name>.engine``.

Usage:
    uv run python scripts/build_vae_int8_engine.py
    uv run python scripts/build_vae_int8_engine.py \\
        --engine-name vae_decode_int8_60s_minmax --calibrator minmax \\
        --calib-dir <path> --opt-frames 1500 --max-frames 1500
    uv run python scripts/build_vae_int8_engine.py \\
        --engine-name vae_decode_int8_60s_pin --pin-first 1 --pin-last 1 \\
        --calib-dir <path> --opt-frames 1500 --max-frames 1500
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from acestep.engine.trt.vae_decode_int8 import (
    VAEDecodeInt8Config,
    build_vae_decode_int8_engine,
)
from acestep.paths import models_dir, trt_engines_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-name", default="vae_decode_int8_15s")
    ap.add_argument("--onnx", default=None,
                    help="Default: <trt_engines>/_onnx_vae/vae_decode/vae_decode.onnx")
    ap.add_argument("--calib-dir", default=None,
                    help="Default: <models>/calibration/vae_latents")
    ap.add_argument("--workspace-gb", type=float, default=8.0)
    ap.add_argument("--min-frames", type=int, default=125)
    ap.add_argument("--opt-frames", type=int, default=250)
    ap.add_argument("--max-frames", type=int, default=375)
    ap.add_argument("--calibrator", choices=["entropy", "minmax"],
                    default="entropy")
    ap.add_argument("--pin-first", type=int, default=0,
                    help="Pin first N Conv layers to fp16 (0 = no pinning)")
    ap.add_argument("--pin-last", type=int, default=0,
                    help="Pin last N Conv layers to fp16 (0 = no pinning)")
    args = ap.parse_args()

    onnx_path = (
        Path(args.onnx) if args.onnx
        else trt_engines_dir() / "_onnx_vae" / "vae_decode" / "vae_decode.onnx"
    )
    calib_dir = (
        Path(args.calib_dir) if args.calib_dir
        else models_dir() / "calibration" / "vae_latents"
    )
    engine_path = trt_engines_dir() / args.engine_name / f"{args.engine_name}.engine"

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    latents = sorted(calib_dir.glob("latent_*.pt")) if calib_dir.exists() else []
    if not latents:
        raise FileNotFoundError(
            f"No calibration latents in {calib_dir}. "
            f"Run scripts/collect_vae_calibration.py first."
        )

    config = VAEDecodeInt8Config(
        workspace_gb=args.workspace_gb,
        min_frames=args.min_frames,
        opt_frames=args.opt_frames,
        max_frames=args.max_frames,
        calibrator_kind=args.calibrator,
        pin_first_conv=args.pin_first,
        pin_last_conv=args.pin_last,
    )
    print(f"ONNX:        {onnx_path}")
    print(f"Calibration: {calib_dir}  ({len(latents)} latents)")
    print(f"Engine:      {engine_path}")
    print(f"Profile:     min={config.min_frames}  opt={config.opt_frames}  max={config.max_frames}")
    print(f"Calibrator:  {config.calibrator_kind}")
    print(f"Pin:         first={config.pin_first_conv}  last={config.pin_last_conv}")
    print()

    build_vae_decode_int8_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        calibration_dir=calib_dir,
        config=config,
    )

    size_mb = engine_path.stat().st_size / 1e6
    print(f"\nDone: {engine_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
