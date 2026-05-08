"""Build VAE decode engine with FP32 only (no FP16) to test if the
TRT 10.16 segfault is FP16-specific."""
from __future__ import annotations

import sys
sys.path.insert(0, "C:/_dev/projects/DEMON_pr17")

from pathlib import Path
from acestep.paths import trt_engines_dir
from acestep.engine.trt.vae_export import (
    build_vae_decode_engine, VAETRTBuildConfig,
)

onnx = trt_engines_dir() / "_onnx_vae" / "vae_decode" / "vae_decode.onnx"
out = trt_engines_dir() / "vae_decode_fp32_60s_test" / "vae_decode_fp32_60s_test.engine"
out.parent.mkdir(parents=True, exist_ok=True)

cfg = VAETRTBuildConfig(
    fp16=False,
    workspace_gb=4.0,
    decode_min_frames=125,
    decode_opt_frames=1500,
    decode_max_frames=1500,
)
print(f"Building {out} fp32 ws=4 ...")
build_vae_decode_engine(str(onnx), str(out), config=cfg)
print(f"DONE: {out}")
