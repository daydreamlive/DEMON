"""Rebuild the 240s VAE encode + decode engines with adjusted opt points.

The default VAETRTBuildConfig has decode_opt_frames=1500 and
encode_opt_samples=2880000 (60s opt point), regardless of seq_max. For
240s engines (max=6000 / 11520000) that's a 4x opt-vs-runtime mismatch.
TRT 10.16 picks narrow kernels at the opt point that fall off badly at
5950 frames in-session, even though they look fine standalone.

Defaults: opt aligned to actual b=1 240s workload (decode=5950, encode=11424000).
Override via env vars: DECODE_OPT (frames), ENCODE_OPT (samples).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acestep.engine.trt.vae_export import (
    VAETRTBuildConfig,
    build_vae_decode_engine,
    build_vae_encode_engine,
)

ROOT = Path("C:/Users/ryanf/.daydream-scope/models/rtmg/trt_engines")
ONNX_DECODE = ROOT / "_onnx_vae" / "vae_decode" / "vae_decode.onnx"
ONNX_ENCODE = ROOT / "_onnx_vae" / "vae_encode" / "vae_encode.onnx"

DEC_DIR = ROOT / "vae_decode_fp16_240s"
ENC_DIR = ROOT / "vae_encode_fp16_240s"

DECODE_OPT = int(os.environ.get("DECODE_OPT", "5950"))
ENCODE_OPT = int(os.environ.get("ENCODE_OPT", "11424000"))

config = VAETRTBuildConfig(
    fp16=True,
    workspace_gb=16.0,
    decode_min_frames=125,
    decode_opt_frames=DECODE_OPT,
    decode_max_frames=6000,
    encode_min_samples=240000,
    encode_opt_samples=ENCODE_OPT,
    encode_max_samples=11520000,
    builder_optimization_level=1,  # keep the Myelin-segfault workaround
)

print(f"VAE 240s rebuild: decode_opt={DECODE_OPT} encode_opt={ENCODE_OPT}", flush=True)

print("\n=== VAE decode 240s ===", flush=True)
t0 = time.time()
out = build_vae_decode_engine(
    onnx_path=ONNX_DECODE,
    engine_path=DEC_DIR / "vae_decode_fp16_240s.engine",
    config=config,
)
print(f"  built in {time.time()-t0:.0f}s -> {out} ({out.stat().st_size/(1<<20):.0f} MB)", flush=True)

print("\n=== VAE encode 240s ===", flush=True)
t0 = time.time()
out = build_vae_encode_engine(
    onnx_path=ONNX_ENCODE,
    engine_path=ENC_DIR / "vae_encode_fp16_240s.engine",
    config=config,
)
print(f"  built in {time.time()-t0:.0f}s -> {out} ({out.stat().st_size/(1<<20):.0f} MB)", flush=True)

print("\nDONE", flush=True)
