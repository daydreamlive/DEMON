"""Build VAE engines without the fp16 builder flag (fp32 path)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acestep.engine.trt.vae_export import (
    VAETRTBuildConfig,
    build_vae_decode_engine,
    build_vae_encode_engine,
)
from acestep.paths import trt_engines_dir

ROOT = trt_engines_dir()
ONNX = ROOT / "_onnx_vae" / "vae_decode" / "vae_decode.onnx"
ONNX_ENC = ROOT / "_onnx_vae" / "vae_encode" / "vae_encode.onnx"

assert ONNX.exists() and ONNX_ENC.exists(), f"missing ONNX files; ensure build.py was run first"

cfg = VAETRTBuildConfig(
    fp16=False,
    decode_min_frames=125,
    decode_opt_frames=1500,
    decode_max_frames=1500,  # 60s
    encode_min_samples=240000,
    encode_opt_samples=2880000,
    encode_max_samples=2880000,
)

# Build to a separate path so we don't clobber the canonical one
out_dec = ROOT / "vae_decode_fp32_60s_test" / "vae_decode_fp32_60s_test.engine"
out_dec.parent.mkdir(parents=True, exist_ok=True)
print(f"building decode -> {out_dec}", flush=True)
build_vae_decode_engine(ONNX, out_dec, config=cfg)
print(f"  size: {out_dec.stat().st_size/1e6:.0f} MB")

print("done")
