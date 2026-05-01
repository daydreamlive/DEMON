"""Build XL turbo TRT engine via fp16 + per-layer attention fp32 islands.

Same recipe as the 2B turbo mixed-fp16 build, plus _AttnFp32Wrapper
around layers whose q_norm/k_norm weights exceed the safety threshold
(XL turbo's outlier layers 0 and 30 with q/k_norm absmax ~14-31).

Goal: match 2B turbo's TRT speed profile (fp16 kernels for the bulk)
while keeping the few overflow-prone layers numerically safe.

Output: trt_engines/decoder_xl-turbo_attnsafe_b8_60s.engine
"""

import os
import sys
import time

# fp16 path uses legacy torchscript exporter; dynamo not needed.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Suppress flash_attn import
import importlib, importlib.util
_orig = importlib.util.find_spec
def _patch(name, *a, **k):
    if "flash_attn" in str(name):
        return None
    return _orig(name, *a, **k)
importlib.util.find_spec = _patch

import torch
torch.set_grad_enabled(False)

from acestep.engine.model_context import ModelContext
from acestep.engine.trt.export import (
    OnnxExportConfig, TRTBuildConfig,
    export_decoder_onnx, build_trt_engine,
)
from acestep.paths import checkpoints_dir, trt_engines_dir

CHECKPOINT = "acestep-v15-xl-turbo"
DURATION_S = 60
BATCH_MAX = 8
WORKSPACE_GB = 16.0


def main():
    trt_dir = trt_engines_dir()
    # v2: every self_attn wrapped in fp32 (vs v1 which only wrapped the
    # q_norm outlier layers). Different output path so v1 artifacts stay.
    onnx_dir = trt_dir / f"_onnx_{CHECKPOINT}" / "decoder_fp16_attn_safe_v2"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "decoder_fp16_attn_safe_v2.onnx"

    seq_max = DURATION_S * 25
    build_cfg = TRTBuildConfig(
        fp16=True,
        bf16=False,
        tf32=True,
        strongly_typed=True,   # 2B-style: dtypes baked into ONNX
        refit=False,
        workspace_gb=WORKSPACE_GB,
        batch_min=1,
        batch_opt=1,
        batch_max=BATCH_MAX,
        seq_min=126,
        seq_opt=750,
        seq_max=seq_max,
        enc_min=32,
        enc_opt=200,
        enc_max=512,
        builder_optimization_level=3,
        variant="xl-turbo",
    )

    base_name = build_cfg.engine_filename().replace(".engine", "")
    # engine_filename() returns "_mixed_" for strongly_typed; rename so
    # this engine is identifiable and doesn't clobber other recipes.
    base_name = base_name.replace("_mixed_", "_attnsafe2_")
    engine_dir = trt_dir / base_name
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / f"{base_name}.engine"

    print("=" * 70)
    print(f"BUILD XL TURBO fp16_attn_safe TRT ENGINE")
    print(f"  checkpoint: {CHECKPOINT}")
    print(f"  duration:   {DURATION_S}s  (seq_max={seq_max})")
    print(f"  batch_max:  {BATCH_MAX}")
    print(f"  precision:  fp16 mixed + per-layer attn fp32 islands, strongly_typed")
    print(f"  workspace:  {WORKSPACE_GB} GB")
    print(f"  onnx:       {onnx_path}")
    print(f"  engine:     {engine_path}")
    print("=" * 70)

    if not onnx_path.exists():
        print(f"\n[1/2] Loading model and exporting ONNX (fp16_attn_safe) ...")
        t0 = time.time()
        ctx = ModelContext(
            project_root=str(checkpoints_dir()),
            config_path=CHECKPOINT,
            device="cuda",
            use_flash_attention=False,
            skip_vae=True,
        )
        print(f"  model loaded in {time.time() - t0:.1f}s")

        t0 = time.time()
        export_cfg = OnnxExportConfig(
            mixed_precision=False,
            precision="fp16_attn_safe",
            for_refit=False,
        )
        with ctx._load_model_context("model"):
            export_decoder_onnx(ctx.model, onnx_path, device="cuda", config=export_cfg)
        print(f"  ONNX exported in {time.time() - t0:.1f}s")
        del ctx
        torch.cuda.empty_cache()
    else:
        print(f"\n[1/2] Reusing existing ONNX at {onnx_path}")

    print(f"\n[2/2] Building TRT engine ...")
    t0 = time.time()
    build_trt_engine(onnx_path, engine_path, config=build_cfg)
    print(f"  Engine built in {time.time() - t0:.1f}s")

    size_gb = engine_path.stat().st_size / (1 << 30)
    print(f"\nDONE  engine={engine_path}  size={size_gb:.2f} GB")


if __name__ == "__main__":
    main()
