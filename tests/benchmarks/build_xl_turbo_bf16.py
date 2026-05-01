"""Build a bf16 TRT engine for XL turbo (one-shot driver).

Bypasses build.py's onnx-dir layout to avoid clobbering the existing
mixed-precision ONNX. Writes to a sibling 'decoder_bf16/' directory.

Output: trt_engines/decoder_xl-turbo_bf16_b1_60s/decoder_xl-turbo_bf16_b1_60s.engine
"""

import os
import sys
import time

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Suppress flash_attn import (not needed for export)
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
    # bf16 mixed precision: bf16 bulk + fp32 islands for AdaLN/timestep/
    # norms AND proj_in/proj_out (TRT lacks bf16 ConvTranspose for the
    # patch embedding shape). Trace via dynamo (legacy chokes on bf16),
    # build with strongly_typed so TRT honors the dtypes exactly. This
    # is the bf16 analog of the 2B mixed-fp16 recipe.
    onnx_dir = trt_dir / f"_onnx_{CHECKPOINT}" / "decoder_bf16_mixed_v2"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "decoder_bf16_mixed_v2.onnx"

    seq_max = DURATION_S * 25  # 25 Hz frame rate
    build_cfg = TRTBuildConfig(
        fp16=False,
        bf16=True,
        tf32=False,
        strongly_typed=True,   # respect dtypes from the bf16 ONNX exactly
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
    # Engine name: rename strongly_typed "mixed" to "bf16mix" so it's
    # obvious which recipe we're using and doesn't clobber prior builds.
    base_name = build_cfg.engine_filename().replace(".engine", "")
    if build_cfg.strongly_typed and build_cfg.bf16:
        base_name = base_name.replace("_mixed_", "_bf16mix2_")
    engine_name = base_name
    engine_dir = trt_dir / engine_name
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / f"{engine_name}.engine"

    print("=" * 70)
    print(f"BUILD XL TURBO BF16 TRT ENGINE  (fp32 ONNX -> bf16+tf32 TRT)")
    print(f"  checkpoint: {CHECKPOINT}")
    print(f"  duration:   {DURATION_S}s  (seq_max={seq_max})")
    print(f"  batch_max:  {BATCH_MAX}")
    print(f"  precision:  fp16={build_cfg.fp16} bf16={build_cfg.bf16} tf32={build_cfg.tf32} "
          f"strongly_typed={build_cfg.strongly_typed}")
    print(f"  workspace:  {WORKSPACE_GB} GB")
    print(f"  onnx:       {onnx_path}")
    print(f"  engine:     {engine_path}")
    print("=" * 70)

    if not onnx_path.exists():
        print(f"\n[1/2] Loading model and exporting ONNX (bf16_mixed, dynamo) ...")
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
        # bf16 bulk + fp32 islands for AdaLN/norms/timestep/proj_in/proj_out.
        # Traced via the dynamo exporter (legacy fails on bf16).
        export_cfg = OnnxExportConfig(
            mixed_precision=False,
            precision="bf16_mixed",
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
