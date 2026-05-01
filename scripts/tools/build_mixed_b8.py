"""Export mixed-precision ONNX and build strongly_typed TRT engine at batch=8.

The mixed-precision ONNX has fp16 for attention/MLP and fp32 for
timestep embedding, AdaLN, and RMSNorm. Building with strongly_typed=True
tells TRT to respect those annotations, so GEMMs run in fp16 while
precision-critical ops stay in fp32.
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
torch.set_grad_enabled(False)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "trt_engines")

# ---------------------------------------------------------------
# Step 1: Export mixed-precision ONNX
# ---------------------------------------------------------------
print("=" * 60)
print("Step 1: Export mixed-precision ONNX")
print("=" * 60)

from acestep.handler import AceStepHandler
from acestep.engine.trt.export import (
    export_decoder_onnx, build_trt_engine,
    OnnxExportConfig, TRTBuildConfig,
)

print("Loading model...")
handler = AceStepHandler()
handler.initialize_service(
    project_root=PROJECT_ROOT,
    config_path="acestep-v15-turbo",
    device="cuda",
    use_flash_attention=False,  # SDPA for export
    compile_model=False,
)

onnx_dir = os.path.join(OUTPUT_DIR, "mixed_v6")
os.makedirs(onnx_dir, exist_ok=True)
onnx_path = os.path.join(onnx_dir, "decoder_mixed_v6.onnx")
print(f"Exporting to {os.path.basename(onnx_path)}...")

t0 = time.time()
export_decoder_onnx(
    handler.model,
    onnx_path,
    device="cuda",
    config=OnnxExportConfig(mixed_precision=True),
)
print(f"ONNX export done in {time.time() - t0:.0f}s")

# Free model memory
del handler
torch.cuda.empty_cache()

# ---------------------------------------------------------------
# Step 2: Build strongly_typed TRT engine at batch=8
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Build strongly_typed TRT engine (batch=8)")
print("=" * 60)

config = TRTBuildConfig(
    fp16=True,
    strongly_typed=True,
    workspace_gb=16.0,
    batch_min=1, batch_opt=8, batch_max=8,
    seq_min=126, seq_opt=1500, seq_max=1500,
    enc_min=32, enc_opt=200, enc_max=512,
    builder_optimization_level=3,
)

engine_path = os.path.join(OUTPUT_DIR, config.engine_filename())
print(f"Building {os.path.basename(engine_path)}...")
print(f"  strongly_typed=True, fp16=True, opt_level=5")
print(f"  ONNX: {os.path.basename(onnx_path)} (mixed precision)")

t0 = time.time()
build_trt_engine(onnx_path, engine_path, config=config)
elapsed = time.time() - t0

size_mb = os.path.getsize(engine_path) / (1 << 20)
print(f"\nDone: {os.path.basename(engine_path)} ({size_mb:.0f} MB) in {elapsed:.0f}s")
