"""Single-target rebuild: decoder_mixed_refit_b8_240s (2B turbo, 240s, refit).

Test that the 2B mixed_precision recipe still builds + runs correctly with TRT 10.16.
If this works, run rebuild_all_engines.py for the rest.
"""
import gc, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_MODULES_CACHE", "C:/Users/ryanf/.cache/huggingface_modules")
os.environ.setdefault("HF_HOME", "C:/Users/ryanf/.cache/huggingface")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.set_grad_enabled(False)

from acestep.engine.model_context import ModelContext
from acestep.engine.trt.export import (
    OnnxExportConfig, TRTBuildConfig,
    export_decoder_onnx, build_trt_engine,
)
from acestep.paths import trt_engines_dir

CHECKPOINT = "acestep-v15-turbo"
NAME = "decoder_mixed_refit_b8_240s"

out_dir = trt_engines_dir() / NAME
engine_path = out_dir / f"{NAME}.engine"
out_dir.mkdir(parents=True, exist_ok=True)

# Move existing TRT 10.13 engine out of the way (keep it)
if engine_path.exists():
    backup = engine_path.with_suffix(".engine.trt10.13.bak")
    if not backup.exists():
        engine_path.rename(backup)
        print(f"backed up old engine -> {backup.name}")
    else:
        engine_path.unlink()
        print(f"old engine removed (backup already exists)")

onnx_dir = trt_engines_dir() / f"_onnx_{CHECKPOINT}" / "decoder_mixed_v6"
onnx_dir.mkdir(parents=True, exist_ok=True)
onnx_path = onnx_dir / "decoder_mixed_v6.onnx"

print("=" * 60)
print(f"Rebuilding {NAME} with TRT 10.16")
print("=" * 60)

print(f"\n[1] loading {CHECKPOINT} ...")
t0 = time.time()
ctx = ModelContext(
    config_path=CHECKPOINT,
    device="cuda",
    compile_decoder=False,
    compile_vae=False,
    skip_vae=True,
)
print(f"    loaded in {time.time()-t0:.1f}s")

print(f"\n[2] exporting ONNX (mixed_precision=True) -> {onnx_path.name}")
t0 = time.time()
export_decoder_onnx(
    ctx.model, onnx_path, device="cuda",
    config=OnnxExportConfig(mixed_precision=True, for_refit=True, batch_size=1, seq_len=1500, enc_len=200),
)
print(f"    exported in {time.time()-t0:.1f}s")

del ctx
gc.collect()
torch.cuda.empty_cache()

print(f"\n[3] building strongly_typed TRT engine (b=8, seq_max=6000, refit) ...")
config = TRTBuildConfig(
    fp16=True,
    strongly_typed=True,
    refit=True,
    workspace_gb=16.0,
    batch_min=1, batch_opt=8, batch_max=8,
    seq_min=126, seq_opt=1500, seq_max=6000,
    enc_min=32, enc_opt=200, enc_max=512,
    builder_optimization_level=3,
)
t0 = time.time()
build_trt_engine(onnx_path, engine_path, config=config)
elapsed = time.time() - t0
sz = engine_path.stat().st_size / 1e9
print(f"\nBuilt {engine_path.name} ({sz:.2f} GB) in {elapsed:.1f}s")
