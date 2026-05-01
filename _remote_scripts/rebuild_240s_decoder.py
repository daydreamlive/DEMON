"""Rebuild the 2B turbo 240s decoder engines with seq_opt at the actual workload.

Default seq_opt=750 (30s) is wrong for a 240s engine — TRT 10.16 picks kernels
that are catastrophically slow far from the opt point. Setting seq_opt=5950
matches the actual b=1 240s workload.

Rebuilds both refit and non-refit variants.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acestep.engine.trt.export import build_trt_engine, TRTBuildConfig

ROOT = Path("C:/Users/ryanf/.daydream-scope/models/rtmg/trt_engines")
ONNX_DIR = ROOT / "_onnx_acestep-v15-turbo"

JOBS = [
    {
        "name": "decoder_mixed_refit_b8_240s",
        "onnx": ONNX_DIR / "decoder_refit" / "decoder_refit.onnx",
        "refit": True,
    },
]

for job in JOBS:
    out_dir = ROOT / job["name"]
    out_path = out_dir / f"{job['name']}.engine"

    config = TRTBuildConfig(
        fp16=True,
        strongly_typed=True,
        refit=job["refit"],
        workspace_gb=16.0,
        batch_min=1,
        batch_opt=1,
        batch_max=8,
        seq_min=126,
        seq_opt=int(os.environ.get("SEQ_OPT", "5950")),
        seq_max=6000,
        enc_min=32,
        enc_opt=200,
        enc_max=512,
        builder_optimization_level=3,
        variant="turbo",
    )

    print(f"\n=== Rebuilding {job['name']} (refit={job['refit']}, seq_opt=5950) ===", flush=True)
    print(f"  onnx: {job['onnx']}", flush=True)
    print(f"  out: {out_path}", flush=True)

    t0 = time.time()
    build_trt_engine(job["onnx"], out_path, config=config)
    print(f"  built in {time.time()-t0:.0f}s", flush=True)
    size = out_path.stat().st_size / (1 << 20)
    print(f"  size: {size:.0f} MB", flush=True)

print("\nDONE", flush=True)
