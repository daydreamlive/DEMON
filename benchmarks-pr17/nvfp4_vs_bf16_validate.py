"""Validate NVFP4 XL decoder vs bf16 baseline: numeric accuracy + speed.

Counterpart to fp8_vs_bf16_validate.py for the NVFP4 plugin engine.
"""
from __future__ import annotations
import os, sys, json, time, ctypes
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
torch.set_grad_enabled(False)

# Load the NVFP4 plugin DLL BEFORE importing tensorrt so the plugin creator
# is registered when the engine deserializer looks for it.
PLUGIN_DLL = Path(__file__).resolve().parent.parent / "acestep" / "engine" / "trt" / \
    "plugins" / "nvfp4_linear" / "nvfp4_linear_plugin.dll"
venv_root = Path(sys.executable).resolve().parent.parent
site_packages = venv_root / "Lib" / "site-packages"
for d in (
    site_packages / "tensorrt_libs",
    site_packages / "torch" / "lib",
    Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
):
    if d.exists():
        os.add_dll_directory(str(d))
torch.cuda.init(); _ = torch.empty(1, device="cuda")
_plugin_dll = ctypes.CDLL(str(PLUGIN_DLL))
print(f"plugin DLL loaded: {PLUGIN_DLL}")

import tensorrt as trt

ENGINES = {
    "bf16": Path(os.path.expanduser(
        "~/.daydream-scope/models/demon/trt_engines/"
        "decoder_xl-turbo_mixed_refit_b4_60s/"
        "decoder_xl-turbo_mixed_refit_b4_60s.engine"
    )),
    "nvfp4": Path(os.path.expanduser(
        "~/.daydream-scope/models/demon/trt_engines/"
        "decoder_xl-turbo_nvfp4_refit_b4_60s/"
        "decoder_xl-turbo_nvfp4_refit_b4_60s.engine"
    )),
    # Also include slowest-fp8 anchor for direct comparison.
    "fp8_slow": Path(os.path.expanduser(
        "~/.daydream-scope/models/demon/trt_engines/"
        "decoder_xl-turbo_fp8_refit_b4_60s/"
        "decoder_xl-turbo_fp8_refit_b4_60s.engine"
    )),
}
CAL = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/calibration/decoder_xl_fp8/calibration.npz"
))
INPUT_NAMES = ("hidden_states", "timestep", "encoder_hidden_states", "context_latents")
OUTPUT_NAME = "velocity"

_TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}
if hasattr(trt, "bfloat16"):
    _TRT_TO_TORCH[trt.bfloat16] = torch.bfloat16


class EngineRunner:
    def __init__(self, label, path):
        self.label = label
        self.path = path
        if not path.exists():
            self.engine = None
            return
        self.size_mb = path.stat().st_size / 1e6
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        rt = trt.Runtime(logger)
        with open(path, "rb") as f:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load {path}")
        self.ctx = self.engine.create_execution_context()
        self.in_dtypes = {
            n: _TRT_TO_TORCH.get(self.engine.get_tensor_dtype(n), torch.float32)
            for n in INPUT_NAMES
        }
        self.out_dtype = _TRT_TO_TORCH.get(
            self.engine.get_tensor_dtype(OUTPUT_NAME), torch.float32,
        )
        self.stream = torch.cuda.Stream()

    def run(self, inputs):
        dev = torch.device("cuda")
        bufs = {}
        for n in INPUT_NAMES:
            t = inputs[n].to(device=dev, dtype=self.in_dtypes[n]).contiguous()
            bufs[n] = t
            self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, t.data_ptr())
        out_shape = tuple(self.ctx.get_tensor_shape(OUTPUT_NAME))
        out = torch.empty(out_shape, dtype=self.out_dtype, device=dev)
        self.ctx.set_tensor_address(OUTPUT_NAME, out.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return out, bufs


def main():
    print("=" * 70)
    print("NVFP4 vs bf16 vs FP8(slow) XL decoder validation")
    print("=" * 70)

    runners = {}
    for label, p in ENGINES.items():
        if not p.exists():
            print(f"  {label}: MISSING ({p})")
            continue
        r = EngineRunner(label, p)
        runners[label] = r
        print(f"  {label}: {r.size_mb:.0f} MB ({p.name})")

    # Use first calibration sample (matches FP8 validation convention)
    print(f"\nLoading calibration samples from {CAL}")
    npz = np.load(CAL)
    print(f"  hidden_states: {npz['hidden_states'].shape}")
    # NVFP4 engine has FIXED seq dim = 1500, so use seq=1500 for all.
    SEQ = 1500
    B = 4
    cal_inputs = {
        "hidden_states": torch.from_numpy(npz["hidden_states"][:B, :SEQ, :]).cuda(),
        "timestep": torch.from_numpy(npz["timestep"][:B]).cuda(),
        "encoder_hidden_states": torch.from_numpy(npz["encoder_hidden_states"][:B]).cuda(),
        "context_latents": torch.from_numpy(npz["context_latents"][:B, :SEQ, :]).cuda(),
    }
    # If hidden_states has only T < 1500 frames, pad to SEQ.
    hs = cal_inputs["hidden_states"]
    if hs.shape[1] < SEQ:
        pad = torch.zeros(B, SEQ - hs.shape[1], hs.shape[2], dtype=hs.dtype, device=hs.device)
        cal_inputs["hidden_states"] = torch.cat([hs, pad], dim=1)
    cl = cal_inputs["context_latents"]
    if cl.shape[1] < SEQ:
        pad = torch.zeros(B, SEQ - cl.shape[1], cl.shape[2], dtype=cl.dtype, device=cl.device)
        cal_inputs["context_latents"] = torch.cat([cl, pad], dim=1)
    print(f"  using B={B}, SEQ={SEQ}")

    # --- Correctness check ---
    print("\n--- Correctness ---")
    outputs = {}
    for label, r in runners.items():
        out, _ = r.run(cal_inputs)
        outputs[label] = out.float()
        print(f"  {label}: out shape={tuple(out.shape)} dtype={out.dtype} "
              f"mean={out.float().mean().item():.4g} std={out.float().std().item():.4g}")

    if "bf16" in outputs:
        for label, out in outputs.items():
            if label == "bf16":
                continue
            ref = outputs["bf16"]
            cos = torch.cosine_similarity(ref.flatten(), out.flatten(), dim=0).item()
            rel = ((out - ref).norm() / ref.norm()).item()
            print(f"  {label} vs bf16: cos = {cos:.5f}  rel_l2 = {rel:.4f}")

    # --- Speed benchmark ---
    print("\n--- Speed (per-tick, batch=4, seq=1500) ---")
    bench = {}
    for label, r in runners.items():
        for _ in range(10):
            r.run(cal_inputs)
        torch.cuda.synchronize()
        N = 30
        t0 = time.perf_counter()
        for _ in range(N):
            r.run(cal_inputs)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N * 1000
        bench[label] = dt
        print(f"  {label}: {dt:.2f} ms/tick")

    print("\n--- Speedups ---")
    if "bf16" in bench:
        anchor = bench["bf16"]
        for label, ms in bench.items():
            print(f"  {label} vs bf16: {anchor/ms:.3f}x")
    if "fp8_slow" in bench and "nvfp4" in bench:
        # "Slowest fp8" reference: the existing FP8 engine the user has built.
        print(f"  nvfp4 vs fp8_slow: {bench['fp8_slow']/bench['nvfp4']:.3f}x")

    return outputs, bench


if __name__ == "__main__":
    outputs, bench = main()
    # Save results.
    Path("benchmarks-pr17").mkdir(exist_ok=True)
    with open("benchmarks-pr17/nvfp4_vs_bf16_results.json", "w") as f:
        json.dump({
            "bench_ms": bench,
            "speedup_vs_bf16": {k: bench["bf16"]/v for k, v in bench.items()} if "bf16" in bench else {},
        }, f, indent=2)
