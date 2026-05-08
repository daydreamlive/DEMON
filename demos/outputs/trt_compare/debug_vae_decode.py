"""Minimal repro: load a TRT VAE decode engine and run a single inference."""
from __future__ import annotations

import sys
from pathlib import Path

# Toggle worktree: pass --pr to use PR-17 code path, --main to use main
USE_PR = "--pr" in sys.argv
ROOT = Path("C:/_dev/projects/DEMON_pr17") if USE_PR else Path("C:/_dev/projects/DEMON")
sys.path.insert(0, str(ROOT))

import torch
torch.set_grad_enabled(False)

from acestep.paths import trt_engine_path

ENGINE = str(trt_engine_path("vae_decode_fp16_60s"))
print(f"engine: {ENGINE}")
print(f"using {'PR-17' if USE_PR else 'main'} acestep")

import tensorrt as trt
print(f"tensorrt: {trt.__version__}")

from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from polygraphy import cuda as pg_cuda

engine = engine_from_bytes(bytes_from_path(ENGINE))
ctx = engine.create_execution_context()
stream = pg_cuda.Stream()
print(f"engine loaded, num_io={engine.num_io_tensors}")

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    dtype = engine.get_tensor_dtype(name)
    print(f"  tensor {i}: {name} mode={mode} dtype={dtype}")

device = torch.device("cuda")
T = 1500
torch.manual_seed(1)
lat = torch.randn(1, 64, T, device=device, dtype=torch.float32).contiguous()
print(f"lat: {lat.shape} {lat.dtype}")

ok = ctx.set_input_shape("latents", tuple(lat.shape))
print(f"set_input_shape: {ok}")
ok = ctx.set_tensor_address("latents", lat.data_ptr())
print(f"set_tensor_address(in): {ok}")

missing = ctx.infer_shapes()
print(f"infer_shapes missing: {missing}")

out_shape = tuple(ctx.get_tensor_shape("audio"))
print(f"output shape: {out_shape}")

audio = torch.empty(out_shape, dtype=torch.float32, device=device)
ok = ctx.set_tensor_address("audio", audio.data_ptr())
print(f"set_tensor_address(out): {ok}")

print("execute_async_v3 ...")
ok = ctx.execute_async_v3(stream.ptr)
print(f"execute_async_v3: {ok}")
stream.synchronize()
print(f"audio sample: {audio[0, 0, :5].cpu().tolist()}")
