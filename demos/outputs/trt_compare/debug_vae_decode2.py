"""Try several shapes and inputs to isolate the 10.16 VAE decode segfault."""
from __future__ import annotations

import sys, os
from pathlib import Path

sys.path.insert(0, "C:/_dev/projects/DEMON_pr17")

import torch
torch.set_grad_enabled(False)

from acestep.paths import trt_engine_path

ENGINE = str(trt_engine_path("vae_decode_fp16_60s"))
print(f"engine: {ENGINE}")

import tensorrt as trt
print(f"tensorrt: {trt.__version__}")

from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from polygraphy import cuda as pg_cuda

engine = engine_from_bytes(bytes_from_path(ENGINE))
ctx = engine.create_execution_context()
stream = pg_cuda.Stream()

device = torch.device("cuda")

# Probe profile range
mn, op, mx = engine.get_tensor_profile_shape("latents", 0)
print(f"profile min={tuple(mn)} opt={tuple(op)} max={tuple(mx)}")

shapes_to_try = [
    (1, 64, 125),   # min
    (1, 64, 250),
    (1, 64, 500),
    (1, 64, 750),
    (1, 64, 1500),  # max
]

for shape in shapes_to_try:
    print(f"\n--- shape {shape} ---", flush=True)
    torch.manual_seed(0)
    lat = torch.zeros(shape, device=device, dtype=torch.float32).contiguous()
    ctx.set_input_shape("latents", shape)
    ctx.set_tensor_address("latents", lat.data_ptr())
    out_shape = tuple(ctx.get_tensor_shape("audio"))
    print(f"out shape {out_shape}", flush=True)
    audio = torch.empty(out_shape, dtype=torch.float32, device=device)
    ctx.set_tensor_address("audio", audio.data_ptr())
    print("execute...", flush=True)
    ok = ctx.execute_async_v3(stream.ptr)
    print(f"ok={ok}", flush=True)
    stream.synchronize()
    print(f"audio[0,0,:3]={audio[0,0,:3].cpu().tolist()}", flush=True)
