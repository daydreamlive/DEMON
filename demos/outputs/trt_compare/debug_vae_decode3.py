"""Try torch CUDA stream + null stream + polygraphy stream."""
from __future__ import annotations

import sys
sys.path.insert(0, "C:/_dev/projects/DEMON_pr17")

import torch
torch.set_grad_enabled(False)

from acestep.paths import trt_engine_path

ENGINE = str(trt_engine_path("vae_decode_fp16_60s"))

import tensorrt as trt
print(f"tensorrt: {trt.__version__}")

from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from polygraphy import cuda as pg_cuda

engine = engine_from_bytes(bytes_from_path(ENGINE))
ctx = engine.create_execution_context()

device = torch.device("cuda")
shape = (1, 64, 125)
torch.manual_seed(0)
lat = torch.zeros(shape, device=device, dtype=torch.float32).contiguous()
ctx.set_input_shape("latents", shape)
ctx.set_tensor_address("latents", lat.data_ptr())
out_shape = tuple(ctx.get_tensor_shape("audio"))
audio = torch.empty(out_shape, dtype=torch.float32, device=device)
ctx.set_tensor_address("audio", audio.data_ptr())

# Try 1: null stream (0)
print("Try 1: null stream (0) ...", flush=True)
ok = ctx.execute_async_v3(0)
print(f"  ok={ok}", flush=True)
torch.cuda.synchronize()
print(f"  audio[0,0,:3]={audio[0,0,:3].cpu().tolist()}", flush=True)

# Try 2: torch stream
print("Try 2: torch stream ...", flush=True)
ts = torch.cuda.Stream(device=device)
print(f"  cuda_stream={ts.cuda_stream}", flush=True)
ok = ctx.execute_async_v3(ts.cuda_stream)
print(f"  ok={ok}", flush=True)
torch.cuda.synchronize()
print(f"  audio[0,0,:3]={audio[0,0,:3].cpu().tolist()}", flush=True)

# Try 3: polygraphy stream
print("Try 3: polygraphy stream ...", flush=True)
ps = pg_cuda.Stream()
print(f"  ptr={ps.ptr}", flush=True)
ok = ctx.execute_async_v3(ps.ptr)
print(f"  ok={ok}", flush=True)
ps.synchronize()
print(f"  audio[0,0,:3]={audio[0,0,:3].cpu().tolist()}", flush=True)
