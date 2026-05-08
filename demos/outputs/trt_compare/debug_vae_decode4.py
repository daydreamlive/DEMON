"""Try TRT Python API directly without polygraphy."""
from __future__ import annotations

import sys
sys.path.insert(0, "C:/_dev/projects/DEMON_pr17")

import torch
torch.set_grad_enabled(False)

from acestep.paths import trt_engine_path

ENGINE = str(trt_engine_path("vae_decode_fp16_60s"))

import tensorrt as trt
print(f"tensorrt: {trt.__version__}")

logger = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(logger, "")

with open(ENGINE, "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

ctx = engine.create_execution_context()
print(f"engine loaded num_io={engine.num_io_tensors}")

device = torch.device("cuda")
shape = (1, 64, 125)
torch.manual_seed(0)
lat = torch.zeros(shape, device=device, dtype=torch.float32).contiguous()
ctx.set_input_shape("latents", shape)
ctx.set_tensor_address("latents", lat.data_ptr())
out_shape = tuple(ctx.get_tensor_shape("audio"))
print(f"out shape {out_shape}")
audio = torch.empty(out_shape, dtype=torch.float32, device=device)
ctx.set_tensor_address("audio", audio.data_ptr())

ts = torch.cuda.Stream(device=device)
print(f"using torch stream {ts.cuda_stream}", flush=True)
ok = ctx.execute_async_v3(ts.cuda_stream)
print(f"ok={ok}", flush=True)
torch.cuda.synchronize()
print(f"audio[0,0,:3]={audio[0,0,:3].cpu().tolist()}")
