"""Test VAE decode at multiple shapes to see if it's shape-dependent.

Args: <T_frames>   e.g. 125, 1450, 1500, 6000
"""
import sys
import torch
import tensorrt as trt

ROOT = "C:/Users/ryanf/.daydream-scope/models/rtmg/trt_engines"
ENG = f"{ROOT}/vae_decode_fp16_240s/vae_decode_fp16_240s.engine"

T = int(sys.argv[1])

print(f"TRT version: {trt.__version__}", flush=True)
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

print("loading ...", flush=True)
with open(ENG, "rb") as f:
    blob = f.read()
engine = runtime.deserialize_cuda_engine(blob)
ctx = engine.create_execution_context()
print(f"  num_layers={engine.num_layers}", flush=True)

B, D = 1, 64
latents = torch.randn(B, D, T, device="cuda", dtype=torch.float32).contiguous()
print(f"input T={T} -> shape {tuple(latents.shape)}", flush=True)

ctx.set_input_shape("latents", tuple(latents.shape))
ctx.set_tensor_address("latents", latents.data_ptr())

out_shape = tuple(ctx.get_tensor_shape("audio"))
print(f"output {out_shape}", flush=True)

audio_buf = torch.empty(out_shape, dtype=torch.float32, device="cuda")
ctx.set_tensor_address("audio", audio_buf.data_ptr())

stream = torch.cuda.Stream()
print("execute_async_v3 ...", flush=True)
ok = ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
torch.cuda.synchronize()
print(f"OK ret={ok}  out range [{audio_buf.min().item():.4f},{audio_buf.max().item():.4f}]  has_nan={torch.isnan(audio_buf).any().item()}", flush=True)
