"""Run VAE decode in isolation (no decoder, no PT model loaded). VERBOSE TRT logs."""
import sys
import torch
import tensorrt as trt

ROOT = "C:/Users/ryanf/.daydream-scope/models/rtmg/trt_engines"
ENG = f"{ROOT}/vae_decode_fp16_240s/vae_decode_fp16_240s.engine"

print(f"TRT version: {trt.__version__}", flush=True)

# Use VERBOSE logger to see kernel-by-kernel activity during execute
logger = trt.Logger(trt.Logger.VERBOSE)
runtime = trt.Runtime(logger)

print("loading engine bytes ...", flush=True)
with open(ENG, "rb") as f:
    blob = f.read()
print(f"  {len(blob)} bytes", flush=True)

print("deserializing ...", flush=True)
engine = runtime.deserialize_cuda_engine(blob)
print(f"  num_io={engine.num_io_tensors}  num_layers={engine.num_layers if hasattr(engine, 'num_layers') else 'n/a'}", flush=True)

ctx = engine.create_execution_context()
print("ctx OK", flush=True)

# fake latents at the t2m b=1 shape: [1, 64, 1450]
B, D, T = 1, 64, 1450
latents = torch.randn(B, D, T, device="cuda", dtype=torch.float32).contiguous()
print(f"input shape {tuple(latents.shape)}", flush=True)

ctx.set_input_shape("latents", tuple(latents.shape))
ctx.set_tensor_address("latents", latents.data_ptr())

out_shape = tuple(ctx.get_tensor_shape("audio"))
print(f"output shape {out_shape}", flush=True)

audio_buf = torch.empty(out_shape, dtype=torch.float32, device="cuda")
ctx.set_tensor_address("audio", audio_buf.data_ptr())

stream = torch.cuda.Stream()
print("calling execute_async_v3 ...", flush=True)
ok = ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
torch.cuda.synchronize()
print(f"execute returned {ok}", flush=True)
print(f"audio range: min={audio_buf.min().item():.4f} max={audio_buf.max().item():.4f}  has_nan={torch.isnan(audio_buf).any().item()}", flush=True)
print("OK", flush=True)
