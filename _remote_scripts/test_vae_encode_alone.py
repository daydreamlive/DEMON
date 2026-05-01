"""Test VAE encode in isolation."""
import torch
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes

ROOT = "C:/Users/ryanf/.daydream-scope/models/rtmg/trt_engines"

print("loading engine ...", flush=True)
engine = engine_from_bytes(bytes_from_path(f"{ROOT}/vae_encode_fp16_240s/vae_encode_fp16_240s.engine"))
ctx = engine.create_execution_context()
print("engine + ctx OK", flush=True)

# audio at 60s 48kHz stereo: [1, 2, 2880000]
B, C, S = 1, 2, 2880000
audio = torch.randn(B, C, S, device="cuda", dtype=torch.float32).contiguous()
print(f"input shape {tuple(audio.shape)}", flush=True)

ctx.set_input_shape("audio", tuple(audio.shape))
ctx.set_tensor_address("audio", audio.data_ptr())

out_shape = tuple(ctx.get_tensor_shape("moments"))
print(f"output shape {out_shape}", flush=True)

moments = torch.empty(out_shape, dtype=torch.float32, device="cuda")
ctx.set_tensor_address("moments", moments.data_ptr())

stream = torch.cuda.Stream()
print("execute_async_v3 ...", flush=True)
ok = ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
torch.cuda.synchronize()
print(f"execute returned {ok}", flush=True)
print(f"moments range: min={moments.min().item():.4f} max={moments.max().item():.4f}", flush=True)
print("OK", flush=True)
