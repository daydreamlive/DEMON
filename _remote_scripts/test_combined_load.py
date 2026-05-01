"""Load decoder + VAE encode + VAE decode in the same process to find the segfault."""
import sys
import torch
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes

ROOT = "C:/Users/ryanf/.daydream-scope/models/rtmg/trt_engines"

print("1. loading decoder ...", flush=True)
d = engine_from_bytes(bytes_from_path(f"{ROOT}/decoder_mixed_refit_b8_240s/decoder_mixed_refit_b8_240s.engine"))
dctx = d.create_execution_context()
print("   decoder ctx OK", flush=True)

print("2. loading vae_encode_fp16_240s ...", flush=True)
e = engine_from_bytes(bytes_from_path(f"{ROOT}/vae_encode_fp16_240s/vae_encode_fp16_240s.engine"))
ectx = e.create_execution_context()
print("   vae encode ctx OK", flush=True)

print("3. loading vae_decode_fp16_240s ...", flush=True)
v = engine_from_bytes(bytes_from_path(f"{ROOT}/vae_decode_fp16_240s/vae_decode_fp16_240s.engine"))
vctx = v.create_execution_context()
print("   vae decode ctx OK", flush=True)

print("ALL OK", flush=True)
