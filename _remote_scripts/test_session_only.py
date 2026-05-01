"""Just create the Session with TRT VAE, encode_text, then decode a dummy latent. Find the segfault site."""
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.paths import default_trt_engines

print("STAGE 1: creating Session", flush=True)
session = Session(
    config_path="acestep-v15-xl-turbo",
    decoder_backend="tensorrt",
    vae_backend="tensorrt",
    trt_engines=default_trt_engines(decoder="decoder_xl-turbo_bf16mix_dynbatch_b8_240s"),
)
print("STAGE 2: session created", flush=True)

print("STAGE 3: encode_text", flush=True)
cond = session.encode_text(
    tags="dance music",
    lyrics="[instrumental]",
    instruction=TASK_INSTRUCTIONS["text2music"],
    bpm=128, duration=58.0, key="F minor",
)
print("STAGE 4: text encoded", flush=True)

print("STAGE 5: generate", flush=True)
latent = session.generate(
    conditioning=cond, seed=42, duration=58.0, steps=8, shift=3.0,
)
print("STAGE 6: generated", flush=True)

print("STAGE 7: decode", flush=True)
audio = session.decode(latent)
print("STAGE 8: decoded", flush=True)

print("ALL OK", flush=True)
