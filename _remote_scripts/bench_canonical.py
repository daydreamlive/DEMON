#!/usr/bin/env python3
"""Bench gen + decode for one (checkpoint, decoder, duration) combo.

Follows the canonical pattern from tests/benchmarks/bench_xl_turbo.py and
bench_base_trt.py: single Session, warmup runs, sync barriers, perf_counter.

Args via env vars:
    BENCH_CHECKPOINT, BENCH_DECODER, BENCH_DURATION, BENCH_INFER_STEPS,
    BENCH_VAE (1 or 0), BENCH_LABEL, BENCH_OUT_JSON

Prints RESULT:{json} with gen + dec medians.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.paths import default_trt_engines, trt_engine_path

CHECKPOINT     = os.environ["BENCH_CHECKPOINT"]
DECODER        = os.environ.get("BENCH_DECODER")
DURATION       = float(os.environ.get("BENCH_DURATION", "58.0"))
INFER_STEPS    = int(os.environ.get("BENCH_INFER_STEPS", "8"))
USE_TRT_VAE    = os.environ.get("BENCH_VAE", "0") == "1"
LABEL          = os.environ.get("BENCH_LABEL", "bench")
WARMUP         = int(os.environ.get("BENCH_WARMUP", "3"))
RUNS           = int(os.environ.get("BENCH_RUNS", "5"))
SEED           = int(os.environ.get("BENCH_SEED", "42"))
KIND           = sys.argv[1]   # "tensorrt" or "pytorch"

vae_suffix = "240s"

if KIND == "tensorrt":
    decoder_backend = "tensorrt"
    vae_backend = "tensorrt" if USE_TRT_VAE else "eager"
    if USE_TRT_VAE:
        trt_engines = default_trt_engines(
            decoder=DECODER,
            vae_decode=f"vae_decode_fp16_{vae_suffix}",
            vae_encode=f"vae_encode_fp16_{vae_suffix}",
        )
    else:
        trt_engines = {"decoder": str(trt_engine_path(DECODER))}
elif KIND == "pytorch":
    decoder_backend = "eager"
    vae_backend = "eager"
    trt_engines = None
else:
    raise SystemExit(f"unknown kind {KIND}")

print(f"[bench] {LABEL} ({KIND})  ckpt={CHECKPOINT}  dec={DECODER}  dur={DURATION}  steps={INFER_STEPS}", flush=True)

session = Session(
    config_path=CHECKPOINT,
    decoder_backend=decoder_backend,
    vae_backend=vae_backend,
    trt_engines=trt_engines,
)

cond = session.encode_text(
    tags="dance music, four on the floor, kick drum, electronic, club, energetic synth bass, bright leads",
    lyrics="[instrumental]",
    instruction=TASK_INSTRUCTIONS["text2music"],
    bpm=128, key="F minor",
    duration=DURATION,
)

# ----- gen warmup -----
print(f"[bench] gen warmup x{WARMUP}", flush=True)
for _ in range(WARMUP):
    latent = session.generate(
        conditioning=cond, seed=SEED,
        steps=INFER_STEPS, shift=3.0, denoise=1.0,
    )

torch.cuda.synchronize()

# ----- gen timed -----
gen_times_ms = []
for i in range(RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    latent = session.generate(
        conditioning=cond, seed=SEED + i,
        steps=INFER_STEPS, shift=3.0, denoise=1.0,
    )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    gen_times_ms.append(elapsed)
    print(f"  gen run {i+1}/{RUNS}: {elapsed:.1f} ms", flush=True)

# ----- dec warmup -----
print(f"[bench] dec warmup x{WARMUP}", flush=True)
for _ in range(WARMUP):
    audio = session.decode(latent)

torch.cuda.synchronize()

# ----- dec timed -----
dec_times_ms = []
for i in range(RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    audio = session.decode(latent)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    dec_times_ms.append(elapsed)
    print(f"  dec run {i+1}/{RUNS}: {elapsed:.1f} ms", flush=True)

# ----- summary -----
gen_times_ms.sort()
dec_times_ms.sort()
gen_med = gen_times_ms[RUNS // 2]
dec_med = dec_times_ms[RUNS // 2]
result = {
    "label": LABEL,
    "kind": KIND,
    "checkpoint": CHECKPOINT,
    "decoder": DECODER or "",
    "duration": DURATION,
    "steps": INFER_STEPS,
    "warmup": WARMUP,
    "runs": RUNS,
    "gen_ms": gen_times_ms,
    "dec_ms": dec_times_ms,
    "gen_med_ms": gen_med,
    "gen_min_ms": min(gen_times_ms),
    "gen_max_ms": max(gen_times_ms),
    "dec_med_ms": dec_med,
    "dec_min_ms": min(dec_times_ms),
    "dec_max_ms": max(dec_times_ms),
    "total_med_ms": gen_med + dec_med,
}
print(f"[bench] gen median={gen_med:.0f}ms  dec median={dec_med:.0f}ms  total={gen_med+dec_med:.0f}ms", flush=True)
print("RESULT:" + json.dumps(result), flush=True)
