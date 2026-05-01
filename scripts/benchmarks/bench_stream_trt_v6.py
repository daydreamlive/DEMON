"""Stream pipeline throughput with mixed-precision v6 TRT engine."""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
torch.set_grad_enabled(False)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRT_ENGINE = os.path.join(PROJECT_ROOT, "trt_engines", "decoder_mixed_b8_60s", "decoder_mixed_b8_60s.engine")

from acestep.engine.session import Session
from acestep.engine.diffusion import DiffusionConfig, DiffusionEngine
from acestep.engine.stream import StreamPipeline, SlotRequest


def make_request(entry, context_latents, seed=42):
    return SlotRequest(
        encoder_hidden_states=entry.encoder_hidden_states,
        encoder_attention_mask=entry.encoder_attention_mask,
        context_latents=context_latents,
        seed=seed,
    )


def bench(pipe, entry, ctx_lat, label, num_gens=24):
    for i in range(num_gens):
        pipe.submit(make_request(entry, ctx_lat, seed=1000 + i))

    completed = []
    for tick_num in range(num_gens + pipe.depth + 2):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = pipe.tick()
        torch.cuda.synchronize()
        tick_ms = (time.perf_counter() - t0) * 1000
        if result is not None:
            completed.append(tick_ms)
        if pipe.active_slots == 0 and not pipe._queue:
            break

    steady = completed[2:] if len(completed) > 2 else completed
    avg = sum(steady) / len(steady) if steady else 0
    med = sorted(steady)[len(steady) // 2] if steady else 0
    print(f"  [{label}] {len(completed)} completions, "
          f"avg={avg:.1f}ms  median={med:.1f}ms  "
          f"({1000/avg:.1f} gen/sec)" if avg > 0 else "")
    return avg


print("Loading model...")
session = Session(
    project_root=os.path.join(PROJECT_ROOT, "checkpoints"),
    use_flash_attention=True,
)
handler = session.handler
device = handler.device
dtype = handler.dtype

cond = session.encode_text(
    tags="electronic ambient, 120 bpm",
    lyrics="[instrumental]",
    duration=60.0,
)
entry = cond.to_entries()[0]

handler._ensure_silence_latent_on_device()
T = int(60.0 * 25)
ctx_lat = handler.silence_latent[:, :T, :].clone().to(device=device, dtype=dtype)
D = ctx_lat.shape[2]
cm = torch.ones(1, T, D, device=device, dtype=dtype)
context_latents = torch.cat([ctx_lat, cm], dim=-1)

config = DiffusionConfig(infer_steps=8, shift=3.0, noise_on_cpu=True)

print("\nPyTorch eager:")
engine_pt = DiffusionEngine(handler.model)
pipe_pt = StreamPipeline(engine_pt, config)
pt_avg = bench(pipe_pt, entry, context_latents, "PyTorch")

print(f"\nTRT mixed v6 ({os.path.basename(TRT_ENGINE)}):")
engine_trt = DiffusionEngine(handler.model, trt_engine_path=TRT_ENGINE)
pipe_trt = StreamPipeline(engine_trt, config)
print(f"  backend={pipe_trt.stats()['backend']}, io_dtype={pipe_trt._trt_io_dtype}")
trt_avg = bench(pipe_trt, entry, context_latents, "TRT")

print(f"\nSpeedup: {pt_avg/trt_avg:.2f}x")
