#!/usr/bin/env python3
"""Profile the full generation pipeline to find remaining optimization targets."""

import os, sys, time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
    os.path.expanduser("~"), ".cache", "torchinductor"
)

import torch
torch._dynamo.config.allow_unspec_int_on_nn_module = True

import soundfile as sf

from acestep.engine.session import Session, PreparedSource
from acestep.engine.diffusion import DiffusionConfig
from acestep.constants import TASK_INSTRUCTIONS
from acestep.nodes import Audio
from acestep.fixtures import audio_fixture


def main():
    # --- Setup (one-time costs, not profiled) ---
    s = Session(
        project_root=project_root,
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines={
            "decoder": os.path.join(project_root, "trt_engines", "decoder_mixed_refit_b8_240s", "decoder_mixed_refit_b8_240s.engine"),
            "vae_encode": os.path.join(project_root, "trt_engines", "vae_encode_fp16_240s", "vae_encode_fp16_240s.engine"),
            "vae_decode": os.path.join(project_root, "trt_engines", "vae_decode_fp16_240s", "vae_decode_fp16_240s.engine"),
        },
    )
    handler = s.handler
    engine = handler._diffusion_engine

    data, sr = sf.read(
        str(audio_fixture("inside_confusion_loop_60s_gsm.wav")),
        dtype="float32",
    )
    waveform = torch.from_numpy(data.T)[:2, :60 * 48000]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]

    source = s.prepare_source(Audio(waveform=waveform, sample_rate=48000))
    cond = s.encode_text(
        tags="deathstep", instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent, bpm=136, duration=60.0, key="G# minor",
    )

    # --- Warmup ---
    print("Warming up...")
    for _ in range(5):
        out = s.generate(
            conditioning=cond, context_latent=source.context_latent,
            source_latent=source.latent, seed=1528, denoise=0.75,
        )
        audio = s.decode(out)

    # --- Profile ---
    print("\n" + "=" * 50)
    print("  PROFILING: 30 runs, denoise=0.75, 8 steps")
    print("=" * 50)

    N = 30
    times_generate = []
    times_decode = []
    times_total = []

    for i in range(N):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = s.generate(
            conditioning=cond, context_latent=source.context_latent,
            source_latent=source.latent, seed=1528 + i, denoise=0.75,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        audio = s.decode(out)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        times_generate.append((t1 - t0) * 1000)
        times_decode.append((t2 - t1) * 1000)
        times_total.append((t2 - t0) * 1000)

    def stats(times, label):
        s = sorted(times)
        print(f"  {label:20s}  mean={sum(s)/len(s):6.1f}ms  min={s[0]:6.1f}ms  p50={s[len(s)//2]:6.1f}ms  p95={s[int(len(s)*0.95)]:6.1f}ms")

    print()
    stats(times_generate, "generate()")
    stats(times_decode, "VAE decode (TRT)")
    stats(times_total, "TOTAL (gen+dec)")

    gen_min = min(times_generate)
    dec_min = min(times_decode)
    print(f"\n  Best total: {gen_min + dec_min:.1f}ms")
    print(f"  Target: 300ms")
    print(f"  Gap: {gen_min + dec_min - 300:.1f}ms")


if __name__ == "__main__":
    main()
