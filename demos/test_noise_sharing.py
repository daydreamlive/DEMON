"""Test noise sharing in StreamPipeline.

Gen 1: cover from source audio (seeds _last_noise).
Gen 2+: pure text-to-music (no source_latents), noise_sharing carries forward.
Runs twice: noise_sharing=0.0 (baseline) and noise_sharing=0.5.
Saves all outputs to _output/noise_sharing/ for comparison.

Usage:
    uv run python demos/test_noise_sharing.py [audio_file]
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

import soundfile as sf

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session, PreparedSource
from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.stream import StreamPipeline, SlotRequest
from acestep.nodes.types import Audio, Latent
from acestep.paths import project_root, checkpoints_dir, select_trt_engines
from acestep.fixtures import audio_fixture

PROJECT_ROOT = project_root()
DEFAULT_AUDIO = audio_fixture("inside_confusion_loop_60s_gsm.wav")
SAMPLE_RATE = 48000
T = 1500  # 60s at 25fps
NUM_GENS = 5  # total generations per run (1 seeded + N-1 pure)
SEED = 1528


def run_pipeline(pipe, entry, context_latents, source_latents, device, dtype, session, label, out_dir):
    """Run NUM_GENS generations through the pipeline and save audio."""
    results = []

    for i in range(NUM_GENS):
        if i == 0:
            # Gen 1: use source audio as stand-in for t2m output
            req = SlotRequest(
                encoder_hidden_states=entry.encoder_hidden_states,
                encoder_attention_mask=entry.encoder_attention_mask,
                context_latents=context_latents,
                seed=SEED,
                source_latents=source_latents,
                denoise=0.7,
            )
        else:
            # Gen 2+: pure generation, no source_latents
            req = SlotRequest(
                encoder_hidden_states=entry.encoder_hidden_states,
                encoder_attention_mask=entry.encoder_attention_mask,
                context_latents=context_latents,
                seed=SEED + i,
            )
        pipe.submit(req)

    # Tick until all results are out
    max_ticks = NUM_GENS + pipe.depth + 5
    for _ in range(max_ticks):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = pipe.tick()
        torch.cuda.synchronize()
        tick_ms = (time.perf_counter() - t0) * 1000

        if result is not None:
            results.append(result)
            print(f"  [{label}] gen {len(results)}/{NUM_GENS}  tick={tick_ms:.0f}ms")

        if len(results) >= NUM_GENS:
            break

    # Decode and save
    for i, lat in enumerate(results):
        t0 = time.perf_counter()
        audio_out = session.decode(Latent(tensor=lat))
        torch.cuda.synchronize()
        dec_ms = (time.perf_counter() - t0) * 1000

        wav = audio_out.waveform.detach().cpu().float().squeeze(0).numpy()  # [C, samples]
        fname = out_dir / f"{label}_gen{i+1}.wav"
        sf.write(str(fname), wav.T, SAMPLE_RATE)
        print(f"  [{label}] saved {fname.name}  decode={dec_ms:.0f}ms")


def main():
    audio_path = DEFAULT_AUDIO
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])

    out_dir = PROJECT_ROOT / "_output" / "noise_sharing"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Noise Sharing Test")
    print("=" * 60)

    # -- Setup (same as realtime_motion). T=1500 → 60s workload. --
    trt_engines = select_trt_engines(duration_s=T / 25.0)

    print("[Setup] Loading model...")
    t0 = time.time()
    session = Session(
        project_root=str(checkpoints_dir()),
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines=trt_engines,
    )
    handler = session.handler
    device, dtype = handler.device, handler.dtype
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    print("[Setup] Loading source audio...")
    data, sr = sf.read(str(audio_path), dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform[:2, :int(60.0 * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    audio_in = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)

    print("[Setup] VAE encode + semantic extract...")
    latent = session.encode_audio(audio_in)
    context_latent = session.extract_hints(latent)
    source = PreparedSource(latent=latent, context_latent=context_latent)

    print("[Setup] Text encode...")
    cond = session.encode_text(
        tags="deathstep, heavy bass, dark atmosphere",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136, duration=60.0, key="G# minor",
    )
    entry = cond.to_entries()[0]

    ctx_lat = source.context_latent.tensor.to(device=device, dtype=dtype)
    D = ctx_lat.shape[2]
    cm = torch.ones(1, T, D, device=device, dtype=dtype)
    context_latents = torch.cat([ctx_lat, cm], dim=-1)
    source_latents = source.latent.tensor.to(device=device, dtype=dtype)

    engine = handler._diffusion_engine

    # -- Run: no noise sharing (baseline) --
    print("\n" + "=" * 60)
    print("Baseline: noise_sharing=0.0")
    print("=" * 60)
    config = DiffusionConfig(infer_steps=8, shift=3.0, noise_on_cpu=True)
    pipe_baseline = StreamPipeline(engine, config, noise_sharing=0.0)
    run_pipeline(pipe_baseline, entry, context_latents, source_latents,
                 device, dtype, session, "baseline", out_dir)

    # -- Run: noise sharing 0.5 --
    print("\n" + "=" * 60)
    print("Noise sharing: noise_sharing=0.5")
    print("=" * 60)
    pipe_shared = StreamPipeline(engine, config, noise_sharing=0.5)
    run_pipeline(pipe_shared, entry, context_latents, source_latents,
                 device, dtype, session, "shared05", out_dir)

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
