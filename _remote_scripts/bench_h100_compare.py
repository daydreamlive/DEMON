#!/usr/bin/env python3
"""H100 t2m A/B: PT eager vs TRT decoder, eager VAE for both passes.

Same prompt, seeds, and steps as the local Windows run so timings and audio
can be compared apples-to-apples. Eager VAE on both sides isolates the
decoder backend as the only difference (we don't have a VAE TRT engine
built on H100, and the goal here is decoder speed/quality, not VAE).
"""

import gc
import os
import sys
import time

os.environ.setdefault("ACESTEP_MODELS_DIR", "/root/.daydream-scope/models/rtmg")
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

import soundfile as sf
import torch

torch.set_grad_enabled(False)

sys.path.insert(0, "/workspace/acestep")

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes import Audio
from acestep.paths import default_trt_engines, trt_engine_path

CHECKPOINT = "acestep-v15-xl-turbo"
DECODER_ENGINE = "decoder_xl-turbo_bf16mix_dynbatch_b8_240s"

OUTPUT_DIR = "/workspace/test_output_h100"

# --- Prompt (matches local text_to_music.py) ---
TAGS = "dance music, four on the floor, kick drum, electronic, club, energetic synth bass, bright leads"
LYRICS = "[instrumental]"
BPM = 128
KEY = "F minor"
DURATION = 58.0  # 60s engine cap

# --- Diffusion knobs (turbo: no CFG) ---
SEEDS = [1528, 42, 9999]
INFER_STEPS = 8
SHIFT = 3.0


def save_audio(audio: Audio, path: str) -> None:
    wav = audio.waveform
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    sf.write(path, wav.detach().cpu().float().numpy().T, audio.sample_rate)
    print(f"  Saved: {path}")


def run_pass(label: str, *, decoder_backend: str, trt_engines):
    print("\n" + "=" * 70)
    print(f"PASS: {label}  (decoder={decoder_backend}, vae=eager)")
    print("=" * 70)

    print(f"\n[1] Creating session ({CHECKPOINT})...")
    t0 = time.time()
    session = Session(
        config_path=CHECKPOINT,
        decoder_backend=decoder_backend,
        vae_backend="eager",
        trt_engines=trt_engines,
    )
    print(f"    Session ready in {time.time() - t0:.1f}s")

    print("\n[2] Encoding text prompt...")
    t0 = time.time()
    cond = session.encode_text(
        tags=TAGS,
        lyrics=LYRICS,
        instruction=TASK_INSTRUCTIONS["text2music"],
        bpm=BPM,
        duration=DURATION,
        key=KEY,
    )
    print(f"    Text encoded in {time.time() - t0:.2f}s")

    timings = []
    print("\n[3] Generating (no CFG, single-batch turbo)...")
    for seed in SEEDS:
        t0 = time.time()
        latent = session.generate(
            conditioning=cond,
            seed=seed,
            duration=DURATION,
            steps=INFER_STEPS,
            shift=SHIFT,
        )
        t_gen = time.time() - t0

        t0 = time.time()
        audio = session.decode(latent)
        t_dec = time.time() - t0

        print(
            f"  seed={seed}: generate={t_gen:.3f}s  decode={t_dec:.3f}s  "
            f"total={t_gen + t_dec:.3f}s"
        )
        timings.append((seed, t_gen, t_dec))
        out_path = os.path.join(OUTPUT_DIR, f"t2m_xl_turbo_h100_{label}_seed_{seed}.wav")
        save_audio(audio, out_path)

    del session
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return timings


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("XL TURBO H100 A/B: PyTorch eager vs TensorRT")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"TRT engine: {DECODER_ENGINE}")
    print(f"Duration:   {DURATION}s")
    print(f"Steps:      {INFER_STEPS}  shift={SHIFT}  (no CFG)")
    print(f"Seeds:      {SEEDS}")

    pt_timings = run_pass("pytorch", decoder_backend="eager", trt_engines=None)

    # Build trt_engines dict but only set the decoder; pass an empty path
    # for VAE since vae_backend stays eager
    trt_engines = {"decoder": str(trt_engine_path(DECODER_ENGINE))}
    trt_timings = run_pass("tensorrt", decoder_backend="tensorrt", trt_engines=trt_engines)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'seed':>10} {'pt_gen':>10} {'pt_dec':>10} {'trt_gen':>10} {'trt_dec':>10} {'speedup':>10}")
    for (s, pg, pd), (_, tg, td) in zip(pt_timings, trt_timings):
        spd = pg / tg if tg > 0 else 0
        print(f"{s:>10} {pg:>10.3f} {pd:>10.3f} {tg:>10.3f} {td:>10.3f} {spd:>9.2f}x")

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
