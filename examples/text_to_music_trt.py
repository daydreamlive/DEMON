#!/usr/bin/env python3
"""Pure text-to-music workflow, TensorRT backend.

Mirrors examples/text_to_music.py but routes the decoder and VAE through
TensorRT engines instead of PyTorch eager. The engine variant (60s vs
240s) is chosen by ``select_trt_engines(duration_s=DURATION)``.
"""

import os
import sys
import time

import soundfile as sf
import torch

torch.set_grad_enabled(False)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes import Audio
from acestep.nodes.types import Curve
from acestep.paths import select_trt_engines


OUTPUT_DIR = os.path.join(project_root, "test_output", "text_to_music_trt")

# --- Prompt (same as eager workflow for A/B comparison) ---
TAGS = "dance music, four on the floor, kick drum, electronic, club, energetic synth bass, bright leads"
LYRICS = "[instrumental]"
BPM = 128
KEY = "F minor"
DURATION = 150.0  # 2:30

# --- Diffusion knobs ---
SEEDS = [1528, 42, 9999]
CFG_SCALE = 7.5
INFER_STEPS = 8
SHIFT = 3.0


def save_audio(audio: Audio, path: str) -> None:
    wav = audio.waveform
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    sf.write(path, wav.detach().cpu().float().numpy().T, audio.sample_rate)
    print(f"  Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("TEXT-TO-MUSIC WORKFLOW (TensorRT)")
    print("=" * 70)
    print(f"Tags:     {TAGS}")
    print(f"BPM:      {BPM}")
    print(f"Key:      {KEY}")
    print(f"Duration: {DURATION}s")
    print(f"Steps:    {INFER_STEPS}  shift={SHIFT}  cfg={CFG_SCALE}")
    print(f"Seeds:    {SEEDS}")

    trt_engines = select_trt_engines(duration_s=DURATION)
    print("\nTRT engines:")
    for k, v in trt_engines.items():
        print(f"  {k}: {os.path.basename(os.path.dirname(v))}")

    print("\n[1] Creating session (TensorRT decoder + VAE)...")
    t0 = time.time()
    session = Session(
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
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
    neg_cond = session.null_conditioning(cond)
    print(f"    Text encoded in {time.time() - t0:.2f}s")

    # Constant CFG curve over the latent T axis (25 frames per second)
    T = int(round(DURATION * 25))
    guidance_curve = Curve(tensor=torch.full((T,), CFG_SCALE, dtype=torch.bfloat16))

    print("\n[3] Generating...")
    for seed in SEEDS:
        t0 = time.time()
        latent = session.generate(
            conditioning=cond,
            negative=neg_cond,
            guidance_curve=guidance_curve,
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
        save_audio(audio, os.path.join(OUTPUT_DIR, f"t2m_seed_{seed}.wav"))

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
