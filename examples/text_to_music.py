#!/usr/bin/env python3
"""Pure text-to-music workflow.

No source audio: just text tags (and optional lyrics) generate music
from scratch via Session. Uses CFG (classifier-free guidance) with
the model's learned null embedding as the negative, which is required
for coherent text-to-music output.
"""

import os
import sys
import time

import soundfile as sf
import torch

# Inference only: disable autograd globally so VAE decode doesn't retain
# activations across chunks (handler.tiled_decode is missing an internal
# no_grad guard, unlike tiled_encode).
torch.set_grad_enabled(False)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes import Audio
from acestep.nodes.types import Curve, Latent


def decode_latent(session: Session, latent: Latent) -> Audio:
    """VAE decode via tiled PyTorch decode with conservative chunk size.

    Bypasses Session.decode (which uses chunk_size=512) so long latents
    don't blow up allocator fragmentation. 256-frame chunks ~= 10s of
    audio per VAE call, which is gentle on transient activations.
    """
    torch.cuda.empty_cache()
    lat_bdt = latent.tensor.transpose(1, 2)
    waveform = session.handler.tiled_decode(
        lat_bdt, chunk_size=256, overlap=32,
    )
    return Audio(waveform=waveform, sample_rate=48000)


OUTPUT_DIR = os.path.join(project_root, "test_output", "text_to_music")

# --- Prompt ---
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
    print("TEXT-TO-MUSIC WORKFLOW")
    print("=" * 70)
    print(f"Tags:     {TAGS}")
    print(f"BPM:      {BPM}")
    print(f"Key:      {KEY}")
    print(f"Duration: {DURATION}s")
    print(f"Steps:    {INFER_STEPS}  shift={SHIFT}  cfg={CFG_SCALE}")
    print(f"Seeds:    {SEEDS}")

    print("\n[1] Creating session...")
    t0 = time.time()
    session = Session()
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
        audio = decode_latent(session, latent)
        t_dec = time.time() - t0

        print(
            f"  seed={seed}: generate={t_gen:.3f}s  decode={t_dec:.3f}s  "
            f"total={t_gen + t_dec:.3f}s"
        )
        save_audio(audio, os.path.join(OUTPUT_DIR, f"t2m_seed_{seed}.wav"))

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
