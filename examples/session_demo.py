#!/usr/bin/env python3
"""Session demo: persistent model, fast iteration.

Shows that after one-time setup (~7s model + ~12s compile warmup),
subsequent generations only cost diffusion + decode (~310ms).
"""

import os
import sys
import time

import soundfile as sf
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes import Audio


SOURCE_AUDIO = os.path.join(project_root, "tests/fixtures", "new_order_confusion_60seconds.wav")
OUTPUT_DIR = os.path.join(project_root, "test_output", "session")


def load_audio(path: str, duration: float = 60.0) -> Audio:
    data, sr = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != 48000:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
    waveform = waveform[:2, : int(duration * 48000)]
    return Audio(waveform=waveform, sample_rate=48000)


def save_audio(audio: Audio, path: str) -> None:
    wav = audio.waveform
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    sf.write(path, wav.detach().cpu().float().numpy().T, audio.sample_rate)
    print(f"  Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("SESSION DEMO")
    print("=" * 70)

    # --- One-time setup ---
    print("\n[1] Creating session (model load + compile)...")
    t0 = time.time()
    session = Session(decoder_backend="compile", vae_backend="compile")
    print(f"    Session ready in {time.time() - t0:.1f}s")

    # --- Prepare source (VAE encode + semantic extract) ---
    print("\n[2] Preparing source audio...")
    t0 = time.time()
    audio = load_audio(SOURCE_AUDIO)
    source = session.prepare_source(audio)
    print(f"    Source prepared in {time.time() - t0:.2f}s")
    print(f"    latent: {list(source.latent.tensor.shape)}")
    print(f"    context: {list(source.context_latent.tensor.shape)}")

    # --- Encode text (one-time for this prompt) ---
    print("\n[3] Encoding text prompt...")
    t0 = time.time()
    cond = session.encode_text(
        tags="deathstep death deaht deaht",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136,
        duration=60.0,
        key="G# minor",
    )
    print(f"    Text encoded in {time.time() - t0:.2f}s")

    # --- Fast iteration: different seeds ---
    print("\n[4] Generating with different seeds (only diffusion + decode)...")
    seeds = [1528, 9999, 42, 7777]
    for seed in seeds:
        t0 = time.time()
        output = session.generate(
            conditioning=cond,
            context_latent=source.context_latent,
            source_latent=source.latent,
            seed=seed,
        )
        t_gen = time.time() - t0

        t0 = time.time()
        result = session.decode(output)
        t_dec = time.time() - t0

        print(f"  seed={seed}: generate={t_gen:.3f}s  decode={t_dec:.3f}s  total={t_gen + t_dec:.3f}s")
        save_audio(result, os.path.join(OUTPUT_DIR, f"seed_{seed}.wav"))

    # --- Change prompt, source stays cached ---
    print("\n[5] New prompt, same source (re-encode text + generate + decode)...")
    t0 = time.time()
    cond2 = session.encode_text(
        tags="ambient angelic synths dreamy pads",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136,
        duration=60.0,
        key="G# minor",
    )
    t_enc = time.time() - t0

    t0 = time.time()
    output = session.generate(
        conditioning=cond2,
        context_latent=source.context_latent,
        source_latent=source.latent,
        seed=1528,
    )
    t_gen = time.time() - t0

    t0 = time.time()
    result = session.decode(output)
    t_dec = time.time() - t0

    print(f"  encode={t_enc:.3f}s  generate={t_gen:.3f}s  decode={t_dec:.3f}s  total={t_enc + t_gen + t_dec:.3f}s")
    save_audio(result, os.path.join(OUTPUT_DIR, "prompt_change.wav"))

    print("\nDone.")


if __name__ == "__main__":
    main()
