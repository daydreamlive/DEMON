#!/usr/bin/env python3
"""Text-to-music A/B: PyTorch eager vs TensorRT for the XL turbo decoder.

Runs the same prompt + seeds twice against the acestep-v15-xl-turbo
checkpoint, once with the PyTorch eager decoder and once with the
TensorRT engine `decoder_xl-turbo_bf16mix_dynbatch_b8_240s`. Outputs
are saved side-by-side for direct A/B listening.

Turbo is CFG-distilled, so no negative conditioning and no guidance
curve are passed (single-batch denoise). DURATION is capped at 58s
because the only XL turbo TRT engine that exists is a 60s build
(max 1500 latent frames).
"""

import gc
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
from acestep.paths import default_trt_engines


CHECKPOINT = "acestep-v15-xl-turbo"
DECODER_ENGINE = "decoder_xl-turbo_bf16mix_dynbatch_b8_240s"
VAE_ENCODE_ENGINE = "vae_encode_fp16_240s"
VAE_DECODE_ENGINE = "vae_decode_fp16_240s"

OUTPUT_DIR = os.path.join(project_root, "test_output", "text_to_music_xl_turbo_compare")

# --- Prompt ---
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


def run_pass(label: str, *, decoder_backend: str, vae_backend: str,
             trt_engines: dict | None) -> None:
    print("\n" + "=" * 70)
    print(f"PASS: {label}  (decoder={decoder_backend}, vae={vae_backend})")
    print("=" * 70)

    print(f"\n[1] Creating session ({CHECKPOINT})...")
    t0 = time.time()
    session = Session(
        config_path=CHECKPOINT,
        decoder_backend=decoder_backend,
        vae_backend=vae_backend,
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
        out_path = os.path.join(OUTPUT_DIR, f"t2m_xl_turbo_{label}_seed_{seed}.wav")
        save_audio(audio, out_path)

    # Free GPU memory before the next pass
    del session
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("XL TURBO A/B: PyTorch eager vs TensorRT")
    print("=" * 70)
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"TRT engine: {DECODER_ENGINE}")
    print(f"Tags:       {TAGS}")
    print(f"BPM:        {BPM}")
    print(f"Key:        {KEY}")
    print(f"Duration:   {DURATION}s")
    print(f"Steps:      {INFER_STEPS}  shift={SHIFT}  (no CFG)")
    print(f"Seeds:      {SEEDS}")

    run_pass(
        "pytorch",
        decoder_backend="eager",
        vae_backend="eager",
        trt_engines=None,
    )

    trt_engines = default_trt_engines(
        decoder=DECODER_ENGINE,
        vae_encode=VAE_ENCODE_ENGINE,
        vae_decode=VAE_DECODE_ENGINE,
    )
    run_pass(
        "tensorrt",
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines=trt_engines,
    )

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")
    print("\nA/B pairs:")
    for seed in SEEDS:
        print(f"  seed {seed}:")
        print(f"    pytorch:  t2m_xl_turbo_pytorch_seed_{seed}.wav")
        print(f"    tensorrt: t2m_xl_turbo_tensorrt_seed_{seed}.wav")


if __name__ == "__main__":
    main()
