#!/usr/bin/env python3
"""Latent noise mask workflow: selective denoising via temporal mask.

Replaces parts of test_denoise_curve_B.py. Demonstrates:
  - CurveWave -> TemporalMask -> SetLatentNoiseMask
  - Generate with masked latent (two-sided blending)
  - Regions with mask=0 are preserved, mask=1 are regenerated
"""

import os
import sys
import time

import soundfile as sf
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from acestep.nodes import Audio
from acestep.nodes.model_nodes import LoadModel
from acestep.nodes.vae_nodes import VAEEncodeAudio, VAEDecodeAudio
from acestep.nodes.cond_nodes import TextEncode
from acestep.nodes.semantic_nodes import SemanticExtract
from acestep.nodes.curve_nodes import CurveWave
from acestep.nodes.mask_nodes import TemporalMask, SetLatentNoiseMask
from acestep.nodes.diffusion_nodes import DiffusionConfigNode, Generate
from acestep.constants import TASK_INSTRUCTIONS

SOURCE_AUDIO = os.path.join(project_root, "tests/fixtures", "new_order_confusion_60seconds.wav")
OUTPUT_DIR = os.path.join(project_root, "test_output", "workflows")


def load_audio(path: str, duration: float = 60.0) -> Audio:
    data, sr = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != 48000:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
    waveform = waveform[:2, :int(duration * 48000)]
    return Audio(waveform=waveform, sample_rate=48000)


def save_audio(audio: Audio, path: str) -> None:
    wav = audio.waveform
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    sf.write(path, wav.cpu().numpy().T, audio.sample_rate)
    print(f"Saved: {path}")


def main():
    print("=" * 70)
    print("WORKFLOW: Latent Noise Mask (selective denoising)")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load model ---
    handles = LoadModel().execute(
        project_root=project_root,
        config_path="acestep-v15-turbo",
        device="cuda",
        use_flash_attention=True,
    )
    model, clip, vae = handles["model"], handles["clip"], handles["vae"]

    # --- Encode source ---
    source_audio = load_audio(SOURCE_AUDIO)
    source_latent = VAEEncodeAudio().execute(vae=vae, audio=source_audio)["latent"]
    T = source_latent.tensor.shape[1]

    # --- Create temporal mask from a pulse curve ---
    # Pulse wave: alternates between preserve (0) and generate (1)
    mask_curve = CurveWave().execute(
        wave_type="pulse",
        frames_per_cycle=300,  # ~12 second cycles
        amplitude=0.5,
        offset=0.5,
        length=T,
    )["curve"]

    mask = TemporalMask().execute(
        latent=source_latent,
        curve=mask_curve,
    )["mask"]

    # --- Attach mask to latent ---
    masked_latent = SetLatentNoiseMask().execute(
        latent=source_latent,
        mask=mask,
    )["latent"]

    print(f"Mask shape: {list(mask.tensor.shape)}")
    print(f"Masked latent has noise mask: {masked_latent.mask is not None}")

    # --- Encode text ---
    context_latent = SemanticExtract().execute(model=model, latent=source_latent)["latent"]

    conditioning = TextEncode().execute(
        clip=clip,
        model=model,
        refer_latent=source_latent,
        tags="driving techno with insane synths",
        instruction=TASK_INSTRUCTIONS["cover"],
        bpm=136,
        duration=60.0,
    )["conditioning"]

    # --- Generate with mask ---
    config = DiffusionConfigNode().execute(
        steps=8, shift=3.0, seed=632, denoise=0.85,
    )["config"]

    t0 = time.time()
    output_latent = Generate().execute(
        model=model,
        config=config,
        positive=conditioning,
        context_latent=context_latent,
        source_latent=masked_latent,
    )["latent"]
    print(f"Generated in {time.time() - t0:.2f}s")

    # --- Decode ---
    output_audio = VAEDecodeAudio().execute(vae=vae, latent=output_latent)["audio"]
    save_audio(output_audio, os.path.join(OUTPUT_DIR, "latent_noise_mask.wav"))

    print("\nDone.")


if __name__ == "__main__":
    main()
