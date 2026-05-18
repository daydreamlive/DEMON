#!/usr/bin/env python3
"""Cover with semantic blending between two audio sources.

Replaces test_cond_blend.py. Demonstrates:
  - Two source audios -> SemanticExtract each (outputs LATENT)
  - LatentBlend with a CurveWave driving the blend factor
  - Single text prompt conditioned on blended structure
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
from acestep.nodes.vae_nodes import VAEEncodeAudio, VAEDecodeAudio, LatentBlend
from acestep.nodes.cond_nodes import TextEncode
from acestep.nodes.semantic_nodes import SemanticExtract
from acestep.nodes.curve_nodes import CurveWave
from acestep.nodes.diffusion_nodes import DiffusionConfigNode, Generate
from acestep.constants import TASK_INSTRUCTIONS
from acestep.fixtures import audio_fixture

SOURCE_A = str(audio_fixture("inside_confusion_loop_60s_gsm.wav"))
SOURCE_B = str(audio_fixture("inside_confusion_loop_60s_gsm.wav"))  # same file for demo
OUTPUT_DIR = os.path.join(project_root, "test_output", "examples")


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
    print("WORKFLOW: Cover with Semantic Blending")
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

    # --- Load and encode two source audios ---
    audio_a = load_audio(SOURCE_A)
    audio_b = load_audio(SOURCE_B)

    latent_a = VAEEncodeAudio().execute(vae=vae, audio=audio_a)["latent"]
    latent_b = VAEEncodeAudio().execute(vae=vae, audio=audio_b)["latent"]
    print(f"Latent A: {list(latent_a.tensor.shape)}")
    print(f"Latent B: {list(latent_b.tensor.shape)}")

    # --- Extract semantic hints from both (output is LATENT) ---
    hints_a = SemanticExtract().execute(model=model, latent=latent_a)["latent"]
    hints_b = SemanticExtract().execute(model=model, latent=latent_b)["latent"]

    # --- Create a blend curve (pulse wave, ~6 second cycle) ---
    T = latent_a.tensor.shape[1]
    blend_curve = CurveWave().execute(
        wave_type="pulse",
        frames_per_cycle=150,
        amplitude=0.5,
        offset=0.5,
        length=T,
    )["curve"]
    print(f"Blend curve: {list(blend_curve.tensor.shape)}")

    # --- Blend extracted hints (LATENT × LATENT → LATENT) ---
    context_latent = LatentBlend().execute(
        latent_a=hints_a,
        latent_b=hints_b,
        blend_curve=blend_curve,
    )["latent"]

    # --- Encode text prompt with blended hints ---
    conditioning = TextEncode().execute(
        clip=clip,
        model=model,
        refer_latent=latent_a,
        tags="jazz piano cover with swing rhythm",
        lyrics="",
        instruction=TASK_INSTRUCTIONS["cover"],
        bpm=136,
        duration=60.0,
        key="C major",
    )["conditioning"]

    # --- Generate ---
    config = DiffusionConfigNode().execute(
        steps=8, shift=3.0, seed=990, denoise=1.0,
    )["config"]

    output_latent = Generate().execute(
        model=model,
        config=config,
        positive=conditioning,
        context_latent=context_latent,
        source_latent=latent_a,
    )["latent"]

    # --- Decode ---
    output_audio = VAEDecodeAudio().execute(vae=vae, latent=output_latent)["audio"]
    save_audio(output_audio, os.path.join(OUTPUT_DIR, "cover_semantic_blend.wav"))

    print("\nDone.")


if __name__ == "__main__":
    main()
