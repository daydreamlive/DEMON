#!/usr/bin/env python3
"""Conditioning average workflow: two prompts blended 50/50.

Replaces test_cond_blend.py. Demonstrates:
  - Two TextEncode calls with different prompts
  - ConditioningAverage to blend them before diffusion
  - Standard cover generation with blended conditioning
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
from acestep.nodes.cond_nodes import TextEncode, ConditioningAverage
from acestep.nodes.semantic_nodes import SemanticExtract
from acestep.nodes.diffusion_nodes import DiffusionConfigNode, Generate
from acestep.constants import TASK_INSTRUCTIONS
from acestep.fixtures import audio_fixture

SOURCE_AUDIO = str(audio_fixture("inside_confusion_loop_60s_gsm.wav"))
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
    print("WORKFLOW: Conditioning Average (50/50 prompt blend)")
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
    context_latent = SemanticExtract().execute(model=model, latent=source_latent)["latent"]

    # --- Encode two different prompts ---
    print("\n[TextEncode] Prompt A: deathstep")
    cond_a = TextEncode().execute(
        clip=clip, model=model,
        refer_latent=source_latent,
        tags="deathstep death deaht deaht",
        lyrics="",
        instruction=TASK_INSTRUCTIONS["cover"],
        bpm=136, duration=60.0, key="G# minor",
    )["conditioning"]

    print("[TextEncode] Prompt B: ambient angelic synths")
    cond_b = TextEncode().execute(
        clip=clip, model=model,
        refer_latent=source_latent,
        tags="ambiet angelic synths a lot of synths",
        lyrics="",
        instruction=TASK_INSTRUCTIONS["cover"],
        bpm=136, duration=60.0, key="G# minor",
    )["conditioning"]

    # --- Blend 50/50 ---
    blended = ConditioningAverage().execute(
        conditioning_a=cond_a,
        conditioning_b=cond_b,
        weight=0.5,
    )["conditioning"]
    print("Blended conditioning (50/50)")

    # --- Generate ---
    config = DiffusionConfigNode().execute(
        steps=8, shift=3.0, seed=1528, denoise=1.0,
    )["config"]

    t0 = time.time()
    output_latent = Generate().execute(
        model=model,
        config=config,
        positive=blended,
        context_latent=context_latent,
        source_latent=source_latent,
    )["latent"]
    print(f"Generated in {time.time() - t0:.2f}s")

    # --- Decode ---
    output_audio = VAEDecodeAudio().execute(vae=vae, latent=output_latent)["audio"]
    save_audio(output_audio, os.path.join(OUTPUT_DIR, "conditioning_average.wav"))

    print("\nDone.")


if __name__ == "__main__":
    main()
