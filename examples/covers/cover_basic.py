#!/usr/bin/env python3
"""Basic cover workflow using the node system.

Demonstrates:
  - LoadModel -> MODEL, CLIP, VAE
  - LoadAudio -> VAEEncode -> LATENT (source)
  - SemanticExtract -> LATENT (context)
  - TextEncode (with refer_latent for timbre) -> CONDITIONING
  - DiffusionConfig -> CONFIG
  - Generate (context_latent + source_latent) -> LATENT (output)
  - VAEDecode -> AUDIO
"""

import os
import sys
import time

import soundfile as sf
import torch

# Project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from acestep.nodes import (
    Audio,
    Latent,
    Config,
    NodeRegistry,
)
from acestep.nodes.model_nodes import LoadModel
from acestep.nodes.vae_nodes import VAEEncodeAudio, VAEDecodeAudio
from acestep.nodes.cond_nodes import TextEncode
from acestep.nodes.semantic_nodes import SemanticExtract
from acestep.nodes.diffusion_nodes import DiffusionConfigNode, Generate
from acestep.constants import TASK_INSTRUCTIONS
from acestep.fixtures import audio_fixture

SOURCE_AUDIO = str(audio_fixture("inside_confusion_loop_60s_gsm.wav"))
OUTPUT_DIR = os.path.join(project_root, "test_output", "examples")


def load_audio(path: str, duration: float = 60.0) -> Audio:
    """Load audio from file into an Audio payload."""
    data, sr = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != 48000:
        from torchaudio.transforms import Resample
        waveform = Resample(sr, 48000)(waveform)
    max_samples = int(duration * 48000)
    waveform = waveform[:2, :max_samples]
    return Audio(waveform=waveform, sample_rate=48000)


def save_audio(audio: Audio, path: str) -> None:
    """Save an Audio payload to a WAV file."""
    wav = audio.waveform
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    sf.write(path, wav.detach().cpu().float().numpy().T, audio.sample_rate)
    print(f"Saved: {path}")


def main():
    print("=" * 70)
    print("WORKFLOW: Basic Cover")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- LoadModel ---
    print("\n[LoadModel]")
    t0 = time.time()
    handles = LoadModel().execute(
        project_root=project_root,
        config_path="acestep-v15-turbo",
        device="cuda",
        use_flash_attention=True,
        compile_decoder=True,
        compile_vae=True,
    )
    model = handles["model"]
    clip = handles["clip"]
    vae = handles["vae"]
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # --- Load source audio ---
    print("\n[LoadAudio]")
    source_audio = load_audio(SOURCE_AUDIO, duration=60.0)
    print(f"  Waveform: {list(source_audio.waveform.shape)}, sr={source_audio.sample_rate}")

    # --- VAE Encode ---
    print("\n[VAEEncodeAudio]")
    t0 = time.time()
    source_latent = VAEEncodeAudio().execute(vae=vae, audio=source_audio)["latent"]
    print(f"  Latent: {list(source_latent.tensor.shape)} ({time.time() - t0:.2f}s)")

    # --- Semantic Extract (structural context for cover) ---
    print("\n[SemanticExtract]")
    t0 = time.time()
    context_latent = SemanticExtract().execute(model=model, latent=source_latent)["latent"]
    print(f"  Context latent: {list(context_latent.tensor.shape)} ({time.time() - t0:.2f}s)")

    # --- Text Encode (cross-attention conditioning) ---
    print("\n[TextEncode]")
    t0 = time.time()
    conditioning = TextEncode().execute(
        clip=clip,
        model=model,
        refer_latent=source_latent,
        tags="deathstep death deaht deaht",
        lyrics="",
        instruction=TASK_INSTRUCTIONS["cover"],
        bpm=136,
        duration=60.0,
        key="G# minor",
        time_signature="4",
        language="en",
    )["conditioning"]
    print(f"  Conditioning encoded ({time.time() - t0:.2f}s)")

    # --- Generate at multiple denoise levels ---
    for denoise_val in [0.5, 0.75, 1.0]:
        print(f"\n[DiffusionConfig + Generate] denoise={denoise_val}")

        config = DiffusionConfigNode().execute(
            steps=8,
            shift=3.0,
            seed=1528,
            denoise=denoise_val,
        )["config"]

        t0 = time.time()
        output_latent = Generate().execute(
            model=model,
            config=config,
            positive=conditioning,
            context_latent=context_latent,
            source_latent=source_latent,
        )["latent"]
        print(f"  Generated: {list(output_latent.tensor.shape)} ({time.time() - t0:.2f}s)")

        # --- VAE Decode ---
        print("  [VAEDecodeAudio]")
        t0 = time.time()
        output_audio = VAEDecodeAudio().execute(vae=vae, latent=output_latent)["audio"]
        print(f"  Decoded: {list(output_audio.waveform.shape)} ({time.time() - t0:.2f}s)")

        name = f"cover_denoise_{int(denoise_val * 100):03d}"
        save_audio(output_audio, os.path.join(OUTPUT_DIR, f"{name}.wav"))

        del output_latent, output_audio
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
