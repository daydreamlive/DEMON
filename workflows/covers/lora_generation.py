#!/usr/bin/env python3
"""LoRA generation workflow: cover with LoRA style adapter.

Demonstrates the unified LoRA path on top of the model handle's
DiffusionEngine — works against eager, torch.compile, and TRT
decoders without any per-backend branching.

Requires a LoRA safetensors file. Update ``LORA_PATH`` below.
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
from acestep.nodes.diffusion_nodes import DiffusionConfigNode, Generate
from acestep.constants import TASK_INSTRUCTIONS
from acestep.fixtures import audio_fixture

SOURCE_AUDIO = str(audio_fixture("inside_confusion_loop_60s_gsm.wav"))
OUTPUT_DIR = os.path.join(project_root, "test_output", "workflows")
LORA_PATH = ""
LORA_STRENGTH = 1.3


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
    print("WORKFLOW: LoRA Generation (daftpunk style)")
    print(f"  LoRA: {LORA_PATH}")
    print(f"  Strength: {LORA_STRENGTH}")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(LORA_PATH):
        print(f"\nLoRA file not found: {LORA_PATH}")
        print("Update LORA_PATH to point to your LoRA safetensors file.")
        return

    # --- Load model ---
    handles = LoadModel().execute(
        project_root=project_root,
        config_path="acestep-v15-turbo",
        device="cuda",
        use_flash_attention=True,
    )
    model, clip, vae = handles["model"], handles["clip"], handles["vae"]

    # --- Apply LoRA via the model's DiffusionEngine ---
    print("\n[apply_lora]")
    engine = model.handler._diffusion_engine
    lora_id = engine.apply_lora(LORA_PATH, strength=LORA_STRENGTH)

    # --- Encode source ---
    source_audio = load_audio(SOURCE_AUDIO)
    source_latent = VAEEncodeAudio().execute(vae=vae, audio=source_audio)["latent"]
    context_latent = SemanticExtract().execute(model=model, latent=source_latent)["latent"]

    conditioning = TextEncode().execute(
        clip=clip, model=model,
        refer_latent=source_latent,
        tags="deathstep death deaht deaht",
        instruction=TASK_INSTRUCTIONS["cover"],
        bpm=136, duration=60.0, key="G# minor",
    )["conditioning"]

    # --- Generate ---
    config = DiffusionConfigNode().execute(
        steps=8, shift=3.0, seed=1528, denoise=1.0,
    )["config"]

    t0 = time.time()
    output_latent = Generate().execute(
        model=model,
        config=config,
        positive=conditioning,
        context_latent=context_latent,
        source_latent=source_latent,
    )["latent"]
    print(f"Generated in {time.time() - t0:.2f}s")

    # --- Remove LoRA (restore base weights) ---
    print("\n[remove_lora]")
    engine.remove_lora(lora_id)

    # --- Decode ---
    output_audio = VAEDecodeAudio().execute(vae=vae, latent=output_latent)["audio"]
    save_audio(output_audio, os.path.join(OUTPUT_DIR, "lora_generation.wav"))

    print("\nDone.")


if __name__ == "__main__":
    main()
