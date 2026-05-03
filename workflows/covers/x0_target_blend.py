#!/usr/bin/env python3
"""x0 target blend workflow: latent-space audio morphing.

Replaces test_x0_target_blend.py. Demonstrates:
  - Two generation passes: first creates the target latent, second
    uses x0_target + x0_target_curve to blend toward it
  - Ramp curve 0.0 -> 0.8: start sounds like source style, end
    sounds like target style
  - Blending is gated to refinement steps (second half of diffusion)

The original test used a LoRA for the target. This workflow uses a
different text prompt instead, avoiding the LoRA dependency while
demonstrating the same engine feature.
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
from acestep.nodes.curve_nodes import CurveRamp
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
    print("WORKFLOW: x0 Target Blend (latent-space morphing)")
    print("  Morph from deathstep toward daft punk style")
    print("  Ramp 0.0 -> 0.8 (gated to refinement steps)")
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
    context_latent = SemanticExtract().execute(model=model, latent=source_latent)["latent"]

    # --- Step 1: Generate the target latent (daft punk style) ---
    print("\n[Pass 1] Generating target latent (daft punk style)...")
    target_cond = TextEncode().execute(
        clip=clip, model=model,
        refer_latent=source_latent,
        tags="daft punk style electronic french house",
        instruction=TASK_INSTRUCTIONS["cover"],
        bpm=136, duration=60.0, key="G# minor",
    )["conditioning"]

    target_config = DiffusionConfigNode().execute(
        steps=8, shift=3.0, seed=1528, denoise=1.0,
    )["config"]

    t0 = time.time()
    target_latent = Generate().execute(
        model=model,
        config=target_config,
        positive=target_cond,
        context_latent=context_latent,
        source_latent=source_latent,
    )["latent"]
    print(f"  Target generated in {time.time() - t0:.2f}s")

    # --- Step 2: Generate with x0 target blending ---
    print("\n[Pass 2] Generating with x0 target blend...")
    source_cond = TextEncode().execute(
        clip=clip, model=model,
        refer_latent=source_latent,
        tags="deathstep death deaht deaht",
        instruction=TASK_INSTRUCTIONS["cover"],
        bpm=136, duration=60.0, key="G# minor",
    )["conditioning"]

    blend_curve = CurveRamp().execute(
        start=0.0, end=0.8, length=T,
    )["curve"]
    print(f"  Blend curve: {blend_curve.tensor[0]:.2f} -> {blend_curve.tensor[-1]:.2f}")

    blend_config = DiffusionConfigNode().execute(
        steps=8, shift=3.0, seed=1528, denoise=1.0,
    )["config"]

    t0 = time.time()
    output_latent = Generate().execute(
        model=model,
        config=blend_config,
        positive=source_cond,
        context_latent=context_latent,
        source_latent=source_latent,
        x0_target=target_latent,
        x0_target_curve=blend_curve,
    )["latent"]
    print(f"  Blended generation in {time.time() - t0:.2f}s")

    # --- Decode ---
    output_audio = VAEDecodeAudio().execute(vae=vae, latent=output_latent)["audio"]
    save_audio(output_audio, os.path.join(OUTPUT_DIR, "x0_target_blend.wav"))

    print("\nDone.")


if __name__ == "__main__":
    main()
