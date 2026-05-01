#!/usr/bin/env python3
"""Per-frame guidance curve (CFG) workflow.

Replaces test_guidance_curve.py. Demonstrates:
  - TextEncode -> positive conditioning
  - ConditioningZeroOut -> negative (unconditional) conditioning
  - CurveRamp -> per-frame guidance scale
  - Generate with negative + guidance_curve for per-frame CFG
  - v_guided = v_uncond + guidance * (v_cond - v_uncond)
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
from acestep.nodes.cond_nodes import TextEncode, ConditioningZeroOut
from acestep.nodes.semantic_nodes import SemanticExtract
from acestep.nodes.curve_nodes import CurveRamp
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
    print("WORKFLOW: Per-frame Guidance Curve (CFG)")
    print("  Ramp 1.0 -> 2.0 across 60s")
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

    # --- Positive conditioning ---
    positive = TextEncode().execute(
        clip=clip, model=model,
        refer_latent=source_latent,
        tags="deathstep death deaht deaht",
        instruction=TASK_INSTRUCTIONS["cover"],
        bpm=136, duration=60.0, key="G# minor",
    )["conditioning"]

    # --- Negative conditioning (zeroed encoder hidden states) ---
    negative = ConditioningZeroOut().execute(
        conditioning=positive,
    )["conditioning"]
    print("Positive + Negative conditioning ready")

    # --- Guidance curve: 1.0 (natural) -> 2.0 (stronger prompt) ---
    cfg_curve = CurveRamp().execute(
        start=1.0, end=2.0, length=T,
    )["curve"]
    print(f"CFG curve: {cfg_curve.tensor[0]:.2f} -> {cfg_curve.tensor[-1]:.2f}")

    # --- Generate with per-frame CFG ---
    config = DiffusionConfigNode().execute(
        steps=8, shift=3.0, seed=1528, denoise=1.0,
    )["config"]

    t0 = time.time()
    output_latent = Generate().execute(
        model=model,
        config=config,
        positive=positive,
        negative=negative,
        context_latent=context_latent,
        source_latent=source_latent,
        guidance_curve=cfg_curve,
    )["latent"]
    print(f"Generated in {time.time() - t0:.2f}s")

    # --- Decode ---
    output_audio = VAEDecodeAudio().execute(vae=vae, latent=output_latent)["audio"]
    save_audio(output_audio, os.path.join(OUTPUT_DIR, "guidance_curve.wav"))

    print("\nDone.")


if __name__ == "__main__":
    main()
