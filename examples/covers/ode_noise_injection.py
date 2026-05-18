#!/usr/bin/env python3
"""ODE noise injection workflow: per-frame controlled creativity.

Replaces test_ode_noise_injection.py. Demonstrates:
  - CurveWave (sine) feeding ode_noise_curve on Generate
  - After each ODE step, small noise is injected per-frame
  - Injection scaled by curve AND current sigma (naturally decreases)
  - High injection = more creative/transformed, low = more faithful
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
from acestep.nodes.diffusion_nodes import DiffusionConfigNode, Generate
from acestep.constants import TASK_INSTRUCTIONS
from acestep.fixtures import audio_fixture

SOURCE_AUDIO = str(audio_fixture("inside_confusion_loop_60s_gsm.wav"))
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
    print("WORKFLOW: ODE Noise Injection (per-frame controlled creativity)")
    print("  1Hz sine wave, range 0.0 -> 0.5")
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

    conditioning = TextEncode().execute(
        clip=clip, model=model,
        refer_latent=source_latent,
        tags="deathstep death deaht deaht",
        instruction=TASK_INSTRUCTIONS["cover"],
        bpm=136, duration=60.0, key="G# minor",
    )["conditioning"]

    # --- Build noise injection curve: 1Hz sine mapped to [0, 0.5] ---
    # CurveWave sine: amplitude * sin(2pi * t/frames_per_cycle) + offset
    # 1Hz at 25fps = 25 frames/cycle
    inject_curve = CurveWave().execute(
        wave_type="sine",
        frames_per_cycle=25,  # 1Hz at 25fps latent rate
        amplitude=0.25,       # sin range [-0.25, 0.25]
        offset=0.25,          # shift to [0.0, 0.5]
        length=T,
    )["curve"]
    print(f"Injection curve: 1Hz sine, range [{inject_curve.tensor.min():.2f}, {inject_curve.tensor.max():.2f}]")

    # --- Generate with ODE noise injection ---
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
        ode_noise_curve=inject_curve,
    )["latent"]
    print(f"Generated in {time.time() - t0:.2f}s")

    # --- Decode ---
    output_audio = VAEDecodeAudio().execute(vae=vae, latent=output_latent)["audio"]
    save_audio(output_audio, os.path.join(OUTPUT_DIR, "ode_noise_injection.wav"))

    print("\nDone.")


if __name__ == "__main__":
    main()
