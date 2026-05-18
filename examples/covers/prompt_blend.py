#!/usr/bin/env python3
"""Multi-prompt temporal blending workflow.

Replaces test_temporal_blend.py. Demonstrates:
  - Source audio cover with two different text prompts
  - CurveWave -> temporal_weight for per-frame crossfade
  - ConditioningCombine to produce multi-condition set
  - Generate runs separate decoder calls, blends velocities per-frame
"""

import os
import sys
import time

import soundfile as sf
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from acestep.nodes import Audio, Mask
from acestep.nodes.model_nodes import LoadModel
from acestep.nodes.vae_nodes import VAEEncodeAudio, VAEDecodeAudio
from acestep.nodes.cond_nodes import TextEncode, ConditioningCombine
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
    print("WORKFLOW: Multi-Prompt Temporal Blend (cover)")
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

    # --- Encode source audio ---
    source_audio = load_audio(SOURCE_AUDIO)
    source_latent = VAEEncodeAudio().execute(vae=vae, audio=source_audio)["latent"]
    T = source_latent.tensor.shape[1]
    context_latent = SemanticExtract().execute(model=model, latent=source_latent)["latent"]

    # --- Encode two different prompts (both as covers of the source) ---
    print("\n[TextEncode] Prompt A: daft punk style")
    cond_a = TextEncode().execute(
        clip=clip, model=model,
        refer_latent=source_latent,
        tags="daft punk style electronic french house",
        lyrics="",
        instruction=TASK_INSTRUCTIONS["cover"],
        bpm=136, duration=60.0, key="G# minor",
    )["conditioning"]

    print("[TextEncode] Prompt B: heavy demon techno")
    cond_b = TextEncode().execute(
        clip=clip, model=model,
        refer_latent=source_latent,
        tags="heavy demon techno, growling bass, industrial",
        lyrics="",
        instruction=TASK_INSTRUCTIONS["cover"],
        bpm=136, duration=60.0, key="G# minor",
    )["conditioning"]

    # --- Create temporal blend curve ---
    # Pulse wave: crossfade between prompt A and prompt B (~6s cycle)
    blend_curve = CurveWave().execute(
        wave_type="pulse",
        frames_per_cycle=151,
        amplitude=0.5,
        offset=0.5,
        length=T,
    )["curve"]

    # Convert curve to mask (clamp to [0,1]) for temporal_weight
    temporal_mask = Mask(tensor=blend_curve.tensor.clamp(0.0, 1.0))

    # --- Combine conditions ---
    combined = ConditioningCombine().execute(
        conditioning_a=cond_a,
        conditioning_b=cond_b,
        temporal_weight_b=temporal_mask,
    )["conditioning"]

    print(f"Combined: {len(combined.to_entries())} entries")

    # --- Generate ---
    config = DiffusionConfigNode().execute(
        steps=8, shift=3.0, seed=1118, denoise=1.0,
    )["config"]

    t0 = time.time()
    output_latent = Generate().execute(
        model=model,
        config=config,
        positive=combined,
        context_latent=context_latent,
        source_latent=source_latent,
    )["latent"]
    print(f"Generated in {time.time() - t0:.2f}s")

    # --- Decode ---
    output_audio = VAEDecodeAudio().execute(vae=vae, latent=output_latent)["audio"]
    save_audio(output_audio, os.path.join(OUTPUT_DIR, "prompt_blend.wav"))

    print("\nDone.")


if __name__ == "__main__":
    main()
