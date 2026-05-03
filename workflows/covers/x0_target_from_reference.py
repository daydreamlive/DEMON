#!/usr/bin/env python3
"""x0 target from encoded reference: blend toward a real audio track.

Unlike x0_target_blend.py (which generates the target via a second
diffusion pass), this workflow VAE-encodes a *different* audio file
and uses those latents directly as x0_target. The result blends the
generation's style/prompt toward the spectral content of the reference.

Use case: you have a reference track whose texture or arrangement you
want the output to gravitate toward, without fully covering it.

  Source:    inside_confusion_loop_60s_gsm.wav  (structure + timbre donor)
  Reference: low_fi_Gm_loop_60s_gnm.wav         (x0 target, encoded latent)
  Prompt:    deathstep

The ramp curve (0.0 -> 0.6) means the first half of the track is
purely prompt-driven, while the second half increasingly pulls
toward the reference's latent content. Blending is further gated
to refinement steps internally (kicks in at step 50%+).
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
REFERENCE_AUDIO = str(audio_fixture("low_fi_Gm_loop_60s_gnm.wav"))
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
    print("WORKFLOW: x0 Target from Encoded Reference")
    print(f"  Source:    {os.path.basename(SOURCE_AUDIO)}")
    print(f"  Reference: {os.path.basename(REFERENCE_AUDIO)}")
    print("  Blend toward reference latent with ramp 0.0 -> 0.6")
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

    # --- Encode source (structure + timbre) ---
    print("\nEncoding source audio...")
    source_audio = load_audio(SOURCE_AUDIO)
    source_latent = VAEEncodeAudio().execute(vae=vae, audio=source_audio)["latent"]
    T = source_latent.tensor.shape[1]
    context_latent = SemanticExtract().execute(model=model, latent=source_latent)["latent"]
    print(f"  Source: {T} frames")

    # --- Encode reference (x0 target) ---
    print("\nEncoding reference audio as x0 target...")
    ref_audio = load_audio(REFERENCE_AUDIO)
    ref_latent = VAEEncodeAudio().execute(vae=vae, audio=ref_audio)["latent"]
    T_ref = ref_latent.tensor.shape[1]
    print(f"  Reference: {T_ref} frames")

    # Trim or pad reference latent to match source frame count
    if T_ref != T:
        ref_t = ref_latent.tensor
        if T_ref > T:
            ref_t = ref_t[:, :T, :]
        else:
            ref_t = torch.nn.functional.pad(ref_t, (0, 0, 0, T - T_ref))
        from acestep.nodes.types import Latent
        ref_latent = Latent(tensor=ref_t)
        print(f"  Adjusted reference to {T} frames (was {T_ref})")

    # --- Encode text conditioning ---
    print("\nEncoding text conditioning...")
    cond = TextEncode().execute(
        clip=clip, model=model,
        refer_latent=source_latent,
        tags="deathstep heavy bass dubstep aggressive",
        instruction=TASK_INSTRUCTIONS["cover"],
        bpm=136, duration=60.0, key="G# minor",
    )["conditioning"]

    # --- Build x0 target curve ---
    blend_curve = CurveRamp().execute(
        start=0.0, end=0.6, length=T,
    )["curve"]
    print(f"  x0 target curve: {blend_curve.tensor[0]:.2f} -> {blend_curve.tensor[-1]:.2f}")

    # --- Generate with x0 target from encoded reference ---
    print("\nGenerating with encoded reference as x0 target...")
    config = DiffusionConfigNode().execute(
        steps=8, shift=3.0, seed=1528, denoise=1.0,
    )["config"]

    t0 = time.time()
    output_latent = Generate().execute(
        model=model,
        config=config,
        positive=cond,
        context_latent=context_latent,
        source_latent=source_latent,
        x0_target=ref_latent,
        x0_target_curve=blend_curve,
    )["latent"]
    print(f"  Generated in {time.time() - t0:.2f}s")

    # --- Decode ---
    output_audio = VAEDecodeAudio().execute(vae=vae, latent=output_latent)["audio"]
    save_audio(output_audio, os.path.join(OUTPUT_DIR, "x0_target_from_reference.wav"))

    print("\nDone.")


if __name__ == "__main__":
    main()
