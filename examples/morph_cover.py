#!/usr/bin/env python3
"""Morph cover: drive a latent A->B morph from the pipeline's SHARED STATE.

This is the canonical example of controlling generation through the
streaming pipeline's shared mutable state. ``StreamPipeline`` keeps a
registry of "shared curves" that every in-flight slot reads live, one
tick at a time, via ``set_shared_curve(name, value)``. Here the shared
curve we drive is ``x0_target_strength`` -- a per-frame morph coefficient
that blends each slot's running x0 prediction toward a *second*
generation's latent::

    x0 = (1 - alpha[t]) * x0_pred  +  alpha[t] * x0_target

So one cover is produced whose character morphs from prompt A into
prompt B across the song, with the whole trajectory living in shared
state (not baked into the request). The same long-lived pipeline serves
every render; we just set / clear the shared curve between them.

Two ingredients make this sound good (and not like the other x0-target
examples, which morph via the per-slot ``Generate`` kwarg at full
structure strength and tend to leave A and B sounding alike):

  1. SHARED STATE, not a per-slot field. ``set_shared_curve`` is the live
     control surface used by the realtime / MCP paths, so the exact same
     recipe works for hot, interactive morphing.
  2. STRUCTURE_REF < 1. We weaken the source's semantic-hint (structure)
     conditioning so the prompt drives harder and the A / B covers
     actually diverge -- without that, the endpoints are too similar to
     hear the morph.

Artifact-safe by construction: both endpoints are clean x0 latents from
the same source / seed / context, so each per-frame blend is a convex
combination inside the latent manifold (no re-noising, no under-denoise
residue, no guidance blow-up).

Run:
    python examples/morph_cover.py

Outputs ref_A / ref_B (the two endpoint covers, so you can hear the
contrast) and three morph variants into test_output/morph_cover/.
"""

import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)  # force repo to FRONT: a sibling ACE-Step
#                                   checkout would otherwise shadow acestep

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes.types import Audio, Latent
from acestep.fixtures import audio_fixture

# ----------------------------------------------------------------------
# Config (these would be VST knobs / DAW params in a host)
# ----------------------------------------------------------------------
SOURCE_AUDIO = str(audio_fixture("inside_confusion_loop_60s_gsm.wav"))

PROMPT_A = "deathcore, brutal, heavy distorted guitars, aggressive, pounding double bass"
PROMPT_B = "ambient, ethereal, dreamy clean synth pads, soft, atmospheric, spacious reverb"
KEY = "G# minor"
BPM = 136

SEED = 1528
STEPS = 8
SHIFT = 3.0
STRUCTURE_REF = 0.5   # semantic-hint (structure) strength; <1 => prompt drives harder
PEAK_TARGET = 0.97    # normalize output so the raw ~1.3-peak decode doesn't clip

OUTPUT_DIR = os.path.join(project_root, "test_output", "morph_cover")


# ----------------------------------------------------------------------
def load_audio(path: str, duration: float = 60.0) -> Audio:
    data, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != 48000:
        import torchaudio
        wav = torchaudio.transforms.Resample(sr, 48000)(wav)
    wav = wav[:2, : int(duration * 48000)]
    pool = 1920 * 5  # keep frame count on the latent grid
    rem = wav.shape[-1] % pool
    if rem:
        wav = wav[:, : wav.shape[-1] - rem]
    return Audio(waveform=wav, sample_rate=48000)


def save_audio(audio: Audio, path: str) -> None:
    wav = audio.waveform
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    arr = wav.detach().cpu().float().numpy()
    peak = float(np.max(np.abs(arr))) or 1.0
    if peak > PEAK_TARGET:
        arr = arr * (PEAK_TARGET / peak)
    sf.write(path, arr.T, audio.sample_rate)
    print(f"  Saved: {os.path.basename(path)}  (peak {peak:.2f})")


def make_session() -> Session:
    """Use TRT engines if they're built, else fall back to torch.compile."""
    try:
        from acestep.paths import checkpoints_dir, select_trt_engines, trt_engine_path
        eng = select_trt_engines(duration_s=60.0)
        trt = {
            "vae_encode": eng["vae_encode"],
            "vae_decode": eng["vae_decode"],
            "decoder": str(trt_engine_path("decoder_mixed_refit_b8_60s")),
        }
        if all(os.path.exists(p) for p in trt.values()):
            print("[session] backend=TensorRT (60s engines)")
            return Session(project_root=str(checkpoints_dir()),
                           config_path="acestep-v15-turbo",
                           decoder_backend="tensorrt", vae_backend="tensorrt",
                           trt_engines=trt, vae_window=0.0)
    except Exception as e:  # noqa: BLE001
        print(f"[session] TRT unavailable ({e}); using torch.compile")
    print("[session] backend=torch.compile")
    return Session(project_root=project_root,
                   decoder_backend="compile", vae_backend="compile")


def smoothstep(a, b, T):
    x = torch.linspace(0.0, 1.0, T, dtype=torch.float32)
    return a + (b - a) * (x * x * x * (x * (x * 6 - 15) + 10))


def swell(lo, hi, T):
    x = torch.linspace(0.0, 1.0, T, dtype=torch.float32)
    return lo + (hi - lo) * (0.5 - 0.5 * torch.cos(2 * np.pi * x))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("MORPH COVER  (shared-state x0_target_strength morph)")
    print("=" * 70)

    s = make_session()

    print("\n[source] preparing...")
    source = s.prepare_source(load_audio(SOURCE_AUDIO))
    T = source.latent.tensor.shape[1]
    print(f"  T={T} frames ({T / 25:.1f}s)")

    # Encode the two endpoint prompts (same source/key/bpm -> aligned latents).
    cond_a = s.encode_text(tags=PROMPT_A, instruction=TASK_INSTRUCTIONS["cover"],
                           refer_latent=source.latent, bpm=BPM, duration=60.0, key=KEY)
    cond_b = s.encode_text(tags=PROMPT_B, instruction=TASK_INSTRUCTIONS["cover"],
                           refer_latent=source.latent, bpm=BPM, duration=60.0, key=KEY)

    handle = s.stream(source=source, conditioning=cond_a,
                      steps=STEPS, shift=SHIFT, pipeline_depth=1)

    # Weaken structure so the prompt (and therefore the morph endpoints)
    # diverge audibly. Blend the context latent toward a same-shape zeros
    # latent -> shape-safe, no empty-latent frame-count mismatch.
    if STRUCTURE_REF < 1.0:
        zeros = Latent(tensor=torch.zeros_like(source.context_latent.tensor))
        handle.context_latent = s.blend_latents(zeros, source.context_latent,
                                                alpha=STRUCTURE_REF)
        print(f"[structure] structure_ref={STRUCTURE_REF} (context weakened)")

    # Warmup builds the lazy StreamPipeline (and compiles/warms the engine);
    # only after a tick can we reach handle.pipeline to set shared curves.
    print("\n[warmup] building pipeline...")
    t0 = time.time()
    handle.conditioning = cond_a
    handle.tick(drain=True, seed=SEED)
    pipe = handle.pipeline
    print(f"  ready in {time.time() - t0:.1f}s; shared-curve control live")

    def render(label, conditioning, x0_target=None, strength_curve=None):
        handle.conditioning = conditioning
        pipe.set_shared_curve("x0_target_strength", strength_curve)  # <-- SHARED STATE
        kw = dict(drain=True, seed=SEED)
        if x0_target is not None:
            kw["x0_target"] = x0_target  # Latent wrapper, via the modulation bundle
        t0 = time.time()
        lat = handle.tick(**kw)
        print(f"[render] {label:14s} {time.time() - t0:5.2f}s")
        save_audio(s.decode(lat), os.path.join(OUTPUT_DIR, f"{label}.wav"))
        return lat

    # Endpoint covers (clear the shared curve so they're pure A / pure B).
    render("ref_A", cond_a)
    lat_b = render("ref_B", cond_b)

    # Morph variants: prompt A as the conditioning, B's latent as the target,
    # the morph trajectory supplied entirely through shared state.
    render("morph_ramp", cond_a, x0_target=lat_b,
           strength_curve=smoothstep(0.0, 1.0, T))   # A -> B over the song
    render("morph_swell", cond_a, x0_target=lat_b,
           strength_curve=swell(0.0, 0.85, T))        # A -> B -> A
    render("morph_partial", cond_a, x0_target=lat_b,
           strength_curve=smoothstep(0.0, 0.6, T))    # gentle drift toward B

    pipe.set_shared_curve("x0_target_strength", None)  # clear shared state
    save_audio(load_audio(SOURCE_AUDIO), os.path.join(OUTPUT_DIR, "source.wav"))

    print(f"\nDone. Listen in {OUTPUT_DIR}")
    print("  Compare ref_A.wav vs ref_B.wav, then play morph_ramp.wav.")


if __name__ == "__main__":
    main()
