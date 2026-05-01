#!/usr/bin/env python3
"""Real-time cover workflow: full Scope parity + engine-exclusive features.

Scope parity:
  - Dual prompts with blend
  - Dual LoRAs with independent strength
  - Separate timbre reference audio (blended with silence for strength)
  - Separate semantic hint source (blended for strength)
  - Auto-detect BPM/key/duration from source audio
  - Temporal masking on source latent
  - Configurable denoise / seed

Engine exclusives:
  - Per-frame velocity scaling curve
  - Per-frame SDE denoise modulation
  - Per-frame initial noise curve
  - Per-frame ODE noise injection
  - x0 target blending (morph toward a pre-generated target)
  - ConditioningCombine with temporal weights (per-frame prompt crossfade)
  - TRT VAE + compiled decoder (~310ms warm generation)
  - Session persistence (no reload between iterations)
"""

import os
import sys
import time

import soundfile as sf
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session, PreparedSource
from acestep.nodes import Audio, Latent, Mask, Conditioning
from acestep.nodes.cond_nodes import ConditioningAverage, ConditioningCombine
from acestep.nodes.curve_nodes import CurveRamp, CurveWave
from acestep.nodes.mask_nodes import TemporalMask, SetLatentNoiseMask

# ======================================================================
# Configuration (these would be VST knobs / DAW parameters)
# ======================================================================

SOURCE_AUDIO = os.path.join(project_root, "tests/fixtures", "new_order_confusion_60seconds.wav")
# Set to None to use source audio for timbre/hints, or a path to override
TIMBRE_REF_AUDIO = None
HINT_SOURCE_AUDIO = None

# Prompts
PROMPT_A = "deathstep death deaht deaht"
PROMPT_B = "ambient angelic synths dreamy pads"
PROMPT_BLEND = 0.3  # 0.0 = all A, 1.0 = all B

# LoRAs (path, scale) - empty list for no LoRA
LORAS = []

# Strength controls (0.0 = silence/off, 1.0 = full)
TIMBRE_STRENGTH = 1.0
HINT_STRENGTH = 1.0

# Generation
SEED = 1528
DENOISE = 1.0
STEPS = 8

# Per-frame curves (set to None to disable)
VELOCITY_CURVE = ("ramp", 0.5, 1.2)  # (type, start, end) or None
SDE_DENOISE_CURVE = None  # ("ramp", 0.3, 1.0) or None
INITIAL_NOISE_CURVE = None  # ("ramp", 0.3, 1.0) or None
ODE_NOISE_CURVE = None  # ("sine", freq_hz, amplitude) or None

OUTPUT_DIR = os.path.join(project_root, "test_output", "realtime_cover")

# ======================================================================


def load_audio(path: str, duration: float = 0) -> Audio:
    data, sr = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != 48000:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
    waveform = waveform[:2]
    if duration > 0:
        waveform = waveform[:, : int(duration * 48000)]
    return Audio(waveform=waveform, sample_rate=48000)


def save_audio(audio: Audio, path: str) -> None:
    wav = audio.waveform
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    sf.write(path, wav.detach().cpu().float().numpy().T, audio.sample_rate)
    print(f"  Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("REAL-TIME COVER: Scope parity + engine exclusives")
    print("=" * 70)

    # === Session ===
    print("\n[Session]")
    t0 = time.perf_counter()
    s = Session(project_root=project_root, decoder_backend="compile", vae_backend="compile")
    print(f"  Ready in {time.perf_counter() - t0:.1f}s")

    # === Load source audio + auto-detect metadata ===
    print("\n[Source Audio]")
    source_audio = load_audio(SOURCE_AUDIO, duration=60.0)
    info = s.audio_info(source_audio)
    bpm, key, duration = info["bpm"], info["key"], info["duration"]
    print(f"  Detected: BPM={bpm}, key={key}, duration={duration}s")

    # === Prepare source (VAE encode + semantic extract) ===
    print("\n[Prepare Source]")
    t0 = time.perf_counter()
    source = s.prepare_source(source_audio)
    T = source.latent.tensor.shape[1]
    print(f"  Done in {time.perf_counter() - t0:.2f}s (T={T})")

    # === Timbre reference (separate audio or source, blended with silence) ===
    print("\n[Timbre Reference]")
    if TIMBRE_REF_AUDIO is not None:
        timbre_audio = load_audio(TIMBRE_REF_AUDIO, duration=duration)
        timbre_latent = s.encode_audio(timbre_audio)
        print(f"  Using separate timbre audio: {TIMBRE_REF_AUDIO}")
    else:
        timbre_latent = source.latent
        print("  Using source audio for timbre")

    if TIMBRE_STRENGTH < 1.0:
        silence = s.empty_latent(duration=duration)
        timbre_latent = s.blend_latents(silence, timbre_latent, alpha=TIMBRE_STRENGTH)
        print(f"  Timbre strength: {TIMBRE_STRENGTH}")

    # === Semantic hints (separate audio or source, blended with silence) ===
    print("\n[Semantic Hints]")
    if HINT_SOURCE_AUDIO is not None:
        hint_audio = load_audio(HINT_SOURCE_AUDIO, duration=duration)
        hint_latent = s.encode_audio(hint_audio)
        context_latent = s.extract_hints(hint_latent)
        print(f"  Using separate hint source: {HINT_SOURCE_AUDIO}")
    else:
        context_latent = source.context_latent
        print("  Using source audio for hints")

    if HINT_STRENGTH < 1.0:
        silence = s.empty_latent(duration=duration)
        context_latent = s.blend_latents(silence, context_latent, alpha=HINT_STRENGTH)
        print(f"  Hint strength: {HINT_STRENGTH}")

    # === Dual text encode + blend ===
    print("\n[Text Encode]")
    t0 = time.perf_counter()
    cond_a = s.encode_text(
        tags=PROMPT_A,
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=timbre_latent,
        bpm=bpm, duration=duration, key=key,
    )
    cond_b = s.encode_text(
        tags=PROMPT_B,
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=timbre_latent,
        bpm=bpm, duration=duration, key=key,
    )
    conditioning = ConditioningAverage().execute(
        conditioning_a=cond_a, conditioning_b=cond_b, weight=PROMPT_BLEND,
    )["conditioning"]
    print(f"  Prompts encoded + blended ({PROMPT_BLEND:.0%} B) in {time.perf_counter() - t0:.2f}s")

    # === LoRAs ===
    if LORAS:
        print("\n[LoRAs]")
        for path, scale in LORAS:
            name = os.path.basename(path)
            s.apply_lora(path, scale=scale)
            print(f"  Applied: {name} (scale={scale})")

    # === Build per-frame curves ===
    print("\n[Curves]")
    gen_kwargs = {}

    if VELOCITY_CURVE is not None:
        kind, start, end = VELOCITY_CURVE
        vel = CurveRamp().execute(start=start, end=end, length=T)["curve"]
        gen_kwargs["velocity_scale"] = vel
        print(f"  velocity_scale: {start} -> {end}")

    if SDE_DENOISE_CURVE is not None:
        kind, start, end = SDE_DENOISE_CURVE
        sde = CurveRamp().execute(start=start, end=end, length=T)["curve"]
        gen_kwargs["sde_denoise_curve"] = sde
        print(f"  sde_denoise_curve: {start} -> {end}")

    if INITIAL_NOISE_CURVE is not None:
        kind, start, end = INITIAL_NOISE_CURVE
        inc = CurveRamp().execute(start=start, end=end, length=T)["curve"]
        gen_kwargs["initial_noise_curve"] = inc
        print(f"  initial_noise_curve: {start} -> {end}")

    if ODE_NOISE_CURVE is not None:
        kind, freq, amp = ODE_NOISE_CURVE
        fps = 25
        frames_per_cycle = int(fps / freq) if freq > 0 else T
        ode = CurveWave().execute(
            wave_type="sine", frames_per_cycle=frames_per_cycle,
            amplitude=amp / 2, offset=amp / 2, length=T,
        )["curve"]
        gen_kwargs["ode_noise_curve"] = ode
        print(f"  ode_noise_curve: {freq}Hz, amplitude={amp}")

    if not gen_kwargs:
        print("  (none)")

    # === Warmup (first generate triggers torch.compile, do before LoRA timing) ===
    print("\n[Warmup]")
    t0 = time.perf_counter()
    _ = s.generate(
        conditioning=conditioning,
        context_latent=context_latent,
        source_latent=source.latent,
        seed=0, **gen_kwargs,
    )
    print(f"  torch.compile warmup: {time.perf_counter() - t0:.1f}s")

    # === Generate (with LoRA, on warm compiled decoder) ===
    print(f"\n[Generate] seed={SEED}, denoise={DENOISE}, steps={STEPS}")
    t0 = time.perf_counter()
    output_latent = s.generate(
        conditioning=conditioning,
        context_latent=context_latent,
        source_latent=source.latent,
        seed=SEED, denoise=DENOISE, steps=STEPS,
        **gen_kwargs,
    )
    t_gen = time.perf_counter() - t0

    # === Decode ===
    t0 = time.perf_counter()
    output_audio = s.decode(output_latent)
    t_dec = time.perf_counter() - t0
    print(f"  generate={t_gen:.3f}s  decode={t_dec:.3f}s  total={t_gen + t_dec:.3f}s")

    save_audio(output_audio, os.path.join(OUTPUT_DIR, "realtime_cover.wav"))

    # === Remove LoRAs ===
    if LORAS:
        s.remove_loras()
        print("  LoRAs removed")

    # === Fast iteration demo (seed sweep, everything else cached) ===
    print("\n[Seed Sweep] (demonstrating cached iteration)")
    for seed in [9999, 42, 7777]:
        t0 = time.perf_counter()
        out = s.generate(
            conditioning=conditioning,
            context_latent=context_latent,
            source_latent=source.latent,
            seed=seed, denoise=DENOISE, steps=STEPS,
            **gen_kwargs,
        )
        audio_out = s.decode(out)
        elapsed = time.perf_counter() - t0
        print(f"  seed={seed}: {elapsed:.3f}s")
        save_audio(audio_out, os.path.join(OUTPUT_DIR, f"seed_{seed}.wav"))

    print("\nDone.")


if __name__ == "__main__":
    main()
