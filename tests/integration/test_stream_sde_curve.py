"""Stream pipeline with per-frame SDE denoise curves.

Instead of sending a scalar denoise per generation, sends a [1, T, 1]
curve that encodes the denoise knob position at 25Hz frame resolution.
This means the knob trajectory is baked into each generation at frame
level, rather than being sampled once per ~130ms generation.

Simulates a user sweeping the denoise knob from 0.3 to 1.0 and back
over ~20 seconds of real time. Each generation's curve encodes the
knob position history at that moment, mapped across the 60s audio
timeline.
"""
if __name__ != "__main__":
    import sys; sys.exit(0)

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

import numpy as np
import soundfile as sf

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session, PreparedSource
from acestep.engine.diffusion import DiffusionConfig, DiffusionEngine
from acestep.engine.stream import StreamPipeline, SlotRequest
from acestep.nodes.types import Audio, Latent
from acestep.fixtures import audio_fixture

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_AUDIO = audio_fixture("inside_confusion_loop_60s_gsm.wav")
OUTPUT_DIR = PROJECT_ROOT / "_debug_tests" / "stream_output"
OUTPUT_FILE = OUTPUT_DIR / "stream_sde_curve_sweep.wav"

SAMPLE_RATE = 48000
SEED = 1528
T = 1500  # 60s at 25fps

TRT_ENGINE = PROJECT_ROOT / "trt_engines" / "decoder_mixed_refit_b8_240s" / "decoder_mixed_refit_b8_240s.engine"
VAE_ENCODE_ENGINE = PROJECT_ROOT / "trt_engines" / "vae_encode_fp16_240s" / "vae_encode_fp16_240s.engine"
VAE_DECODE_ENGINE = PROJECT_ROOT / "trt_engines" / "vae_decode_fp16_240s" / "vae_decode_fp16_240s.engine"


def load_audio(path, duration=60.0):
    data, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform[:2, :int(duration * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    return Audio(waveform=waveform, sample_rate=SAMPLE_RATE)


def make_curve(knob_value, T):
    """Constant curve from a single knob value."""
    return torch.full((1, T, 1), knob_value)


def make_ramp_curve(start, end, T):
    """Linear ramp curve across the temporal dimension."""
    return torch.linspace(start, end, T).reshape(1, T, 1)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Stream SDE Curve - Per-Frame Denoise Sweep")
print("=" * 60)

timings = {}

def timed(label, quiet=False):
    class _Timer:
        def __enter__(self_):
            torch.cuda.synchronize()
            self_.t0 = time.perf_counter()
            return self_
        def __exit__(self_, *exc):
            torch.cuda.synchronize()
            self_.ms = (time.perf_counter() - self_.t0) * 1000
            timings.setdefault(label, []).append(self_.ms)
            if not quiet:
                print(f"  [{label}] {self_.ms:.1f}ms")
    return _Timer()


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
with timed("model_load"):
    print("\n[Setup] Loading model...")
    session = Session(
        project_root=str(PROJECT_ROOT / "checkpoints"),
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines={
            "decoder": str(TRT_ENGINE),
            "vae_encode": str(VAE_ENCODE_ENGINE),
            "vae_decode": str(VAE_DECODE_ENGINE),
        },
    )

handler = session.handler
device = handler.device
dtype = handler.dtype

with timed("load_audio"):
    audio = load_audio(SOURCE_AUDIO)

with timed("vae_encode"):
    latent = session.encode_audio(audio)
with timed("semantic_extract"):
    context_latent = session.extract_hints(latent)
source = PreparedSource(latent=latent, context_latent=context_latent)
print(f"  Source: T={source.latent.tensor.shape[1]} frames")

with timed("text_encode"):
    cond = session.encode_text(
        tags="deathstep, heavy bass, dark atmosphere",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136,
        duration=60.0,
        key="G# minor",
    )
entry = cond.to_entries()[0]

ctx_lat = source.context_latent.tensor.to(device=device, dtype=dtype)
D = ctx_lat.shape[2]
cm = torch.ones(1, T, D, device=device, dtype=dtype)
context_latents = torch.cat([ctx_lat, cm], dim=-1)
source_latents = source.latent.tensor.to(device=device, dtype=dtype)

# ------------------------------------------------------------------
# Build curve timeline
# ------------------------------------------------------------------
# Simulate knob sweep: smooth ramp across frames, changing each gen.
# Each generation gets a curve that ramps from knob_low to knob_high,
# where the endpoints shift over time to simulate the knob moving.
num_submissions = 80
curves = []
for i in range(num_submissions):
    # Knob sweeps 0.3 -> 1.0 -> 0.3 over all submissions
    t = i / num_submissions
    knob_center = 0.3 + 0.35 * (1.0 - np.cos(2 * np.pi * t))
    # Each curve is a ramp around the center: +-0.15 across the timeline
    knob_lo = max(0.0, knob_center - 0.15)
    knob_hi = min(1.0, knob_center + 0.15)
    curves.append(make_ramp_curve(knob_lo, knob_hi, T))

total_ticks = num_submissions + 8 + 8

print(f"\n[Timeline] {num_submissions} submissions, ~{total_ticks} ticks")
print(f"  Each submission sends a ramp curve [1, {T}, 1]")
print(f"  Knob center sweeps 0.3 -> 1.0 -> 0.3")

# ------------------------------------------------------------------
# Also run ODE scalar denoise for comparison
# ------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("A) ODE scalar denoise (baseline)")
print("=" * 60)

engine = handler._diffusion_engine
config = DiffusionConfig(infer_steps=8, shift=3.0, noise_on_cpu=True)
pipe_ode = StreamPipeline(engine, config)

submit_idx = 0
num_completed_ode = 0
ode_start = time.time()

for tick_num in range(total_ticks):
    if submit_idx < num_submissions:
        # Use the curve center as scalar denoise
        t = submit_idx / num_submissions
        dn = 0.3 + 0.35 * (1.0 - np.cos(2 * np.pi * t))
        pipe_ode.submit(SlotRequest(
            encoder_hidden_states=entry.encoder_hidden_states,
            encoder_attention_mask=entry.encoder_attention_mask,
            context_latents=context_latents,
            seed=SEED,
            source_latents=source_latents,
            denoise=round(dn, 3),
        ))
        submit_idx += 1

    result = pipe_ode.tick()
    if result is not None:
        num_completed_ode += 1

    if pipe_ode.active_slots == 0 and submit_idx >= num_submissions:
        break

ode_ms = (time.time() - ode_start) * 1000
print(f"  {num_completed_ode} gens in {ode_ms:.0f}ms "
      f"({ode_ms/max(num_completed_ode,1):.1f}ms/gen, tick only)")

# ------------------------------------------------------------------
# SDE curve run
# ------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("B) SDE per-frame curve")
print("=" * 60)

pipe_sde = StreamPipeline(engine, config)

submit_idx = 0
num_completed_sde = 0
sde_start = time.time()

for tick_num in range(total_ticks):
    if submit_idx < num_submissions:
        pipe_sde.submit(SlotRequest(
            encoder_hidden_states=entry.encoder_hidden_states,
            encoder_attention_mask=entry.encoder_attention_mask,
            context_latents=context_latents,
            seed=SEED,
            source_latents=source_latents,
            denoise=0.75,  # schedule still uses this for timestep range
            sde_denoise_curve=curves[submit_idx],
        ))
        submit_idx += 1

    result = pipe_sde.tick()
    if result is not None:
        num_completed_sde += 1

    if pipe_sde.active_slots == 0 and submit_idx >= num_submissions:
        break

sde_ms = (time.time() - sde_start) * 1000
print(f"  {num_completed_sde} gens in {sde_ms:.0f}ms "
      f"({sde_ms/max(num_completed_sde,1):.1f}ms/gen, tick only)")

# ------------------------------------------------------------------
# SDE curve run with decode (full pipeline)
# ------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("C) SDE per-frame curve + decode")
print("=" * 60)

pipe_full = StreamPipeline(engine, config)

submit_idx = 0
num_completed_full = 0
num_decoded = 0
last_latent = None
last_wav = None
mse_values = []
skip_threshold = 1e-3
slice_duration = 0.3
slice_samples = int(slice_duration * SAMPLE_RATE)
playback_start = 5.0
playback_offset = int(playback_start * SAMPLE_RATE)
output_chunks = []
prev_curve_mean = None
tick_times = []
decode_times = []

full_start = time.time()

# Track which curve index each result maps to
curve_indices = []

for tick_num in range(total_ticks):
    if submit_idx < num_submissions:
        pipe_full.submit(SlotRequest(
            encoder_hidden_states=entry.encoder_hidden_states,
            encoder_attention_mask=entry.encoder_attention_mask,
            context_latents=context_latents,
            seed=SEED,
            source_latents=source_latents,
            denoise=0.75,
            sde_denoise_curve=curves[submit_idx],
        ))
        submit_idx += 1

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = pipe_full.tick()

    if result is not None:
        torch.cuda.synchronize()
        tick_ms = (time.perf_counter() - t0) * 1000
        tick_times.append(tick_ms)

        # Figure out which curve this result used
        submit_tick = tick_num - config.infer_steps
        if 0 <= submit_tick < num_submissions:
            curve_idx = submit_tick
            c = curves[curve_idx]
            curve_lo = c[0, 0, 0].item()
            curve_hi = c[0, -1, 0].item()
            curve_mean = c.mean().item()
        else:
            curve_idx = -1
            curve_lo = curve_hi = curve_mean = -1.0

        skipped = False
        if last_latent is not None:
            mse = (result - last_latent).pow(2).mean().item()
            mse_values.append(mse)
            if mse < skip_threshold and last_wav is not None:
                wav = last_wav
                skipped = True

        last_latent = result.clone()

        dec_ms = 0.0
        if not skipped:
            dec_t0 = time.perf_counter()
            audio_out = session.decode(Latent(tensor=result))
            torch.cuda.synchronize()
            dec_ms = (time.perf_counter() - dec_t0) * 1000
            decode_times.append(dec_ms)
            wav = audio_out.waveform.detach().cpu().float().squeeze(0)
            last_wav = wav
            num_decoded += 1

        start = playback_offset + num_completed_full * slice_samples
        end = start + slice_samples
        if end <= wav.shape[1]:
            chunk = wav[:, start:end]
        else:
            chunk = torch.zeros(wav.shape[0], slice_samples)
            avail = wav.shape[1] - start
            if avail > 0:
                chunk[:, :avail] = wav[:, start:start+avail]

        output_chunks.append(chunk)
        num_completed_full += 1

        dec_str = "SKIP" if skipped else f"{dec_ms:5.1f}ms"
        mse_str = f"mse={mse:.2e}" if len(mse_values) > 0 else ""
        if round(curve_mean, 2) != prev_curve_mean or num_completed_full % 20 == 0:
            print(f"  #{num_completed_full:3d} curve={curve_lo:.2f}->{curve_hi:.2f}  "
                  f"tick={tick_ms:5.1f}ms  decode={dec_str:>7s}  {mse_str}  "
                  f"(playback {start/SAMPLE_RATE:.1f}s-{end/SAMPLE_RATE:.1f}s)")
            prev_curve_mean = round(curve_mean, 2)

    if pipe_full.active_slots == 0 and submit_idx >= num_submissions:
        break

full_ms = (time.time() - full_start) * 1000
num_skipped = num_completed_full - num_decoded
print(f"\n[Run] {num_completed_full} generations in {full_ms:.0f}ms "
      f"({full_ms/max(num_completed_full,1):.1f}ms avg incl. decode)")
print(f"  Decoded: {num_decoded}, Skipped: {num_skipped} "
      f"({100*num_skipped/max(num_completed_full,1):.0f}% skip rate)")
if mse_values:
    sorted_mse = sorted(mse_values)
    print(f"  MSE: min={sorted_mse[0]:.2e}  median={sorted_mse[len(sorted_mse)//2]:.2e}  "
          f"max={sorted_mse[-1]:.2e}")

# Save output
if output_chunks:
    output_wav = torch.cat(output_chunks, dim=1)
    sf.write(str(OUTPUT_FILE), output_wav.numpy().T, SAMPLE_RATE, format="WAV")
    print(f"\n[Save] Output: {output_wav.shape[1]/SAMPLE_RATE:.1f}s, {output_wav.shape}")
    print(f"  Saved: {OUTPUT_FILE}")

# Also save source for A/B
source_out = OUTPUT_DIR / "source_reference.wav"
src_wav = audio.waveform
if src_wav.dim() == 3:
    src_wav = src_wav.squeeze(0)
sf.write(str(source_out), src_wav.numpy().T, SAMPLE_RATE, format="WAV")
print(f"  Source: {source_out}")

# ------------------------------------------------------------------
# Timing summary
# ------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("TIMING SUMMARY")
print(f"{'=' * 60}")
for label in ["model_load", "load_audio", "vae_encode",
               "semantic_extract", "text_encode"]:
    vals = timings.get(label, [])
    if not vals:
        continue
    print(f"  {label:22s}  {sum(vals):8.1f}ms  (1 call)")

if tick_times:
    steady = tick_times[2:] if len(tick_times) > 2 else tick_times
    t_avg = sum(steady) / len(steady)
    t_min, t_max = min(steady), max(steady)
    print(f"  {'tick':22s}  {sum(tick_times):8.1f}ms total  "
          f"avg={t_avg:6.1f}ms  min={t_min:6.1f}ms  max={t_max:6.1f}ms  "
          f"({len(tick_times)} calls)")

if decode_times:
    d_avg = sum(decode_times) / len(decode_times)
    d_min, d_max = min(decode_times), max(decode_times)
    print(f"  {'vae_decode':22s}  {sum(decode_times):8.1f}ms total  "
          f"avg={d_avg:6.1f}ms  min={d_min:6.1f}ms  max={d_max:6.1f}ms  "
          f"({len(decode_times)} calls)")

# Amortized per-gen
if tick_times:
    tick_total = sum(tick_times)
    decode_total = sum(decode_times) if decode_times else 0
    avg_tick = tick_total / num_completed_full
    avg_decode = decode_total / num_completed_full
    print(f"\n  Per-generation (amortized over {num_completed_full} gens):")
    print(f"    tick={avg_tick:.1f}ms + decode={avg_decode:.1f}ms "
          f"({num_decoded} decoded, {num_skipped} skipped) "
          f"= {avg_tick + avg_decode:.1f}ms")

# Comparison
print(f"\n  ODE vs SDE tick comparison:")
print(f"    A) ODE scalar:  {ode_ms/max(num_completed_ode,1):6.1f}ms/gen")
print(f"    B) SDE curve:   {sde_ms/max(num_completed_sde,1):6.1f}ms/gen")
overhead = (sde_ms/max(num_completed_sde,1)) / (ode_ms/max(num_completed_ode,1))
print(f"    SDE overhead: {overhead:.2f}x")

print(f"\n  Key insight: SDE curve encodes knob trajectory at 25Hz frame")
print(f"  resolution within each generation, vs ~{1000/(full_ms/max(num_completed_full,1)):.0f}Hz generation rate")
print("=" * 60)
