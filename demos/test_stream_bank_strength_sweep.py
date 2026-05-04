"""Definitive A/B test for the streamA2A FeatureBank.

Sweeps an ordered list of seeds through the streaming pipeline once per
bank strength in {0.0, 0.25, 0.5, 0.75, 1.0} and writes one WAV per
strength. With strength=0 the bank is fully masked (no influence) and
each seed produces a wildly different generation; as strength grows,
successive generations should "lock onto" the previous ones via per-step
K/V re-injection and the output should sound increasingly stable across
seed changes. The five WAVs are byte-comparable because the only thing
that varies between runs is the bank strength: same seeds in the same
order, same source, same conditioning, same denoise, fresh stream per
run so bank state starts empty.

Eager decoder + eager VAE only -- the bank-aware TRT engine doesn't
build at this duration and ``StreamPipeline.enable_feature_bank``
explicitly refuses when a TRT engine is loaded.
"""

if __name__ != "__main__":
    import sys
    sys.exit(0)

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
from acestep.engine.session import Session
from acestep.nodes.types import Audio, Latent
from acestep.nodes.vae_nodes import EmptyLatent, LatentBlend
from acestep.paths import checkpoints_dir, project_root

PROJECT_ROOT = project_root()
DEFAULT_SOURCE_AUDIO = PROJECT_ROOT / "tests/fixtures" / "new_order_confusion_60seconds.wav"
OUTPUT_DIR = PROJECT_ROOT / "_debug_tests" / "bank_strength_sweep"

SAMPLE_RATE = 48000

STRENGTHS = (1.5, 2.0)

# CLI flags
_args = sys.argv[1:]


def _get_arg(name, default=None, cast=str):
    if name in _args:
        return cast(_args[_args.index(name) + 1])
    return default


# Sweep length / generator. 64 seeds at 0.3s/slice = ~19s of audio per
# strength run, which is long enough to clearly hear identity locking
# without being ridiculous. Seeds are sequential ints so PyTorch's PRNG
# produces well-decorrelated noise per gen at strength=0.
num_seeds = _get_arg("--num-seeds", 64, int)
seed_base = _get_arg("--seed-base", 1528, int)
SEEDS = list(range(seed_base, seed_base + num_seeds))

# A single denoise value is held across the whole sweep so the bank's
# effect is the only varying input. 1.0 = full noise -> full inheritance
# from the bank when strength is high; lower values dilute the test.
fixed_denoise = _get_arg("--denoise", 1.0, float)

# hint_strength: blend the source's context latent toward silence
# before each tick. 1.0 = full source structure, 0.0 = no structural
# guidance. Pulling this down (e.g. 0.15) frees the model to drift
# more across gens, which makes the bank's identity-locking effect
# stand out: at strength=0 each gen wanders freely, at strength=1 the
# bank should still pull successive gens together despite the loose
# structural anchor.
hint_strength = _get_arg("--hint-strength", 0.15, float)

depth = _get_arg("--depth", 8, int)
vae_window = _get_arg("--vae-window", 0.0, float)

source_audio_override = _get_arg("--source-audio", None, str)
if source_audio_override is not None:
    _p = Path(source_audio_override)
    SOURCE_AUDIO = _p if _p.is_absolute() else PROJECT_ROOT / _p
else:
    SOURCE_AUDIO = DEFAULT_SOURCE_AUDIO

# DCW defaults to ON (matches v0.1.7 + the cover graph test). Off via
# --no-dcw for cleaner A/B if the operator suspects DCW is masking the
# bank's effect.
use_dcw = "--no-dcw" not in _args
dcw_mode = _get_arg("--dcw-mode", "double", str)
dcw_scaler = _get_arg("--dcw-scaler", 0.05, float)
dcw_high_scaler = _get_arg("--dcw-high-scaler", 0.02, float)
dcw_wavelet = _get_arg("--dcw-wavelet", "haar", str)


def load_audio(path, duration=60.0):
    data, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform[:2, : int(duration * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, : waveform.shape[-1] - rem]
    return Audio(waveform=waveform, sample_rate=SAMPLE_RATE)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Bank-strength A/B sweep (eager backend)")
print("=" * 60)
print(f"  strengths : {STRENGTHS}")
print(f"  seeds     : {len(SEEDS)} sequential starting at {seed_base}")
print(f"  denoise   : {fixed_denoise}")
print(f"  depth     : {depth}")
print(f"  source    : {SOURCE_AUDIO.name}")
print(f"  output    : {OUTPUT_DIR}")
print()

# ------------------------------------------------------------------
# One-time setup: model, source, conditioning. Reused for all 5
# strength runs so the only thing that varies per WAV is bank strength.
# ------------------------------------------------------------------
print("[Setup] Loading model (eager / eager)...")
t0 = time.time()
session = Session(
    project_root=str(checkpoints_dir()),
    decoder_backend="eager",
    vae_backend="eager",
    trt_engines={},
    vae_window=vae_window,
)
print(f"  Model loaded in {time.time() - t0:.1f}s")

print("[Setup] Loading source audio...")
audio = load_audio(SOURCE_AUDIO)

print("[Setup] Preparing source...")
source = session.prepare_source(audio)
T = source.latent.tensor.shape[1]
print(f"  Source: T={T} frames ({T / 25:.1f}s)")

# Build the blended context latent once. Each fresh stream gets its
# .context_latent reassigned to this so every strength run sees the
# same structural hint. Mirrors PipelineRunner._update_hint_strength.
if hint_strength < 1.0:
    _silence = EmptyLatent().execute(model=session.model, duration=T / 25.0)["latent"]
    blended_context = LatentBlend().execute(
        latent_a=_silence,
        latent_b=source.context_latent,
        alpha=hint_strength,
    )["latent"]
    print(f"  hint_strength: {hint_strength:.2f} (context blended toward silence)")
else:
    blended_context = source.context_latent
    print(f"  hint_strength: {hint_strength:.2f} (full source structure)")

print("[Setup] Encoding cover conditioning...")
cond = session.encode_text(
    tags="deathcore, heavy, DISTORTED GUITARS, BRUTAL",
    instruction=TASK_INSTRUCTIONS["cover"],
    refer_latent=source.latent,
    bpm=136,
    duration=60.0,
    key="G# minor",
)


def run_strength(strength: float) -> Path:
    """Run the seed sweep for one bank strength, return the saved WAV path."""
    print(f"\n{'=' * 60}")
    print(f"  STRENGTH = {strength:.2f}")
    print(f"{'=' * 60}")

    # Fresh stream per run so the StreamDenoise node lazy-builds a fresh
    # StreamPipeline (and therefore a fresh bank state) each time.
    stream = session.stream(
        source=source,
        conditioning=cond,
        steps=8,
        shift=3.0,
        pipeline_depth=depth,
        dcw_enabled=use_dcw,
        dcw_mode=dcw_mode,
        dcw_scaler=dcw_scaler,
        dcw_high_scaler=dcw_high_scaler,
        dcw_wavelet=dcw_wavelet,
    )
    # Apply the precomputed hint blend to this stream's context latent.
    stream.context_latent = blended_context

    total_submissions = len(SEEDS)
    # +depth for warmup (first finished gen lands at tick depth) and
    # +depth for drain (so all submitted gens finish).
    total_ticks = total_submissions + depth + depth

    slice_duration = 0.3
    slice_samples = int(slice_duration * SAMPLE_RATE)
    playback_start = 5.0
    playback_offset_samples = int(playback_start * SAMPLE_RATE)

    output_chunks = []
    last_latent = None
    last_wav = None
    last_win_start_sample = 0
    skip_threshold = 1e-3
    num_skipped = 0
    num_completed = 0
    submit_idx = 0
    bank_installed = False

    run_t0 = time.time()
    for tick_num in range(total_ticks):
        # Lazy bank install: needs to happen AFTER the first tick has
        # constructed the StreamPipeline. enable_feature_bank refuses
        # when a TRT engine is loaded; we're eager-only so it succeeds.
        if not bank_installed and stream.pipeline is not None:
            stream.pipeline.enable_feature_bank(strength=strength)
            bank_installed = True
            bank = stream.pipeline.feature_bank
            n_layers = len(bank.banked)
            print(
                f"  [bank] installed on {n_layers} layers, "
                f"strength={bank.strength:.2f}"
            )

        if submit_idx < total_submissions:
            seed_i = SEEDS[submit_idx]
            result_latent = stream.tick(denoise=fixed_denoise, seed=seed_i)
            submit_idx += 1
        else:
            raw = stream.pipeline.tick()
            result_latent = Latent(tensor=raw) if raw is not None else None

        if result_latent is None:
            if stream.stream_node.active_slots == 0 and submit_idx >= total_submissions:
                break
            continue

        result = result_latent.tensor

        start = playback_offset_samples + num_completed * slice_samples
        end = start + slice_samples

        skipped = False
        if last_latent is not None:
            mse = (result - last_latent).pow(2).mean().item()
            if mse < skip_threshold and last_wav is not None:
                local_start = start - last_win_start_sample
                local_end = local_start + slice_samples
                if 0 <= local_start and local_end <= last_wav.shape[1]:
                    wav = last_wav
                    skipped = True
                    num_skipped += 1
        last_latent = result.clone()

        if not skipped:
            if vae_window > 0:
                t_start = start / SAMPLE_RATE
                audio_out = session.decode(result_latent, t_start=t_start)
                wav = audio_out.waveform.detach().cpu().float().squeeze(0)
                win_start_sample = audio_out.start_sample
            else:
                audio_out = session.decode(result_latent)
                wav = audio_out.waveform.detach().cpu().float().squeeze(0)
                win_start_sample = 0
            last_wav = wav
            last_win_start_sample = win_start_sample

        local_start = start - last_win_start_sample
        local_end = local_start + slice_samples
        if local_end <= wav.shape[1]:
            chunk = wav[:, local_start:local_end]
        else:
            chunk = torch.zeros(wav.shape[0], slice_samples)
            available = wav.shape[1] - local_start
            if available > 0:
                chunk[:, :available] = wav[:, local_start : local_start + available]
        output_chunks.append(chunk)
        num_completed += 1

        # Per-gen progress every 8 gens; quieter than the cover graph
        # because we're running this 5 times.
        if num_completed % 8 == 0 or num_completed == 1:
            bank = stream.pipeline.feature_bank
            entries = bank.num_entries() if bank is not None else 0
            print(
                f"    gen {num_completed:3d}/{total_submissions}  "
                f"seed_idx={submit_idx - 1:3d}  bank_entries={entries:4d}  "
                f"({'skipped' if skipped else 'decoded'})"
            )

        if stream.stream_node.active_slots == 0 and submit_idx >= total_submissions:
            break

    run_ms = (time.time() - run_t0) * 1000
    print(
        f"  done: {num_completed} gens in {run_ms:.0f}ms "
        f"({num_skipped} skipped); bank_entries={stream.pipeline.feature_bank.num_entries()}"
    )

    output_wav = torch.cat(output_chunks, dim=1)
    out_name = f"bank_sweep_strength{strength:.2f}.wav".replace("0.00", "0p00").replace(
        "0.25", "0p25"
    ).replace("0.50", "0p50").replace("0.75", "0p75").replace("1.00", "1p00")
    out_path = OUTPUT_DIR / out_name
    sf.write(str(out_path), output_wav.numpy().T, SAMPLE_RATE, format="WAV")
    print(f"  saved: {out_path.name} ({output_wav.shape[1] / SAMPLE_RATE:.1f}s)")

    # Disable the bank on this stream's pipeline so the decoder's
    # self_attn modules go back to their unpatched state before the
    # next strength run swaps in a fresh bank. enable_feature_bank is
    # idempotent (re-uses the patch hook and just rebinds .bank), but
    # disabling is the cleanest way to be sure no stale references
    # leak between runs.
    if stream.pipeline.feature_bank is not None:
        stream.pipeline.disable_feature_bank()

    return out_path


# Save the source audio once for reference (same across strength runs).
src_out = OUTPUT_DIR / "source_reference.wav"
src_wav = audio.waveform
if src_wav.dim() == 3:
    src_wav = src_wav.squeeze(0)
sf.write(str(src_out), src_wav.numpy().T, SAMPLE_RATE, format="WAV")
print(f"\n[Setup] Source reference saved to {src_out.name}")

# Run the sweep.
results = []
for s in STRENGTHS:
    results.append(run_strength(s))

print("\n" + "=" * 60)
print("Sweep complete. Outputs:")
for s, p in zip(STRENGTHS, results):
    print(f"  strength={s:.2f}  ->  {p}")
print("=" * 60)
print(
    "\nA/B by ear: at strength=0 every gen should sound different "
    "(seeds dominate); as strength climbs the new gens should track "
    "the previous one's timbre/character more strongly. The transition "
    "from 0.00 -> 1.00 is the bank's effect."
)
