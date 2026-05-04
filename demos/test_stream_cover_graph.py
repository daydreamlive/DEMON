"""Offline stress test for the streaming pipeline.

Sweeps denoise strength over many ticks and splices the resulting
generations into a single WAV at advancing playback positions, so the
listener hears the song progress while the denoise character shifts.
Useful for validating performance and listening to what the stream
pipeline produces across a range of denoise values without interactive
I/O.

Drives the pipeline through ``Session.stream`` + ``handle.tick(**kwargs)``.
Drain-phase ticks (after all submissions are queued) go through
``handle.pipeline.tick()`` directly.
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
from acestep.paths import checkpoints_dir, project_root, trt_engine_path, select_trt_engines
from acestep.fixtures import audio_fixture

PROJECT_ROOT = project_root()
DEFAULT_SOURCE_AUDIO = audio_fixture("inside_confusion_loop_60s_gsm.wav")
OUTPUT_DIR = PROJECT_ROOT / "demos" / "outputs" / "stream_cover_graph"

SAMPLE_RATE = 48000
SEED = 1528

LORA_PATH = ""
LORA_STRENGTH = 1.4

# CLI flags
_args = sys.argv[1:]


def _get_arg(name, default=None, cast=str):
    if name in _args:
        return cast(_args[_args.index(name) + 1])
    return default


vae_window = _get_arg("--vae-window", 0.0, float)
depth = _get_arg("--depth", 8, int)
use_fast_vae = "--fast-vae" in _args
use_lora = "--lora" in _args
if use_lora:
    _i = _args.index("--lora")
    if _i + 1 < len(_args) and not _args[_i + 1].startswith("--"):
        LORA_PATH = _args[_i + 1]

# DCW (Differential Correction in Wavelet domain) — on by default,
# opt out via --no-dcw. Defaults match upstream v0.1.7
# (mode=double, scaler=0.05, high=0.02, haar).
use_dcw = "--no-dcw" not in _args
dcw_mode = _get_arg("--dcw-mode", "double", str)
dcw_scaler = _get_arg("--dcw-scaler", 0.05, float)
dcw_high_scaler = _get_arg("--dcw-high-scaler", 0.02, float)
dcw_wavelet = _get_arg("--dcw-wavelet", "haar", str)

# Optional explicit output filename. Relative paths land in OUTPUT_DIR;
# absolute paths are used as-is. When omitted, a filename is synthesized
# from the active config so A/B runs don't overwrite each other.
explicit_output = _get_arg("--output", None, str)

# Backend selection: choose checkpoint and decoder engine.
# Defaults preserve the original 2B turbo + mixed engine behavior.
checkpoint = _get_arg("--checkpoint", "acestep-v15-turbo")

# Default decoder engine depends on the checkpoint. This demo runs at 60s,
# so the 60s variant is the default; pass --decoder-engine to override.
if "--decoder-engine" in _args:
    decoder_engine_name = _get_arg("--decoder-engine")
elif checkpoint == "acestep-v15-xl-turbo":
    # No 60s variant of the refit XL decoder is built today; keep the 240s
    # default for this checkpoint until one is available.
    decoder_engine_name = "decoder_xl-turbo_bf16_refit_b8_240s"
else:
    decoder_engine_name = "decoder_mixed_refit_b8_60s"

# Pure-PyTorch decoder mode (skip TRT decoder; VAE engines still used).
no_trt_decoder = "--no-trt-decoder" in _args
# Pure-PyTorch everywhere: skip TRT decoder AND TRT VAE engines.
no_acceleration = "--no-acceleration" in _args
if no_acceleration:
    no_trt_decoder = True

source_audio_override = _get_arg("--source-audio", None, str)
if source_audio_override is not None:
    _p = Path(source_audio_override)
    SOURCE_AUDIO = _p if _p.is_absolute() else PROJECT_ROOT / _p
else:
    SOURCE_AUDIO = DEFAULT_SOURCE_AUDIO

TRT_ENGINE = trt_engine_path(decoder_engine_name)

# Output filename encodes (config, backend, engine) so different recipes
# don't overwrite each other.
_ckpt_tag = checkpoint.replace("acestep-v15-", "")  # turbo / xl-turbo / ...
if no_trt_decoder:
    _backend_tag = "pt"
else:
    # Use the engine name as the backend tag, stripping common prefixes
    # so the filename stays readable (e.g. "decoder_xl-turbo_attnsafe_b8_60s"
    # -> "trt_attnsafe_b8_60s").
    _eng = decoder_engine_name
    for prefix in ("decoder_xl-turbo_", "decoder_"):
        if _eng.startswith(prefix):
            _eng = _eng[len(prefix) :]
            break
    _backend_tag = f"trt_{_eng}"

if explicit_output is not None:
    _p = Path(explicit_output)
    OUTPUT_FILE = _p if _p.is_absolute() else OUTPUT_DIR / _p
else:
    if not use_dcw:
        _dcw_tag = "_nodcw"
    elif (
        dcw_mode == "double"
        and dcw_scaler == 0.05
        and dcw_high_scaler == 0.02
        and dcw_wavelet == "haar"
    ):
        # Default DCW config (now on by default) — keep the filename clean.
        _dcw_tag = ""
    else:
        _dcw_tag = (
            f"_dcw-{dcw_mode}-s{dcw_scaler:g}"
            + (f"-h{dcw_high_scaler:g}" if dcw_mode == "double" else "")
            + (f"-{dcw_wavelet}" if dcw_wavelet != "haar" else "")
        )
    OUTPUT_FILE = OUTPUT_DIR / (
        f"stream_cover_graph_{_ckpt_tag}_{_backend_tag}{_dcw_tag}.wav"
    )


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
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
print("Stream Cover - offline pipeline stress test")
print("=" * 60)

# ------------------------------------------------------------------
# Setup -- all through Session API
# ------------------------------------------------------------------
# load_audio() defaults to 60s, so 60s engines are the right default.
if no_acceleration:
    trt_engines = {}
else:
    _base_engines = select_trt_engines(duration_s=60.0)
    trt_engines = {
        "vae_encode": _base_engines["vae_encode"],
        "vae_decode": _base_engines["vae_decode"],
    }
if use_fast_vae and not no_acceleration:
    fast_name = "dreamvae_decode_fp16_60s"
    if not Path(str(trt_engine_path(fast_name))).exists():
        print(f"[Setup] WARNING: {fast_name} engine missing, using {Path(trt_engines['vae_decode']).stem}")
    else:
        trt_engines["vae_decode"] = str(trt_engine_path(fast_name))
if not no_trt_decoder:
    trt_engines["decoder"] = str(TRT_ENGINE)

print(f"\n[Setup] checkpoint={checkpoint}")
print(
    f"[Setup] decoder backend={'PT' if no_trt_decoder else 'TRT (' + decoder_engine_name + ')'}"
)
print(f"[Setup] vae backend={'PT (eager)' if no_acceleration else 'TRT'}  vae_window={vae_window}  depth={depth}")
if use_dcw:
    print(
        f"[Setup] DCW=ON  mode={dcw_mode}  scaler={dcw_scaler}  "
        f"high_scaler={dcw_high_scaler}  wavelet={dcw_wavelet}"
    )
else:
    print("[Setup] DCW=OFF")
print(f"[Setup] output={OUTPUT_FILE.name}")

with timed("model_load"):
    print("[Setup] Loading model...")
    session = Session(
        project_root=str(checkpoints_dir()),
        config_path=checkpoint,
        decoder_backend="tensorrt" if not no_trt_decoder else "eager",
        vae_backend="eager" if no_acceleration else "tensorrt",
        trt_engines=trt_engines,
        vae_window=vae_window,
    )

if use_lora:
    engine_obj = session.handler._diffusion_engine
    if engine_obj is not None and engine_obj.lora_available:
        with timed("apply_lora"):
            print(
                f"[Setup] Applying LoRA: {Path(LORA_PATH).name} (strength={LORA_STRENGTH})"
            )
            engine_obj.apply_lora(LORA_PATH, strength=LORA_STRENGTH)
    else:
        print("[Setup] WARNING: --lora requested but LoRA backend unavailable")

with timed("load_audio"):
    print("[Setup] Loading source audio...")
    audio = load_audio(SOURCE_AUDIO)

print("[Setup] Preparing source...")
with timed("prepare_source"):
    source = session.prepare_source(audio)
T = source.latent.tensor.shape[1]
print(f"  Source: T={T} frames ({T / 25:.1f}s)")

with timed("text_encode"):
    print("[Setup] Encoding cover conditioning...")
    cond = session.encode_text(
        tags="deathcore, heavy, DISTORTED GUITARS, BRUTAL",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136,
        duration=60.0,
        key="G# minor",
    )

# ------------------------------------------------------------------
# Create stream handle -- all through Session API, no engine internals.
# The underlying StreamPipeline is lazily built inside StreamDenoise on
# the first tick, so there is no backend to report here yet.
# ------------------------------------------------------------------
with timed("stream_setup"):
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
print("  Stream handle ready (pipeline builds on first tick)")

# ------------------------------------------------------------------
# Denoise timeline (identical to raw version)
# ------------------------------------------------------------------
num_ticks_per_cycle = 80
num_cycles = 2
total_submissions = num_ticks_per_cycle * num_cycles

denoise_per_tick = []
for i in range(total_submissions):
    t = i / total_submissions
    dn = 0.25 + 0.375 * (1.0 - np.cos(2 * np.pi * num_cycles * t))
    denoise_per_tick.append(round(dn, 3))

total_ticks = total_submissions + 8 + 8

print(f"\n[Timeline] {total_submissions} submissions, ~{total_ticks} ticks")
print(f"  Denoise range: {min(denoise_per_tick):.3f} - {max(denoise_per_tick):.3f}")

# ------------------------------------------------------------------
# Run pipeline -- submit+tick through StreamHandle.tick() while
# submissions remain, then drain-tick through the raw pipeline.
# ------------------------------------------------------------------
submit_idx = 0
num_completed = 0

slice_duration = 0.3
slice_samples = int(slice_duration * SAMPLE_RATE)
playback_start = 5.0
playback_offset_samples = int(playback_start * SAMPLE_RATE)
output_chunks = []
prev_dn = None
last_latent = None
last_wav = None
skip_threshold = 1e-3
num_skipped = 0
mse_values = []
last_win_start_sample = 0

print(
    f"\n[Run] Starting pipeline ({slice_duration}s slices, skip_threshold={skip_threshold})..."
)
run_start = time.time()

for tick_num in range(total_ticks):
    torch.cuda.synchronize()
    iter_t0 = time.perf_counter()

    if submit_idx < len(denoise_per_tick):
        # Active phase: StreamDenoise submits a fresh request every tick.
        dn = denoise_per_tick[submit_idx]
        result_latent = stream.tick(denoise=dn, seed=SEED)
        submit_idx += 1
    else:
        # Drain phase: tick the underlying ring buffer without submitting.
        raw = stream.pipeline.tick()
        result_latent = Latent(tensor=raw) if raw is not None else None

    if result_latent is not None:
        result = result_latent.tensor
        torch.cuda.synchronize()
        tick_ms = (time.perf_counter() - iter_t0) * 1000
        timings.setdefault("tick", []).append(tick_ms)

        submit_tick = tick_num - stream.pipeline.config.infer_steps
        if 0 <= submit_tick < len(denoise_per_tick):
            dn_submitted = denoise_per_tick[submit_tick]
        else:
            dn_submitted = -1.0

        start = playback_offset_samples + num_completed * slice_samples
        end = start + slice_samples

        skipped = False
        if last_latent is not None:
            mse = (result - last_latent).pow(2).mean().item()
            mse_values.append(mse)
            if mse < skip_threshold and last_wav is not None:
                local_start = start - last_win_start_sample
                local_end = local_start + slice_samples
                if 0 <= local_start and local_end <= last_wav.shape[1]:
                    wav = last_wav
                    skipped = True
                    num_skipped += 1

        last_latent = result.clone()

        if not skipped:
            dec_t0 = time.perf_counter()
            if vae_window > 0:
                t_start = start / SAMPLE_RATE
                audio_out = session.decode(result_latent, t_start=t_start)
                wav = audio_out.waveform.detach().cpu().float().squeeze(0)
                win_start_sample = audio_out.start_sample
            else:
                audio_out = session.decode(result_latent)
                wav = audio_out.waveform.detach().cpu().float().squeeze(0)
                win_start_sample = 0
            torch.cuda.synchronize()
            dec_ms = (time.perf_counter() - dec_t0) * 1000
            timings.setdefault("vae_decode", []).append(dec_ms)
            last_wav = wav
            last_win_start_sample = win_start_sample
            local_start = start - win_start_sample
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

        dec_str = "SKIP" if skipped else f"{dec_ms:5.1f}ms"
        mse_str = (
            f"mse={mse:.2e}" if last_latent is not None and num_completed > 1 else ""
        )
        if dn_submitted != prev_dn or num_completed % 20 == 0:
            print(
                f"  #{num_completed:3d} dn={dn_submitted:.2f}  "
                f"tick={tick_ms:5.1f}ms  decode={dec_str:>7s}  {mse_str}  "
                f"(playback {start / SAMPLE_RATE:.1f}s-{end / SAMPLE_RATE:.1f}s)"
            )
            prev_dn = dn_submitted
    else:
        torch.cuda.synchronize()
        tick_ms = (time.perf_counter() - iter_t0) * 1000
        timings.setdefault("tick", []).append(tick_ms)

    if stream.stream_node.active_slots == 0 and submit_idx >= len(denoise_per_tick):
        break

run_ms = (time.time() - run_start) * 1000
num_decoded = num_completed - num_skipped
print(
    f"\n[Run] {num_completed} generations in {run_ms:.0f}ms "
    f"({run_ms / max(num_completed, 1):.1f}ms avg incl. decode)"
)
print(
    f"  Decoded: {num_decoded}, Skipped: {num_skipped} "
    f"({100 * num_skipped / max(num_completed, 1):.0f}% skip rate)"
)
if mse_values:
    sorted_mse = sorted(mse_values)
    print(
        f"  MSE: min={sorted_mse[0]:.2e}  median={sorted_mse[len(sorted_mse) // 2]:.2e}  "
        f"max={sorted_mse[-1]:.2e}"
    )

# Concatenate all chunks
output_wav = torch.cat(output_chunks, dim=1)
total_duration = output_wav.shape[1] / SAMPLE_RATE

print(f"\n[Save] Output: {total_duration:.1f}s, {output_wav.shape}")
sf.write(str(OUTPUT_FILE), output_wav.numpy().T, SAMPLE_RATE, format="WAV")
print(f"  Saved: {OUTPUT_FILE}")

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
for label in [
    "model_load",
    "load_audio",
    "prepare_source",
    "text_encode",
    "stream_setup",
    "tick",
    "vae_decode",
]:
    vals = timings.get(label, [])
    if not vals:
        continue
    total = sum(vals)
    avg = total / len(vals)
    mn, mx = min(vals), max(vals)
    if len(vals) == 1:
        print(f"  {label:22s}  {total:8.1f}ms  (1 call)")
    else:
        print(
            f"  {label:22s}  {total:8.1f}ms total  "
            f"avg={avg:6.1f}ms  min={mn:6.1f}ms  max={mx:6.1f}ms  "
            f"({len(vals)} calls)"
        )

tick_vals = timings.get("tick", [])
decode_vals = timings.get("vae_decode", [])
if tick_vals:
    tick_total = sum(tick_vals)
    decode_total = sum(decode_vals) if decode_vals else 0
    avg_tick = tick_total / num_completed
    avg_decode_amortized = decode_total / num_completed
    print(f"\n  Per-generation (amortized over {num_completed} gens):")
    print(
        f"    tick={avg_tick:.1f}ms + decode={avg_decode_amortized:.1f}ms "
        f"({num_decoded} decoded, {num_skipped} skipped) "
        f"= {avg_tick + avg_decode_amortized:.1f}ms"
    )

print("\n" + "=" * 60)
