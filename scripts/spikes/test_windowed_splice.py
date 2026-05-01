"""Verify windowed decode + splice produces identical output to full decode.

Takes a single latent, decodes it both ways:
  1. Full decode (reference): extract playback slices directly
  2. Windowed decode: decode a window around each playback position,
     use start_sample to align, extract the same slice

Compares the slices sample-by-sample. Any mismatch means the splice
logic has an alignment bug.

Usage:
    python test_windowed_splice.py                  # random latent
    python test_windowed_splice.py --latent foo.pt  # real music latent [1, T, 64]
"""
if __name__ != "__main__":
    import sys; sys.exit(0)

import os, sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
torch.set_grad_enabled(False)

from acestep.engine.session import Session
from acestep.nodes.types import Latent

PROJECT_ROOT = Path(__file__).parent.parent

SAMPLE_RATE = 48000
FRAMES_PER_SEC = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FRAMES_PER_SEC  # 1920

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
latent_path = None
if "--latent" in sys.argv:
    idx = sys.argv.index("--latent")
    latent_path = sys.argv[idx + 1]

# ---------------------------------------------------------------------------
# Session (VAE TRT only, no DiT needed)
# ---------------------------------------------------------------------------
VAE_ENCODE_ENGINE = PROJECT_ROOT / "trt_engines" / "vae_encode_fp16_240s" / "vae_encode_fp16_240s.engine"
VAE_DECODE_ENGINE = PROJECT_ROOT / "trt_engines" / "vae_decode_fp16_240s" / "vae_decode_fp16_240s.engine"

print("Loading session (VAE TRT only)...")
session = Session(
    project_root=str(PROJECT_ROOT / "checkpoints"),
    vae_backend="tensorrt",
    trt_engines={
        "vae_encode": str(VAE_ENCODE_ENGINE),
        "vae_decode": str(VAE_DECODE_ENGINE),
    },
    vae_window=15.0,
    vae_overlap=0.5,
)

# ---------------------------------------------------------------------------
# Latent
# ---------------------------------------------------------------------------
if latent_path:
    print(f"Loading latent from {latent_path}")
    tensor = torch.load(latent_path, map_location="cuda", weights_only=True)
    if tensor.dim() == 3 and tensor.shape[1] == 64:
        tensor = tensor.transpose(1, 2)  # [1, 64, T] -> [1, T, 64]
    latent = Latent(tensor=tensor.to(device="cuda"))
    T = tensor.shape[1]
    print(f"  Loaded: [1, {T}, 64] ({T / FRAMES_PER_SEC:.1f}s)")
else:
    T = 1500  # 60s
    torch.manual_seed(42)
    tensor = torch.randn(1, T, 64, device="cuda")
    latent = Latent(tensor=tensor)
    print(f"Using random latent: [1, {T}, 64] ({T / FRAMES_PER_SEC:.1f}s)")

total_samples = T * SAMPLES_PER_FRAME

# ---------------------------------------------------------------------------
# 1. Full decode (reference)
# ---------------------------------------------------------------------------
print("\nFull decode (reference)...")
session._vae_window = 0.0  # force full decode
ref_audio = session.decode(latent)
ref_wav = ref_audio.waveform.detach().cpu().float().squeeze(0)  # [C, samples]
print(f"  Reference shape: {ref_wav.shape}")

# ---------------------------------------------------------------------------
# 2. Windowed decode at advancing playback positions
# ---------------------------------------------------------------------------
session._vae_window = 15.0
session._vae_overlap = 0.5

slice_duration = 0.3  # 300ms slices, same as test_stream_cover
slice_samples = int(slice_duration * SAMPLE_RATE)

# Walk through playback positions spread across the full duration,
# including positions that don't land on frame boundaries
test_starts_sec = [0.0, 0.017, 2.5, 5.0, 10.123, 20.0, 30.777, 44.5, 55.0, 58.0]
# Filter to positions where the slice fits within the audio
test_starts_sec = [t for t in test_starts_sec if int(t * SAMPLE_RATE) + slice_samples <= ref_wav.shape[1]]

print(f"\nTesting {len(test_starts_sec)} playback positions (slice={slice_duration}s)...")
print(f"{'':>4}  {'t_start':>8}  {'start_smp':>10}  {'offset':>7}  {'max_err':>10}  {'mean_err':>12}  {'pass':>5}")
print("-" * 72)

all_pass = True

for t_start in test_starts_sec:
    start_sample = int(t_start * SAMPLE_RATE)
    end_sample = start_sample + slice_samples

    # --- Reference: slice directly from full decode ---
    ref_slice = ref_wav[:, start_sample:end_sample]

    # --- Windowed path: same logic as test_stream_cover / realtime_motion ---
    audio_out = session.decode(latent, t_start=t_start)
    win_wav = audio_out.waveform.detach().cpu().float().squeeze(0)

    # Align using quantized start_sample (integer, no float precision issues)
    win_start_sample = audio_out.start_sample
    local_start = start_sample - win_start_sample
    local_end = local_start + slice_samples

    if local_end <= win_wav.shape[1]:
        win_slice = win_wav[:, local_start:local_end]
    else:
        win_slice = torch.zeros_like(ref_slice)
        avail = win_wav.shape[1] - local_start
        if avail > 0:
            win_slice[:, :avail] = win_wav[:, local_start:local_start + avail]

    # --- Compare ---
    diff = (ref_slice - win_slice).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    passed = max_err < 1e-4
    if not passed:
        all_pass = False

    offset_ms = (start_sample - win_start_sample) / SAMPLE_RATE * 1000
    tag = "OK" if passed else "FAIL"
    print(f"  {t_start:8.3f}s  {win_start_sample:>10d}  "
          f"{offset_ms:6.1f}ms  {max_err:10.6f}  {mean_err:12.8f}  {tag:>5}")

    if not passed:
        abs_diff = diff[0].numpy()
        peak_idx = int(np.argmax(abs_diff))
        win_total = win_wav.shape[1]
        print(f"         peak at slice[{peak_idx}], "
              f"window={win_total} samples, local=[{local_start}:{local_end}]")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
if all_pass:
    print("ALL PASSED: windowed splice output matches full decode at every position.")
else:
    print("FAILURES DETECTED: windowed splice does not match full decode.")
    sys.exit(1)
