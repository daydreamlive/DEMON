#!/usr/bin/env python3
"""Benchmark: windowed VAE encode vs full-buffer encode.

Measures the GPU time for encoding different window sizes to validate
the performance advantage of windowed encoding.

Usage:
    uv run python scripts/benchmarks/bench_windowed_encode.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
torch.set_grad_enabled(False)

from acestep.engine.audio_input import AudioCapture, LiveAudioEncoder, SAMPLE_RATE, SAMPLES_PER_FRAME
from acestep.engine.session import Session
from acestep.nodes.types import Audio

PROJECT_ROOT = Path(__file__).parent.parent.parent


def bench_raw_vae_encode(session, duration_seconds: float, n_runs: int = 10) -> dict:
    """Benchmark raw VAE encode for a given audio duration."""
    n_samples = int(duration_seconds * SAMPLE_RATE)
    n_samples = (n_samples // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    waveform = torch.randn(1, 2, n_samples)
    audio = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)

    # Warmup
    session.encode_audio(audio)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        session.encode_audio(audio)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "duration_s": duration_seconds,
        "n_samples": n_samples,
        "n_frames": n_samples // SAMPLES_PER_FRAME,
        "mean_ms": np.mean(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
    }


def bench_windowed_encoder(session, new_audio_seconds: float, n_runs: int = 10) -> dict:
    """Benchmark the full windowed encode cycle (including margin + EMA splice)."""
    capture = AudioCapture(buffer_seconds=60.0)

    # Bootstrap: fill 60 seconds
    data = np.random.randn(60 * SAMPLE_RATE, 2).astype(np.float32) * 0.1
    with capture._lock:
        capture._buffer[:] = data
        capture._write_pos = 0
        capture._total_written = 60 * SAMPLE_RATE

    encoder = LiveAudioEncoder(
        capture, session,
        target_duration=60.0,
        overlap_seconds=1.0,
        ema_alpha=0.8,
        extract_hints=False,
    )

    # Bootstrap encode
    encoder._encode_full()
    torch.cuda.synchronize()

    # Benchmark windowed encodes
    new_samples = int(new_audio_seconds * SAMPLE_RATE)
    times = []
    for _ in range(n_runs):
        # Simulate new audio arrival
        with capture._lock:
            capture._total_written += new_samples

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        encoder._encode_windowed()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "new_audio_s": new_audio_seconds,
        "window_frames": encoder._last_window_frames,
        "mean_ms": np.mean(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
    }


def main():
    trt_engines = {
        "decoder": str(PROJECT_ROOT / "trt_engines/decoder_mixed_refit_b8_240s/decoder_mixed_refit_b8_240s.engine"),
        "vae_encode": str(PROJECT_ROOT / "trt_engines/vae_encode_fp16_240s/vae_encode_fp16_240s.engine"),
        "vae_decode": str(PROJECT_ROOT / "trt_engines/vae_decode_fp16_240s/vae_decode_fp16_240s.engine"),
    }

    print("Loading model...")
    session = Session(
        project_root=str(PROJECT_ROOT / "checkpoints"),
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines=trt_engines,
    )
    print("Ready.\n")

    print("=" * 70)
    print("RAW VAE ENCODE BENCHMARK (different durations)")
    print("=" * 70)
    print(f"{'Duration':>10} {'Frames':>8} {'Mean':>8} {'Min':>8} {'P50':>8} {'P95':>8}")
    print("-" * 70)

    for dur in [5.0, 7.0, 10.0, 15.0, 30.0, 60.0]:
        r = bench_raw_vae_encode(session, dur, n_runs=10)
        print(f"{r['duration_s']:>8.0f}s {r['n_frames']:>8} "
              f"{r['mean_ms']:>7.1f}ms {r['min_ms']:>7.1f}ms "
              f"{r['p50_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms")

    print()
    print("=" * 70)
    print("WINDOWED ENCODE BENCHMARK (different new audio amounts)")
    print("=" * 70)
    print(f"{'New Audio':>10} {'Core Fr':>8} {'Mean':>8} {'Min':>8} {'P50':>8} {'P95':>8}")
    print("-" * 70)

    for new_s in [1.0, 2.0, 3.0, 5.0, 10.0]:
        r = bench_windowed_encoder(session, new_s, n_runs=10)
        print(f"{r['new_audio_s']:>8.0f}s {r['window_frames']:>8} "
              f"{r['mean_ms']:>7.1f}ms {r['min_ms']:>7.1f}ms "
              f"{r['p50_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms")

    print()
    print("Comparison: windowed 2s new audio vs full 60s encode")
    full = bench_raw_vae_encode(session, 60.0, n_runs=10)
    windowed = bench_windowed_encoder(session, 2.0, n_runs=10)
    speedup = full["mean_ms"] / windowed["mean_ms"]
    print(f"  Full 60s:     {full['mean_ms']:.1f}ms")
    print(f"  Windowed 2s:  {windowed['mean_ms']:.1f}ms")
    print(f"  Speedup:      {speedup:.1f}x")


if __name__ == "__main__":
    main()
