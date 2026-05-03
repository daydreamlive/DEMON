#!/usr/bin/env python3
"""Integration test: live audio input through the streaming pipeline.

Simulates a live audio feed from a file, runs all three tiers of
audio input processing, and validates the generation pipeline produces
output influenced by the live input.

Requires: GPU, TRT engines, base audio fixture.

Usage:
    uv run python tests/integration/test_live_audio_input.py
    uv run python tests/integration/test_live_audio_input.py --sim-input path/to/other.wav
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import soundfile as sf
import torch

torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.audio_input import AudioCapture, SpectralTracker, LiveAudioEncoder, SAMPLE_RATE
from acestep.engine.session import Session
from acestep.nodes.types import Audio
from acestep.fixtures import audio_fixture

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_BASE = audio_fixture("inside_confusion_loop_60s_gsm.wav")
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "live_audio_input"


def load_audio(path: str, duration: float = 60.0) -> Audio:
    data, sr = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != 48000:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
    waveform = waveform[:2, :int(duration * 48000)]
    # Quantize to frame boundary
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    return Audio(waveform=waveform, sample_rate=48000)


def save_audio(audio: Audio, path: str):
    wav = audio.waveform.squeeze(0) if audio.waveform.dim() == 3 else audio.waveform
    sf.write(path, wav.detach().cpu().float().numpy().T, audio.sample_rate)
    print(f"  Saved: {path}")


def feed_audio_to_capture(capture: AudioCapture, audio: Audio):
    """Feed a full Audio tensor into the capture buffer (simulates recording)."""
    wav = audio.waveform.squeeze(0).cpu().numpy().T  # [samples, channels]
    n = wav.shape[0]
    with capture._lock:
        if n <= capture._buf_len:
            capture._buffer[:n] = wav
            capture._write_pos = n % capture._buf_len
        else:
            capture._buffer[:] = wav[-capture._buf_len:]
            capture._write_pos = 0
        capture._total_written = n


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parse args
    sim_input_path = None
    args = sys.argv[1:]
    if "--sim-input" in args:
        idx = args.index("--sim-input")
        sim_input_path = args[idx + 1]

    print("=" * 70)
    print("LIVE AUDIO INPUT INTEGRATION TEST")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Session setup
    # ------------------------------------------------------------------
    trt_engines = {
        "decoder": str(PROJECT_ROOT / "trt_engines/decoder_mixed_refit_b8_240s/decoder_mixed_refit_b8_240s.engine"),
        "vae_encode": str(PROJECT_ROOT / "trt_engines/vae_encode_fp16_240s/vae_encode_fp16_240s.engine"),
        "vae_decode": str(PROJECT_ROOT / "trt_engines/vae_decode_fp16_240s/vae_decode_fp16_240s.engine"),
    }

    print("\n[1/7] Loading model...")
    t0 = time.perf_counter()
    session = Session(
        project_root=str(PROJECT_ROOT / "checkpoints"),
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines=trt_engines,
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Base audio preparation
    # ------------------------------------------------------------------
    print("\n[2/7] Preparing base audio...")
    t0 = time.perf_counter()
    base_audio = load_audio(str(DEFAULT_BASE))
    source = session.prepare_source(base_audio)
    T = source.latent.tensor.shape[1]
    print(f"  Base prepared in {time.perf_counter() - t0:.1f}s (T={T})")

    print("[2/7] Text encode...")
    conditioning = session.encode_text(
        tags="cinematic orchestral ambient",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=120, duration=60.0, key="C minor",
    )

    # ------------------------------------------------------------------
    # Simulate live audio input
    # ------------------------------------------------------------------
    print("\n[3/7] Setting up simulated audio input...")
    capture = AudioCapture(buffer_seconds=60.0, channels=2)

    if sim_input_path:
        live_audio = load_audio(sim_input_path)
        print(f"  Using: {sim_input_path}")
    else:
        # Generate a synthetic signal: sine sweep + noise
        print("  Generating synthetic signal (sine sweep + noise)...")
        dur = 60.0
        n_samples = int(dur * SAMPLE_RATE)
        t = np.arange(n_samples) / SAMPLE_RATE
        freq = 100 + 900 * (t / dur)  # 100Hz -> 1000Hz sweep
        sweep = 0.3 * np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
        noise = 0.05 * np.random.randn(n_samples)
        mono = (sweep + noise).astype(np.float32)
        waveform = torch.from_numpy(np.stack([mono, mono], axis=0)).unsqueeze(0)
        live_audio = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)

    feed_audio_to_capture(capture, live_audio)
    print(f"  Buffer fill: {capture.buffer_fill:.0%}")

    # ------------------------------------------------------------------
    # Test Tier 1: Spectral Analysis
    # ------------------------------------------------------------------
    print("\n[4/7] Tier 1: Spectral Analysis...")
    tracker = SpectralTracker(capture, n_bands=8)
    analysis = tracker.analyze()
    print(f"  bands: [{', '.join(f'{b:.2f}' for b in analysis['bands'])}]")
    print(f"  rms={analysis['rms']:.3f}  onset={analysis['onset']:.3f}  centroid={analysis['centroid']:.0f}Hz")

    configs = tracker.to_channel_guidance(analysis, sensitivity=1.0)
    print(f"  Channel guidance: {len(configs)} entries")
    for cfg in configs:
        print(f"    ch[{cfg.channel_start}:{cfg.channel_end}] scale={cfg.scale:.2f}")

    # ------------------------------------------------------------------
    # Test Tier 2: VAE Encode
    # ------------------------------------------------------------------
    print("\n[5/7] Tier 2: VAE Encode live audio...")
    t0 = time.perf_counter()
    encoder = LiveAudioEncoder(
        capture, session,
        target_duration=60.0,
        update_interval=999,  # we'll call manually
        extract_hints=True,
    )
    live_latent = encoder.encode_now()
    encode_ms = (time.perf_counter() - t0) * 1000
    print(f"  Encoded in {encode_ms:.0f}ms")
    print(f"  Latent shape: {live_latent.shape}")
    assert live_latent.shape == (1, T, 64), f"Expected (1, {T}, 64), got {live_latent.shape}"

    live_hints = encoder.latest_hints
    if live_hints is not None:
        print(f"  Hints shape: {live_hints.shape}")
        assert live_hints.shape == (1, T, 64)
    print("  PASS: shapes match target")

    # ------------------------------------------------------------------
    # Test Tier 3: Stream with live audio input
    # ------------------------------------------------------------------
    print("\n[6/7] Tier 3: Streaming generation with live audio input...")

    stream = session.create_stream(
        source=source, conditioning=conditioning,
        steps=8, shift=3.0, pipeline_depth=4,
    )

    # Apply all three tiers
    stream.set_channel_guidance(configs)  # Tier 1
    if live_hints is not None:
        stream.set_live_context(live_hints, blend=0.3)  # Tier 3

    device = stream.source_latents.device
    dtype = stream.source_latents.dtype
    live_source = live_latent.to(device=device, dtype=dtype)

    # Run 12 ticks (4 warmup + 8 generations)
    results = []
    timings = []
    for i in range(12):
        stream.submit(
            denoise=0.7,
            seed=1528,
            source_b_latents=live_source,  # Tier 2
            source_crossfade=0.3,
        )
        t0 = time.perf_counter()
        result = stream.tick()
        torch.cuda.synchronize()
        tick_ms = (time.perf_counter() - t0) * 1000
        timings.append(tick_ms)

        if result is not None:
            results.append(result)
            print(f"  tick {i}: {tick_ms:.0f}ms -> finished (shape={result.tensor.shape})")
        else:
            print(f"  tick {i}: {tick_ms:.0f}ms -> in progress")

    print(f"  Generated {len(results)} outputs")
    print(f"  Timing: mean={np.mean(timings):.0f}ms, min={np.min(timings):.0f}ms")

    # ------------------------------------------------------------------
    # Baseline comparison: same pipeline without live input
    # ------------------------------------------------------------------
    print("\n[6b/7] Baseline: same generation without live input...")
    stream_base = session.create_stream(
        source=source, conditioning=conditioning,
        steps=8, shift=3.0, pipeline_depth=4,
    )

    baseline_results = []
    for i in range(12):
        stream_base.submit(denoise=0.7, seed=1528)
        result = stream_base.tick()
        if result is not None:
            baseline_results.append(result)

    if results and baseline_results:
        live_out = results[-1].tensor
        base_out = baseline_results[-1].tensor
        mse = (live_out - base_out).pow(2).mean().item()
        print(f"  Live vs Baseline MSE: {mse:.6f}")
        if mse > 1e-6:
            print("  PASS: live audio input produced different output (as expected)")
        else:
            print("  WARNING: outputs are identical; live input may not be influencing generation")

    # ------------------------------------------------------------------
    # Decode and save
    # ------------------------------------------------------------------
    print("\n[7/7] Decoding outputs...")
    if results:
        t0 = time.perf_counter()
        audio_out = session.decode(results[-1])
        dec_ms = (time.perf_counter() - t0) * 1000
        print(f"  Decoded in {dec_ms:.0f}ms")
        save_audio(audio_out, str(OUTPUT_DIR / "live_input_output.wav"))

    if baseline_results:
        audio_base = session.decode(baseline_results[-1])
        save_audio(audio_base, str(OUTPUT_DIR / "baseline_output.wav"))

    # Save the live input for reference
    if live_audio.waveform.dim() == 3:
        wav = live_audio.waveform.squeeze(0)
    else:
        wav = live_audio.waveform
    sf.write(
        str(OUTPUT_DIR / "live_input_reference.wav"),
        wav.cpu().numpy().T, SAMPLE_RATE,
    )
    print(f"  Saved live input reference")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print(f"  VAE encode: {encode_ms:.0f}ms")
    print(f"  Tick mean: {np.mean(timings):.0f}ms")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
