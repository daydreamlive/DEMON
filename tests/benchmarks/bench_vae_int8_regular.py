"""Compare VAE decode for the ACE-Step Oobleck VAE: PT eager vs TRT fp16 vs TRT int8.

Two tests, both run by default:

  Test A (synthetic):  generate one latent, decode it three ways.
                       Reports per-path latency, peak alloc, MSE vs PT.
                       Writes pt_eager.wav / trt_fp16.wav / trt_int8.wav.

  Test B (wav round-trip):
                       Load --wav, encode via PT VAE, decode three ways,
                       save round-tripped wavs alongside the original.
                       Reports MSE vs the PT round-trip (decode-only error)
                       AND vs the original wav (full round-trip error).

Usage:
    uv run python tests/benchmarks/bench_vae_int8_regular.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
torch.set_grad_enabled(False)

import soundfile as sf

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes.vae_nodes import _trt_vae_decode
from acestep.paths import trt_engines_dir


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _bench_one(label, fn, reference=None):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    mse = None
    if reference is not None:
        mse = ((out.float() - reference.float()) ** 2).mean().item()
    if mse is not None:
        print(f"  [{label:>16}] {dt_ms:7.1f} ms  peak_alloc={peak_gb:.3f} GB  mse={mse:.2e}")
    else:
        print(f"  [{label:>16}] {dt_ms:7.1f} ms  peak_alloc={peak_gb:.3f} GB")
    return {"label": label, "ms": dt_ms, "peak_alloc_gb": peak_gb, "mse": mse}, out


def _save_wav(audio_bcs: torch.Tensor, path: Path, sr: int = 48000):
    wav = audio_bcs.detach().squeeze(0).float().cpu().numpy().T  # [samples, channels]
    sf.write(str(path), wav, sr)
    print(f"  wav: {path}")


def _summarize(runs):
    ms = sorted(r["ms"] for r in runs)
    peak = max(r["peak_alloc_gb"] for r in runs)
    med = ms[len(ms) // 2]
    mses = [r["mse"] for r in runs if r["mse"] is not None]
    mse = sum(mses) / len(mses) if mses else None
    return med, peak, mse


def _print_table(rows, mse_label="MSE vs PT"):
    print("\n" + "=" * 78)
    print(f"{'Path':<18}{'Median ms':>12}{'Peak alloc GB':>18}{mse_label:>30}")
    print("-" * 78)
    for name, med, peak, mse in rows:
        s = f"{mse:.2e}" if mse is not None else "(reference)"
        print(f"{name:<18}{med:>12.1f}{peak:>18.3f}{s:>30}")
    print("=" * 78)


def _audio_metrics(out: torch.Tensor, ref: torch.Tensor):
    """RMSE + PSNR (assuming roughly [-1, 1] range)."""
    diff = (out.float() - ref.float()).flatten()
    rmse = diff.pow(2).mean().sqrt().item()
    peak = max(ref.float().abs().max().item(), 1e-9)
    psnr = 20.0 * math.log10(peak / max(rmse, 1e-12))
    return rmse, psnr


# -------------------------------------------------------------------
# Test A: synthetic latent
# -------------------------------------------------------------------

def run_synthetic(session, fp16_path, int8_path, args, out_dir, device, results=None):
    print("\n" + "#" * 78)
    print("# Test A: synthetic latent")
    print("#" * 78)
    handler = session.handler

    print("Generating a latent...")
    cond = session.encode_text(
        tags="electronic ambient, evolving pads, 120 bpm",
        lyrics="[instrumental]",
        duration=args.duration,
        instruction=TASK_INSTRUCTIONS["text2music"],
    )
    neg = session.null_conditioning(cond)
    latent = session.generate(
        conditioning=cond, negative=neg,
        seed=args.seed, duration=args.duration,
        steps=args.steps, shift=3.0, denoise=1.0,
    )
    lat_bdt = latent.tensor.transpose(1, 2).contiguous()
    chunk = lat_bdt[:, :, :args.chunk_frames].contiguous()
    print(f"  chunk: shape={tuple(chunk.shape)}  dtype={chunk.dtype}")

    def pt_decode():
        return handler.vae.decode(chunk.to(handler.vae.dtype)).sample

    print("\nWarmup (1 each)...")
    _ = pt_decode()
    _ = _trt_vae_decode(chunk, str(fp16_path), device)
    _ = _trt_vae_decode(chunk, str(int8_path), device)
    torch.cuda.synchronize()

    print(f"\n{args.runs} timed runs each:")
    pt_runs, fp16_runs, int8_runs = [], [], []
    pt_out = fp16_out = int8_out = None
    for _ in range(args.runs):
        r, o = _bench_one("PT eager bf16", pt_decode); pt_runs.append(r); pt_out = o
    reference = pt_out.detach().clone()
    for _ in range(args.runs):
        r, o = _bench_one("TRT fp16",
                          lambda: _trt_vae_decode(chunk, str(fp16_path), device),
                          reference=reference); fp16_runs.append(r); fp16_out = o
    for _ in range(args.runs):
        r, o = _bench_one("TRT int8",
                          lambda: _trt_vae_decode(chunk, str(int8_path), device),
                          reference=reference); int8_runs.append(r); int8_out = o

    print("\nSaving wavs (synthetic)...")
    _save_wav(pt_out, out_dir / "synth_pt_eager.wav")
    _save_wav(fp16_out, out_dir / "synth_trt_fp16.wav")
    _save_wav(int8_out, out_dir / "synth_trt_int8.wav")

    rows = []
    for runs, name in [(pt_runs, "PT eager bf16"), (fp16_runs, "TRT fp16"), (int8_runs, "TRT int8")]:
        med, peak, mse = _summarize(runs)
        rows.append((name, med, peak, mse))
    _print_table(rows, mse_label="MSE vs PT")

    rmse_fp16_int8, psnr_fp16_int8 = _audio_metrics(int8_out, fp16_out)
    print(f"  fp16 vs int8 (audio):  RMSE={rmse_fp16_int8:.6f}  PSNR={psnr_fp16_int8:.2f} dB")

    if results is not None:
        results["synthetic"] = {
            "pt": {"median_ms": rows[0][1], "peak_alloc_gb": rows[0][2]},
            "fp16": {"median_ms": rows[1][1], "peak_alloc_gb": rows[1][2], "mse_vs_pt": rows[1][3]},
            "int8": {"median_ms": rows[2][1], "peak_alloc_gb": rows[2][2], "mse_vs_pt": rows[2][3]},
            "fp16_vs_int8": {"rmse": rmse_fp16_int8, "psnr_db": psnr_fp16_int8},
        }


# -------------------------------------------------------------------
# Test B: wav round-trip
# -------------------------------------------------------------------

def run_wav_roundtrip(session, fp16_path, int8_path, args, out_dir, device, results=None):
    print("\n" + "#" * 78)
    print(f"# Test B: wav round-trip — {args.wav}")
    print("#" * 78)
    handler = session.handler

    wav_data, sr = sf.read(str(args.wav), dtype="float32", always_2d=True)
    if sr != 48000:
        raise RuntimeError(f"Expected 48kHz, got {sr}")
    if wav_data.shape[1] == 1:
        wav_data = np.repeat(wav_data, 2, axis=1)

    # Trim to chunk_frames * 1920 samples (1500 frames -> 60s -> 2_880_000 samples).
    samples_per_frame = 1920  # 48000 / 25
    n_samples = args.chunk_frames * samples_per_frame
    if wav_data.shape[0] < n_samples:
        raise RuntimeError(
            f"Wav has {wav_data.shape[0]} samples, need {n_samples} for {args.chunk_frames} latent frames"
        )
    wav_data = wav_data[:n_samples]

    audio = torch.from_numpy(wav_data.T).unsqueeze(0).to(device)  # [1, 2, S]
    print(f"  input audio: shape={tuple(audio.shape)}  ({n_samples / sr:.1f}s)")

    print("Encoding via PT VAE (mean of moments — deterministic)...")
    vae_input = audio.to(handler.vae.dtype)
    enc_out = handler.vae.encode(vae_input)
    latents = enc_out.latent_dist.mean  # [1, 64, T]
    print(f"  latents: shape={tuple(latents.shape)}  dtype={latents.dtype}")

    if latents.shape[2] != args.chunk_frames:
        # vae_encoder downsample factor may give a slightly different T;
        # use whatever it returns.
        print(f"  (encoder gave T={latents.shape[2]}, not {args.chunk_frames})")

    chunk = latents.contiguous()

    def pt_decode():
        return handler.vae.decode(chunk).sample

    print("\nWarmup (1 each)...")
    _ = pt_decode()
    _ = _trt_vae_decode(chunk, str(fp16_path), device)
    _ = _trt_vae_decode(chunk, str(int8_path), device)
    torch.cuda.synchronize()

    print(f"\n{args.runs} timed runs each:")
    pt_runs, fp16_runs, int8_runs = [], [], []
    pt_out = fp16_out = int8_out = None
    for _ in range(args.runs):
        r, o = _bench_one("PT eager bf16", pt_decode); pt_runs.append(r); pt_out = o
    reference = pt_out.detach().clone()
    for _ in range(args.runs):
        r, o = _bench_one("TRT fp16",
                          lambda: _trt_vae_decode(chunk, str(fp16_path), device),
                          reference=reference); fp16_runs.append(r); fp16_out = o
    for _ in range(args.runs):
        r, o = _bench_one("TRT int8",
                          lambda: _trt_vae_decode(chunk, str(int8_path), device),
                          reference=reference); int8_runs.append(r); int8_out = o

    # Trim/clip outputs to original length and save round-tripped wavs
    print("\nSaving wavs (round-trip)...")
    sf.write(str(out_dir / "wav_original.wav"),
             audio.squeeze(0).float().cpu().numpy().T, sr)
    print(f"  wav: {out_dir / 'wav_original.wav'}")
    _save_wav(pt_out, out_dir / "wav_pt_eager.wav")
    _save_wav(fp16_out, out_dir / "wav_trt_fp16.wav")
    _save_wav(int8_out, out_dir / "wav_trt_int8.wav")

    # Decode-only error (vs PT)
    rows_decode = []
    for runs, name in [(pt_runs, "PT eager bf16"), (fp16_runs, "TRT fp16"), (int8_runs, "TRT int8")]:
        med, peak, mse = _summarize(runs)
        rows_decode.append((name, med, peak, mse))
    _print_table(rows_decode, mse_label="MSE vs PT decode")

    # Full round-trip error (vs original audio): truncate decoded to source length
    src = audio.float()
    s_len = src.shape[-1]

    def _aligned(x: torch.Tensor) -> torch.Tensor:
        # crop or pad decoded to match source length
        if x.shape[-1] >= s_len:
            return x[..., :s_len]
        pad = s_len - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad))

    print("\nFull round-trip vs original audio:")
    print(f"{'Path':<18}{'RMSE':>14}{'PSNR (dB)':>14}")
    print("-" * 46)
    rt_metrics = {}
    for out, key, name in [(pt_out, "pt", "PT eager bf16"),
                           (fp16_out, "fp16", "TRT fp16"),
                           (int8_out, "int8", "TRT int8")]:
        out_aligned = _aligned(out.float())
        rmse, psnr = _audio_metrics(out_aligned, src)
        rt_metrics[key] = {"rmse": rmse, "psnr_db": psnr}
        print(f"{name:<18}{rmse:>14.6f}{psnr:>14.2f}")

    rmse_fp16_int8, psnr_fp16_int8 = _audio_metrics(int8_out, fp16_out)
    print(f"\n  fp16 vs int8 (audio): RMSE={rmse_fp16_int8:.6f}  PSNR={psnr_fp16_int8:.2f} dB")

    if results is not None:
        results["wav_roundtrip"] = {
            "wav": str(args.wav),
            "samples": int(s_len),
            "decode": {
                "pt": {"median_ms": rows_decode[0][1], "peak_alloc_gb": rows_decode[0][2]},
                "fp16": {"median_ms": rows_decode[1][1], "peak_alloc_gb": rows_decode[1][2], "mse_vs_pt": rows_decode[1][3]},
                "int8": {"median_ms": rows_decode[2][1], "peak_alloc_gb": rows_decode[2][2], "mse_vs_pt": rows_decode[2][3]},
            },
            "vs_original_audio": rt_metrics,
            "fp16_vs_int8": {"rmse": rmse_fp16_int8, "psnr_db": psnr_fp16_int8},
        }


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8-engine", default="vae_decode_int8_60s")
    ap.add_argument("--fp16-engine", default="vae_decode_fp16_60s")
    ap.add_argument("--chunk-frames", type=int, default=1500,
                    help="Latent frames decoded per pass (1500 = 60s, "
                         "matching fp16/int8 60s profiles)")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="Generate duration for synthetic test")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wav", default="tests/fixtures/inside_confusion.wav",
                    help="Wav fixture for round-trip test")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--skip-wav", action="store_true")
    ap.add_argument("--out-dir", default="bench_outputs/vae_int8")
    ap.add_argument("--json-out", default=None,
                    help="Write structured results to this JSON path")
    ap.add_argument("--label", default=None,
                    help="Label this run in the JSON (variant name)")
    args = ap.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    int8_path = trt_engines_dir() / args.int8_engine / f"{args.int8_engine}.engine"
    fp16_path = trt_engines_dir() / args.fp16_engine / f"{args.fp16_engine}.engine"
    if not int8_path.exists():
        raise FileNotFoundError(f"Missing INT8 engine: {int8_path}")
    if not fp16_path.exists():
        raise FileNotFoundError(f"Missing fp16 engine: {fp16_path}")

    print(f"INT8: {int8_path}")
    print(f"FP16: {fp16_path}")
    print(f"Chunk: [1, 64, {args.chunk_frames}]  ({args.chunk_frames / 25.0:.1f}s)")

    print("\nLoading session (eager VAE for PT reference)...")
    session = Session(decoder_backend="eager", vae_backend="eager", use_flash_attention=True)

    results = {
        "label": args.label or args.int8_engine,
        "int8_engine": str(int8_path),
        "fp16_engine": str(fp16_path),
        "engine_size_mb": int8_path.stat().st_size / 1e6,
        "chunk_frames": args.chunk_frames,
        "runs": args.runs,
    }

    if not args.skip_synthetic:
        run_synthetic(session, fp16_path, int8_path, args, out_dir, device, results)
    if not args.skip_wav:
        run_wav_roundtrip(session, fp16_path, int8_path, args, out_dir, device, results)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults JSON: {args.json_out}")


if __name__ == "__main__":
    main()
