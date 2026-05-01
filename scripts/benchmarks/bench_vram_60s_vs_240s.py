"""Definitive VRAM comparison: 60s engine vs 240s engine, identical 60s input.

Each engine is measured in its own subprocess so CUDA contexts and TRT
allocations cannot bleed between runs. We report driver-level VRAM
(torch.cuda.mem_get_info) which captures TRT and PyTorch allocations alike.

Stages measured per engine:
  - baseline   : VRAM after CUDA init, before loading anything.
  - post_load  : VRAM after engine + execution context construction.
  - post_alloc : VRAM after the first inference (TRT lazy workspace bound).
  - peak       : Max VRAM observed across all warmup + timed iterations.
  - final      : VRAM after all iterations complete.

The "peak" line is the answer to the user's question. Differences between
60s and 240s with identical input come from TRT's optimization profile:
the 240s engine reserves workspace sized for its max shape (T=6000) even
when run at smaller shapes (T=1500), and may keep the larger reservation
resident regardless of the executed shape.

Usage::

    uv run python scripts/benchmarks/bench_vram_60s_vs_240s.py

Results from the most recent run live alongside this file in
``vram_60s_vs_240s_results.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent
TRT_ROOT = Path.home() / ".daydream-scope" / "models" / "demon" / "trt_engines"


def parent_main():
    """Driver that subprocesses each engine and prints a comparison table."""
    pairs = [
        ("decoder",     TRT_ROOT / "decoder_mixed_refit_b8_60s"  / "decoder_mixed_refit_b8_60s.engine",
                        TRT_ROOT / "decoder_mixed_refit_b8_240s" / "decoder_mixed_refit_b8_240s.engine"),
        ("vae_decode",  TRT_ROOT / "vae_decode_fp16_60s"         / "vae_decode_fp16_60s.engine",
                        TRT_ROOT / "vae_decode_fp16_240s"        / "vae_decode_fp16_240s.engine"),
        ("vae_encode",  TRT_ROOT / "vae_encode_fp16_60s"         / "vae_encode_fp16_60s.engine",
                        TRT_ROOT / "vae_encode_fp16_240s"        / "vae_encode_fp16_240s.engine"),
    ]

    rows = []
    for mode, eng_60, eng_240 in pairs:
        for label, eng_path in (("60s", eng_60), ("240s", eng_240)):
            if not eng_path.exists():
                print(f"[SKIP] {mode} {label}: engine not found at {eng_path}", flush=True)
                continue
            print(f"\n[RUN ] {mode:11s} {label:4s}  {eng_path.name}", flush=True)
            t0 = time.time()
            result = run_child(mode, eng_path)
            elapsed = time.time() - t0
            if result is None:
                print(f"[FAIL] {mode} {label}: subprocess returned no JSON", flush=True)
                continue
            result["mode"] = mode
            result["engine_label"] = label
            result["engine_path"] = str(eng_path)
            result["elapsed_s"] = round(elapsed, 1)
            rows.append(result)
            _print_one(result)

    print("\n" + "=" * 96)
    print("SUMMARY (all numbers in MB; identical 60-second input fed to each engine)")
    print("=" * 96)
    hdr = f"{'mode':<12}{'engine':<6}{'baseline':>10}{'post_load':>11}{'post_alloc':>12}{'peak':>10}{'final':>10}{'mean_ms':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['mode']:<12}{r['engine_label']:<6}"
              f"{r['baseline_mb']:>10.0f}{r['post_load_mb']:>11.0f}"
              f"{r['post_alloc_mb']:>12.0f}{r['peak_mb']:>10.0f}{r['final_mb']:>10.0f}"
              f"{r['mean_ms']:>10.1f}")

    print("\n" + "=" * 96)
    print("DELTA (240s minus 60s, same input)")
    print("=" * 96)
    by_mode: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], {})[r["engine_label"]] = r
    print(f"{'mode':<12}{'load_delta':>14}{'alloc_delta':>14}{'peak_delta':>14}")
    print("-" * 54)
    for mode, d in by_mode.items():
        if "60s" not in d or "240s" not in d:
            continue
        load_d = d["240s"]["post_load_mb"] - d["60s"]["post_load_mb"]
        alloc_d = d["240s"]["post_alloc_mb"] - d["60s"]["post_alloc_mb"]
        peak_d = d["240s"]["peak_mb"] - d["60s"]["peak_mb"]
        print(f"{mode:<12}{load_d:>+14.0f}{alloc_d:>+14.0f}{peak_d:>+14.0f}")

    out_path = RESULTS_DIR / "bench_vram_60s_vs_240s.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nDetailed JSON written to {out_path}")


def run_child(mode: str, engine_path: Path) -> dict | None:
    """Spawn a fresh python subprocess running --child for one engine."""
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--child", "--mode", mode, "--engine", str(engine_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(f"  stderr:\n{proc.stderr}", flush=True)
        return None
    last_json = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                last_json = json.loads(line)
            except json.JSONDecodeError:
                pass
    if last_json is None and proc.stderr:
        print(f"  stderr:\n{proc.stderr}", flush=True)
    return last_json


def _print_one(r: dict):
    print(f"       baseline={r['baseline_mb']:.0f}MB  post_load={r['post_load_mb']:.0f}MB  "
          f"post_alloc={r['post_alloc_mb']:.0f}MB  peak={r['peak_mb']:.0f}MB  "
          f"final={r['final_mb']:.0f}MB  mean={r['mean_ms']:.1f}ms")


# ---------------------------------------------------------------------------
# Child mode: runs in a fresh subprocess, owns its own CUDA context.
# ---------------------------------------------------------------------------

def child_main(mode: str, engine_path: Path):
    import torch
    torch.set_grad_enabled(False)

    device = torch.device("cuda")
    # Force CUDA init so the context cost is in baseline.
    torch.cuda.synchronize()
    _ = torch.empty(1, device=device)
    torch.cuda.synchronize()

    def vram_mb() -> float:
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return (total - free) / (1024 * 1024)

    baseline_mb = vram_mb()

    if mode == "decoder":
        keepalive, timings, peak_mb, post_load_mb, post_alloc_mb = bench_decoder(engine_path, device, vram_mb)
    elif mode == "vae_decode":
        keepalive, timings, peak_mb, post_load_mb, post_alloc_mb = bench_vae_decode(engine_path, device, vram_mb)
    elif mode == "vae_encode":
        keepalive, timings, peak_mb, post_load_mb, post_alloc_mb = bench_vae_encode(engine_path, device, vram_mb)
    else:
        raise ValueError(f"unknown mode {mode}")

    # Measure with engine + tensors still resident (keepalive prevents GC).
    final_mb = vram_mb()
    del keepalive

    result = {
        "baseline_mb": round(baseline_mb, 1),
        "post_load_mb": round(post_load_mb, 1),
        "post_alloc_mb": round(post_alloc_mb, 1),
        "peak_mb": round(peak_mb, 1),
        "final_mb": round(final_mb, 1),
        "mean_ms": round(sum(timings) / len(timings), 2) if timings else -1,
        "min_ms": round(min(timings), 2) if timings else -1,
        "max_ms": round(max(timings), 2) if timings else -1,
        "n_iters": len(timings),
    }
    print(json.dumps(result), flush=True)


def _load_engine(engine_path: Path):
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes
    engine = engine_from_bytes(bytes_from_path(str(engine_path)))
    ctx = engine.create_execution_context()
    return engine, ctx


def _trt_dtype_to_torch(trt_dtype):
    import tensorrt as trt
    import torch
    table = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "bfloat16"):
        table[trt.bfloat16] = torch.bfloat16
    return table.get(trt_dtype, torch.float32)


def _get_stream():
    from polygraphy import cuda as pg_cuda
    return pg_cuda.Stream()


# 60 seconds of latent at 25 fps = 1500 frames; CFG batch = 2.
DECODER_B = 2
DECODER_T = 1500
DECODER_ENC_L = 200


def bench_decoder(engine_path, device, vram_mb):
    import torch
    engine, ctx = _load_engine(engine_path)
    post_load_mb = vram_mb()

    in_dtypes = {n: _trt_dtype_to_torch(engine.get_tensor_dtype(n))
                 for n in ("hidden_states", "timestep", "encoder_hidden_states", "context_latents")}
    out_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype("velocity"))

    B, T, L = DECODER_B, DECODER_T, DECODER_ENC_L
    hs = torch.empty((B, T, 64), dtype=in_dtypes["hidden_states"], device=device).normal_()
    ts = torch.full((B,), 0.5, dtype=in_dtypes["timestep"], device=device)
    enc = torch.empty((B, L, 2048), dtype=in_dtypes["encoder_hidden_states"], device=device).normal_()
    ctx_lat = torch.empty((B, T, 128), dtype=in_dtypes["context_latents"], device=device).normal_()

    ctx.set_input_shape("hidden_states", (B, T, 64))
    ctx.set_input_shape("timestep", (B,))
    ctx.set_input_shape("encoder_hidden_states", (B, L, 2048))
    ctx.set_input_shape("context_latents", (B, T, 128))
    ctx.set_tensor_address("hidden_states", hs.data_ptr())
    ctx.set_tensor_address("timestep", ts.data_ptr())
    ctx.set_tensor_address("encoder_hidden_states", enc.data_ptr())
    ctx.set_tensor_address("context_latents", ctx_lat.data_ptr())

    out_shape = tuple(ctx.get_tensor_shape("velocity"))
    out = torch.empty(out_shape, dtype=out_dtype, device=device)
    ctx.set_tensor_address("velocity", out.data_ptr())

    stream = _get_stream()
    timings, peak_mb, post_alloc_mb = _run_iters(ctx, stream, vram_mb, post_load_mb)
    keepalive = (engine, ctx, hs, ts, enc, ctx_lat, out, stream)
    return keepalive, timings, peak_mb, post_load_mb, post_alloc_mb


def bench_vae_decode(engine_path, device, vram_mb):
    import torch
    engine, ctx = _load_engine(engine_path)
    post_load_mb = vram_mb()

    # latents [B, D, T], 60s = 1500 frames, B=1
    B, D, T = 1, 64, 1500
    lat = torch.empty((B, D, T), dtype=torch.float32, device=device).normal_()
    ctx.set_input_shape("latents", (B, D, T))
    ctx.set_tensor_address("latents", lat.data_ptr())

    out_shape = tuple(ctx.get_tensor_shape("audio"))
    out = torch.empty(out_shape, dtype=torch.float32, device=device)
    ctx.set_tensor_address("audio", out.data_ptr())

    stream = _get_stream()
    timings, peak_mb, post_alloc_mb = _run_iters(ctx, stream, vram_mb, post_load_mb)
    keepalive = (engine, ctx, lat, out, stream)
    return keepalive, timings, peak_mb, post_load_mb, post_alloc_mb


def bench_vae_encode(engine_path, device, vram_mb):
    import torch
    engine, ctx = _load_engine(engine_path)
    post_load_mb = vram_mb()

    # audio [B, 2, samples], 60s @ 48kHz
    B, C, S = 1, 2, 60 * 48000
    audio = torch.empty((B, C, S), dtype=torch.float32, device=device).normal_()
    ctx.set_input_shape("audio", (B, C, S))
    ctx.set_tensor_address("audio", audio.data_ptr())

    out_shape = tuple(ctx.get_tensor_shape("moments"))
    out = torch.empty(out_shape, dtype=torch.float32, device=device)
    ctx.set_tensor_address("moments", out.data_ptr())

    stream = _get_stream()
    timings, peak_mb, post_alloc_mb = _run_iters(ctx, stream, vram_mb, post_load_mb)
    keepalive = (engine, ctx, audio, out, stream)
    return keepalive, timings, peak_mb, post_load_mb, post_alloc_mb


def _run_iters(ctx, stream, vram_mb, post_load_mb):
    import torch
    peak_mb = post_load_mb
    post_alloc_mb = None
    timings: list[float] = []

    # Warmup (5 iterations)
    for i in range(5):
        if not ctx.execute_async_v3(stream.ptr):
            raise RuntimeError("TRT execute_async_v3 returned False")
        stream.synchronize()
        cur = vram_mb()
        peak_mb = max(peak_mb, cur)
        if i == 0:
            post_alloc_mb = cur

    # Timed (10 iterations)
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if not ctx.execute_async_v3(stream.ptr):
            raise RuntimeError("TRT execute_async_v3 returned False")
        stream.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)
        peak_mb = max(peak_mb, vram_mb())

    if post_alloc_mb is None:
        post_alloc_mb = post_load_mb
    return timings, peak_mb, post_alloc_mb


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true",
                   help="internal: run a single engine measurement and emit JSON")
    p.add_argument("--mode", choices=("decoder", "vae_decode", "vae_encode"))
    p.add_argument("--engine", type=Path)
    args = p.parse_args()

    if args.child:
        if args.mode is None or args.engine is None:
            sys.exit("--child requires --mode and --engine")
        child_main(args.mode, args.engine)
    else:
        parent_main()


if __name__ == "__main__":
    main()
