"""Isolation benchmark for the 2B-turbo TRT engines.

Captures wall-clock latencies for the decoder + VAE encode + VAE decode
TRT engines, plus output tensor checksums so a sister run on a different
TRT version can be diffed numerically.

Outputs:
    timings_<tag>.json  -- per-config mean/min/max/p50/p95 ms + cuda peak
    outputs_<tag>.pt    -- {config -> output_tensor (cpu)} for quality diff
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

torch.set_grad_enabled(False)

from acestep.paths import trt_engine_path
from acestep.engine.trt.runtime import TRTDecoder
from acestep.nodes.vae_nodes import (
    _get_trt_vae,
    _trt_vae_decode,
    _trt_vae_encode,
    _get_trt_stream,
)


SEED = 1528
DEVICE = torch.device("cuda")


def percentile(values, p):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, p))


def stats(values):
    return {
        "count": len(values),
        "mean_ms": float(mean(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
    }


def reset_peak():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_gib():
    return {
        "alloc_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
    }


def bench_decoder(results, outputs):
    engine_path = trt_engine_path("decoder_mixed_refit_b8_60s")
    print(f"\n=== decoder ({engine_path.name}) ===")
    dec = TRTDecoder(str(engine_path), device=DEVICE)

    configs = [
        {"name": "decoder_B1_T750_E200", "B": 1, "T": 750, "L": 200},
        {"name": "decoder_B8_T750_E200", "B": 8, "T": 750, "L": 200},
        {"name": "decoder_B8_T1500_E200", "B": 8, "T": 1500, "L": 200},
    ]

    for cfg in configs:
        B, T, L = cfg["B"], cfg["T"], cfg["L"]
        torch.manual_seed(SEED)
        hs = torch.randn(B, T, 64, device=DEVICE, dtype=torch.float32)
        ts = torch.full((B,), 0.5, device=DEVICE, dtype=torch.float32)
        enc = torch.randn(B, L, 2048, device=DEVICE, dtype=torch.float32)
        ctx = torch.randn(B, T, 128, device=DEVICE, dtype=torch.float32)

        # Warm up
        for _ in range(5):
            out = dec(hs, ts, enc, ctx)
        torch.cuda.synchronize()

        reset_peak()
        times = []
        for _ in range(30):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = dec(hs, ts, enc, ctx)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

        s = stats(times)
        s.update(peak_gib())
        s["shape_in"] = list(hs.shape)
        s["shape_out"] = list(out.shape)
        s["dtype_out"] = str(out.dtype)
        results[cfg["name"]] = s
        # Capture full-precision copy for diffing across TRT versions.
        outputs[cfg["name"]] = out.detach().to(torch.float32).cpu().clone()
        print(
            f"  {cfg['name']}: mean={s['mean_ms']:.2f}ms p50={s['p50_ms']:.2f} p95={s['p95_ms']:.2f} "
            f"min={s['min_ms']:.2f} max={s['max_ms']:.2f}  peak_alloc={s['alloc_gib']:.2f}GiB"
        )


def bench_vae_decode(results, outputs):
    engine_path = trt_engine_path("vae_decode_fp16_60s")
    print(f"\n=== vae_decode ({engine_path.name}) ===")
    # Prime the cache.
    _get_trt_vae(str(engine_path), DEVICE)

    configs = [
        {"name": "vae_decode_T750", "T": 750},
        {"name": "vae_decode_T1500", "T": 1500},
    ]
    for cfg in configs:
        T = cfg["T"]
        torch.manual_seed(SEED)
        latents = torch.randn(1, 64, T, device=DEVICE, dtype=torch.float32)

        for _ in range(3):
            audio = _trt_vae_decode(latents, str(engine_path), DEVICE)
        torch.cuda.synchronize()

        reset_peak()
        times = []
        for _ in range(15):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            audio = _trt_vae_decode(latents, str(engine_path), DEVICE)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

        s = stats(times)
        s.update(peak_gib())
        s["shape_in"] = list(latents.shape)
        s["shape_out"] = list(audio.shape)
        results[cfg["name"]] = s
        outputs[cfg["name"]] = audio.detach().to(torch.float32).cpu().clone()
        print(
            f"  {cfg['name']}: mean={s['mean_ms']:.2f}ms p50={s['p50_ms']:.2f} p95={s['p95_ms']:.2f} "
            f"min={s['min_ms']:.2f} max={s['max_ms']:.2f}"
        )


def bench_vae_encode(results, outputs):
    engine_path = trt_engine_path("vae_encode_fp16_60s")
    print(f"\n=== vae_encode ({engine_path.name}) ===")
    _get_trt_vae(str(engine_path), DEVICE)

    SR = 48000
    # 60s and 30s clips (the encoder's profile permits 5-60s; pick mid + max).
    configs = [
        {"name": "vae_encode_30s", "samples": 30 * SR},
        {"name": "vae_encode_60s", "samples": 60 * SR},
    ]
    for cfg in configs:
        N = cfg["samples"]
        torch.manual_seed(SEED)
        # Sample-level noise sounds like static; that's fine -- we just need
        # deterministic dense input. Real audio waveforms vary widely.
        audio = (torch.randn(1, 2, N, device=DEVICE, dtype=torch.float32) * 0.2).clamp(-1, 1)

        # The encoder samples internally (mean + std * randn). Re-seed so
        # the captured "moments path" is deterministic across runs.
        for _ in range(3):
            torch.manual_seed(SEED + 7)
            latent = _trt_vae_encode(audio, str(engine_path), DEVICE)
        torch.cuda.synchronize()

        reset_peak()
        times = []
        for _ in range(15):
            torch.manual_seed(SEED + 7)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            latent = _trt_vae_encode(audio, str(engine_path), DEVICE)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

        s = stats(times)
        s.update(peak_gib())
        s["shape_in"] = list(audio.shape)
        s["shape_out"] = list(latent.shape)
        results[cfg["name"]] = s
        # Capture moments-derived latent. Numerical diff across TRT versions
        # is meaningful as long as we re-seed before each measured call.
        outputs[cfg["name"]] = latent.detach().to(torch.float32).cpu().clone()
        print(
            f"  {cfg['name']}: mean={s['mean_ms']:.2f}ms p50={s['p50_ms']:.2f} p95={s['p95_ms']:.2f} "
            f"min={s['min_ms']:.2f} max={s['max_ms']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Output filename tag (e.g. trt10_13)")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--skip", nargs="*", default=[], choices=["decoder", "vae_encode", "vae_decode"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import tensorrt as trt
    print(f"TensorRT: {trt.__version__}")
    print(f"Torch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Tag: {args.tag}")

    results = {
        "_meta": {
            "tensorrt": trt.__version__,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "tag": args.tag,
            "seed": SEED,
        }
    }
    outputs = {}

    if "decoder" not in args.skip:
        bench_decoder(results, outputs)
    if "vae_decode" not in args.skip:
        bench_vae_decode(results, outputs)
    if "vae_encode" not in args.skip:
        bench_vae_encode(results, outputs)

    json_path = out_dir / f"timings_{args.tag}.json"
    out_path = out_dir / f"outputs_{args.tag}.pt"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    torch.save(outputs, out_path)
    print(f"\nWrote {json_path}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
