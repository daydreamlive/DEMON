"""Steady-state StreamPipeline tick throughput on the production path.

Drives ``Session.stream`` + ``StreamDenoise`` (the same path the
streaming server runs) at a fixed depth and measures wall time per
tick and finished latents per second. No audio decode — this isolates
the DiT tick loop, which dominates the stream. Run the same arguments
on two revisions to compare them.

Usage:
    .venv/Scripts/python.exe scripts/benchmarks/bench_stream_tick.py
    .venv/Scripts/python.exe scripts/benchmarks/bench_stream_tick.py --cfg
    .venv/Scripts/python.exe scripts/benchmarks/bench_stream_tick.py \
        --duration 30 --depth 4 --steps 8 --ticks 200 --no-dcw
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

torch.set_grad_enabled(False)

POOL = 1920 * 5  # vae_encode sample-count alignment quantum


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream tick throughput")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="Source/song duration in seconds (default: 30)")
    ap.add_argument("--depth", type=int, default=4,
                    help="pipeline_depth / ring slots (default: 4)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--denoise", type=float, default=1.0)
    ap.add_argument("--ticks", type=int, default=200,
                    help="Timed steady-state ticks (default: 200)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--cfg", action="store_true",
                    help="Enable full CFG (null negative + guidance curve)")
    ap.add_argument("--guidance-scale", type=float, default=2.5)
    ap.add_argument("--dcw", dest="dcw", action="store_true", default=True)
    ap.add_argument("--no-dcw", dest="dcw", action="store_false",
                    help="Disable DCW (enables the batched fast-path flush)")
    args = ap.parse_args()

    from acestep.constants import TASK_INSTRUCTIONS
    from acestep.engine.session import Session
    from acestep.nodes.types import Audio, Curve
    from acestep.paths import default_trt_engines

    torch.manual_seed(0)
    samples = (int(args.duration * 48000) // POOL) * POOL
    audio = Audio(waveform=torch.randn(2, samples) * 0.05, sample_rate=48000)

    sess = Session(
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines=default_trt_engines(),
    )
    src = sess.prepare_source(audio)
    cond = sess.encode_text(
        tags="electronic ambient, synthesizer, driving beat",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=src.latent,
        bpm=120,
        duration=args.duration,
    )
    handle = sess.stream(
        source=src,
        conditioning=cond,
        steps=args.steps,
        shift=3.0,
        pipeline_depth=args.depth,
        dcw_enabled=args.dcw,
    )

    tick_kwargs: dict = {"denoise": args.denoise}
    if args.cfg:
        T = src.latent.tensor.shape[1]
        tick_kwargs["negative"] = sess.null_conditioning(cond)
        tick_kwargs["guidance_curve"] = Curve(
            tensor=torch.full((T,), args.guidance_scale, dtype=torch.bfloat16),
        )

    label = (
        f"depth={args.depth} steps={args.steps} T={src.latent.tensor.shape[1]} "
        f"cfg={'on' if args.cfg else 'off'} dcw={'on' if args.dcw else 'off'}"
    )
    print(f"[bench_stream_tick] {label}")

    finished = 0
    for i in range(args.warmup):
        if handle.tick(seed=1000 + i, **tick_kwargs) is not None:
            finished += 1
    torch.cuda.synchronize()
    print(f"warmup: {args.warmup} ticks, {finished} finished")

    finished = 0
    t0 = time.perf_counter()
    for i in range(args.ticks):
        if handle.tick(seed=5000 + i, **tick_kwargs) is not None:
            finished += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    ms_tick = wall / args.ticks * 1000
    lat_s = finished / wall
    print(
        f"ticks={args.ticks} finished={finished} wall={wall:.2f}s  "
        f"ms/tick={ms_tick:.2f}  latents/s={lat_s:.3f}  "
        f"audio_x_realtime={lat_s * args.duration:.1f}"
    )

    handle.close()
    sess.close()


if __name__ == "__main__":
    main()
