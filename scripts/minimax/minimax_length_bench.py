"""How the MiniMax-Music3 renderer scales with song length.

The family's default song is 7.999 s (200 AR frames, 689 latent
frames) because that is upstream's inference chunk size -- NOT a
trained span; nothing upstream states one, and the DiT config carries
no length bound. Sessions may run any length the autoregressive stage
produces. So "generations per second" here counts one whole song of
whatever that length is, and comparing it against a family whose unit
is a minute needs either a normalization or an actual measurement at
that length.

This script is the measurement. It times one DiT forward across latent
lengths, reports the throughput three ways -- forwards/s, 8 s
generations/s, and 60 s-equivalent generations/s -- and says where the
attention term stops being free.

Two caveats it cannot measure and will not pretend to:

* A number here says the arithmetic is affordable, NOT that the audio
  is good. Upstream renders everything in 689-frame windows, so longer
  single-pass spans are simply untested rather than known-bad -- one
  1240-frame render measured clean, which is a data point and not a
  result.
* TensorRT engines are built per length profile. The shipped engine is
  ``l2_689_689``; anything longer runs eager here.

    .venv/Scripts/python.exe scripts/minimax/minimax_length_bench.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# A sibling ACE-Step checkout shadows `acestep` otherwise.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from acestep.engine.minimax_adapter import (  # noqa: E402
    MINIMAX_COND_DIM,
    MINIMAX_LATENT_CHANNELS,
    MINIMAX_SAMPLE_RATE,
    MINIMAX_UPSAMPLE,
)
from acestep.engine.minimax_context import get_minimax_context  # noqa: E402

LATENT_RATE_HZ = float(MINIMAX_SAMPLE_RATE) / float(MINIMAX_UPSAMPLE)
#: Upstream's inference window. Not a trained span -- see the module
#: docstring and docs/MINIMAX.md section 3b.
UPSTREAM_WINDOW_FRAMES = 689


def frames_for(seconds: float) -> int:
    return int(seconds * LATENT_RATE_HZ)


@torch.no_grad()
def time_forward(dit, frames: int, batch: int, dtype, device,
                 *, warmup=2, iters=6) -> float:
    """Median ms for one forward at ``[batch, C, frames]``."""
    x = torch.randn(batch, MINIMAX_LATENT_CHANNELS, frames,
                    device=device, dtype=dtype)
    t = torch.full((batch,), 0.5, device=device, dtype=dtype)
    cond = torch.zeros(batch, frames, MINIMAX_COND_DIM,
                       device=device, dtype=dtype)
    for _ in range(warmup):
        dit(x, t, cond)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        dit(x, t, cond)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", default="8,16,30,60",
                    help="song lengths to time")
    ap.add_argument("--batches", default="1,4",
                    help="batch sizes (the ring's depth)")
    ap.add_argument("--steps", type=int, default=16,
                    help="sampler steps, for the generations/s column")
    ap.add_argument("--guidance", action="store_true", default=True,
                    help="count two forwards per step (CFG on)")
    ap.add_argument("--no-guidance", dest="guidance", action="store_false")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float32"))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    ctx = get_minimax_context(dtype=dtype, ar_policy="absent")
    per_step = 2 if args.guidance else 1
    fwd_per_gen = args.steps * per_step

    secs = [float(v) for v in args.seconds.split(",") if v]
    batches = [int(v) for v in args.batches.split(",") if v]

    print(f"dtype={args.dtype}  steps={args.steps}  "
          f"guidance={'on' if args.guidance else 'off'}  "
          f"-> {fwd_per_gen} forwards per generation")
    print(f"trained length is {TRAINED_FRAMES} frames "
          f"({TRAINED_FRAMES / LATENT_RATE_HZ:.2f} s); longer is "
          f"out of distribution\n")
    print(f"{'song_s':>7s}{'frames':>8s}{'B':>3s}{'ms/fwd':>9s}"
          f"{'ms/sample':>11s}{'vs_linear':>10s}{'gen/s':>8s}"
          f"{'60s_gen/s':>11s}{'xRT':>8s}")

    rows = []
    base = None
    for s in secs:
        frames = frames_for(s)
        for b in batches:
            try:
                dit = ctx.make_dit(latent_frames=frames)
                ms = time_forward(dit, frames, b, dtype, ctx.device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{s:7.0f}{frames:8d}{b:3d}       OOM")
                continue
            per_sample = ms / b
            if base is None:
                base = per_sample / frames
            # 1.0 means the forward grew exactly in proportion to length;
            # above that is the attention term showing up.
            vs_linear = (per_sample / frames) / base
            # One generation is `fwd_per_gen` forwards. At batch B the
            # ring has B of them in flight, so throughput is B/gen_time.
            gen_s = b / (ms * fwd_per_gen / 1000.0)
            rows.append(dict(song_s=s, frames=frames, batch=b, ms=ms,
                             ms_per_sample=per_sample, vs_linear=vs_linear,
                             gens_per_s=gen_s,
                             gens60_per_s=gen_s * s / 60.0,
                             realtime=gen_s * s))
            print(f"{s:7.0f}{frames:8d}{b:3d}{ms:9.1f}{per_sample:11.1f}"
                  f"{vs_linear:10.2f}{gen_s:8.2f}"
                  f"{gen_s * s / 60.0:11.3f}{gen_s * s:8.1f}x")

    print("\n  gen/s counts whole songs of that length. 60s_gen/s "
          "normalizes to a\n  one-minute span so it is comparable across "
          "families with different\n  fixed durations. xRT is seconds of "
          "audio produced per second of GPU.")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
