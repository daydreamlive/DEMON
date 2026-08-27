"""What the MiniMax-Music3 DAV decoder actually costs, and how much
context a decoded sample depends on.

Two questions a streaming backend has to answer before it can claim a
decode strategy, and neither can be answered from a config file:

1. **Cost vs length.** If decoding the whole song is flat and cheap,
   a backend can cache one full decode and index into it. If it grows
   with length, the backend must decode only the window it is about to
   play, and the cost of the whole-song shortcut is paid on every
   fresh latent.

2. **Receptive field.** A windowed decode is only legitimate if a
   sample's value does not depend on latent frames far outside the
   window. Measured directly: decode a long latent, decode a slice of
   it, and find how many samples in from the slice edge the two agree.
   That width is the guard margin the backend owes the decoder --
   without it every window boundary is a discontinuity, and with too
   much of it every render pays for audio it throws away.

    .venv/Scripts/python.exe scripts/minimax/minimax_decode_profile.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# A sibling ACE-Step checkout shadows `acestep` otherwise.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from acestep.engine.minimax_adapter import (  # noqa: E402
    MINIMAX_LATENT_CHANNELS,
    MINIMAX_SAMPLE_RATE,
    MINIMAX_UPSAMPLE,
)
from acestep.engine.minimax_context import get_minimax_context  # noqa: E402

LATENT_RATE_HZ = float(MINIMAX_SAMPLE_RATE) / float(MINIMAX_UPSAMPLE)


@torch.no_grad()
def time_decode(codec, latent, *, warmup=2, iters=6) -> float:
    for _ in range(warmup):
        codec.decode_full(latent)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        codec.decode_full(latent)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def cost_vs_length(codec, dev, dtype, lengths) -> list:
    print(f"{'frames':>8s}{'audio_s':>9s}{'ms':>9s}{'ms/s_audio':>12s}"
          f"{'xRT':>10s}{'peak_GB':>9s}")
    rows = []
    for L in lengths:
        torch.cuda.reset_peak_memory_stats()
        lat = torch.randn(1, MINIMAX_LATENT_CHANNELS, L, device=dev, dtype=dtype)
        try:
            ms = time_decode(codec, lat)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{L:8d}       OOM")
            continue
        secs = L / LATENT_RATE_HZ
        gb = torch.cuda.max_memory_allocated() / 1e9
        rows.append(dict(frames=L, audio_s=secs, ms=ms,
                         ms_per_audio_s=ms / secs,
                         realtime=secs / (ms / 1000.0), peak_gb=gb))
        print(f"{L:8d}{secs:9.2f}{ms:9.2f}{ms / secs:12.3f}"
              f"{secs / (ms / 1000.0):10.0f}x{gb:9.2f}")
    return rows


@torch.no_grad()
def receptive_field(codec, dev, *, total=689, slice_len=192) -> dict:
    """Decode error as a function of distance from a window edge.

    Decode a long latent, decode a contiguous slice of it, and profile
    how the disagreement decays as you move inward from the slice's
    edge. That decay length IS the receptive field, and it is the guard
    margin a windowed decode owes the decoder.

    Deliberately fp32: at bf16 the interior noise floor is ~1e-2, which
    swamps the very signal being measured and makes a threshold-crossing
    estimate bounce around meaninglessly. A profile also beats a
    threshold outright -- it shows whether the error decays sharply
    (a genuine finite receptive field) or drifts (a global operation
    somewhere, which would make windowing unsound at any margin).
    """
    lat = torch.randn(1, MINIMAX_LATENT_CHANNELS, total, device=dev,
                      dtype=torch.float32)
    full = codec.decode_full(lat).float()

    start = total // 2 - slice_len // 2
    part = codec.decode_full(lat[:, :, start:start + slice_len]).float()

    up = MINIMAX_UPSAMPLE
    ref = full[:, start * up:(start + slice_len) * up]
    n = min(ref.shape[1], part.shape[1])
    ref, part = ref[:, :n], part[:, :n]

    scale = ref.abs().mean().clamp_min(1e-9)
    err = ((part - ref).abs().mean(dim=0) / scale).cpu().numpy()

    print(f"  slice {slice_len} frames ({slice_len / LATENT_RATE_HZ:.2f} s), "
          f"error by distance in from the leading edge:")
    prof = {}
    for guard in (0, 1, 2, 4, 8, 16, 32, 64):
        lo = guard * up
        hi = n - guard * up
        if hi <= lo:
            continue
        v = float(err[lo:hi].mean())
        peak = float(err[lo:hi].max())
        prof[guard] = dict(mean=v, peak=peak)
        print(f"    guard {guard:3d} latent frames "
              f"({guard / LATENT_RATE_HZ * 1000:7.1f} ms): "
              f"mean rel err {v:.3e}   peak {peak:.3e}")
    return dict(slice_frames=slice_len, profile=prof)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="8,16,32,64,128,256,689,1378,2583,5167",
                    help="latent frame counts to time")
    ap.add_argument("--slices", default="192,689",
                    help="slice lengths for the receptive-field probe")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float32"))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    ctx = get_minimax_context(dtype=dtype, ar_policy="absent")
    codec = ctx.make_codec()

    print(f"DAV decode cost vs length (dtype={args.dtype}, "
          f"{LATENT_RATE_HZ:.3f} latent frames/s, {MINIMAX_UPSAMPLE}x upsample)\n")
    lengths = [int(v) for v in args.lengths.split(",") if v]
    cost = cost_vs_length(codec, ctx.device, dtype, lengths)

    print("\nreceptive field: windowed decode vs the same span of a full decode\n")
    fp32_codec = get_minimax_context(
        dtype=torch.float32, ar_policy="absent",
    ).make_codec()
    rf = [receptive_field(fp32_codec, ctx.device, slice_len=int(v))
          for v in args.slices.split(",") if v]

    print("\n  A flat ms/s_audio means whole-song decode is affordable and a "
          "backend\n  may cache it. A rising one means the render path must "
          "decode only the\n  window it is about to play, with the measured "
          "guard margin on each side.")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"cost": cost, "receptive_field": rf}, indent=2))
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
