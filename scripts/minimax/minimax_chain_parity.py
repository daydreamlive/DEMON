"""Chain-level parity gate for the MiniMax-Music3 backend.

Proves the whole DEMON-side chain against ground truth in one shot: the
reimplemented DiT, the ``[B,T,C]`` transpose boundary, and — the part
most likely to be silently wrong — the time/sign conversion between
MiniMax's flow-matching convention and DEMON's.

MiniMax runs ``t`` from 0 (noise) to 1 (data) and steps Euler forward.
DEMON runs ``s`` from 1 down to 0 with ``x0 = xt - v*s``. Substituting
``s = 1-t`` makes the interpolants identical, so the adapter converts
only the two scalars: ``t = 1-s`` and ``v_demon = -v_minimax``. Get
either backwards and the model denoises away from the data manifold —
which sounds like plausible audio, not like an error, so it needs a
numeric gate rather than a listen.

The fixture carries both endpoints of a known-good reference run
(``initial_noise`` and ``final_latent``, seed 7, 30 steps, CFG 1.7),
captured from the diffusers pipeline. Driving the adapter from one
endpoint must land on the other.

Companion to ``minimax_dit_parity.py``, which compares the DiT module
against diffusers layer by layer; this one is end-to-end and needs no
diffusers install.

    .venv/Scripts/python.exe scripts/minimax/minimax_chain_parity.py \
        --fixture <minimax_music3_8s_seed7.safetensors>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# A sibling ACE-Step checkout shadows `acestep` otherwise.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from acestep.engine.minimax_adapter import MiniMaxAdapter  # noqa: E402
from acestep.engine.minimax_context import get_minimax_context  # noqa: E402

# bf16 accumulation over 30 steps, with an op order that differs from
# the reference by a transpose, lands around 1.7e-2 relative RMS. The
# cosine bar is what actually discriminates: a sign or direction error
# produces output uncorrelated with the reference (measured r ~ 0.08),
# not a near miss.
COS_BAR = 0.99


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=1.7)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    args = ap.parse_args()

    data = load_file(args.fixture)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    ctx = get_minimax_context(dtype=dtype, ar_policy="absent")
    dev = ctx.device

    cond = data["encoder_hidden_states"].unsqueeze(0).to(dev, ctx.dtype)
    noise = data["initial_noise"].unsqueeze(0).to(dev, ctx.dtype)
    ref = data["final_latent"].unsqueeze(0).to(dev, torch.float32)
    zero = torch.zeros_like(cond)
    frames = cond.shape[1]

    adapter = MiniMaxAdapter(
        ctx.make_dit(latent_frames=frames),
        schedule_builder=None,   # this script drives the loop itself
        device=dev,
        dtype=ctx.dtype,
    )

    # DEMON convention: descending to 0. The reference walks t = 1-s up.
    schedule = torch.linspace(1.0, 0.0, args.steps + 1)
    x = noise.movedim(1, 2).contiguous()     # native -> engine layout

    with torch.no_grad():
        for i in range(args.steps):
            s_i = float(schedule[i])
            args_row = ([s_i], [None], [None], [None])
            v_c = adapter.batched_forward(
                x, *args_row, [{"encoder_hidden_states": cond}],
            )
            v_u = adapter.batched_forward(
                x, *args_row, [{"encoder_hidden_states": zero}],
            )
            v = v_u + args.cfg * (v_c - v_u)
            # ds is negative; combined with the adapter's negated
            # velocity this reproduces the reference's +dt step exactly.
            x = x + (float(schedule[i + 1]) - s_i) * v

    got = x.movedim(1, 2).float()
    cos = torch.nn.functional.cosine_similarity(
        got.flatten(), ref.flatten(), dim=0,
    ).item()
    rel = (
        (got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    ).item()

    print(f"  frames                 {frames}")
    print(f"  steps / cfg            {args.steps} / {args.cfg}")
    print(f"  final-latent cosine    {cos:.6f}   (bar {COS_BAR})")
    print(f"  relative RMS error     {rel:.5f}")
    print(f"  std got / ref          {got.std().item():.4f} / {ref.std().item():.4f}")

    ok = cos > COS_BAR
    print(f"\n  {'PASS' if ok else 'FAIL'}: chain "
          f"{'matches' if ok else 'DIVERGES FROM'} reference trajectory")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
