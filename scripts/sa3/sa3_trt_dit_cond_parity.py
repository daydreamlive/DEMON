"""Parity check: TRT DiT vs eager DiT on a REAL cond bundle.

The spike bench (sa3_bench_medium_dit_trt.py) only ever fed the engine
random t5_hidden tensors; this script runs the production staging path
(SA3TRTDit._stage_bundle over prepare_sa3_conditioning output) against
the torch DiT on identical x/t and reports cos + rel_rms per step.

Run:
    .venv/Scripts/python.exe scripts/sa3/sa3_trt_dit_cond_parity.py [--duration 54]

``--duration`` selects which built engine profile gets exercised (the
padded latent window must land inside an engine's L range): 54 → L=646,
24 → L=323/324.

Expected numbers (fp16mixed/STRONGLY_TYPED engines, 5090): cos ≥ 0.9998
per step when the cond's ``padding_mask`` is all-valid (duration 54 →
L=646). At durations where the padded window has masked tail frames
(e.g. 24 → 323 valid of L=324) the engine — which has no padding_mask
input; upstream's runtime always generates fully-valid windows — treats
the tail as valid and cos drops to ~0.991-0.999. That residual is the
mask-semantics difference, not engine numerics: giving eager an
all-valid mask restores cos 0.9999 against the same TRT output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "sa3"))

import torch  # noqa: E402

from sa3_reference_generate import checkpoint_dir, load_local_model  # noqa: E402
from sa3_stream_pipeline import prepare_sa3_conditioning  # noqa: E402

PROMPT = "funky ass shit"
STEPS = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=54.0)
    args = ap.parse_args()
    duration = float(args.duration)
    from acestep.engine.sa3_trt import SA3TRTDit, find_dit_engine

    print("[load] SA3 medium...", flush=True)
    sam = load_local_model(checkpoint_dir("medium"), device="cuda", model_half=True)
    sam.model.eval()

    cond = prepare_sa3_conditioning(
        sam, prompt=PROMPT, duration=duration, steps=STEPS,
    )
    bundle = cond.cond_bundle
    L = cond.latent_frames
    print(f"[cond] latent_frames={L} "
          f"cross_attn={tuple(bundle['cross_attn_cond'].shape)} "
          f"mask_sum={float(bundle['cross_attn_mask'].float().sum()):.0f}")

    engine_path = find_dit_engine("medium", L)
    if engine_path is None:
        raise RuntimeError(f"no TRT DiT engine for L={L}")
    trt_dit = SA3TRTDit(engine_path, latent_frames=L, seconds_total=duration)

    dtype = next(sam.model.model.parameters()).dtype
    g = torch.Generator(device="cuda").manual_seed(1528)

    for t_val in (1.0, 0.7, 0.4, 0.1):
        x = torch.randn(1, 256, L, device="cuda", dtype=dtype, generator=g)
        t_b = torch.tensor([t_val], device="cuda", dtype=dtype)
        with torch.no_grad():
            v_eager = sam.model.model(x, t_b, **bundle).float()
        v_trt = trt_dit.step_bundle(x, t_val, bundle).float().clone()
        cos = torch.nn.functional.cosine_similarity(
            v_eager.flatten(), v_trt.flatten(), dim=0,
        ).item()
        rel = ((v_trt - v_eager).norm() / v_eager.norm()).item()
        print(f"t={t_val:.2f}  cos={cos:.6f}  rel_rms={rel:.4e}  "
              f"|eager|={v_eager.norm().item():.2f} |trt|={v_trt.norm().item():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
