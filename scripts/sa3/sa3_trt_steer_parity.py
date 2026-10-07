"""Gate for the SA3 steering-input DiT engine (``sa3_m_dit_steer_*``).

Runs the production staging path (``SA3TRTDit``) on a REAL cond bundle
and reports, per timestep:

1. zero steering on the steering engine vs the previous engine (same
   fp16mixed graph without the input): bit-identity and cos;
2. zero steering vs eager (the existing per-step bar, cos >= 0.9998);
3. NONZERO steering, TRT vs eager with the same shift added by forward
   hooks on the trunk blocks (the eager delivery path), at the same bar,
   plus how far the shift moved the output (so the check is not vacuous);

then the per-step engine time of the previous engine, the steering
engine at zero and at nonzero steering, and the fp8 engine when present.

Run (idle GPU, no server):
    .venv/Scripts/python.exe scripts/sa3/sa3_trt_steer_parity.py [--duration 54]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "sa3"))

import torch  # noqa: E402

from sa3_reference_generate import checkpoint_dir, load_local_model  # noqa: E402
from sa3_stream_pipeline import prepare_sa3_conditioning  # noqa: E402

PROMPT = "driving cinematic synthwave, analog arpeggios, gated reverb snare"
STEPS = 8
T_VALUES = (1.0, 0.7, 0.4, 0.1)
STEER_BLOCKS = (5, 15)


def _cos(a, b) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0,
    ).item()


def _rel(a, ref) -> float:
    return ((a.float() - ref.float()).norm() / ref.float().norm()).item()


def _time_steps(dit, x, t, bundle, steering=None, iters=40) -> float:
    def once():
        if dit.steering_shape is not None:
            dit.step_bundle(x, t, bundle, steering=steering)
        else:
            dit.step_bundle(x, t, bundle)

    for _ in range(5):
        once()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        once()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=54.0)
    ap.add_argument("--json", default=None, help="write the numbers here")
    args = ap.parse_args()
    duration = float(args.duration)

    from acestep.engine.sa3_internals import wrapper_blocks
    from acestep.engine.sa3_trt import SA3TRTDit, find_dit_engine, trt_engines_dir

    print("[load] SA3 medium...", flush=True)
    sam = load_local_model(checkpoint_dir("medium"), device="cuda", model_half=True)
    sam.model.eval()
    dit_eager = sam.model.model
    blocks = wrapper_blocks(dit_eager)

    cond = prepare_sa3_conditioning(sam, prompt=PROMPT, duration=duration, steps=STEPS)
    bundle = cond.cond_bundle
    L = cond.latent_frames
    print(f"[cond] latent_frames={L}")

    plain_path = find_dit_engine("medium", L)  # selection without steering
    steer_path = find_dit_engine("medium", L, want_steering=True)
    if steer_path is None or "_steer_" not in steer_path.parent.name:
        raise RuntimeError(f"no steering engine covers L={L}")
    base_path = None
    for sub in sorted(trt_engines_dir().iterdir()):
        if sub.name == steer_path.parent.name.replace("_steer_", "_"):
            base_path = sub / f"{sub.name}.trt"
    if base_path is None or not base_path.is_file():
        raise RuntimeError("previous fp16mixed engine for the same profile not found")
    print(f"[engines] previous={base_path.parent.name} steer={steer_path.parent.name} "
          f"default_selection={plain_path.parent.name if plain_path else None}")

    base = SA3TRTDit(base_path, latent_frames=L, seconds_total=duration)
    steer = SA3TRTDit(steer_path, latent_frames=L, seconds_total=duration)
    assert base.steering_shape is None
    nb, hidden = steer.steering_shape
    assert nb == len(blocks), (nb, len(blocks))
    print(f"[steer] engine input [1, {nb}, {hidden}]")

    dtype = next(dit_eager.parameters()).dtype
    g = torch.Generator(device="cuda").manual_seed(1528)

    # Scale the test shift to the residual stream: 0.25 x the mean per-token
    # norm of the block outputs it lands on (measured on an eager pass).
    norms = {}

    def _cap(i):
        def h(_m, _inp, out):
            norms[i] = out.float().norm(dim=-1).mean().item()
        return h

    hs = [blocks[i].register_forward_hook(_cap(i)) for i in STEER_BLOCKS]
    x0 = torch.randn(1, 256, L, device="cuda", dtype=dtype, generator=g)
    with torch.no_grad():
        dit_eager(x0, torch.tensor([0.7], device="cuda", dtype=dtype), **bundle)
    for h in hs:
        h.remove()
    gv = torch.Generator(device="cuda").manual_seed(7)
    steering = torch.zeros(1, nb, hidden, device="cuda", dtype=torch.float32)
    for i in STEER_BLOCKS:
        v = torch.randn(hidden, device="cuda", generator=gv)
        steering[0, i] = v / v.norm() * 0.25 * norms[i]
    print(f"[steer] test shift norms: " + ", ".join(
        f"block {i}: {steering[0, i].norm().item():.2f} (token norm {norms[i]:.2f})"
        for i in STEER_BLOCKS))

    def _eager_hook(i):
        def h(_m, _inp, out):
            return out + steering[0, i].to(out.dtype).view(1, 1, -1)
        return h

    rows = []
    for t_val in T_VALUES:
        x = torch.randn(1, 256, L, device="cuda", dtype=dtype, generator=g)
        t_b = torch.tensor([t_val], device="cuda", dtype=dtype)
        with torch.no_grad():
            v_eager = dit_eager(x, t_b, **bundle).float()
            hooks = [blocks[i].register_forward_hook(_eager_hook(i)) for i in STEER_BLOCKS]
            try:
                v_eager_s = dit_eager(x, t_b, **bundle).float()
            finally:
                for h in hooks:
                    h.remove()
        v_base = base.step_bundle(x, t_val, bundle).float().clone()
        v_zero = steer.step_bundle(x, t_val, bundle, steering=None).float().clone()
        v_s = steer.step_bundle(x, t_val, bundle, steering=steering).float().clone()
        v_zero2 = steer.step_bundle(x, t_val, bundle, steering=None).float().clone()
        row = {
            "t": t_val,
            "zero_vs_prev_bitexact": bool(torch.equal(v_zero, v_base)),
            "zero_vs_prev_cos": _cos(v_zero, v_base),
            "zero_vs_prev_maxabs": (v_zero - v_base).abs().max().item(),
            "zero_vs_eager_cos": _cos(v_zero, v_eager),
            "prev_vs_eager_cos": _cos(v_base, v_eager),
            "steer_trt_vs_eager_cos": _cos(v_s, v_eager_s),
            "steer_trt_vs_eager_rel": _rel(v_s, v_eager_s),
            "steer_effect_cos": _cos(v_s, v_zero),
            "steer_effect_rel": _rel(v_s, v_zero),
            "zero_after_steer_bitexact": bool(torch.equal(v_zero2, v_zero)),
        }
        rows.append(row)
        print(
            f"t={t_val:.2f} zero-vs-prev bitexact={row['zero_vs_prev_bitexact']} "
            f"cos={row['zero_vs_prev_cos']:.6f} | zero-vs-eager cos={row['zero_vs_eager_cos']:.6f} "
            f"(prev-vs-eager {row['prev_vs_eager_cos']:.6f}) | steered TRT-vs-eager "
            f"cos={row['steer_trt_vs_eager_cos']:.6f} rel={row['steer_trt_vs_eager_rel']:.3e} | "
            f"effect rel={row['steer_effect_rel']:.3f} | rezero bitexact={row['zero_after_steer_bitexact']}",
            flush=True,
        )

    # Per-step engine time.
    x = torch.randn(1, 256, L, device="cuda", dtype=dtype, generator=g)
    timing = {
        "prev_ms": _time_steps(base, x, 0.5, bundle),
        "steer_zero_ms": _time_steps(steer, x, 0.5, bundle, steering=None),
        "steer_active_ms": _time_steps(steer, x, 0.5, bundle, steering=steering),
    }
    for sub in sorted(trt_engines_dir().iterdir()):
        if sub.name == f"sa3_m_dit_fp8_l1_{L}_{L}" or (
            sub.name.startswith("sa3_m_dit_fp8_l") and sub.name.endswith(f"_{L}")
        ):
            fp8 = SA3TRTDit(sub / f"{sub.name}.trt", latent_frames=L, seconds_total=duration)
            timing["fp8_ms"] = _time_steps(fp8, x, 0.5, bundle)
            timing["fp8_engine"] = sub.name
            del fp8
            break
    print("[timing] per step: " + json.dumps(timing))

    gate = min(
        min(r["zero_vs_eager_cos"] for r in rows),
        min(r["steer_trt_vs_eager_cos"] for r in rows),
    )
    out = {
        "L": L, "duration": duration, "engine": str(steer_path),
        "previous_engine": str(base_path), "rows": rows, "timing": timing,
        "min_cos_vs_eager": gate,
        "pass": gate >= 0.9998 and all(r["zero_after_steer_bitexact"] for r in rows),
    }
    print(f"[gate] min cos vs eager {gate:.6f} -> {'PASS' if out['pass'] else 'FAIL'}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
