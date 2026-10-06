"""Parity check: TRT DiT vs eager DiT on a REAL cond bundle.

The spike bench (sa3_bench_medium_dit_trt.py) only ever fed the engine
random t5_hidden tensors; this script runs the production staging path
(SA3TRTDit._stage_bundle over prepare_sa3_conditioning output) against
the torch DiT on identical x/t and reports cos + rel_rms per step.

Run:
    .venv/Scripts/python.exe scripts/sa3/sa3_trt_dit_cond_parity.py [--duration 54]
    .venv/Scripts/python.exe scripts/sa3/sa3_trt_dit_cond_parity.py --model small-music

Two checks per run: the original four fixed-t probes on random x, then
the real 8-step pingpong trajectory under the session schedule
(``cond.sched_args``). On the trajectory both backends see the SAME
eager-driven ``x_t`` at every step, so the per-step cosine isolates
engine numerics from compounding; a second pass runs each backend's own
trajectory (same seeded renoise) and reports the compounded final-latent
cosine. ``--json`` writes every number to a file.

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
GATE_COS = 0.9998
STEPS = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=54.0)
    ap.add_argument("--model", default="medium",
                    help="SA3 catalog id: medium | small-music")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--json", default=None, help="write results here")
    ap.add_argument("--production", action="store_true",
                    help="condition through the production SA3Context "
                         "(prepare_cond: song-length seconds label + the "
                         "context's outro pad), i.e. the exact latent "
                         "window and padding mask a streaming session gets, "
                         "instead of the spike helper's duration+6 s window")
    ap.add_argument("--fp32-ref", action="store_true",
                    help="eager reference in fp32 instead of the production "
                         "fp16 load: separates engine error from the fp16 "
                         "eager reference's own error (which alone measures "
                         "cos ~0.9996-0.9999 vs fp32 on short prompts)")
    ap.add_argument("--engine", default=None,
                    help="engine dir name under trt_engines/ to force "
                         "(default: what find_dit_engine selects)")
    args = ap.parse_args()
    duration = float(args.duration)
    from acestep.engine.sa3_trt import SA3TRTDit, find_dit_engine

    print(f"[load] SA3 {args.model}...", flush=True)
    seconds_label = duration
    if args.production:
        from acestep.engine.sa3_context import SA3Context

        context = SA3Context(args.model, model_half=not args.fp32_ref)
        sam = context.sam
        cond = context.prepare_cond(prompt=args.prompt, duration=duration, steps=STEPS)
        seconds_label = context.cond_seconds_total(duration)
    else:
        sam = load_local_model(
            checkpoint_dir(args.model), device="cuda", model_half=not args.fp32_ref,
        )
        cond = prepare_sa3_conditioning(
            sam, prompt=args.prompt, duration=duration, steps=STEPS,
        )
    sam.model.eval()
    bundle = cond.cond_bundle
    L = cond.latent_frames
    print(f"[cond] latent_frames={L} "
          f"cross_attn={tuple(bundle['cross_attn_cond'].shape)} "
          f"mask_sum={float(bundle['cross_attn_mask'].float().sum()):.0f} "
          f"seconds_label={seconds_label}")
    pad = bundle.get("padding_mask")
    valid = int(pad.sum()) if pad is not None else L
    print(f"[cond] padding_mask valid frames {valid}/{L}")

    if args.engine:
        from acestep.engine.sa3_trt import trt_engines_dir
        engine_path = trt_engines_dir() / args.engine / f"{args.engine}.trt"
    else:
        engine_path = find_dit_engine(args.model, L)
    if engine_path is None:
        raise RuntimeError(f"no TRT DiT engine for L={L}")
    trt_dit = SA3TRTDit(engine_path, latent_frames=L, seconds_total=seconds_label)

    dtype = next(sam.model.model.parameters()).dtype
    g = torch.Generator(device="cuda").manual_seed(1528)

    print(f"[engine] {engine_path.parent.name}")
    results = {"model": args.model, "engine": engine_path.parent.name,
               "latent_frames": int(L), "valid_frames": valid,
               "duration_s": duration, "seconds_label": seconds_label,
               "production_cond": bool(args.production),
               "prompt": args.prompt, "eager_ref": "fp32" if args.fp32_ref else "fp16",
               "fixed_t": [], "trajectory": []}
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
        results["fixed_t"].append({"t": t_val, "cos": cos, "rel_rms": rel})

    # ---- real trajectory: 8-step pingpong under the session schedule ----
    import stable_audio_3.inference.sampling as sampling

    sa = dict(cond.sched_args)
    esl = sa.get("effective_seq_len")
    if torch.is_tensor(esl):
        esl = esl.detach().cpu()
    sched = sampling.build_schedule(
        steps=STEPS, sigma_max=1.0, dist_shift=sa["dist_shift"],
        effective_seq_len=esl, fallback_seq_len=sa["fallback_seq_len"],
        include_endpoint=True, device="cpu",
    )
    if sched.dim() == 2:
        sched = sched[0]
    sched = sched.float().tolist()

    def velocity(kind, x, t_val):
        if kind == "eager":
            t_b = torch.tensor([t_val], device="cuda", dtype=dtype)
            with torch.no_grad():
                return sam.model.model(x.to(dtype), t_b, **bundle).float()
        return trt_dit.step_bundle(x.to(dtype), t_val, bundle).float().clone()

    def run(kind, *, shadow: bool):
        gen = torch.Generator(device="cuda").manual_seed(7)
        x = torch.randn(1, 256, L, device="cuda", generator=gen)
        for i in range(STEPS):
            t_cur, t_nxt = sched[i], sched[i + 1]
            v = velocity(kind, x, t_cur)
            if shadow:
                v_t = velocity("trt", x, t_cur)
                cos = torch.nn.functional.cosine_similarity(
                    v.flatten(), v_t.flatten(), dim=0).item()
                rel = ((v_t - v).norm() / v.norm()).item()
                print(f"step {i}  t={t_cur:.4f}  cos={cos:.6f}  rel_rms={rel:.4e}")
                results["trajectory"].append(
                    {"step": i, "t": t_cur, "cos": cos, "rel_rms": rel})
            denoised = x - t_cur * v
            noise = torch.randn(x.shape, device="cuda", generator=gen)
            x = (1 - t_nxt) * denoised + t_nxt * noise
        return x

    print(f"[trajectory] schedule={[round(v, 4) for v in sched]}")
    x_eager = run("eager", shadow=True)
    x_trt = run("trt", shadow=False)
    final_cos = torch.nn.functional.cosine_similarity(
        x_eager.flatten(), x_trt.flatten(), dim=0).item()
    results["final_latent_cos"] = final_cos
    min_cos = min(r["cos"] for r in results["trajectory"])
    results["min_step_cos"] = min_cos
    results["gate"] = GATE_COS
    results["pass"] = bool(min_cos >= GATE_COS)
    print(f"[trajectory] min per-step cos={min_cos:.6f} (gate {GATE_COS}) "
          f"-> {'PASS' if results['pass'] else 'FAIL'}; compounded final "
          f"latent cos={final_cos:.6f}")
    if args.json:
        import json
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
