"""Eager-vs-refit LoRA output parity (notes/SA3_LORA_PLAN.md Phase 2 item 5).

Runs the eager parametrized DiT against the refit-mirrored TRT engine on
identical inputs through the LoRA lifecycle:

* baseline (no LoRA) per-step cos — the engine-numerics floor;
* one LoRA at strengths 0.0 / 0.5 / 1.0 (strength 0 must leave the TRT
  outputs value-identical to baseline: a zero delta merges to the exact
  base weights);
* a second stacked LoRA;
* disable of the second (the mirror's dirty-set must push the first
  LoRA's merged weights back — TRT outputs value-identical to the
  single-LoRA phase);
* disable of the first (full restore — TRT outputs value-identical to
  baseline);
* full 8-step pingpong denoise (upstream's own sampler) with the LoRA
  at strength 1.0, eager vs TRT-shimmed, same seed: final-latent cos.

Per-step comparisons use the same cond bundle on both paths, so
conditioner-side LoRA keys (which can never reach the TRT graph — the
seconds_total tail is baked in as constants, plan D5) do not skew the
DiT-level verdict.

Run:
    .venv/Scripts/python.exe scripts/sa3/lora_refit_parity.py --lora <trained.safetensors>
    (add --lora2 <file> for the stacking phase; defaults to the smoke
    synthetic adapter, synthesizing it if absent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists())
for _p in (str(_REPO_ROOT),):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import torch  # noqa: E402

PROMPT = "warm analog house groove, 124 bpm"
DURATION = 54.0
STEPS = 8
T_VALUES = (1.0, 0.7, 0.4, 0.1)
SEED = 1528
# Engine-numerics floor from the pre-LoRA TRT parity signoff
# (scripts/sa3/sa3_trt_dit_cond_parity.py): fp16mixed cos >= 0.9998/step
# on all-valid windows.
COS_FLOOR = 0.9998


class _TRTShim(torch.nn.Module):
    """Routes DiT forwards to the TRT engine inside upstream's sampler.
    Holds the eager module so ``parameters()`` keeps reporting the model
    dtype to ``generate``-style callers."""

    def __init__(self, trt_dit, eager_dit):
        super().__init__()
        self._trt = trt_dit
        self._eager = eager_dit

    def forward(self, x, t, **kwargs):
        t_val = float(t.flatten()[0].item()) if torch.is_tensor(t) else float(t)
        # step_bundle returns its persistent fp32 output buffer —
        # materialize and match the eager dtype so sampler arithmetic
        # (and randn_like draws) stay identical across paths.
        return self._trt.step_bundle(x, t_val, kwargs).to(dtype=x.dtype, copy=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--lora", required=True, help="trained SA3 LoRA (primary)")
    ap.add_argument("--lora2", default=None,
                    help="second LoRA for the stacking phase "
                         "(default: the smoke synthetic adapter)")
    args = ap.parse_args()

    from acestep.engine.sa3_context import SA3Context
    from acestep.engine.sa3_lora import SA3LoRAManager
    from acestep.engine.sa3_trt import SA3TRTDit, find_dit_engine
    from acestep.engine.sa3_trt_lora import SA3TRTRefitMirror, find_refit_manifest

    print(f"loading SA3Context({args.model!r})...")
    ctx = SA3Context(model_id=args.model)
    sam = ctx.sam

    cond = ctx.prepare_cond(prompt=PROMPT, duration=DURATION, steps=STEPS)
    L = cond.latent_frames
    from acestep.engine.sa3_stream_helpers import stack_sa3_cond_bundles
    stacked = stack_sa3_cond_bundles([cond.cond_bundle])
    print(f"[cond] latent_frames={L}")

    engine_path = find_dit_engine(args.model, L, want_refittable=True)
    if engine_path is None:
        raise RuntimeError(f"no refittable TRT DiT engine for L={L}; "
                           f"build with sa3_build --dit --refit")
    trt_dit = SA3TRTDit(engine_path, latent_frames=L, seconds_total=DURATION)
    if not trt_dit.refittable:
        raise RuntimeError(f"selected engine is not refit-built: {engine_path}")
    manifest = find_refit_manifest(engine_path)
    if manifest is None:
        raise RuntimeError("no refit manifest; run gen_sa3_refit_manifest.py")
    mirror = SA3TRTRefitMirror(trt_dit.engine, sam.model.model, manifest)

    mgr = SA3LoRAManager(
        model_root=sam.model.model,
        conditioner_root=sam.model.conditioner,
        checkpoint_id=ctx.model_id,
    )

    lora2 = args.lora2
    if lora2 is None:
        from acestep.paths import models_dir
        import lora_smoke as smoke  # scripts/sa3 on sys.path via ensure_sa3_paths

        default2 = models_dir() / "sa3" / "smoke" / "smoke_synthetic_lora.safetensors"
        if not default2.is_file():
            default2.parent.mkdir(parents=True, exist_ok=True)
            smoke._synthesize_lora(ctx, default2.parent)
        lora2 = str(default2)

    import torch.nn.utils.parametrize as parametrize

    def compare(tag: str):
        """Per-step eager-vs-TRT at fixed t values on identical x.
        Returns (worst_cos, {t: trt_out})."""
        worst = 1.0
        trt_outs = {}
        g = torch.Generator(device=ctx.device.type).manual_seed(SEED)
        for t_val in T_VALUES:
            x = torch.randn(1, ctx.latent_channels, L, device=ctx.device,
                            dtype=ctx.dtype, generator=g)
            t_b = torch.full((1,), t_val, device=ctx.device, dtype=ctx.dtype)
            with torch.no_grad(), parametrize.cached():
                v_eager = ctx.dit(x, t_b, **stacked).float()
            v_trt = trt_dit.step_bundle(x, t_val, stacked).float().clone()
            cos = torch.nn.functional.cosine_similarity(
                v_eager.flatten(), v_trt.flatten(), dim=0).item()
            rel = ((v_trt - v_eager).norm() / v_eager.norm()).item()
            worst = min(worst, cos)
            trt_outs[t_val] = v_trt
            print(f"  [{tag}] t={t_val:.2f} cos={cos:.6f} rel_rms={rel:.4e}")
        return worst, trt_outs

    def outs_equal(a: dict, b: dict) -> bool:
        return all(torch.equal(a[t], b[t]) for t in T_VALUES)

    failures = []

    def gate(name: str, ok: bool):
        print(f"  -> {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)

    print("\n=== baseline (no LoRA) ===")
    worst, base_outs = compare("base")
    gate(f"baseline worst cos {worst:.6f} >= {COS_FLOOR}", worst >= COS_FLOOR)

    lid1 = mgr.register_lora(args.lora)
    mgr.prewarm_lora(lid1).result(timeout=180)
    print(f"\nlora1={Path(args.lora).name} "
          f"touches_conditioner={mgr.touches_conditioner(lid1)}")

    print(f"\n=== lora1 @ 0.0 ===")
    mgr.enable_lora(lid1, strength=0.0)
    mirror.sync(reason="parity-enable-s0")
    worst, s0_outs = compare("s0.0")
    gate("strength-0 TRT outputs identical to baseline", outs_equal(s0_outs, base_outs))
    gate(f"s0 worst cos {worst:.6f} >= {COS_FLOOR}", worst >= COS_FLOOR)

    s1_outs = None
    for s in (0.5, 1.0):
        print(f"\n=== lora1 @ {s} ===")
        mgr.set_lora_strength(lid1, s)
        mirror.sync(reason=f"parity-s{s}")
        worst, outs = compare(f"s{s}")
        gate(f"s{s} worst cos {worst:.6f} >= {COS_FLOOR}", worst >= COS_FLOOR)
        if s == 1.0:
            s1_outs = outs

    print(f"\n=== + lora2 (stacked) ===")
    lid2 = mgr.register_lora(lora2)
    mgr.prewarm_lora(lid2).result(timeout=180)
    mgr.enable_lora(lid2, strength=0.7)
    mirror.sync(reason="parity-stack")
    worst, _ = compare("stack")
    gate(f"stacked worst cos {worst:.6f} >= {COS_FLOOR}", worst >= COS_FLOOR)

    print(f"\n=== disable lora2 (dirty-set restore) ===")
    mgr.disable_lora(lid2)
    mirror.sync(reason="parity-drop2")
    worst, outs = compare("drop2")
    gate("post-drop2 TRT outputs identical to lora1@1.0 phase",
         outs_equal(outs, s1_outs))

    print(f"\n=== disable lora1 (full restore) ===")
    mgr.disable_lora(lid1)
    mirror.sync(reason="parity-drop1")
    worst, outs = compare("drop1")
    gate("post-drop-all TRT outputs identical to baseline",
         outs_equal(outs, base_outs))

    print(f"\n=== full {STEPS}-step pingpong denoise, lora1 @ 1.0, seed {SEED} ===")
    from stable_audio_3.inference.sampling import sample_flow_pingpong

    mgr.enable_lora(lid1, strength=1.0)
    mirror.sync(reason="parity-fullloop")
    sigmas = ctx.make_schedule_builder(cond, STEPS)(1.0).detach().float()

    def full_loop(model) -> torch.Tensor:
        torch.manual_seed(SEED)
        x = torch.randn(1, ctx.latent_channels, L, device=ctx.device, dtype=ctx.dtype)
        with torch.no_grad(), parametrize.cached():
            return sample_flow_pingpong(
                model, x, sigmas.to(ctx.device), disable_tqdm=True, **stacked,
            ).float().clone()

    lat_eager = full_loop(ctx.dit)
    lat_trt = full_loop(_TRTShim(trt_dit, ctx.dit))
    cos = torch.nn.functional.cosine_similarity(
        lat_eager.flatten(), lat_trt.flatten(), dim=0).item()
    rel = ((lat_trt - lat_eager).norm() / lat_eager.norm()).item()
    print(f"  final-latent cos={cos:.6f} rel_rms={rel:.4e}")
    gate(f"final-latent cos {cos:.6f} >= 0.99", cos >= 0.99)

    mgr.close()
    verdict = "PASS" if not failures else f"FAIL ({', '.join(failures)})"
    print(f"\n=== lora_refit_parity: {verdict} ===")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
