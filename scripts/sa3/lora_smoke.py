"""SA3 LoRA GPU smoke (notes/SA3_LORA_PLAN.md Phase 1).

Drives the PRODUCTION manager (:class:`acestep.engine.sa3_lora.SA3LoRAManager`)
against a loaded medium SA3Context: register → prewarm → enable at
strength → DiT forward effect check → strength sweep 0→2 → disable →
VRAM accounting → teardown + re-session hygiene (the process-cached
model must come back bitwise pristine, twice).

Pass ``--lora <path>`` to smoke a real trained file (underfit output).
Without it, a synthetic rank-8 adapter is written against the live
module tree (DiT attention/FF + the seconds_total conditioner Linear) —
every mechanic is exercised identically; only musical quality needs a
trained file.

Run:
    .venv/Scripts/python.exe scripts/sa3/lora_smoke.py
    .venv/Scripts/python.exe scripts/sa3/lora_smoke.py --lora path/to/file.safetensors
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists())
for _p in (str(_REPO_ROOT),):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import torch  # noqa: E402


def _vram_mb() -> float:
    return torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def _synthesize_lora(ctx, out_dir: Path, rank: int = 8) -> Path:
    """Write a trainer-faithful synthetic file against the LIVE tree:
    every self_attn/cross_attn/ff Linear in the DiT plus the
    seconds_total conditioner Linear, fp16, lora_config in the header."""
    import torch.nn as nn

    from stable_audio_3.models.lora.utils import save_lora_safetensors

    g = torch.Generator().manual_seed(1528)
    sd = {}
    n = 0
    for name, mod in ctx.sam.model.model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not any(k in name for k in ("self_attn", "cross_attn", ".ff.")):
            continue
        w2 = mod.weight.view(mod.weight.shape[0], -1)
        prefix = f"{name}.parametrizations.weight.0"
        sd[f"{prefix}.lora_A"] = torch.randn(rank, w2.shape[1], generator=g) * 0.05
        sd[f"{prefix}.lora_B"] = torch.randn(w2.shape[0], rank, generator=g) * 0.05
        n += 1
    for name, mod in ctx.sam.model.conditioner.named_modules():
        if isinstance(mod, nn.Linear):
            w2 = mod.weight.view(mod.weight.shape[0], -1)
            prefix = f"{name}.parametrizations.weight.0"
            sd[f"{prefix}.lora_A"] = torch.randn(rank, w2.shape[1], generator=g) * 0.05
            sd[f"{prefix}.lora_B"] = torch.randn(w2.shape[0], rank, generator=g) * 0.05
            n += 1
    path = out_dir / "smoke_synthetic_lora.safetensors"
    save_lora_safetensors(
        sd,
        {"rank": rank, "alpha": rank, "adapter_type": "lora",
         "base_model": f"{ctx.model_id}-base"},
        path,
    )
    print(f"synthesized {n}-module rank-{rank} adapter -> {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--lora", default=None, help="path to a trained SA3 LoRA")
    args = ap.parse_args()

    from acestep.engine.sa3_context import SA3Context
    from acestep.engine.sa3_lora import SA3LoRAManager

    print(f"loading SA3Context({args.model!r})...")
    ctx = SA3Context(model_id=args.model)
    sam = ctx.sam

    # Snapshot / probe helpers from the phase-0.5 harness (scripts/sa3
    # is on sys.path via ensure_sa3_paths, which SA3Context ran).
    import lora_derisk_phase05 as h

    pristine = h.snapshot_state(sam)
    cond = ctx.prepare_cond(
        prompt="warm analog house groove, 124 bpm", duration=20.0, steps=8,
    )
    from acestep.engine.sa3_stream_helpers import stack_sa3_cond_bundles
    stacked = stack_sa3_cond_bundles([cond.cond_bundle])
    x = torch.randn(
        1, ctx.latent_channels, cond.latent_frames,
        device=ctx.device, dtype=ctx.dtype,
        generator=torch.Generator(device=ctx.device.type).manual_seed(7),
    )
    t = torch.full((1,), 0.5, device=ctx.device, dtype=ctx.dtype)

    def forward():
        import torch.nn.utils.parametrize as parametrize
        with torch.no_grad(), parametrize.cached():
            return ctx.dit(x, t, **stacked).clone()

    baseline = forward()
    ok = True

    if args.lora:
        lora_path = Path(args.lora)
    else:
        # Outside the repo AND outside every LoRA scan root (loras_dir +
        # lora_extra_dirs), so the synthetic file can never show up in a
        # real catalog.
        from acestep.paths import models_dir

        out_dir = models_dir() / "sa3" / "smoke"
        out_dir.mkdir(parents=True, exist_ok=True)
        lora_path = _synthesize_lora(ctx, out_dir)

    def run_session(label: str) -> bool:
        nonlocal ok
        session_ok = True
        mgr = SA3LoRAManager(
            model_root=sam.model.model,
            conditioner_root=sam.model.conditioner,
            checkpoint_id=ctx.model_id,
        )
        v0 = _vram_mb()
        lid = mgr.register_lora(str(lora_path))
        mgr.prewarm_lora(lid).result(timeout=120)
        d = mgr.get_lora(lid)
        print(f"[{label}] materialized: {d.materialized_bytes / 1e6:.1f} MB, "
              f"conditioner={mgr.touches_conditioner(lid)}")

        t0 = time.perf_counter()
        mgr.enable_lora(lid, strength=1.0)
        print(f"[{label}] enable_ms={(time.perf_counter() - t0) * 1000:.1f} "
              f"vram_delta_mb={_vram_mb() - v0:.1f}")

        out_1 = forward()
        if torch.equal(out_1, baseline):
            print(f"[{label}] FAIL: enabled adapter had no forward effect")
            session_ok = False

        # Strength sweep 0 -> 2 (the knob range): monotone engagement and
        # exact identity at 0.
        sweep_ms = []
        prev = None
        for s in (0.0, 0.5, 1.0, 1.5, 2.0):
            t0 = time.perf_counter()
            mgr.set_lora_strength(lid, s)
            sweep_ms.append((time.perf_counter() - t0) * 1000)
            out_s = forward()
            if s == 0.0 and not torch.equal(out_s, baseline):
                print(f"[{label}] FAIL: strength 0 is not bit-identical to base")
                session_ok = False
            if prev is not None and torch.equal(out_s, prev):
                print(f"[{label}] FAIL: strength {s} output identical to previous")
                session_ok = False
            prev = out_s
        print(f"[{label}] strength sweep applied, max set_ms={max(sweep_ms):.2f}")

        v_enabled = _vram_mb()
        mgr.disable_lora(lid)
        out_off = forward()
        if not torch.equal(out_off, baseline):
            print(f"[{label}] FAIL: post-disable forward differs from baseline")
            session_ok = False
        print(f"[{label}] disable reclaimed {v_enabled - _vram_mb():.1f} MB")

        # Enable again, then close WITHOUT disabling — the session-death
        # path. The process-cached model must come back pristine.
        mgr.enable_lora(lid, strength=0.8)
        mgr.close()
        if not h.no_parametrizations_left(sam):
            print(f"[{label}] FAIL: parametrizations left after close()")
            session_ok = False
        if not h.compare_state(sam, pristine, f"{label} post-close"):
            session_ok = False
        out_post = forward()
        if not torch.equal(out_post, baseline):
            print(f"[{label}] FAIL: post-close forward differs from baseline")
            session_ok = False
        print(f"[{label}] {'PASS' if session_ok else 'FAIL'}")
        ok &= session_ok
        return session_ok

    # Two back-to-back "sessions" on the same cached context — the
    # cross-session hygiene contract.
    run_session("session-1")
    run_session("session-2")

    print(f"\n=== lora_smoke: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
