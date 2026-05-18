#!/usr/bin/env python3
"""Benchmark TRT vs TRT + Triton fused Euler step, end-to-end.

The fused Triton kernel in ``acestep.engine.kernels`` replaces the
post-DiT Euler integration step. Since TRT runs the DiT and the
integrator runs afterward on the resulting velocity tensor, the two
layers are orthogonal: the fused kernel slots in regardless of decoder
backend.

This bench measures end-to-end wall time for ``Session.generate()`` with
a TRT-backed decoder under two configurations:

    baseline   -- production path. ``StreamPipeline._get_compiled`` wraps
                  ``ode_steps.step_ode_euler`` in ``torch.compile``.
    fused      -- ``_get_compiled`` is monkey-patched so the Euler call
                  bypasses torch.compile and calls
                  ``acestep.engine.kernels.fused_euler_step`` instead.

The fused kernel only covers the vanilla fast path (vs == ones sentinel,
onc == zeros sentinel). The bench's workload (single prompt, no CFG, no
SDE, no latent mask, no x0_target) hits exactly that path -- any deviation
falls back to the original step_ode_euler so quality is preserved.

A numerical-equivalence check runs once on a captured latent before
timing, so a non-trivial divergence aborts before the bench wastes time.

Usage:
    uv run python scripts/benchmarks/bench_fused_euler.py
    uv run python scripts/benchmarks/bench_fused_euler.py --duration 30 --steps 8 --iters 20
    uv run python scripts/benchmarks/bench_fused_euler.py --checkpoint acestep-v15-turbo
    uv run python scripts/benchmarks/bench_fused_euler.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Callable, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

torch.set_grad_enabled(False)

from acestep.engine import ode_steps
from acestep.engine.kernels import fused_euler_step
from acestep.engine.session import Session
from acestep.engine.stream import StreamPipeline
from acestep.paths import available_trt_engines


DEFAULT_PROMPT = "ambient electronic, slow tempo, warm pads, gentle texture"


# ── triton wrapper that matches step_ode_euler's signature ──────────


def _triton_step_ode_euler(
    xt: torch.Tensor,
    vt: torch.Tensor,
    t_curr: float,
    t_next: float,
    vs: torch.Tensor,
    onc: torch.Tensor,
) -> torch.Tensor:
    """Drop-in for ``ode_steps.step_ode_euler`` using the Triton kernel.

    Covers only the production fast path (``vs == ones`` sentinel and
    ``onc == zeros`` sentinel). Falls back to the original on anything
    else so quality stays identical.
    """
    # Shape-only sentinel check (no .item() sync per step). Non-trivial
    # velocity / ode-noise curves are shaped [1, T, 1] or [B, T, 1]; the
    # ones / zeros sentinels stay [1, 1, 1]. The bench's verification
    # pass catches any case where this assumption gets violated.
    if vs.shape != (1, 1, 1) or onc.shape != (1, 1, 1):
        return _ORIG_STEP_ODE_EULER(xt, vt, t_curr, t_next, vs, onc)
    # The Triton kernel walks tensors with linear offsets, so non-contig
    # views (slices of a stacked buffer, etc.) would read wrong memory.
    # ``xt.clone()`` already gives us a fresh contig output buffer; vt
    # needs an explicit ``.contiguous()`` because it can come from a
    # view of the DiT's batched output.
    out = xt.contiguous().clone()
    vt = vt.contiguous()
    fused_euler_step(out, vt, t_curr, t_next)
    return out


# Captured at module-load so the wrapper can fall back cleanly even after
# a future test monkey-patches ode_steps directly.
_ORIG_STEP_ODE_EULER = ode_steps.step_ode_euler


# ── _get_compiled patcher ───────────────────────────────────────────


_ORIG_GET_COMPILED = StreamPipeline._get_compiled


def _patched_get_compiled(self, fn: Callable) -> Callable:
    """Inject the Triton wrapper for ``step_ode_euler``; leave the rest alone."""
    if fn is ode_steps.step_ode_euler:
        # Stash and return uncompiled; bypassing torch.compile is the point.
        self._compiled_cache[fn] = _triton_step_ode_euler
        return _triton_step_ode_euler
    return _ORIG_GET_COMPILED(self, fn)


def _enable_fused() -> None:
    StreamPipeline._get_compiled = _patched_get_compiled


def _disable_fused() -> None:
    StreamPipeline._get_compiled = _ORIG_GET_COMPILED


# ── workload builder ───────────────────────────────────────────────


def _build_session(
    checkpoint: str, duration_s: float,
) -> tuple[Session, dict]:
    """Construct a Session with a TRT decoder + VAE backend."""
    try:
        engines, max_dur = available_trt_engines(
            duration_s=duration_s, checkpoint=checkpoint,
        )
    except Exception as e:
        raise SystemExit(
            f"No TRT engines available for checkpoint={checkpoint} "
            f"duration={duration_s}s ({e}). Build them first; see docs/TRT.md."
        )

    session = Session(
        config_path=checkpoint,
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines=engines,
    )
    return session, engines


def _encode_prompt(session: Session, prompt: str, duration_s: float):
    """Encode the prompt into Conditioning. The default text2music
    instruction is applied internally by ``encode_text``.
    """
    return session.encode_text(tags=prompt, duration=duration_s)


# ── numerical check ────────────────────────────────────────────────


def _generate_once(
    session: Session, conditioning, *, duration_s: float, steps: int, seed: int,
):
    """Run one generation. Returns the output Latent."""
    from acestep.nodes.types import Latent

    # Empty source latent at the right T so denoise=1 produces fresh audio.
    return session.generate(
        conditioning=conditioning,
        steps=steps,
        seed=seed,
        denoise=1.0,
        duration=duration_s,
    )


def _verify_equivalence(
    session: Session, conditioning, *, duration_s: float, steps: int, seed: int,
) -> dict:
    """Generate once with baseline and once with fused on the same seed,
    compare output latents elementwise.
    """
    _disable_fused()
    base_lat = _generate_once(
        session, conditioning, duration_s=duration_s, steps=steps, seed=seed,
    )

    _enable_fused()
    try:
        fused_lat = _generate_once(
            session, conditioning, duration_s=duration_s, steps=steps, seed=seed,
        )
    finally:
        _disable_fused()

    a = base_lat.tensor.float()
    b = fused_lat.tensor.float()
    return {
        "shape": list(a.shape),
        "max_abs_err": (a - b).abs().max().item(),
        "mean_abs_err": (a - b).abs().mean().item(),
        "base_norm": a.norm().item(),
        "fused_norm": b.norm().item(),
    }


# ── timing harness ──────────────────────────────────────────────────


def _time_runs(
    fn: Callable[[], None], iters: int, warmup: int,
) -> list[float]:
    """Wall-clock per-iter ms with CUDA sync at boundaries.

    Wall clock (perf_counter + cuda.synchronize) rather than CUDA events
    because Session.generate launches many kernels including Python-side
    work; total wall time is the relevant signal.
    """
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _summarize(times_ms: list[float]) -> dict:
    return {
        "min_ms": min(times_ms),
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "p95_ms": sorted(times_ms)[int(len(times_ms) * 0.95)],
        "iters": len(times_ms),
    }


# ── main ────────────────────────────────────────────────────────────


def _print_results(baseline: dict, fused: dict, *, steps: int) -> None:
    print()
    print("=" * 78)
    print(f"{'variant':<10}  {'min':>9}  {'median':>9}  {'p95':>9}  {'mean':>9}")
    print("-" * 78)
    for label, s in [("baseline", baseline), ("fused", fused)]:
        print(
            f"{label:<10}  {s['min_ms']:>8.2f}m  {s['median_ms']:>8.2f}m"
            f"  {s['p95_ms']:>8.2f}m  {s['mean_ms']:>8.2f}m"
        )
    print("-" * 78)

    delta_med = baseline["median_ms"] - fused["median_ms"]
    pct = 100.0 * delta_med / baseline["median_ms"] if baseline["median_ms"] else 0
    speedup = baseline["median_ms"] / fused["median_ms"] if fused["median_ms"] else 0
    print(
        f"delta (median): {delta_med:+.2f} ms ({pct:+.2f}%) "
        f"-> {speedup:.3f}x speedup"
    )
    print(f"per-step delta: {delta_med / max(steps, 1):+.3f} ms")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="acestep-v15-turbo",
        help="Model checkpoint (must have TRT engines built).",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Audio duration in seconds.",
    )
    parser.add_argument(
        "--steps", type=int, default=8,
        help="Diffusion steps per generation (= Euler steps timed per call).",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help="Text prompt for the bench workload.",
    )
    parser.add_argument(
        "--iters", type=int, default=20,
        help="Timed iterations per variant.",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Warmup iterations per variant (first call hits TRT + compile).",
    )
    parser.add_argument(
        "--seed", type=int, default=1528,
        help="Generation seed (same across variants for verification).",
    )
    parser.add_argument(
        "--equivalence-atol", type=float, default=1e-2,
        help="Max absolute latent error allowed before aborting (bf16 ~3e-3).",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip the numerical equivalence check (not recommended).",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Write full result dict as JSON.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required.", file=sys.stderr)
        return 2

    print(f"device     : {torch.cuda.get_device_name(0)}")
    print(f"checkpoint : {args.checkpoint}")
    print(f"duration   : {args.duration}s")
    print(f"steps      : {args.steps}")
    print(f"iters      : {args.iters} (warmup={args.warmup})")
    print()

    print("Building Session (TRT decoder + VAE)...")
    t0 = time.perf_counter()
    session, engines = _build_session(args.checkpoint, args.duration)
    print(f"  Session ready in {time.perf_counter() - t0:.1f}s")
    print(f"  TRT engines: {list(engines.keys())}")

    print("Encoding prompt...")
    conditioning = _encode_prompt(session, args.prompt, args.duration)

    if not args.skip_verify:
        print("Verifying baseline vs fused on identical seed...")
        t0 = time.perf_counter()
        verif = _verify_equivalence(
            session, conditioning,
            duration_s=args.duration, steps=args.steps, seed=args.seed,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"  shape={verif['shape']}  "
            f"max_abs_err={verif['max_abs_err']:.3e}  "
            f"mean_abs_err={verif['mean_abs_err']:.3e}  "
            f"({elapsed:.1f}s)"
        )
        if verif["max_abs_err"] > args.equivalence_atol:
            print(
                f"\nFAIL: max_abs_err={verif['max_abs_err']:.3e} exceeds "
                f"--equivalence-atol={args.equivalence_atol:.0e}.\n"
                "The Triton kernel is producing materially different output. "
                "Investigate before trusting timing.",
                file=sys.stderr,
            )
            return 3
        print("  equivalence OK")
        print()

    # --- Timed runs ---
    def workload():
        _generate_once(
            session, conditioning,
            duration_s=args.duration, steps=args.steps, seed=args.seed,
        )

    print("Timing baseline (production: torch.compile'd step_ode_euler)...")
    _disable_fused()
    baseline_times = _time_runs(workload, args.iters, args.warmup)
    baseline = _summarize(baseline_times)
    print(f"  median = {baseline['median_ms']:.2f} ms")

    print("Timing fused (Triton fused_euler_step)...")
    _enable_fused()
    try:
        fused_times = _time_runs(workload, args.iters, args.warmup)
    finally:
        _disable_fused()
    fused = _summarize(fused_times)
    print(f"  median = {fused['median_ms']:.2f} ms")

    _print_results(baseline, fused, steps=args.steps)

    if args.json:
        out = {
            "device": torch.cuda.get_device_name(0),
            "checkpoint": args.checkpoint,
            "duration_s": args.duration,
            "steps": args.steps,
            "iters": args.iters,
            "warmup": args.warmup,
            "seed": args.seed,
            "engines": list(engines.keys()),
            "baseline": baseline,
            "fused": fused,
            "baseline_times_ms": baseline_times,
            "fused_times_ms": fused_times,
        }
        if not args.skip_verify:
            out["equivalence"] = verif
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
