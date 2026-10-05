"""Parity and latency gate for the MiniMax-Music3 TensorRT DiT.

Answers the only two questions that decide whether an engine ships:
does it compute the same thing as the torch module, and is it faster.

**Real conditioning, never random.** Every input comes from
``minimax_music3_8s_seed7.safetensors``: a real 200-frame
autoregressive capture put through the condition encoder, plus the
reference trajectory's own endpoints. Cosine similarity against
random-normal inputs is a known false positive for engines of this
shape (a graph can be badly wrong and still score 0.9999 on noise,
because noise has no structure to destroy). The fixture's conditioning
is the structure.

Two gates, and they fail differently:

1. **Per-step.** One velocity forward at t = 0.05 / 0.3 / 0.6 / 0.95
   against the eager module, on the fixture's own initial latent and
   conditioning. Bar: **cosine >= 0.9998**. This catches a wrong graph,
   a mis-bound buffer, or a precision island TRT re-cast.
2. **Compounded.** The full 30-step CFG-1.7 trajectory from the
   fixture's ``initial_noise``, compared against its ``final_latent``.
   This is the one that catches errors too small to see per step and
   large enough to matter after thirty of them, the failure mode that
   retired SA3's BF16 engine (per-step cos 0.99 -> final-latent cos
   0.81).

The reference for gate 1 is eager **fp32**, not eager bf16. bf16 itself
only reaches cos ~0.998-0.9997/step against fp32 on this model, so
gating an engine against it would grade against a ruler more bent than
the thing being measured. bf16 is reported alongside for context,
because bf16 is what the eager production path actually runs.

Usage::

    .venv/Scripts/python.exe scripts/minimax/minimax_trt_parity.py \
        --precision fp16
    # reuse the cached eager references (they are dtype- and
    # fixture-keyed, so this is safe across engine rebuilds)
    .venv/Scripts/python.exe scripts/minimax/minimax_trt_parity.py \
        --precision fp32 --reuse-ref
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# A sibling ACE-Step checkout shadows `acestep` otherwise.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from acestep.engine.minimax_dit import MiniMaxDiT  # noqa: E402
from acestep.engine.minimax_helpers import minimax_root, resolve_model_dir  # noqa: E402
from acestep.engine.minimax_trt import (  # noqa: E402
    MiniMaxTRTDit,
    find_dit_engine_path,
    list_dit_engines,
    trt_engines_dir,
)

#: Per-step bar. Same number the SA3 fp16mixed engine is held to.
COSINE_BAR = 0.9998
#: The four flow-matching times the module-level parity gate uses, so
#: the two harnesses are directly comparable.
TIMESTEPS = (0.05, 0.30, 0.60, 0.95)
#: The fixture's own reference trajectory.
TRAJECTORY_STEPS = 30
TRAJECTORY_CFG = 1.7
#: Eager bf16 lands here on the compounded gate. Not a bar, a landmark:
#: an engine materially below it has lost something bf16 kept.
EAGER_BF16_TRAJECTORY_COS = 0.999868

EAGER_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def default_fixture() -> Path:
    return minimax_root() / "fixtures" / "minimax_music3_8s_seed7.safetensors"


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    a = reference.detach().to(torch.float64).flatten()
    b = candidate.detach().to(torch.float64).flatten()
    diff = (a - b).norm()
    return {
        "cosine": float(torch.dot(a, b) / (a.norm() * b.norm())),
        "rel_rms": float(diff / a.norm()),
        "max_abs": float((a - b).abs().max()),
        "snr_db": float(20 * torch.log10(a.norm() / diff)) if float(diff) > 0 else float("inf"),
    }


def vram_used_gb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9


def median_ms(fn, *, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def trajectory(step_fn, noise: torch.Tensor, cond: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """The fixture's reference trajectory, in MiniMax's own convention.

    ``t`` runs 0 (noise) to 1 (data) on a uniform grid and Euler steps
    forward: ``x += v * dt``. The DEMON-convention rewrite lives in
    :class:`~acestep.engine.minimax_adapter.MiniMaxAdapter`; driving the
    module directly here keeps this gate independent of it, so a broken
    adapter cannot make a broken engine look right (or vice versa).
    """
    x = noise.clone()
    dt = 1.0 / TRAJECTORY_STEPS
    for i in range(TRAJECTORY_STEPS):
        t = i * dt
        v_cond = step_fn(x, t, cond)
        v_uncond = step_fn(x, t, zero)
        x = x + dt * (v_uncond + TRAJECTORY_CFG * (v_cond - v_uncond))
    return x


# ------------------------------------------------------------------
# eager reference
# ------------------------------------------------------------------


def eager_reference(
    *, fixture: dict, model_dir, device, dtypes: tuple[str, ...], bench: bool,
) -> tuple[dict, dict]:
    """Reference velocities + trajectories from the torch module.

    Each dtype is loaded, used and released before the next, so the peak
    is one model rather than three; this runs on the same card the
    engine is about to want.
    """
    root = Path(resolve_model_dir(model_dir))
    tensors: dict[str, torch.Tensor] = {}
    report: dict = {}

    for label in dtypes:
        dtype = EAGER_DTYPES[label]
        print(f"[eager] loading {label} ...", flush=True)
        t0 = time.time()
        dit = MiniMaxDiT.from_pretrained(root / "transformer", dtype=dtype, device=device)
        print(f"[eager] loaded in {time.time() - t0:.0f}s", flush=True)

        x = fixture["initial_noise"].to(device, dtype)
        cond = fixture["cond"].to(device, dtype)
        zero = torch.zeros_like(cond)

        with torch.no_grad():
            for t in TIMESTEPS:
                ts = torch.full((1,), t, device=device, dtype=dtype)
                out = dit(x, ts, cond)
                tensors[f"{label}/step/t{t:.2f}"] = out.float().cpu()

            def step_fn(xt, t, c, _dit=dit, _dtype=dtype):
                ts = torch.full((1,), t, device=xt.device, dtype=_dtype)
                return _dit(xt.to(_dtype), ts, c).to(xt.dtype)

            # The latent accumulates in the module's own dtype, which is
            # what the eager production path does and what the published
            # bf16 landmark was measured with. The engine's trajectory
            # accumulates in fp32 because its IO is fp32; that difference
            # is part of what the comparison is showing.
            t0 = time.time()
            final = trajectory(step_fn, x, cond, zero)
            traj_s = time.time() - t0
            tensors[f"{label}/trajectory"] = final.float().cpu()
        report[f"{label}_trajectory_s"] = traj_s

        if bench:
            with torch.no_grad():
                ts1 = torch.full((1,), 0.5, device=device, dtype=dtype)
                report[f"{label}_b1_ms"] = median_ms(
                    lambda: dit(x, ts1, cond), iters=20, warmup=5,
                )
                x4 = x.repeat(4, 1, 1).contiguous()
                c4 = cond.repeat(4, 1, 1).contiguous()
                ts4 = torch.full((4,), 0.5, device=device, dtype=dtype)
                report[f"{label}_b4_ms"] = median_ms(
                    lambda: dit(x4, ts4, c4), iters=10, warmup=3,
                )
            print(
                f"[eager] {label}: B=1 {report[f'{label}_b1_ms']:.1f} ms  "
                f"B=4 {report[f'{label}_b4_ms']:.1f} ms",
                flush=True,
            )

        del dit
        torch.cuda.empty_cache()

    return tensors, report


# ------------------------------------------------------------------
# engine build facts
# ------------------------------------------------------------------


def engine_facts(engine_path: Path) -> dict:
    facts = {
        "path": str(engine_path),
        "name": engine_path.parent.name,
        "size_gb": engine_path.stat().st_size / 1e9,
    }
    sidecar = Path(str(engine_path) + ".metadata.json")
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        facts["built_at"] = meta.get("built_at")
        facts["tensorrt"] = meta.get("tensorrt_version")
        facts["config"] = meta.get("config")
    report_csv = engine_path.parent.parent / "build_report.csv"
    if report_csv.is_file():
        with report_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                cfg = facts.get("config") or {}
                want = (
                    f"{cfg.get('precision', '')} "
                    f"b{cfg.get('min_batch', '')}-{cfg.get('max_batch', '')} "
                    f"l{cfg.get('min_latents', '')}_{cfg.get('opt_latents', '')}_"
                    f"{cfg.get('max_latents', '')}"
                )
                if want and want in row.get("engine", "") and row.get("status") == "OK":
                    facts["build_time_s"] = float(row["build_time_s"])
    return facts


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--precision", choices=("fp16", "fp32"), default=None,
                    help="Which engine to gate (default: discovery order)")
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--latent-frames", type=int, default=None,
                    help="Defaults to the fixture's own length")
    ap.add_argument("--eager-dtypes", nargs="+", default=["float32", "bfloat16"],
                    choices=sorted(EAGER_DTYPES))
    ap.add_argument("--reuse-ref", action="store_true",
                    help="Reuse a cached eager reference instead of recomputing")
    ap.add_argument("--ref-cache", default=None)
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    fixture_path = Path(args.fixture) if args.fixture else default_fixture()
    if not fixture_path.is_file():
        raise SystemExit(f"fixture not found: {fixture_path}")
    raw = load_file(str(fixture_path))
    cond = raw["encoder_hidden_states"].unsqueeze(0)
    fixture = {
        "cond": cond,
        "initial_noise": raw["initial_noise"].unsqueeze(0),
        "final_latent": raw["final_latent"].unsqueeze(0).to(device, torch.float32),
    }
    latent_frames = args.latent_frames or cond.shape[1]
    print(f"fixture {fixture_path.name}  L={latent_frames}", flush=True)

    engine_path = find_dit_engine_path(latent_frames, precision=args.precision, batch=1)
    if engine_path is None:
        built = [e["name"] for e in list_dit_engines()]
        raise SystemExit(
            f"no batch-1 MiniMax DiT engine covering L={latent_frames} at "
            f"precision={args.precision or 'any'} under {trt_engines_dir()}. "
            f"Built: {built or 'none'}. "
            "Build one with `python -m acestep.engine.trt.minimax_build`."
        )
    facts = engine_facts(engine_path)
    print(f"engine  {facts['name']}  ({facts['size_gb']:.2f} GB)", flush=True)

    cache = Path(args.ref_cache) if args.ref_cache else (
        fixture_path.parent / f"trt_parity_eager_ref_L{latent_frames}.safetensors"
    )
    report: dict = {"engine": facts, "fixture": str(fixture_path), "L": latent_frames}

    if args.reuse_ref and cache.is_file():
        print(f"[eager] reusing {cache}", flush=True)
        refs = load_file(str(cache))
        eager_report = {}
    else:
        refs, eager_report = eager_reference(
            fixture=fixture, model_dir=args.model_dir, device=device,
            dtypes=tuple(args.eager_dtypes), bench=not args.no_bench,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in refs.items()}, str(cache))
        print(f"[eager] cached references -> {cache}", flush=True)
    report["eager"] = eager_report

    # ---- engine -------------------------------------------------------
    torch.cuda.empty_cache()
    before_gb = vram_used_gb()
    t0 = time.time()
    dit = MiniMaxTRTDit(engine_path, latent_frames=latent_frames)
    load_s = time.time() - t0
    after_gb = vram_used_gb()
    report["engine"]["load_s"] = load_s
    report["engine"]["vram_gb"] = after_gb - before_gb
    print(
        f"[trt] loaded in {load_s:.1f}s, +{after_gb - before_gb:.2f} GB VRAM "
        f"(device total in use {after_gb:.2f} GB)",
        flush=True,
    )

    x = fixture["initial_noise"].to(device, torch.float32)
    cond_d = cond.to(device, torch.float32)
    zero_d = torch.zeros_like(cond_d)
    cond_bundle = {"encoder_hidden_states": cond_d}
    zero_bundle = {"encoder_hidden_states": zero_d}

    # ---- gate 1: per-step ---------------------------------------------
    step_rows = []
    for t in TIMESTEPS:
        got = dit.step_bundle(x, t, cond_bundle).float().cpu()
        for label in args.eager_dtypes:
            key = f"{label}/step/t{t:.2f}"
            if key not in refs:
                continue
            row = metrics(refs[key], got)
            row.update({"t": t, "vs": label})
            step_rows.append(row)
    report["per_step"] = step_rows

    # ---- gate 2: compounded trajectory --------------------------------
    def trt_step(xt, t, bundle_cond):
        bundle = cond_bundle if bundle_cond is cond_d else zero_bundle
        return dit.step_bundle(xt, t, bundle).float().clone()

    t0 = time.time()
    final = trajectory(trt_step, x, cond_d, zero_d)
    report["trajectory_s"] = time.time() - t0
    traj = {"vs_fixture": metrics(fixture["final_latent"], final)}
    for label in args.eager_dtypes:
        key = f"{label}/trajectory"
        if key in refs:
            traj[f"vs_eager_{label}"] = metrics(refs[key].to(device), final)
            traj[f"eager_{label}_vs_fixture"] = metrics(
                fixture["final_latent"], refs[key].to(device),
            )
    report["trajectory"] = traj

    # ---- bench ---------------------------------------------------------
    if not args.no_bench:
        bench: dict = {}
        bench["b1_ms"] = median_ms(
            lambda: dit.step_bundle(x, 0.5, cond_bundle), iters=30, warmup=10,
        )
        # The production engine is batch-1 and the adapter loops slots, so
        # the honest depth-4 number is four sequential forwards, not a
        # batched one. A batched engine is reported too when one is built.
        bench["b4_looped_ms"] = median_ms(
            lambda: [dit.step_bundle(x, 0.5, cond_bundle) for _ in range(4)],
            iters=10, warmup=3,
        )
        batched_path = find_dit_engine_path(
            latent_frames, precision=args.precision or dit.precision, batch=4,
        )
        if batched_path is not None and batched_path != engine_path:
            try:
                batched = _batched_engine(batched_path, latent_frames)
                x4 = x.repeat(4, 1, 1).contiguous()
                c4 = cond_d.repeat(4, 1, 1).contiguous()
                t4 = torch.full((4,), 0.5, device=device, dtype=torch.float32)
                bench["b4_batched_engine"] = batched_path.parent.name
                bench["b4_batched_ms"] = median_ms(
                    lambda: batched(x4, t4, c4), iters=10, warmup=3,
                )
                del batched
            except Exception as exc:  # a bench-only engine must never fail the gate
                bench["b4_batched_error"] = repr(exc)
        report["bench"] = bench

    # ---- verdict -------------------------------------------------------
    print()
    print("  per-step parity (bar cosine >= %.4f)" % COSINE_BAR)
    print(f"  {'t':>6} {'vs eager':>10} {'cosine':>10} {'rel RMS':>10} {'SNR dB':>8}  verdict")
    # The gate grades against eager fp32 when it is available; bf16 rows
    # are printed for context but never decide the verdict (bf16 itself
    # only reaches ~0.998 against fp32 on this model).
    gate_rows = [r for r in step_rows if r["vs"] == "float32"] or step_rows
    worst = min((r["cosine"] for r in gate_rows), default=float("nan"))
    for row in step_rows:
        ok = row["cosine"] >= COSINE_BAR
        print(
            f"  {row['t']:>6.2f} {row['vs']:>10} {row['cosine']:>10.6f} "
            f"{row['rel_rms']:>10.2e} {row['snr_db']:>8.1f}  "
            f"{'PASS' if ok else 'FAIL'}"
        )
    print()
    print(f"  compounded trajectory ({TRAJECTORY_STEPS} steps, CFG {TRAJECTORY_CFG})")
    for key, row in traj.items():
        print(f"  {key:>28}  cos {row['cosine']:.6f}   rel RMS {row['rel_rms']:.4f}")
    print(f"  {'eager bf16 landmark':>28}  cos {EAGER_BF16_TRAJECTORY_COS:.6f}")

    if "bench" in report:
        print()
        print("  latency (median ms per forward)")
        for key, value in report["bench"].items():
            if key.endswith("_ms"):
                print(f"  {key:>28}  {value:8.1f}")
        for key, value in eager_report.items():
            if key.endswith("_ms"):
                print(f"  {('eager ' + key):>28}  {value:8.1f}")
        print(f"  {'engine size GB':>28}  {facts['size_gb']:8.2f}")
        if "build_time_s" in facts:
            print(f"  {'build time s':>28}  {facts['build_time_s']:8.0f}")
        print(f"  {'engine VRAM GB':>28}  {report['engine']['vram_gb']:8.2f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json_out}")

    ok = bool(gate_rows) and all(r["cosine"] >= COSINE_BAR for r in gate_rows)
    against = gate_rows[0]["vs"] if gate_rows else "nothing"
    print(
        f"\n  {'PASS' if ok else 'FAIL'}: worst per-step cosine vs eager "
        f"{against} {worst:.6f} (bar {COSINE_BAR})"
    )
    return 0 if ok else 1


def _batched_engine(path: Path, latent_frames: int):
    """A batch-N engine wrapped just enough to time it.

    :class:`MiniMaxTRTDit` is deliberately batch-1 (it is the production
    contract), so the benchmark-only batched engine gets a local, minimal
    binding rather than a loosening of that class.
    """
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes
    from acestep.nodes.vae_nodes import _get_trt_stream

    engine = engine_from_bytes(bytes_from_path(str(path)))
    ctx = engine.create_execution_context()
    if ctx is None:
        raise RuntimeError("could not create execution context (CUDA OOM)")
    pg_stream = _get_trt_stream()
    stream = torch.cuda.ExternalStream(pg_stream.ptr)

    def run(x, t, cond):
        for name, tensor in (
            ("hidden_states", x), ("timestep", t), ("encoder_hidden_states", cond),
        ):
            ctx.set_input_shape(name, tuple(tensor.shape))
            ctx.set_tensor_address(name, tensor.data_ptr())
        out_shape = tuple(ctx.get_tensor_shape("velocity"))
        out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
        ctx.set_tensor_address("velocity", out.data_ptr())
        stream.wait_stream(torch.cuda.current_stream())
        if not ctx.execute_async_v3(pg_stream.ptr):
            raise RuntimeError("batched engine execution failed")
        stream.synchronize()
        return out

    return run


if __name__ == "__main__":
    raise SystemExit(main())
