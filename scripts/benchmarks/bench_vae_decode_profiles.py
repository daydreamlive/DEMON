"""VRAM profile across VAE decode TRT engines — accurate edition.

Question: how much VRAM does each VAE decode engine commit, broken down
by source (weights, TRT activation workspace, runtime overhead), so we
can decide whether swapping the canonical 240 s VAE decoder for a
narrow-profile engine while windowed decode is active is worth it.

Three engines are compared (all standard teacher VAE, fp16, same ONNX):

  * vae_decode_fp16_5s_fixed   min=opt=max=125
  * vae_decode_fp16_3to30s     min=75 opt=125 max=750
  * vae_decode_fp16_240s       min=125 opt=1500 max=6000   (canonical)

Two complementary measurements:

  1. **TRT-reported, GPU-noise-free.** ``engine.device_memory_size_v2``
     is the workspace TRT will reserve for the engine, queried from the
     deserialized engine without ever doing inference. This is the
     authoritative activation-memory number. Combined with the
     on-disk engine size (which is dominated by serialised weights),
     this gives us a per-engine VRAM budget that doesn't depend on what
     else is on the GPU.

  2. **Subprocess-isolated GPU measurement.** Per (engine, runtime_T)
     pair we spawn a fresh Python subprocess that initialises CUDA,
     samples the baseline VRAM 3 times to detect external noise, then
     loads the engine, creates a context, sets a shape, allocates I/O
     tensors, and runs warmup + timed iterations. We report driver-
     level VRAM via ``torch.cuda.mem_get_info()`` at each stage. On
     Windows WDDM, per-process accounting via NVML returns ``None`` —
     so this baseline-subtracted delta is the only practical
     per-process number, and the per-stage breakdown lets us see when
     each piece commits.

  3. **Shape sweep** for the dynamic-range engines confirms TRT's
     default ``STATIC`` allocation strategy: workspace is sized for the
     profile's max shape and committed at context creation, so it
     doesn't grow when the runtime input grows.

Usage::

    uv run python scripts/benchmarks/bench_vae_decode_profiles.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent
TRT_ROOT = Path.home() / ".daydream-scope" / "models" / "demon" / "trt_engines"


ENGINES: tuple[tuple[str, Path], ...] = (
    ("5s_fixed", TRT_ROOT / "vae_decode_fp16_5s_fixed" / "vae_decode_fp16_5s_fixed.engine"),
    ("3to30s",   TRT_ROOT / "vae_decode_fp16_3to30s"   / "vae_decode_fp16_3to30s.engine"),
    ("240s",     TRT_ROOT / "vae_decode_fp16_240s"     / "vae_decode_fp16_240s.engine"),
)


# (engine_label, runtime_T_frames, comment)
RUN_PLAN: tuple[tuple[str, int, str], ...] = (
    ("5s_fixed", 125, "5 s window — primary"),
    ("3to30s",   125, "5 s window — primary"),
    ("240s",     125, "5 s window — primary"),

    # Sweep dynamic engines: confirm workspace is shape-invariant.
    ("3to30s",   250, "10 s — sweep"),
    ("3to30s",   500, "20 s — sweep"),
    ("3to30s",   750, "30 s — sweep (max)"),
    ("240s",     750, "30 s — sweep"),
    ("240s",    1500, "60 s — sweep (240s opt)"),
    ("240s",    3000, "120 s — sweep"),
    ("240s",    6000, "240 s — sweep (max)"),
)


# -----------------------------------------------------------------------
# 1. TRT-reported authoritative numbers (no GPU work).
# -----------------------------------------------------------------------

def inspect_engine(engine_path: Path) -> dict:
    """Return TRT-reported memory numbers for an engine.

    Runs in the parent process; deserialising creates a CUDA context
    but does not bind a per-engine workspace, so the returned numbers
    are independent of any subsequent GPU measurement.
    """
    import tensorrt as trt

    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        engine = rt.deserialize_cuda_engine(f.read())

    on_disk = engine_path.stat().st_size
    # Single profile (idx 0); v2 is the supported API in TRT 10.x.
    activation = engine.get_device_memory_size_for_profile_v2(0)

    profile_shape = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            mn, op, mx = engine.get_tensor_profile_shape(name, 0)
            profile_shape[name] = {"min": list(mn), "opt": list(op), "max": list(mx)}

    return {
        "engine_path": str(engine_path),
        "on_disk_mb": round(on_disk / (1 << 20), 1),
        "activation_mb": round(activation / (1 << 20), 1),
        "weight_streaming_mb": round(engine.streamable_weights_size / (1 << 20), 1),
        "profile_shape": profile_shape,
    }


# -----------------------------------------------------------------------
# 2. Subprocess-isolated GPU measurement.
# -----------------------------------------------------------------------

def parent_main():
    # ---------- 1. Engine inspection (no GPU pressure yet) ----------
    print("=" * 96)
    print("ENGINE INSPECTION  (TRT-reported, independent of GPU state)")
    print("=" * 96)
    inspections = {}
    hdr = f"{'engine':<12}{'on_disk_MB':>13}{'activation_MB':>16}{'profile (T frames)':>30}"
    print(hdr)
    print("-" * len(hdr))
    for label, path in ENGINES:
        if not path.exists():
            print(f"{label:<12}  MISSING: {path}")
            continue
        info = inspect_engine(path)
        inspections[label] = info
        ps = info["profile_shape"].get("latents", {})
        rng = f"{ps.get('min', ['?'])[-1]} / {ps.get('opt', ['?'])[-1]} / {ps.get('max', ['?'])[-1]}"
        print(f"{label:<12}{info['on_disk_mb']:>13.1f}{info['activation_mb']:>16.1f}"
              f"{rng:>30}")

    # ---------- 2. GPU stage-breakdown across (engine, T) pairs ----------
    print("\n" + "=" * 110)
    print("GPU STAGE BREAKDOWN  (each row = fresh subprocess; baseline 3x stability check)")
    print("=" * 110)
    rows = []
    engine_paths = dict(ENGINES)
    for label, T, note in RUN_PLAN:
        path = engine_paths.get(label)
        if path is None or not path.exists():
            print(f"[skip] {label} T={T}: engine missing")
            continue
        result = run_child(path, T)
        if result is None:
            print(f"[fail] {label} T={T}: subprocess returned no JSON")
            continue
        result["engine_label"] = label
        result["runtime_T"] = T
        result["note"] = note
        rows.append(result)
        print(
            f"  {label:<10} T={T:>4}  "
            f"baseline={result['baseline_mb']:>5.0f} "
            f"(stab+/-{result['baseline_stability_mb']:>4.1f}) | "
            f"+deserialize={result['delta_after_deserialize']:>+5.0f} "
            f"+context={result['delta_after_context']:>+5.0f} "
            f"+shape={result['delta_after_shape']:>+5.0f} "
            f"+io={result['delta_after_io']:>+5.0f} "
            f"+exec1={result['delta_after_first_exec']:>+5.0f} | "
            f"peak_d={result['delta_peak']:>+5.0f}  "
            f"({result['mean_ms']:>5.2f} ms)"
        )

    # ---------- 3. Apples-to-apples summary at T=125 ----------
    print("\n" + "=" * 96)
    print("PER-PROCESS COST AT T=125  (5 s window — the windowed-decode workload)")
    print("=" * 96)
    by_label_125 = {r["engine_label"]: r for r in rows if r["runtime_T"] == 125}
    if {"5s_fixed", "3to30s", "240s"} <= set(by_label_125):
        ref = by_label_125["5s_fixed"]
        hdr = (
            f"{'engine':<10}{'GPU committed (d)':>22}"
            f"{'TRT activation':>17}{'TRT on-disk':>14}"
            f"{'vs 5s_fixed':>14}"
        )
        print(hdr)
        print("-" * len(hdr))
        for label in ("5s_fixed", "3to30s", "240s"):
            r = by_label_125[label]
            inf = inspections.get(label, {})
            committed = r["delta_peak"]
            delta_vs_ref = committed - ref["delta_peak"]
            print(
                f"{label:<10}{committed:>22.0f}{inf.get('activation_mb', 0):>17.1f}"
                f"{inf.get('on_disk_mb', 0):>14.1f}{delta_vs_ref:>+14.0f}"
            )

    # ---------- 4. Shape-invariance check ----------
    print("\n" + "=" * 96)
    print("SHAPE INVARIANCE  (does committed VRAM grow when runtime T grows?)")
    print("=" * 96)
    for engine_label in ("3to30s", "240s"):
        related = [r for r in rows if r["engine_label"] == engine_label]
        if len(related) < 2:
            continue
        peaks = [r["delta_peak"] for r in related]
        rng = max(peaks) - min(peaks)
        print(f"  {engine_label:<10}: peak-d across {len(related)} shapes = "
              f"min {min(peaks):.0f} / max {max(peaks):.0f} (spread {rng:.0f} MB)")
        for r in related:
            print(f"     T={r['runtime_T']:>4}  peak_d = {r['delta_peak']:>5.0f} MB  "
                  f"(latency {r['mean_ms']:.2f} ms)")

    out_path = RESULTS_DIR / "bench_vae_decode_profiles.json"
    out_path.write_text(json.dumps({
        "engine_inspection": inspections,
        "runs": rows,
    }, indent=2))
    print(f"\nDetailed JSON written to {out_path}")


def run_child(engine_path: Path, T: int) -> dict | None:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--child", "--engine", str(engine_path), "--frames", str(T),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(f"  stderr:\n{proc.stderr}", flush=True)
        return None
    last_json = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                last_json = json.loads(line)
            except json.JSONDecodeError:
                pass
    if last_json is None and proc.stderr:
        print(f"  stderr:\n{proc.stderr}", flush=True)
    return last_json


# -----------------------------------------------------------------------
# Child subprocess: own CUDA context, single (engine, T) measurement.
# -----------------------------------------------------------------------

def child_main(engine_path: Path, T: int):
    import torch
    torch.set_grad_enabled(False)

    device = torch.device("cuda")
    torch.cuda.synchronize()
    _ = torch.empty(1, device=device)
    torch.cuda.synchronize()

    def vram_mb() -> float:
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return (total - free) / (1024 * 1024)

    # 3-sample baseline for noise / external-process detection.
    baseline_samples = []
    for _ in range(3):
        baseline_samples.append(vram_mb())
        time.sleep(0.05)
    baseline_mb = statistics.median(baseline_samples)
    baseline_stability = max(baseline_samples) - min(baseline_samples)

    # ---- Stage 1: deserialize engine (no execution context yet) ----
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes
    engine = engine_from_bytes(bytes_from_path(str(engine_path)))
    after_deserialize_mb = vram_mb()

    # ---- Stage 2: create execution context (TRT default = STATIC reserves now) ----
    ctx = engine.create_execution_context()
    after_context_mb = vram_mb()

    # ---- Stage 3: set input shape ----
    ctx.set_input_shape("latents", (1, 64, T))
    after_shape_mb = vram_mb()

    # ---- Stage 4: allocate I/O tensors (ours, via torch) ----
    lat = torch.empty((1, 64, T), dtype=torch.float32, device=device).normal_()
    ctx.set_tensor_address("latents", lat.data_ptr())
    out_shape = tuple(ctx.get_tensor_shape("audio"))
    out = torch.empty(out_shape, dtype=torch.float32, device=device)
    ctx.set_tensor_address("audio", out.data_ptr())
    after_io_mb = vram_mb()

    # ---- Stage 5: warmup + timed iterations ----
    from polygraphy import cuda as pg_cuda
    stream = pg_cuda.Stream()

    if not ctx.execute_async_v3(stream.ptr):
        raise RuntimeError("TRT execute_async_v3 failed (warmup #1)")
    stream.synchronize()
    after_first_exec_mb = vram_mb()

    peak_mb = max(
        after_deserialize_mb, after_context_mb, after_shape_mb,
        after_io_mb, after_first_exec_mb,
    )

    for _ in range(4):
        if not ctx.execute_async_v3(stream.ptr):
            raise RuntimeError("TRT execute_async_v3 failed (warmup)")
        stream.synchronize()
        peak_mb = max(peak_mb, vram_mb())

    timings: list[float] = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if not ctx.execute_async_v3(stream.ptr):
            raise RuntimeError("TRT execute_async_v3 failed (timed)")
        stream.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)
        peak_mb = max(peak_mb, vram_mb())

    final_mb = vram_mb()
    keepalive = (engine, ctx, lat, out, stream)

    result = {
        # Raw (driver-global) VRAM at each stage.
        "baseline_mb": round(baseline_mb, 1),
        "baseline_stability_mb": round(baseline_stability, 1),
        "after_deserialize_mb": round(after_deserialize_mb, 1),
        "after_context_mb": round(after_context_mb, 1),
        "after_shape_mb": round(after_shape_mb, 1),
        "after_io_mb": round(after_io_mb, 1),
        "after_first_exec_mb": round(after_first_exec_mb, 1),
        "peak_mb": round(peak_mb, 1),
        "final_mb": round(final_mb, 1),
        # Stage deltas (cumulative, baseline-subtracted) — the headline numbers.
        "delta_after_deserialize": round(after_deserialize_mb - baseline_mb, 1),
        "delta_after_context": round(after_context_mb - baseline_mb, 1),
        "delta_after_shape": round(after_shape_mb - baseline_mb, 1),
        "delta_after_io": round(after_io_mb - baseline_mb, 1),
        "delta_after_first_exec": round(after_first_exec_mb - baseline_mb, 1),
        "delta_peak": round(peak_mb - baseline_mb, 1),
        # Latency, secondary.
        "mean_ms": round(sum(timings) / len(timings), 3) if timings else -1,
        "min_ms": round(min(timings), 3) if timings else -1,
        "max_ms": round(max(timings), 3) if timings else -1,
        "n_iters": len(timings),
    }
    print(json.dumps(result), flush=True)
    del keepalive


# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true",
                   help="internal: run a single measurement and emit JSON")
    p.add_argument("--engine", type=Path)
    p.add_argument("--frames", type=int)
    args = p.parse_args()

    if args.child:
        if args.engine is None or args.frames is None:
            sys.exit("--child requires --engine and --frames")
        child_main(args.engine, args.frames)
    else:
        parent_main()


if __name__ == "__main__":
    main()
