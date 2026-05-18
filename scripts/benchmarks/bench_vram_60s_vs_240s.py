"""VRAM + latency sweep across every TRT engine profile we have on disk.

Auto-discovers every ``*.engine`` under ``~/.daydream-scope/models/demon/trt_engines``
(skipping ``.bak`` files and ``_onnx*`` source dirs) and benchmarks each at
batch depths of 2, 4, and 8 (clamped to whatever the engine's optimization
profile actually allows). Each (engine, depth) is measured in its own
subprocess so CUDA contexts and TRT allocations cannot bleed between runs.

Driver-level VRAM is reported via ``torch.cuda.mem_get_info`` so TRT and
PyTorch allocations are both captured.

Stages measured per (engine, depth):
  - baseline   : VRAM after CUDA init, before loading anything.
  - post_load  : VRAM after engine + execution context construction.
  - post_alloc : VRAM after the first inference (TRT lazy workspace bound).
  - peak       : Max VRAM observed across all warmup + timed iterations.
  - final      : VRAM after all iterations complete.

The child runs entirely off the engine's optimization profile: it asks
TensorRT for each input's max shape, overrides dim 0 to the requested depth
(clamped), and feeds random tensors of the engine's declared dtype. No
hardcoded shape tables, so new engines just work.

Usage::

    uv run python scripts/benchmarks/bench_vram_60s_vs_240s.py
    uv run python scripts/benchmarks/bench_vram_60s_vs_240s.py --depths 2,4,8
    uv run python scripts/benchmarks/bench_vram_60s_vs_240s.py --filter decoder

Detailed JSON is written next to this file in
``bench_vram_60s_vs_240s.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent
TRT_ROOT = Path.home() / ".daydream-scope" / "models" / "demon" / "trt_engines"


def discover_engines():
    """Return a sorted list of (label, engine_path) for every real engine."""
    out = []
    for sub in sorted(TRT_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith("_"):
            continue
        for eng in sorted(sub.glob("*.engine")):
            if eng.name.endswith(".bak"):
                continue
            out.append((sub.name, eng))
    return out


def parent_main(args):
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    engines = discover_engines()
    if args.filter:
        engines = [(l, p) for (l, p) in engines if args.filter in l]
    if not engines:
        print(f"No engines found under {TRT_ROOT}", flush=True)
        return

    print(f"Found {len(engines)} engines, testing depths {depths}", flush=True)
    print(f"Total runs: {len(engines) * len(depths)}", flush=True)

    rows = []
    out_path = RESULTS_DIR / "bench_vram_60s_vs_240s.json"

    for label, eng_path in engines:
        for depth in depths:
            print(f"\n[RUN ] {label:<48s} depth={depth}", flush=True)
            t0 = time.time()
            result = run_child(eng_path, depth)
            elapsed = time.time() - t0
            if result is None:
                print(f"[FAIL] {label} depth={depth}: subprocess returned no JSON", flush=True)
                rows.append({"label": label, "engine_path": str(eng_path),
                             "depth_requested": depth, "status": "FAIL",
                             "elapsed_s": round(elapsed, 1)})
                out_path.write_text(json.dumps(rows, indent=2))
                continue
            result["label"] = label
            result["engine_path"] = str(eng_path)
            result["depth_requested"] = depth
            result["elapsed_s"] = round(elapsed, 1)
            rows.append(result)
            _print_one(result)
            # Persist after every run so partial results survive a crash.
            out_path.write_text(json.dumps(rows, indent=2))

    print("\n" + "=" * 130)
    print("SUMMARY (VRAM in MB; mean inference latency in ms; depth_actual is depth_requested clamped to engine max batch)")
    print("=" * 130)
    hdr = (f"{'engine':<48}{'d_req':>6}{'d_act':>6}{'baseline':>10}"
           f"{'post_load':>11}{'post_alloc':>12}{'peak':>10}{'final':>10}{'mean_ms':>10}{'status':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("status") == "FAIL":
            print(f"{r['label']:<48}{r['depth_requested']:>6}{'-':>6}"
                  f"{'-':>10}{'-':>11}{'-':>12}{'-':>10}{'-':>10}{'-':>10}{'FAIL':>8}")
            continue
        if r.get("status") == "OOM":
            print(f"{r['label']:<48}{r['depth_requested']:>6}{r.get('depth_actual','-'):>6}"
                  f"{'-':>10}{'-':>11}{'-':>12}{'-':>10}{'-':>10}{'-':>10}{'OOM':>8}")
            continue
        print(f"{r['label']:<48}{r['depth_requested']:>6}{r['depth_actual']:>6}"
              f"{r['baseline_mb']:>10.0f}{r['post_load_mb']:>11.0f}"
              f"{r['post_alloc_mb']:>12.0f}{r['peak_mb']:>10.0f}{r['final_mb']:>10.0f}"
              f"{r['mean_ms']:>10.1f}{'OK':>8}")

    print(f"\nDetailed JSON written to {out_path}")


def run_child(engine_path: Path, depth: int) -> dict | None:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--child", "--engine", str(engine_path), "--depth", str(depth),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    last_json = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                last_json = json.loads(line)
            except json.JSONDecodeError:
                pass
    if last_json is None:
        if proc.stderr:
            err_tail = "\n".join(proc.stderr.splitlines()[-12:])
            print(f"  stderr (tail):\n{err_tail}", flush=True)
    return last_json


def _print_one(r: dict):
    if r.get("status") == "OOM":
        print(f"       OOM at depth_actual={r.get('depth_actual','?')}: {r.get('error','')[:120]}")
        return
    print(f"       depth_actual={r['depth_actual']}  baseline={r['baseline_mb']:.0f}MB  "
          f"post_load={r['post_load_mb']:.0f}MB  post_alloc={r['post_alloc_mb']:.0f}MB  "
          f"peak={r['peak_mb']:.0f}MB  final={r['final_mb']:.0f}MB  mean={r['mean_ms']:.1f}ms  "
          f"input_shapes={r.get('input_shapes', {})}")


# ---------------------------------------------------------------------------
# Child mode: runs in a fresh subprocess, owns its own CUDA context.
# ---------------------------------------------------------------------------

def child_main(engine_path: Path, depth: int):
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

    baseline_mb = vram_mb()

    try:
        result = _bench_engine(engine_path, depth, device, vram_mb, baseline_mb)
    except torch.cuda.OutOfMemoryError as e:
        result = {
            "status": "OOM",
            "depth_actual": depth,
            "baseline_mb": round(baseline_mb, 1),
            "error": str(e),
        }
    except Exception as e:
        result = {
            "status": "FAIL",
            "depth_actual": depth,
            "baseline_mb": round(baseline_mb, 1),
            "error": f"{type(e).__name__}: {e}",
        }

    print(json.dumps(result), flush=True)


def _bench_engine(engine_path, depth, device, vram_mb, baseline_mb):
    import torch
    import tensorrt as trt

    engine, ctx = _load_engine(engine_path)
    post_load_mb = vram_mb()

    input_names = []
    output_names = []
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            input_names.append(n)
        else:
            output_names.append(n)

    # Determine actual depth: clamp requested to engine's max batch.
    # We assume dim 0 is the batch dim across all inputs (true for every
    # engine in this repo); take the min of all per-input maxes to be safe.
    actual_depth = depth
    for n in input_names:
        _, _, max_s = engine.get_tensor_profile_shape(n, 0)
        if len(max_s) >= 1 and max_s[0] > 0:
            actual_depth = min(actual_depth, int(max_s[0]))

    keepalive = []
    chosen_shapes: dict[str, tuple] = {}
    for n in input_names:
        min_s, opt_s, max_s = engine.get_tensor_profile_shape(n, 0)
        shape = list(max_s)  # use max time/feature dims to stress VRAM
        if len(shape) >= 1:
            shape[0] = min(actual_depth, int(max_s[0]))
        # Make sure no dim is non-positive (defensive).
        shape = [max(int(s), 1) for s in shape]
        chosen_shapes[n] = tuple(shape)

        dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(n))
        t = _alloc_input(shape, dtype, device)
        ctx.set_input_shape(n, tuple(shape))
        ctx.set_tensor_address(n, t.data_ptr())
        keepalive.append(t)

    for n in output_names:
        out_shape = tuple(ctx.get_tensor_shape(n))
        out_shape = tuple(max(int(s), 1) for s in out_shape)
        out_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(n))
        out = torch.empty(out_shape, dtype=out_dtype, device=device)
        ctx.set_tensor_address(n, out.data_ptr())
        keepalive.append(out)

    stream = _get_stream()
    timings, peak_mb, post_alloc_mb = _run_iters(ctx, stream, vram_mb, post_load_mb)
    final_mb = vram_mb()

    # Hold refs until we report.
    _keepalive_holder = (engine, ctx, stream, keepalive)
    del _keepalive_holder

    return {
        "status": "OK",
        "depth_actual": int(actual_depth),
        "baseline_mb": round(baseline_mb, 1),
        "post_load_mb": round(post_load_mb, 1),
        "post_alloc_mb": round(post_alloc_mb, 1),
        "peak_mb": round(peak_mb, 1),
        "final_mb": round(final_mb, 1),
        "mean_ms": round(sum(timings) / len(timings), 2) if timings else -1,
        "min_ms": round(min(timings), 2) if timings else -1,
        "max_ms": round(max(timings), 2) if timings else -1,
        "n_iters": len(timings),
        "input_shapes": {k: list(v) for k, v in chosen_shapes.items()},
    }


def _alloc_input(shape, dtype, device):
    import torch
    shape = tuple(int(s) for s in shape)
    if dtype in (torch.float32, torch.float16, torch.float64):
        return torch.empty(shape, dtype=dtype, device=device).normal_()
    if dtype == torch.bfloat16:
        # normal_ supports bf16 in modern torch; fall back if it doesn't.
        try:
            return torch.empty(shape, dtype=dtype, device=device).normal_()
        except RuntimeError:
            f = torch.empty(shape, dtype=torch.float32, device=device).normal_()
            return f.to(torch.bfloat16)
    if dtype == torch.bool:
        return torch.zeros(shape, dtype=dtype, device=device)
    # int8 / int32 / int64 / fallback
    return torch.zeros(shape, dtype=dtype, device=device)


def _load_engine(engine_path: Path):
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes
    engine = engine_from_bytes(bytes_from_path(str(engine_path)))
    ctx = engine.create_execution_context()
    return engine, ctx


def _trt_dtype_to_torch(trt_dtype):
    import tensorrt as trt
    import torch
    table = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "bfloat16"):
        table[trt.bfloat16] = torch.bfloat16
    if hasattr(trt, "int64"):
        table[trt.int64] = torch.int64
    if hasattr(trt, "uint8"):
        table[trt.uint8] = torch.uint8
    return table.get(trt_dtype, torch.float32)


def _get_stream():
    from polygraphy import cuda as pg_cuda
    return pg_cuda.Stream()


def _run_iters(ctx, stream, vram_mb, post_load_mb):
    import torch
    peak_mb = post_load_mb
    post_alloc_mb = None
    timings: list[float] = []

    for i in range(3):
        if not ctx.execute_async_v3(stream.ptr):
            raise RuntimeError("TRT execute_async_v3 returned False (warmup)")
        stream.synchronize()
        cur = vram_mb()
        peak_mb = max(peak_mb, cur)
        if i == 0:
            post_alloc_mb = cur

    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if not ctx.execute_async_v3(stream.ptr):
            raise RuntimeError("TRT execute_async_v3 returned False (timed)")
        stream.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)
        peak_mb = max(peak_mb, vram_mb())

    if post_alloc_mb is None:
        post_alloc_mb = post_load_mb
    return timings, peak_mb, post_alloc_mb


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true",
                   help="internal: run a single engine measurement and emit JSON")
    p.add_argument("--engine", type=Path)
    p.add_argument("--depth", type=int)
    p.add_argument("--depths", default="2,4,8",
                   help="comma-separated list of depths to test (parent only)")
    p.add_argument("--filter", default=None,
                   help="only run engines whose dir-name contains this substring")
    args = p.parse_args()

    if args.child:
        if args.engine is None or args.depth is None:
            sys.exit("--child requires --engine and --depth")
        child_main(args.engine, args.depth)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
