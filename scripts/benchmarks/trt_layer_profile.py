"""Per-layer profile of a decoder TRT engine via trt.IProfiler.

Answers "where does the tick actually go?" at layer granularity —
attention vs GEMM vs norm/pointwise — which whole-engine timing
(``TRTDecoder.benchmark``) and tick timing can't see.

Caveat: engines whose graph Myelin fused into ``{ForeignNode[...]}``
blobs report those blobs as single opaque layers. The category rollup
calls this out explicitly so a "myelin 95%" result reads as "rebuild
with --profiling-verbosity detailed for more", not as an answer.

Usage:
    .venv/Scripts/python.exe scripts/benchmarks/trt_layer_profile.py
    .venv/Scripts/python.exe scripts/benchmarks/trt_layer_profile.py \
        --engine trt_engines/decoder_turbo_fp16_b8_60s/decoder_turbo_fp16_b8_60s.engine \
        --batch 4 --seq-len 750 --enc-len 200 --iters 50 --top 40
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

torch.set_grad_enabled(False)


# ── engine discovery ─────────────────────────────────────────────────

def find_decoder_engine(project_root: str) -> str | None:
    """Auto-detect a decoder TRT engine in trt_engines/, preferring turbo."""
    trt_dir = os.path.join(project_root, "trt_engines")
    if not os.path.isdir(trt_dir):
        return None
    candidates = []
    for name in sorted(os.listdir(trt_dir)):
        if "decoder" not in name or name.startswith("_"):
            continue
        engine = os.path.join(trt_dir, name, f"{name}.engine")
        if os.path.isfile(engine):
            candidates.append(engine)
    if not candidates:
        return None
    for c in candidates:
        if "turbo" in os.path.basename(c) and "xl" not in os.path.basename(c):
            return c
    return candidates[0]


# ── profiler ─────────────────────────────────────────────────────────

class LayerProfiler:
    """Accumulates per-layer times across executions.

    Instantiated lazily as a trt.IProfiler subclass because tensorrt
    import must happen after arg parsing (so --help works anywhere).
    """

    def __new__(cls):
        import tensorrt as trt

        class _Impl(trt.IProfiler):
            def __init__(self):
                trt.IProfiler.__init__(self)
                self.layer_ms: dict[str, float] = {}

            def report_layer_time(self, layer_name: str, ms: float) -> None:
                self.layer_ms[layer_name] = self.layer_ms.get(layer_name, 0.0) + ms

        return _Impl()


# Order matters: first match wins. "myelin" before "gemm" so fused
# ForeignNode blobs don't get claimed by a substring of their contents.
CATEGORIES: list[tuple[str, re.Pattern]] = [
    ("myelin-fused", re.compile(r"ForeignNode|myelin", re.I)),
    ("attention", re.compile(r"attn|attention|softmax|mha|bmm", re.I)),
    ("gemm", re.compile(r"gemm|matmul|linear|\bfc\b|conv", re.I)),
    ("norm", re.compile(r"norm", re.I)),
    ("pointwise", re.compile(r"PWN|pointwise|elementwise|activation|silu|gelu|sigmoid|\bmul\b|\badd\b|cast|shuffle|reshape|slice|concat", re.I)),
]


def categorize(layer_name: str) -> str:
    for cat, pat in CATEGORIES:
        if pat.search(layer_name):
            return cat
    return "other"


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Per-layer TRT decoder profile")
    parser.add_argument("--engine", default=None,
                        help="Path to decoder engine (auto-detected if omitted)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size, i.e. active ring slots (default: 4)")
    parser.add_argument("--seq-len", type=int, default=750,
                        help="Latent frames T (default: 750 = 30s)")
    parser.add_argument("--enc-len", type=int, default=200,
                        help="Encoder sequence length L (default: 200)")
    parser.add_argument("--t-value", type=float, default=0.5,
                        help="Timestep value fed to every row (default: 0.5)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--top", type=int, default=40,
                        help="How many layers to print (default: 40; 0 = all)")
    args = parser.parse_args()

    import tensorrt as trt  # noqa: F401  (registers plugins on import)
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    engine_path = args.engine or find_decoder_engine(project_root)
    if engine_path is None:
        print("No decoder engine found in trt_engines/. Pass --engine.")
        sys.exit(1)

    device = torch.device("cuda")
    B = args.batch
    # Decoder engines require even T (the stream pads odd T the same way).
    T = args.seq_len + (args.seq_len % 2)
    L = args.enc_len

    print(f"engine: {engine_path}")
    print(f"shapes: B={B} T={T} L={L}  t={args.t_value}")

    engine = engine_from_bytes(bytes_from_path(engine_path))
    context = engine.create_execution_context()

    trt_to_torch = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "bfloat16"):
        trt_to_torch[trt.bfloat16] = torch.bfloat16

    # Enumerate I/O from the engine itself so the script works on both
    # spectral engines (extra "steering" input) and older ones.
    input_names, output_names = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    def resolved_shape(name: str) -> tuple[int, ...]:
        dims = list(engine.get_tensor_shape(name))
        if name == "hidden_states":
            return (B, T, dims[2])
        if name == "timestep":
            return (B,)
        if name == "encoder_hidden_states":
            return (B, L, dims[2])
        if name == "context_latents":
            return (B, T, dims[2])
        if name == "steering":
            return (B, dims[1], dims[2])  # L, D static on spectral engines
        raise RuntimeError(
            f"Unknown engine input {name!r} with shape {dims}; "
            "teach resolved_shape() about it."
        )

    bufs: dict[str, torch.Tensor] = {}
    for name in input_names:
        shape = resolved_shape(name)
        dt = trt_to_torch.get(engine.get_tensor_dtype(name), torch.float32)
        if name == "timestep":
            buf = torch.full(shape, args.t_value, dtype=dt, device=device)
        elif name == "steering":
            buf = torch.zeros(shape, dtype=dt, device=device)
        else:
            buf = torch.randn(shape, dtype=torch.float32, device=device).to(dt)
        bufs[name] = buf
        if not context.set_input_shape(name, shape):
            raise RuntimeError(f"engine rejected input shape for {name}: {shape}")
        if not context.set_tensor_address(name, buf.data_ptr()):
            raise RuntimeError(f"engine rejected input address for {name}")

    missing = context.infer_shapes()
    if missing:
        raise RuntimeError(f"shapes insufficiently specified: {missing}")

    for name in output_names:
        shape = tuple(context.get_tensor_shape(name))
        dt = trt_to_torch.get(engine.get_tensor_dtype(name), torch.float32)
        buf = torch.empty(shape, dtype=dt, device=device)
        bufs[name] = buf
        if not context.set_tensor_address(name, buf.data_ptr()):
            raise RuntimeError(f"engine rejected output address for {name}")

    stream = torch.cuda.Stream()

    def execute() -> None:
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TRT execution failed")
        stream.synchronize()

    for _ in range(args.warmup):
        execute()

    # Unprofiled wall time first: attaching IProfiler serializes layer
    # execution, so profiled totals overstate the real latency.
    import time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        execute()
    wall_ms = (time.perf_counter() - t0) * 1000 / args.iters

    prof = LayerProfiler()
    context.profiler = prof
    for _ in range(args.iters):
        execute()

    layer_avg = {name: total / args.iters for name, total in prof.layer_ms.items()}
    profiled_ms = sum(layer_avg.values())

    print(f"\nwall (unprofiled): {wall_ms:.2f} ms/iter")
    print(f"profiled layer sum: {profiled_ms:.2f} ms/iter "
          f"({len(layer_avg)} layers; profiling adds serialization overhead)\n")

    rollup: dict[str, float] = {}
    for name, ms in layer_avg.items():
        rollup[categorize(name)] = rollup.get(categorize(name), 0.0) + ms

    print(f"{'category':<14s} {'ms/iter':>9s} {'share':>7s}")
    print("-" * 32)
    for cat, ms in sorted(rollup.items(), key=lambda kv: -kv[1]):
        print(f"{cat:<14s} {ms:>9.3f} {ms / profiled_ms:>6.1%}")
    if rollup.get("myelin-fused", 0.0) / max(profiled_ms, 1e-9) > 0.5:
        print("\nNOTE: most time is inside opaque Myelin ForeignNode blobs."
              "\nRebuild the engine with profiling verbosity 'detailed' to"
              "\nsplit them before drawing conclusions.")

    top = args.top if args.top > 0 else len(layer_avg)
    print(f"\ntop {min(top, len(layer_avg))} layers:")
    print(f"{'ms/iter':>9s} {'share':>7s}  layer")
    print("-" * 80)
    for name, ms in sorted(layer_avg.items(), key=lambda kv: -kv[1])[:top]:
        print(f"{ms:>9.3f} {ms / profiled_ms:>6.1%}  {name}")


if __name__ == "__main__":
    main()
