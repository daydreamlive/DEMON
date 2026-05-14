"""NVFP4 feasibility stub: can TRT 10.16 build & run an NVFP4 GEMM on RTX 5090?

Builds a tiny ONNX with one MatMul where both inputs flow through NVFP4
Q-DQ chains:

    bf16 activation -> QuantizeLinear (FLOAT4E2M1, per-block FP8 scale, block_size=16)
                    -> DequantizeLinear -> bf16
                    \\
                     MatMul -> bf16 output
                    /
    FLOAT4E2M1 weight initializer with per-block FP8 scale
                    -> DequantizeLinear -> bf16

Then parses with TRT, builds an engine with VERBOSE logging, and dumps:
  - Whether the build succeeded
  - Whether 'FP4' / 'NVFP4' / 'E2M1' appears in the build log (tactic selection)
  - Engine size
  - Forward latency on RTX 5090

If the build fails or the tactic isn't picked, this tells us NVFP4 isn't
viable for our DiT graph as-is.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import tensorrt as trt
import torch


# Small but realistic shapes (~3072 -> 3072 matches DiT MLP gate_proj dims).
M = 6000      # batch * seq = 4 * 1500
K = 3072      # in_features
N = 3072      # out_features
BLOCK = 16    # NVFP4 block size along K

OUT_DIR = Path(__file__).parent / "nvfp4_stub_out"
OUT_DIR.mkdir(exist_ok=True)
ONNX_PATH = OUT_DIR / "nvfp4_stub.onnx"
ENGINE_PATH = OUT_DIR / "nvfp4_stub.engine"
LOG_PATH = OUT_DIR / "nvfp4_stub_build.log"


# ------------------------------------------------------------------
# FP4 weight + scale generation
# ------------------------------------------------------------------

def _make_fp4_weight_and_scales(K: int, N: int, block: int):
    """Return (packed_bytes, weight_dims, scale_fp8_bytes, scale_dims).

    weight_dims is (K, N) for ONNX MatMul layout. FLOAT4E2M1 in ONNX
    expects raw_data with 2 fp4 values per byte in row-major order
    (low nibble = first element). For this stub we generate random
    bytes — TRT only needs them to parse and pick a tactic, not be
    numerically meaningful.
    """
    assert K % block == 0
    assert (K * N) % 2 == 0, "Total FP4 elements must be even to pack"
    rng = np.random.default_rng(0)
    # K*N FP4 values -> K*N/2 bytes. Random bytes are valid FP4
    # encodings (any 4-bit pattern is a valid FP4 E2M1 value).
    raw_bytes = rng.integers(0, 256, size=K * N // 2, dtype=np.uint8).tobytes()
    weight_dims = (K, N)

    # FP8 E4M3FN scale, one per block along K. Use realistic small values.
    scale_fp32 = torch.full(
        (K // block, N), 0.05, dtype=torch.float32,
    )
    scale_fp8 = scale_fp32.to(torch.float8_e4m3fn).contiguous()
    scale_bytes = scale_fp8.view(torch.uint8).numpy().tobytes()
    scale_dims = (K // block, N)

    return raw_bytes, weight_dims, scale_bytes, scale_dims


def _make_fp8_zero_point_initializer(name: str) -> TensorProto:
    init = TensorProto()
    init.name = name
    init.data_type = int(TensorProto.FLOAT8E4M3FN)
    init.raw_data = bytes([0])
    return init


def _make_fp4_zero_point_initializer(name: str, dims: tuple = ()) -> TensorProto:
    """FLOAT4E2M1 zero_point. Scalar by default; per-block when dims given."""
    init = TensorProto()
    init.name = name
    init.data_type = int(TensorProto.FLOAT4E2M1)
    if dims:
        n_values = 1
        for d in dims:
            n_values *= d
        # Packed: 2 fp4 per byte, all zeros = byte 0x00.
        n_bytes = (n_values + 1) // 2
        init.dims.extend(list(dims))
        init.raw_data = bytes(n_bytes)
    else:
        init.raw_data = bytes([0])
    return init


def _make_scalar_fp8_scale_initializer(name: str, value: float) -> TensorProto:
    """Per-tensor FP8 E4M3FN scale (scalar)."""
    t = torch.tensor(value, dtype=torch.float8_e4m3fn)
    raw = t.view(torch.uint8).numpy().tobytes()
    init = TensorProto()
    init.name = name
    init.data_type = int(TensorProto.FLOAT8E4M3FN)
    init.raw_data = raw
    return init


def _make_blocked_fp8_scale_initializer(name: str, scale_bytes: bytes, dims: tuple) -> TensorProto:
    init = TensorProto()
    init.name = name
    init.data_type = int(TensorProto.FLOAT8E4M3FN)
    init.dims.extend(list(dims))
    init.raw_data = scale_bytes
    return init


def _make_blocked_fp32_scale_initializer(name: str, dims: tuple, value: float = 0.05) -> TensorProto:
    """For probing whether TRT prefers FP32 scales over FP8 for FP4 DQ."""
    arr = np.full(dims, value, dtype=np.float32)
    init = TensorProto()
    init.name = name
    init.data_type = int(TensorProto.FLOAT)
    init.dims.extend(list(dims))
    init.raw_data = arr.tobytes()
    return init


def _make_blocked_fp16_scale_initializer(name: str, dims: tuple, value: float = 0.05) -> TensorProto:
    arr = np.full(dims, value, dtype=np.float16)
    init = TensorProto()
    init.name = name
    init.data_type = int(TensorProto.FLOAT16)
    init.dims.extend(list(dims))
    init.raw_data = arr.tobytes()
    return init


def _make_blocked_bf16_scale_initializer(name: str, dims: tuple, value: float = 0.05) -> TensorProto:
    t = torch.full(dims, value, dtype=torch.bfloat16).contiguous()
    raw = t.view(torch.uint16).numpy().tobytes()
    init = TensorProto()
    init.name = name
    init.data_type = int(TensorProto.BFLOAT16)
    init.dims.extend(list(dims))
    init.raw_data = raw
    return init


def _make_fp4_weight_initializer(name: str, raw_bytes: bytes, dims: tuple) -> TensorProto:
    init = TensorProto()
    init.name = name
    init.data_type = int(TensorProto.FLOAT4E2M1)
    init.dims.extend(list(dims))
    init.raw_data = raw_bytes
    return init


# ------------------------------------------------------------------
# Build the stub ONNX
# ------------------------------------------------------------------

def build_bf16_baseline_onnx() -> Path:
    """Plain bf16 MatMul (no quant) as the throughput baseline."""
    input_t = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [M, K])
    output_t = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [M, N])
    rng = np.random.default_rng(0)
    w_bf16 = torch.from_numpy(rng.standard_normal((K, N)).astype(np.float32)).to(torch.bfloat16).contiguous()
    raw = w_bf16.view(torch.uint16).numpy().tobytes()
    w_init = TensorProto()
    w_init.name = "weight"
    w_init.data_type = int(TensorProto.BFLOAT16)
    w_init.dims.extend([K, N])
    w_init.raw_data = raw
    mm = helper.make_node("MatMul", inputs=["input", "weight"], outputs=["output"], name="MatMul")
    graph = helper.make_graph(
        nodes=[mm], name="bf16_baseline",
        inputs=[input_t], outputs=[output_t], initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    model.ir_version = 10
    out = OUT_DIR / "bf16_baseline.onnx"
    onnx.save(model, str(out))
    print(f"[onnx] wrote {out} ({out.stat().st_size/1024:.1f} KB)  mode=BF16-baseline")
    return out


def build_modelopt_pattern_onnx() -> Path:
    """The canonical NVFP4 pattern from ModelOpt's nvfp4_exporter.

    Weight side (W4A16): two-DQ structure
      sw_f8_per_block  (FP8E4M3)   [K/BLOCK, N]
      sw_f32_per_tensor (FP32)     []
      DQ1: (sw_f8_per_block, sw_f32_per_tensor) -> sw_f32_combined (FP32 [K/BLOCK, N])
      DQ2: (w_f4, sw_f32_combined) axis=0 block_size=BLOCK -> w_fp32 [K, N]
      Cast: w_fp32 -> w_bf16
      MatMul: input_bf16 @ w_bf16 -> output_bf16
    """
    input_t = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [M, K])
    output_t = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [M, N])

    # Weight FP4 + FP8 per-block scale + FP32 per-tensor scale.
    w_bytes, w_dims, w_scale_bytes, w_scale_dims = _make_fp4_weight_and_scales(K, N, BLOCK)
    w_init = _make_fp4_weight_initializer("weight_f4", w_bytes, w_dims)
    sw_f8_init = _make_blocked_fp8_scale_initializer(
        "sw_f8_per_block", w_scale_bytes, w_scale_dims,
    )
    # Per-tensor FP32 scale. Single scalar.
    sw_f32_per_tensor = np.array(1.0, dtype=np.float32)
    sw_f32_init = TensorProto()
    sw_f32_init.name = "sw_f32_per_tensor"
    sw_f32_init.data_type = int(TensorProto.FLOAT)
    sw_f32_init.raw_data = sw_f32_per_tensor.tobytes()

    nodes = [
        # DQ1: (sw_f8_per_block, sw_f32_per_tensor) -> sw_f32_combined
        # Both inputs only — DequantizeLinear can omit zero_point.
        helper.make_node(
            "DequantizeLinear",
            inputs=["sw_f8_per_block", "sw_f32_per_tensor"],
            outputs=["sw_f32_combined"],
            name="ScaleDQ",
        ),
        # DQ2: (w_f4, sw_f32_combined) -> w_fp32
        helper.make_node(
            "DequantizeLinear",
            inputs=["weight_f4", "sw_f32_combined"],
            outputs=["weight_fp32"],
            name="WeightDQ",
            axis=0,
            block_size=BLOCK,
        ),
        # Cast to bf16 for MatMul.
        helper.make_node(
            "Cast",
            inputs=["weight_fp32"],
            outputs=["weight_bf16"],
            name="WeightCast",
            to=int(TensorProto.BFLOAT16),
        ),
        helper.make_node(
            "MatMul",
            inputs=["input", "weight_bf16"],
            outputs=["output"],
            name="MatMul",
        ),
    ]
    initializers = [w_init, sw_f8_init, sw_f32_init]
    graph = helper.make_graph(
        nodes=nodes,
        name="nvfp4_modelopt_pattern",
        inputs=[input_t],
        outputs=[output_t],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    model.ir_version = 10
    onnx.save(model, str(ONNX_PATH))
    print(f"[onnx] wrote {ONNX_PATH} ({ONNX_PATH.stat().st_size/1024:.1f} KB)  mode=MODELOPT-2DQ")
    return ONNX_PATH


def build_stub_onnx(*, with_act_quant: bool = False, scale_dtype: str = "fp32") -> Path:
    """W4A16 (default) or W4A4 stub.

    W4A16:
      bf16 input -> MatMul <- DequantizeLinear <- FP4 weight + per-block scale

    W4A4 (with_act_quant=True):
      bf16 input -> QuantizeLinear (block) -> FP4
                 -> DequantizeLinear -> bf16
                 -> MatMul <- DequantizeLinear <- FP4 weight + per-block scale

    ``scale_dtype`` controls the per-block scale tensor's dtype:
      "fp8"  : FP8E4M3FN (NVFP4 native scale type)
      "fp32" : FP32 (TRT may prefer this for FP4 DQ)
      "fp16" : FP16
      "bf16" : BFLOAT16
    """
    # Input: [M, K] bf16
    input_t = helper.make_tensor_value_info(
        "input", TensorProto.BFLOAT16, [M, K],
    )
    output_t = helper.make_tensor_value_info(
        "output", TensorProto.BFLOAT16, [M, N],
    )

    nodes = []
    initializers = []

    # Weight FP4 init + per-block scale (dtype configurable).
    w_bytes, w_dims, w_scale_bytes, w_scale_dims = _make_fp4_weight_and_scales(K, N, BLOCK)
    w_init = _make_fp4_weight_initializer("weight", w_bytes, w_dims)
    if scale_dtype == "fp8":
        w_scale_init = _make_blocked_fp8_scale_initializer(
            "weight_scale", w_scale_bytes, w_scale_dims,
        )
    elif scale_dtype == "fp32":
        w_scale_init = _make_blocked_fp32_scale_initializer("weight_scale", w_scale_dims)
    elif scale_dtype == "fp16":
        w_scale_init = _make_blocked_fp16_scale_initializer("weight_scale", w_scale_dims)
    elif scale_dtype == "bf16":
        w_scale_init = _make_blocked_bf16_scale_initializer("weight_scale", w_scale_dims)
    else:
        raise ValueError(f"unknown scale_dtype: {scale_dtype}")
    w_zp_init = _make_fp4_zero_point_initializer("weight_zp", dims=w_scale_dims)
    initializers.extend([w_init, w_scale_init, w_zp_init])

    matmul_lhs_name = "input"
    if with_act_quant:
        # Activation per-block scale: [M, K/block]. Use same dtype as weight scale.
        act_scale_dims = (M, K // BLOCK)
        if scale_dtype == "fp8":
            act_scale_const = torch.full(act_scale_dims, 0.5, dtype=torch.float8_e4m3fn).contiguous()
            act_scale_bytes = act_scale_const.view(torch.uint8).numpy().tobytes()
            act_scale_init = _make_blocked_fp8_scale_initializer("act_scale", act_scale_bytes, act_scale_dims)
        elif scale_dtype == "fp32":
            act_scale_init = _make_blocked_fp32_scale_initializer("act_scale", act_scale_dims, value=0.1)
        elif scale_dtype == "fp16":
            act_scale_init = _make_blocked_fp16_scale_initializer("act_scale", act_scale_dims, value=0.1)
        elif scale_dtype == "bf16":
            act_scale_init = _make_blocked_bf16_scale_initializer("act_scale", act_scale_dims, value=0.1)
        act_zp_init = _make_fp4_zero_point_initializer("act_zp", dims=act_scale_dims)
        initializers.extend([act_scale_init, act_zp_init])

        # If scale is non-bf16, we need to Cast input to scale dtype first
        # so Quantize accepts it.
        if scale_dtype != "bf16":
            q_input_name = "input_cast_for_q"
            cast_target = {
                "fp32": int(TensorProto.FLOAT),
                "fp16": int(TensorProto.FLOAT16),
                "fp8": int(TensorProto.FLOAT8E4M3FN),  # likely won't work
            }[scale_dtype]
            nodes.append(helper.make_node(
                "Cast", inputs=["input"], outputs=[q_input_name],
                name="InputCastForQ", to=cast_target,
            ))
        else:
            q_input_name = "input"

        nodes.append(helper.make_node(
            "QuantizeLinear",
            inputs=[q_input_name, "act_scale", "act_zp"],
            outputs=["act_fp4"],
            name="ActQ",
            axis=-1,
            block_size=BLOCK,
        ))
        nodes.append(helper.make_node(
            "DequantizeLinear",
            inputs=["act_fp4", "act_scale", "act_zp"],
            outputs=["act_dq_raw"],
            name="ActDQ",
            axis=-1,
            block_size=BLOCK,
        ))
        # Cast back to bf16 so MatMul inputs match.
        if scale_dtype != "bf16":
            nodes.append(helper.make_node(
                "Cast", inputs=["act_dq_raw"], outputs=["act_dq"],
                name="ActDQCast", to=int(TensorProto.BFLOAT16),
            ))
        else:
            nodes.append(helper.make_node(
                "Identity", inputs=["act_dq_raw"], outputs=["act_dq"],
                name="ActDQIdent",
            ))
        matmul_lhs_name = "act_dq"

    # DQ outputs same dtype as scale. We Cast to BF16 afterward to match MatMul input.
    nodes.append(helper.make_node(
        "DequantizeLinear",
        inputs=["weight", "weight_scale", "weight_zp"],
        outputs=["weight_dq_raw"],
        name="WeightDQ",
        axis=0,
        block_size=BLOCK,
    ))
    nodes.append(helper.make_node(
        "Cast",
        inputs=["weight_dq_raw"],
        outputs=["weight_dq"],
        name="WeightDQCast",
        to=int(TensorProto.BFLOAT16),
    ))
    nodes.append(helper.make_node(
        "MatMul",
        inputs=[matmul_lhs_name, "weight_dq"],
        outputs=["output"],
        name="MatMul",
    ))

    graph = helper.make_graph(
        nodes=nodes,
        name="nvfp4_stub_w4a4" if with_act_quant else "nvfp4_stub_w4a16",
        inputs=[input_t],
        outputs=[output_t],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 23)],  # opset 23 has block_size + FP4
    )
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, str(ONNX_PATH))
    print(f"[onnx] wrote {ONNX_PATH} ({ONNX_PATH.stat().st_size/1024:.1f} KB)  mode={'W4A4' if with_act_quant else 'W4A16'}")
    return ONNX_PATH


# ------------------------------------------------------------------
# Build with TRT and inspect
# ------------------------------------------------------------------

class _CapturingLogger(trt.ILogger):
    def __init__(self, severity=trt.ILogger.VERBOSE):
        super().__init__()
        self.min_severity = severity
        self.lines: list[str] = []

    def log(self, severity, msg):
        if severity <= self.min_severity:
            self.lines.append(f"[{severity.name}] {msg}")


def build_with_trt(onnx_path: Path, *, strongly_typed: bool = True) -> tuple[bytes, list[str]]:
    logger = _CapturingLogger(trt.ILogger.VERBOSE)
    builder = trt.Builder(logger)
    flags = 0
    if strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f"[parser] ERROR: {parser.get_error(i)}")
        raise RuntimeError("ONNX parse failed")

    cfg = builder.create_builder_config()
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
    # Precision flags only valid for non-strongly-typed networks.
    if not strongly_typed:
        cfg.set_flag(trt.BuilderFlag.FP4)
        cfg.set_flag(trt.BuilderFlag.FP8)
        cfg.set_flag(trt.BuilderFlag.BF16)

    print("[trt] building engine ...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, cfg)
    dt = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError("TRT build returned None")
    raw = bytes(serialized)
    print(f"[trt] built in {dt:.1f}s, {len(raw)/1024:.1f} KB serialized")
    return raw, logger.lines


def benchmark_engine(engine_bytes: bytes) -> dict:
    logger = trt.Logger(trt.Logger.WARNING)
    rt = trt.Runtime(logger)
    engine = rt.deserialize_cuda_engine(engine_bytes)
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()
    dev = torch.device("cuda")

    inp = torch.randn(M, K, device=dev, dtype=torch.bfloat16).contiguous()
    ctx.set_input_shape("input", (M, K))
    ctx.set_tensor_address("input", inp.data_ptr())
    miss = ctx.infer_shapes()
    if miss:
        raise RuntimeError(f"shapes underspecified: {miss}")
    out_shape = tuple(ctx.get_tensor_shape("output"))
    out = torch.empty(out_shape, dtype=torch.bfloat16, device=dev)
    ctx.set_tensor_address("output", out.data_ptr())

    # Warmup
    for _ in range(10):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    # Bench
    iters = 200
    t0 = time.perf_counter()
    for _ in range(iters):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1000
    # FLOPS estimate
    flops = 2 * M * K * N
    tflops = flops / (dt / 1000) / 1e12
    return {
        "tick_ms": dt,
        "tflops": tflops,
        "out_shape": out_shape,
    }


def main() -> None:
    print("=" * 70)
    print("NVFP4 feasibility stub")
    print("=" * 70)
    print(f"shapes: M={M} K={K} N={N} block={BLOCK}")
    print(f"FLOPS per matmul: {2*M*K*N/1e9:.2f} G")
    print()

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--w4a4", action="store_true",
                    help="Also Q-DQ activations to FP4 (default: W4A16).")
    ap.add_argument("--scale", choices=("fp8", "fp32", "fp16", "bf16"),
                    default="fp32",
                    help="Per-block scale dtype (default: fp32).")
    ap.add_argument("--mode", choices=("nvfp4", "bf16", "modelopt"),
                    default="nvfp4",
                    help="'bf16' builds a plain bf16 MatMul; 'modelopt' uses ModelOpt's 2-DQ pattern.")
    ap.add_argument("--non-strongly-typed", action="store_true",
                    help="Use non-strongly-typed network + FP4/FP8 builder flags.")
    args = ap.parse_args()

    if args.mode == "bf16":
        onnx_path = build_bf16_baseline_onnx()
    elif args.mode == "modelopt":
        onnx_path = build_modelopt_pattern_onnx()
    else:
        onnx_path = build_stub_onnx(with_act_quant=args.w4a4, scale_dtype=args.scale)
    engine_bytes, log_lines = build_with_trt(onnx_path, strongly_typed=not args.non_strongly_typed)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"[log] {len(log_lines)} lines -> {LOG_PATH}")
    ENGINE_PATH.write_bytes(engine_bytes)
    print(f"[engine] wrote {ENGINE_PATH} ({len(engine_bytes)/1024:.1f} KB)")

    # Scan log for tactic / precision hints.
    print()
    print("Log keywords (case-insensitive):")
    hits = {"fp4": 0, "nvfp4": 0, "e2m1": 0, "fp8": 0, "blocked": 0, "block_size": 0, "mxfp": 0, "tactic": 0}
    for line in log_lines:
        for k in hits:
            if k in line.lower():
                hits[k] += 1
    for k, v in hits.items():
        print(f"  '{k}': {v} occurrences")

    print()
    print("Interesting log lines:")
    keep_pat = re.compile(r"fp4|nvfp4|e2m1|fp8|tactic|blocked|block_size|fallback", re.I)
    interesting = [ln for ln in log_lines if keep_pat.search(ln)]
    for ln in interesting[:60]:
        print(f"  {ln[:300]}")

    print()
    print("Benchmark:")
    try:
        bench = benchmark_engine(engine_bytes)
        print(f"  tick: {bench['tick_ms']:.3f} ms")
        print(f"  TFLOPS: {bench['tflops']:.1f}")
        print(f"  RTX 5090 peaks: FP8={trt.__name__ and 330} TFLOPS  NVFP4={660} TFLOPS")
        print(f"  Utilization: of FP8 peak {bench['tflops']/330*100:.1f}%, of NVFP4 peak {bench['tflops']/660*100:.1f}%")
    except Exception as e:
        print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
