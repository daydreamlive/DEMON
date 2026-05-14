"""FP8 W8A8 stub mirroring our production fp8_onnx.py pattern.

Builds a single MatMul where:
  - Weight: FP8 E4M3FN initializer with per-output-channel BF16 scale
  - Activation: bf16 -> QuantizeLinear (per-tensor BF16 scale, FP8 zp)
              -> DequantizeLinear -> bf16
  - MatMul: bf16 @ bf16 -> bf16

This matches the pattern fp8_onnx.py emits for our DiT MatMuls. Builds
with VERBOSE logging so we can see what GEMM tactic TRT picks — answering
the gating question for the FP8-plugin step-function bet.

Compares against:
  - bf16 baseline (no quant)
  - torch._scaled_mm FP8 (Blackwell tensor cores, known fast)
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
from onnx import TensorProto, helper
import tensorrt as trt
import torch

OUT_DIR = Path(__file__).parent / "fp8_stub_out"
OUT_DIR.mkdir(exist_ok=True)

# DiT MLP-ish shape (M aligned to 128).
M = 6144
K = 3072
N = 3072


def _bf16_init(name: str, t_bf16: torch.Tensor) -> TensorProto:
    raw = t_bf16.contiguous().view(torch.uint16).numpy().tobytes()
    init = TensorProto()
    init.name = name
    init.data_type = int(TensorProto.BFLOAT16)
    init.dims.extend(list(t_bf16.shape))
    init.raw_data = raw
    return init


def _scalar_bf16_init(name: str, value: float) -> TensorProto:
    t = torch.tensor(value, dtype=torch.bfloat16)
    raw = t.view(torch.uint16).numpy().tobytes()
    init = TensorProto()
    init.name = name
    init.data_type = int(TensorProto.BFLOAT16)
    init.raw_data = raw
    return init


def _fp8_scalar_zp(name: str) -> TensorProto:
    init = TensorProto()
    init.name = name
    init.data_type = int(TensorProto.FLOAT8E4M3FN)
    init.raw_data = bytes([0])
    return init


def build_fp8_w8a8_onnx() -> Path:
    """Mirror of our production fp8_onnx.py emission:
       - FP8 weight + per-output-channel BF16 scale
       - bf16 activation -> Q (per-tensor BF16 scale, FP8 scalar zp) -> DQ -> bf16
       - MatMul bf16 @ bf16 -> bf16
    """
    input_t = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [M, K])
    output_t = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [M, N])

    # Weight (FP8 E4M3 packed bytes). One byte per element.
    rng = np.random.default_rng(0)
    w_fp32 = torch.from_numpy(rng.standard_normal((K, N)).astype(np.float32) * 0.05)
    # Per-output-channel scale (along last axis = N).
    scale_fp32 = w_fp32.abs().amax(dim=0).clamp(min=1e-6) / 448.0  # [N]
    w_scaled = w_fp32 / scale_fp32.unsqueeze(0)
    w_fp8 = w_scaled.to(torch.float8_e4m3fn).contiguous()
    w_raw = w_fp8.view(torch.uint8).numpy().tobytes()

    w_init = TensorProto()
    w_init.name = "weight"
    w_init.data_type = int(TensorProto.FLOAT8E4M3FN)
    w_init.dims.extend([K, N])
    w_init.raw_data = w_raw

    # Scale as bf16, shape [N], axis=-1 in DequantizeLinear.
    scale_bf16 = scale_fp32.to(torch.bfloat16)
    w_scale_init = _bf16_init("weight_scale", scale_bf16)
    w_zp_init = _fp8_scalar_zp("weight_zp")

    # Per-tensor activation Q-DQ.
    act_scale_init = _scalar_bf16_init("act_scale", 0.5)
    act_zp_init = _fp8_scalar_zp("act_zp")

    nodes = [
        helper.make_node(
            "QuantizeLinear",
            inputs=["input", "act_scale", "act_zp"],
            outputs=["input_fp8"],
            name="ActQ",
        ),
        helper.make_node(
            "DequantizeLinear",
            inputs=["input_fp8", "act_scale", "act_zp"],
            outputs=["input_dq"],
            name="ActDQ",
        ),
        helper.make_node(
            "DequantizeLinear",
            inputs=["weight", "weight_scale", "weight_zp"],
            outputs=["weight_dq"],
            name="WeightDQ",
            axis=-1,
        ),
        helper.make_node(
            "MatMul",
            inputs=["input_dq", "weight_dq"],
            outputs=["output"],
            name="MatMul",
        ),
    ]
    graph = helper.make_graph(
        nodes=nodes, name="fp8_w8a8_stub",
        inputs=[input_t], outputs=[output_t],
        initializer=[w_init, w_scale_init, w_zp_init, act_scale_init, act_zp_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    out = OUT_DIR / "fp8_w8a8_stub.onnx"
    onnx.save(model, str(out))
    print(f"[onnx] wrote {out} ({out.stat().st_size/1024:.1f} KB)")
    return out


def build_bf16_baseline_onnx() -> Path:
    inp = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [M, K])
    out = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [M, N])
    rng = np.random.default_rng(0)
    w = torch.from_numpy(rng.standard_normal((K, N)).astype(np.float32) * 0.05).to(torch.bfloat16).contiguous()
    w_init = _bf16_init("weight", w)
    mm = helper.make_node("MatMul", inputs=["input", "weight"], outputs=["output"], name="MatMul")
    g = helper.make_graph([mm], "bf16", [inp], [out], [w_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])
    m.ir_version = 10
    p = OUT_DIR / "bf16_baseline.onnx"
    onnx.save(m, str(p))
    return p


class _CapLog(trt.ILogger):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []
    def log(self, sev, msg):
        self.lines.append(f"[{sev.name}] {msg}")


def build_and_inspect(onnx_path: Path, *, tag: str) -> dict:
    print(f"\n[{tag}] building")
    logger = _CapLog()
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f"[{tag}] parse error: {parser.get_error(i)}")
        return {"ok": False}

    cfg = builder.create_builder_config()
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        return {"ok": False}
    raw = bytes(serialized)
    log_path = OUT_DIR / f"{tag}_build.log"
    log_path.write_text("\n".join(logger.lines), encoding="utf-8")

    # Parse log for tactic names on MatMul/gemm layers.
    tactic_pat = re.compile(r"TacticName:\s*(\S+)", re.I)
    layer_pat = re.compile(r"LayerType:\s*(\S+),\s*Inputs", re.I)
    matmul_tactics = []
    for ln in logger.lines:
        if "LayerType: gemm" in ln or "LayerType: kgen" in ln:
            tac_m = tactic_pat.search(ln)
            if tac_m:
                matmul_tactics.append(tac_m.group(1))

    print(f"[{tag}] log -> {log_path}")
    print(f"[{tag}] matmul/kgen tactics found: {len(matmul_tactics)}")
    for t in matmul_tactics:
        print(f"  {t[:160]}")

    # Benchmark.
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = rt.deserialize_cuda_engine(raw)
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()
    dev = torch.device("cuda")
    inp = torch.randn(M, K, device=dev, dtype=torch.bfloat16).contiguous()
    ctx.set_input_shape("input", (M, K))
    ctx.set_tensor_address("input", inp.data_ptr())
    if ctx.infer_shapes():
        print(f"[{tag}] shapes underspecified")
        return {"ok": False}
    out_shape = tuple(ctx.get_tensor_shape("output"))
    out = torch.empty(out_shape, dtype=torch.bfloat16, device=dev)
    ctx.set_tensor_address("output", out.data_ptr())
    for _ in range(10):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    iters = 200
    t0 = time.perf_counter()
    for _ in range(iters):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1000
    tflops = 2 * M * K * N / (dt / 1000) / 1e12
    print(f"[{tag}] tick {dt:.3f} ms  {tflops:.1f} TFLOPS")
    return {"tick_ms": dt, "tflops": tflops, "tactics": matmul_tactics}


def main():
    print("=" * 72)
    print(f"FP8 W8A8 stub matching production fp8_onnx.py pattern  ({M}x{K}x{N})")
    print("=" * 72)

    bf16_path = build_bf16_baseline_onnx()
    fp8_path = build_fp8_w8a8_onnx()

    bf16_r = build_and_inspect(bf16_path, tag="bf16_baseline")
    fp8_r = build_and_inspect(fp8_path, tag="fp8_w8a8")

    # Compare against torch._scaled_mm FP8.
    print(f"\n[torch._scaled_mm FP8] reference call")
    dev = torch.device("cuda")
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.randn(K, N, device=dev, dtype=torch.bfloat16).t().contiguous().to(torch.float8_e4m3fn).t()
    sa = torch.tensor(1.0, device=dev, dtype=torch.float32)
    sb = torch.tensor(1.0, device=dev, dtype=torch.float32)
    for _ in range(20):
        torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    dt_torch = (time.perf_counter() - t0) / 200 * 1000
    tf_torch = 2 * M * K * N / (dt_torch / 1000) / 1e12
    print(f"[torch._scaled_mm FP8]   {dt_torch:.3f} ms  {tf_torch:.1f} TFLOPS")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if bf16_r.get("tick_ms"):
        print(f"  TRT bf16 baseline    : {bf16_r['tick_ms']:.3f} ms  {bf16_r['tflops']:.1f} TFLOPS")
    if fp8_r.get("tick_ms"):
        print(f"  TRT fp8 W8A8 (prod pattern): {fp8_r['tick_ms']:.3f} ms  {fp8_r['tflops']:.1f} TFLOPS")
    print(f"  torch._scaled_mm FP8 : {dt_torch:.3f} ms  {tf_torch:.1f} TFLOPS")
    if bf16_r.get("tick_ms") and fp8_r.get("tick_ms"):
        print(f"  TRT FP8/BF16 speedup : {bf16_r['tick_ms']/fp8_r['tick_ms']:.2f}x")
        print(f"  torch/TRT FP8 ratio  : {fp8_r['tick_ms']/dt_torch:.2f}x  (room left in TRT path)")


if __name__ == "__main__":
    main()
