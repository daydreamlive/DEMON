"""Experimental VAE TRT build with multiple flag configurations.

Usage:
    build_vae_experiment.py <component> <mode> <out_engine>

component: decode | encode
mode:
    fp16            - default, FP16 flag set
    fp32            - no precision flag at all (pure fp32)
    strongly_typed  - STRONGLY_TYPED (respects ONNX dtypes, no FP16 flag)
    fp16_optlvl0    - FP16 + builder_optimization_level=0
    fp16_optlvl1    - FP16 + builder_optimization_level=1
    fp16_no_jit     - FP16 + tactic sources without JIT_CONVOLUTIONS
    fp16_cublas_only- FP16 + only CUBLAS / CUBLAS_LT tactic sources
    fp16_static     - FP16 + min=opt=max profile (static shape)
    fp16_no_edge_mask - FP16 + tactic sources without EDGE_MASK_CONVOLUTIONS
    fp16_disable_strict_types - FP16 + DIRECT_IO

In all cases ProfilingVerbosity.DETAILED is set so we can later inspect.
"""
import os
import sys
import time
from pathlib import Path

import tensorrt as trt

ROOT = Path("C:/Users/ryanf/.daydream-scope/models/rtmg/trt_engines")
ONNX = {
    "decode": ROOT / "_onnx_vae/vae_decode/vae_decode.onnx",
    "encode": ROOT / "_onnx_vae/vae_encode/vae_encode.onnx",
}
INPUT = {
    "decode": ("latents", 64, 125, 1500, 1500),     # name, dim, min, opt, max (60s for max)
    "encode": ("audio", 2, 240000, 2880000, 2880000),
    "decode240": ("latents", 64, 125, 6000, 6000),  # 240s
    "encode240": ("audio", 2, 240000, 11520000, 11520000),
}
ONNX["decode240"] = ONNX["decode"]
ONNX["encode240"] = ONNX["encode"]

component = sys.argv[1]
mode = sys.argv[2]
out_engine = Path(sys.argv[3])

assert component in ONNX, f"unknown component {component}"
assert ONNX[component].exists(), f"missing onnx {ONNX[component]}"

print(f"TRT version: {trt.__version__}", flush=True)
print(f"component={component} mode={mode}", flush=True)
print(f"out: {out_engine}", flush=True)

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

print("parsing onnx ...", flush=True)
ok = parser.parse_from_file(str(ONNX[component].resolve()))
if not ok:
    for i in range(parser.num_errors):
        print(f"  parse error: {parser.get_error(i)}", flush=True)
    sys.exit(1)
print(f"  network: {network.num_layers} layers, {network.num_inputs} inputs, {network.num_outputs} outputs", flush=True)

# Print all inputs
for i in range(network.num_inputs):
    t = network.get_input(i)
    print(f"  input[{i}] {t.name} dtype={t.dtype} shape={tuple(t.shape)}", flush=True)
for i in range(network.num_outputs):
    t = network.get_output(i)
    print(f"  output[{i}] {t.name} dtype={t.dtype} shape={tuple(t.shape)}", flush=True)

build_config = builder.create_builder_config()
build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)
build_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
print(f"  profiling verbosity: DETAILED", flush=True)

# Mode-specific flags
if mode == "fp16":
    build_config.set_flag(trt.BuilderFlag.FP16)
elif mode == "fp32":
    pass  # no precision flag
elif mode == "strongly_typed":
    build_config.set_flag(trt.BuilderFlag.STRONGLY_TYPED)
elif mode == "fp16_optlvl0":
    build_config.set_flag(trt.BuilderFlag.FP16)
    build_config.builder_optimization_level = 0
elif mode == "fp16_optlvl1":
    build_config.set_flag(trt.BuilderFlag.FP16)
    build_config.builder_optimization_level = 1
elif mode == "fp16_optlvl2":
    build_config.set_flag(trt.BuilderFlag.FP16)
    build_config.builder_optimization_level = 2
elif mode == "fp16_optlvl4":
    build_config.set_flag(trt.BuilderFlag.FP16)
    build_config.builder_optimization_level = 4
elif mode == "fp16_optlvl5":
    build_config.set_flag(trt.BuilderFlag.FP16)
    build_config.builder_optimization_level = 5
elif mode == "fp16_no_jit":
    build_config.set_flag(trt.BuilderFlag.FP16)
    # default sources minus JIT_CONVOLUTIONS
    sources = (
        (1 << int(trt.TacticSource.CUBLAS))
        | (1 << int(trt.TacticSource.CUBLAS_LT))
        | (1 << int(trt.TacticSource.CUDNN))
        | (1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS))
    )
    build_config.set_tactic_sources(sources)
elif mode == "fp16_cublas_only":
    build_config.set_flag(trt.BuilderFlag.FP16)
    sources = (
        (1 << int(trt.TacticSource.CUBLAS))
        | (1 << int(trt.TacticSource.CUBLAS_LT))
    )
    build_config.set_tactic_sources(sources)
elif mode == "fp16_no_edge_mask":
    build_config.set_flag(trt.BuilderFlag.FP16)
    sources = (
        (1 << int(trt.TacticSource.CUBLAS))
        | (1 << int(trt.TacticSource.CUBLAS_LT))
        | (1 << int(trt.TacticSource.CUDNN))
        | (1 << int(trt.TacticSource.JIT_CONVOLUTIONS))
    )
    build_config.set_tactic_sources(sources)
elif mode == "fp16_static":
    build_config.set_flag(trt.BuilderFlag.FP16)
elif mode == "fp16_direct_io":
    build_config.set_flag(trt.BuilderFlag.FP16)
    build_config.set_flag(trt.BuilderFlag.DIRECT_IO)
elif mode == "fp16_obey":
    build_config.set_flag(trt.BuilderFlag.FP16)
    build_config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
else:
    raise ValueError(f"unknown mode {mode}")

# Print active flags + tactic sources for clarity
print("  flags:", flush=True)
for f in trt.BuilderFlag.__members__.values():
    if build_config.get_flag(f):
        print(f"    + {f}", flush=True)
print(f"  tactic_sources: {build_config.get_tactic_sources()}", flush=True)
print(f"  builder_optimization_level: {build_config.builder_optimization_level}", flush=True)

# Optimization profile
input_name, input_dim, mn, op, mx = INPUT[component]
if mode == "fp16_static":
    mn = op
    mx = op

profile = builder.create_optimization_profile()
profile.set_shape(
    input_name,
    min=(1, input_dim, mn),
    opt=(1, input_dim, op),
    max=(1, input_dim, mx),
)
build_config.add_optimization_profile(profile)
print(f"  profile: {input_name} min={mn} opt={op} max={mx}", flush=True)

print("building ...", flush=True)
t0 = time.time()
serialized = builder.build_serialized_network(network, build_config)
t1 = time.time()
if serialized is None:
    print("BUILD FAILED", flush=True)
    sys.exit(2)

blob = bytes(serialized)
print(f"build OK in {t1-t0:.1f}s, size={len(blob)/(1<<20):.1f} MB", flush=True)

out_engine.parent.mkdir(parents=True, exist_ok=True)
with open(out_engine, "wb") as f:
    f.write(blob)
print(f"saved -> {out_engine}", flush=True)
