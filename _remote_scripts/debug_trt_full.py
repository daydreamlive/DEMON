#!/usr/bin/env python3
"""Debug full TRT build with INFO logger to find where it dies."""
import sys, time
sys.path.insert(0, "/workspace/acestep")
import tensorrt as trt

print(f"TRT version: {trt.__version__}", flush=True)

onnx_path = "/workspace/build/onnx/decoder_bf16_mixed_dynbatch.onnx"
engine_path = "/workspace/build/test.engine"

# Use INFO level (matches the failing build run)
trt_logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(trt_logger)
print("STAGE 1: builder OK", flush=True)

import os
USE_STRONG = os.environ.get("STRONG", "1") == "1"
net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
if USE_STRONG:
    net_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
print(f"strongly_typed={USE_STRONG}", flush=True)
network = builder.create_network(net_flags)
parser = trt.OnnxParser(network, trt_logger)
print("STAGE 2: network + parser OK", flush=True)

t0 = time.time()
ok = parser.parse_from_file(onnx_path)
print(f"STAGE 3: parse returned {ok} in {time.time()-t0:.1f}s", flush=True)
print(f"  num_errors={parser.num_errors}, layers={network.num_layers}", flush=True)

build_config = builder.create_builder_config()
build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * (1 << 30))
build_config.set_flag(trt.BuilderFlag.TF32)
if not USE_STRONG:
    build_config.set_flag(trt.BuilderFlag.BF16)
    build_config.set_flag(trt.BuilderFlag.FP16)
    print("set BF16 + FP16 builder flags (non-strongly-typed)", flush=True)
import os
build_config.builder_optimization_level = int(os.environ.get("OPT_LVL", "0"))
print("STAGE 4: build_config OK", flush=True)

profile = builder.create_optimization_profile()
profile.set_shape("hidden_states", min=(1, 126, 64), opt=(8, 750, 64), max=(8, 1500, 64))
profile.set_shape("timestep", min=(1,), opt=(8,), max=(8,))
profile.set_shape("encoder_hidden_states", min=(1, 32, 2048), opt=(8, 200, 2048), max=(8, 512, 2048))
profile.set_shape("context_latents", min=(1, 126, 128), opt=(8, 750, 128), max=(8, 1500, 128))
build_config.add_optimization_profile(profile)
print("STAGE 5: profile added", flush=True)

print("STAGE 6: calling build_serialized_network...", flush=True)
t0 = time.time()
serialized = builder.build_serialized_network(network, build_config)
print(f"STAGE 7: build returned in {time.time()-t0:.1f}s, type={type(serialized)}", flush=True)
if serialized is None:
    print("BUILD FAILED: serialized is None", flush=True)
    sys.exit(1)

with open(engine_path, "wb") as f:
    f.write(serialized)
print(f"STAGE 8: wrote {engine_path}", flush=True)
print(f"  size: {open(engine_path,'rb').seek(0,2)} bytes", flush=True)
