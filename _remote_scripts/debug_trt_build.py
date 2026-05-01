#!/usr/bin/env python3
"""Debug TRT build with verbose logger to see what's happening."""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/acestep")
import tensorrt as trt

print(f"TRT version: {trt.__version__}", flush=True)

onnx_path = "/workspace/build/onnx/decoder_bf16_mixed_dynbatch.onnx"
engine_path = "/workspace/build/test.engine"

trt_logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(trt_logger)
print("builder created", flush=True)

net_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | \
            (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
network = builder.create_network(net_flags)
parser = trt.OnnxParser(network, trt_logger)
print("parser created, calling parse_from_file...", flush=True)

t0 = time.time()
ok = parser.parse_from_file(onnx_path)
print(f"parse_from_file returned {ok} in {time.time()-t0:.1f}s", flush=True)
print(f"num_errors: {parser.num_errors}", flush=True)
for i in range(parser.num_errors):
    print(f"  err[{i}]: {parser.get_error(i)}", flush=True)

print(f"Network: {network.num_inputs} inputs, {network.num_outputs} outputs, {network.num_layers} layers", flush=True)
