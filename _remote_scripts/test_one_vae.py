"""Test ANY VAE engine. Args: <engine> <input_name> <dim> <T>"""
import sys
import torch
import tensorrt as trt

eng = sys.argv[1]
in_name = sys.argv[2]
dim = int(sys.argv[3])
T = int(sys.argv[4])

print(f"TRT version: {trt.__version__}", flush=True)
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

print("loading ...", flush=True)
with open(eng, "rb") as f:
    blob = f.read()
engine = runtime.deserialize_cuda_engine(blob)
assert engine is not None
print(f"  num_layers={engine.num_layers}", flush=True)

ctx = engine.create_execution_context()

x = torch.randn(1, dim, T, device="cuda", dtype=torch.float32).contiguous()
ctx.set_input_shape(in_name, tuple(x.shape))
ctx.set_tensor_address(in_name, x.data_ptr())

# find the output name
out_name = None
for i in range(engine.num_io_tensors):
    n = engine.get_tensor_name(i)
    if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
        out_name = n
        break
print(f"output: {out_name}", flush=True)

out_shape = tuple(ctx.get_tensor_shape(out_name))
print(f"  shape {out_shape}", flush=True)
y = torch.empty(out_shape, dtype=torch.float32, device="cuda")
ctx.set_tensor_address(out_name, y.data_ptr())

stream = torch.cuda.Stream()
print("execute_async_v3 ...", flush=True)
ok = ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
torch.cuda.synchronize()
print(f"OK ret={ok}  out [{y.min().item():.4f},{y.max().item():.4f}]  has_nan={torch.isnan(y).any().item()}", flush=True)
