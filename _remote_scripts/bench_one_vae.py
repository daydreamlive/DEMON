"""Benchmark one VAE engine. Args: <engine> <input_name> <dim> <T> [iters=20]"""
import sys
import time
import torch
import tensorrt as trt

eng = sys.argv[1]
in_name = sys.argv[2]
dim = int(sys.argv[3])
T = int(sys.argv[4])
iters = int(sys.argv[5]) if len(sys.argv) > 5 else 20

print(f"TRT version: {trt.__version__}", flush=True)
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open(eng, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()

x = torch.randn(1, dim, T, device="cuda", dtype=torch.float32).contiguous()
ctx.set_input_shape(in_name, tuple(x.shape))
ctx.set_tensor_address(in_name, x.data_ptr())

out_name = None
for i in range(engine.num_io_tensors):
    n = engine.get_tensor_name(i)
    if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
        out_name = n
out_shape = tuple(ctx.get_tensor_shape(out_name))
y = torch.empty(out_shape, dtype=torch.float32, device="cuda")
ctx.set_tensor_address(out_name, y.data_ptr())

stream = torch.cuda.Stream()

# warmup
for _ in range(3):
    ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
torch.cuda.synchronize()

times = []
for _ in range(iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

times.sort()
median = times[len(times)//2]
mn = min(times)
mx = max(times)
print(f"engine: {eng}", flush=True)
print(f"  T={T}  iters={iters}", flush=True)
print(f"  min={mn*1000:.2f}ms  median={median*1000:.2f}ms  max={mx*1000:.2f}ms", flush=True)
