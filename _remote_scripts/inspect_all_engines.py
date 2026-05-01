"""Print declared input/output dtypes for all decoder TRT engines."""
from pathlib import Path
import tensorrt as trt

ROOT = Path(r"C:\Users\ryanf\.daydream-scope\models\rtmg\trt_engines")
INPUT_NAMES = ("hidden_states", "timestep", "encoder_hidden_states", "context_latents", "velocity")

logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)

dirs = sorted([d for d in ROOT.iterdir() if d.is_dir() and d.name.startswith("decoder_")])
print(f"{'engine':<55} {'hs':<8} {'ts':<8} {'enc':<8} {'ctx':<8} {'out':<8}")
print("-" * 100)
for d in dirs:
    eng = d / f"{d.name}.engine"
    if not eng.exists():
        continue
    try:
        engine = runtime.deserialize_cuda_engine(eng.read_bytes())
    except Exception as e:
        print(f"{d.name:<55} ERR: {e}")
        continue
    dtypes = {}
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        dtypes[n] = engine.get_tensor_dtype(n).name
    line = f"{d.name:<55}"
    for n in INPUT_NAMES:
        line += f" {dtypes.get(n, '-'):<8}"
    print(line)
