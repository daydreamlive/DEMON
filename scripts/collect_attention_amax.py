"""Capture per-tensor absmax for the 128 attention-MatMul input tensors.

The attention path in the bf16 DiT ONNX contains 128 MatMul nodes whose
inputs are dynamic (not weight initializers). To quantize these to FP8
via Q-DQ chains we need actual amax for each input tensor. This script
runs the bf16 ONNX through ONNX Runtime, exposing the 256 distinct
attention-MatMul input tensors as additional graph outputs, captures
their absmax across a handful of calibration samples, and writes the
result as a JSON the FP8 patcher can consume.

We use the CUDA EP. Falls back to CPU if CUDA refuses.

Output: ``<MODELS_DIR>/calibration/decoder_xl_fp8/attention_amax.json``
"""
from __future__ import annotations

import json
import os
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
import onnxruntime as ort

REPO = Path(__file__).resolve().parent.parent
SRC = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/trt_engines/"
    "_onnx_acestep-v15-xl-turbo/decoder_refit/"
    "decoder_refit_dynbatch.onnx"
))
CAL_NPZ = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/calibration/decoder_xl_fp8/calibration.npz"
))
OUT_JSON = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/calibration/decoder_xl_fp8/attention_amax.json"
))


def find_attention_inputs(model: onnx.ModelProto) -> tuple[list[str], list[dict]]:
    """Return (unique_input_tensor_names, matmul_info)."""
    g = model.graph
    inits = {i.name for i in g.initializer}
    producer_op = {}
    for nd in g.node:
        for out in nd.output:
            producer_op[out] = nd.op_type

    matmul_info = []
    inputs_set = set()
    for nd in g.node:
        if nd.op_type != "MatMul" or len(nd.input) < 2:
            continue
        a, b = nd.input[0], nd.input[1]
        if a in inits or b in inits:
            continue
        # Skip MatMuls whose inputs are already DQ outputs (post-patch),
        # which doesn't matter on the source bf16 ONNX but is defensive.
        if producer_op.get(a) == "DequantizeLinear" or producer_op.get(b) == "DequantizeLinear":
            continue
        info = {
            "matmul_name": nd.name,
            "input_0": a,
            "input_0_producer": producer_op.get(a, "Input"),
            "input_1": b,
            "input_1_producer": producer_op.get(b, "Input"),
        }
        matmul_info.append(info)
        inputs_set.add(a)
        inputs_set.add(b)
    return sorted(inputs_set), matmul_info


def expose_outputs(model: onnx.ModelProto, tensor_names: list[str]) -> onnx.ModelProto:
    """Add each tensor name to graph outputs so ORT exposes its value."""
    g = model.graph
    existing = {o.name for o in g.output}
    for name in tensor_names:
        if name in existing:
            continue
        # ValueInfo with unknown shape/dtype is fine; ORT figures it out.
        vi = onnx.ValueInfoProto()
        vi.name = name
        g.output.append(vi)
    return model


def main():
    print(f"Loading bf16 ONNX from {SRC}")
    model = onnx.load(str(SRC), load_external_data=True)
    inputs, matmul_info = find_attention_inputs(model)
    print(f"Found {len(matmul_info)} attention MatMuls; {len(inputs)} distinct input tensors")

    print("Patching graph to expose attention inputs as outputs ...")
    expose_outputs(model, inputs)
    # Serialize to a temp file ORT can load (external data is messy with
    # SerializeToString on a >2GB model — write to disk).
    tmp_path = SRC.with_name(SRC.stem + "_attn_amax_probe.onnx")
    print(f"Writing intermediate ONNX to {tmp_path}")
    # Reset external_data so onnx writes a new sidecar.
    for init in model.graph.initializer:
        init.ClearField("external_data")
        init.data_location = onnx.TensorProto.DEFAULT
    onnx.save(
        model, str(tmp_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=tmp_path.stem + ".data",
    )

    print("Loading into ONNX Runtime (CUDA EP) ...")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(tmp_path), so, providers=providers)
    print(f"ORT providers in use: {sess.get_providers()}")

    print(f"Loading calibration .npz from {CAL_NPZ}")
    cal = np.load(str(CAL_NPZ))
    n_total = cal["hidden_states"].shape[0]
    # Use a modest subset for speed — amax converges fast.
    n_samples = min(64, n_total)
    print(f"Using {n_samples} samples")

    # Build name -> running max.
    amax: dict[str, float] = {n: 0.0 for n in inputs}

    in_names = [i.name for i in sess.get_inputs()]
    print(f"ORT input names: {in_names}")

    t0 = time.perf_counter()
    B = 4
    batches = n_samples // B
    for bi in range(batches):
        s = slice(bi * B, (bi + 1) * B)
        feed = {}
        for name in in_names:
            arr = cal[name][s]
            # ORT needs the exact dtype the model expects.
            in_info = next(i for i in sess.get_inputs() if i.name == name)
            # in_info.type is like 'tensor(bfloat16)'.
            if "bfloat16" in in_info.type:
                # ORT's CUDA EP requires the numpy dtype to be bfloat16, but
                # numpy doesn't have it native. Workaround: ML_FLOAT16 cast.
                # Use np.float16 as a stand-in.
                arr = arr.astype(np.float16)
            elif "float16" in in_info.type:
                arr = arr.astype(np.float16)
            elif "float32" in in_info.type or in_info.type == "tensor(float)":
                arr = arr.astype(np.float32)
            feed[name] = arr
        outs = sess.run(inputs, feed)
        for tensor_name, val in zip(inputs, outs):
            # Some captured outputs might be in fp16 / bf16; cast to float32.
            v = np.asarray(val, dtype=np.float32)
            m = float(np.abs(v).max())
            if m > amax[tensor_name]:
                amax[tensor_name] = m
        if (bi + 1) % 4 == 0:
            print(f"  batch {bi+1}/{batches}  elapsed={time.perf_counter()-t0:.1f}s")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")

    # Pair each MatMul with its inputs' amax.
    matmul_records = []
    for mi in matmul_info:
        mi_out = dict(mi)
        mi_out["input_0_amax"] = amax[mi["input_0"]]
        mi_out["input_1_amax"] = amax[mi["input_1"]]
        matmul_records.append(mi_out)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({
            "tensor_amax": amax,
            "matmuls": matmul_records,
            "n_samples": n_samples,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {OUT_JSON}")

    # Diagnostics.
    sorted_amax = sorted(amax.items(), key=lambda kv: -kv[1])
    print("\nTop 10 attention input tensors by amax:")
    for name, val in sorted_amax[:10]:
        print(f"  {val:>10.3f}  {name}")
    print("\nBottom 10 attention input tensors by amax:")
    for name, val in sorted_amax[-10:]:
        print(f"  {val:>10.3f}  {name}")


if __name__ == "__main__":
    main()
