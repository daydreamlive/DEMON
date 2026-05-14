"""Find attention MatMul nodes (dynamic-operand GEMMs) in the bf16 ONNX.

These are the MatMul ops whose neither input is a constant initializer
(both come from the graph). They run as bf16 @ bf16 in our current
production engine because fp8_onnx.py skips them — its quantization is
keyed on weight initializers.

Output: a list of node names + their input tensor names. We'll later
extend fp8_onnx.py to insert per-tensor FP8 Q-DQ on BOTH operands of
these nodes, so TRT picks FP8 GEMM tactics for them too.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import onnx

SRC = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/trt_engines/"
    "_onnx_acestep-v15-xl-turbo/decoder_refit/"
    "decoder_refit_dynbatch.onnx"
))


def main():
    print(f"Loading: {SRC}")
    model = onnx.load(str(SRC), load_external_data=False)
    g = model.graph

    inits = {i.name: i for i in g.initializer}
    print(f"Initializers: {len(inits)}")

    # Producer map: tensor name -> producer node
    producers: dict[str, str] = {}
    for n in g.node:
        for out in n.output:
            producers[out] = n.name

    all_matmuls = [n for n in g.node if n.op_type == "MatMul"]
    print(f"Total MatMul nodes: {len(all_matmuls)}")

    weight_init = []
    weight_dynamic = []
    dynamic_dynamic = []
    other = []
    for n in all_matmuls:
        a, b = n.input[0], n.input[1]
        a_is_init = a in inits
        b_is_init = b in inits
        if b_is_init and not a_is_init:
            weight_init.append(n)
        elif a_is_init and not b_is_init:
            other.append(n)  # weight on left — unusual
        elif not a_is_init and not b_is_init:
            dynamic_dynamic.append(n)
        else:
            weight_dynamic.append(n)  # both init (rare)

    print()
    print(f"  MatMul with weight initializer on input[1]:  {len(weight_init)}  (production fp8 path quantizes these)")
    print(f"  MatMul with weight on input[0]:              {len(other)}")
    print(f"  MatMul with BOTH inputs from graph:          {len(dynamic_dynamic)}  (attention bmm/baddbmm — what we want)")
    print(f"  MatMul with both inputs initializers:        {len(weight_dynamic)}")

    print()
    print("Sample of dynamic-dynamic MatMul names + inputs:")
    for n in dynamic_dynamic[:10]:
        a, b = n.input[0], n.input[1]
        ap = producers.get(a, "<input>")
        bp = producers.get(b, "<input>")
        print(f"  {n.name}")
        print(f"    in[0]={a}  (from {ap})")
        print(f"    in[1]={b}  (from {bp})")

    # Categorize dynamic-dynamic by location (layer / attention type).
    print()
    print("Per-name-pattern bucket:")
    pat_counts = Counter()
    for n in dynamic_dynamic:
        nm = n.name.lower()
        if "self_attn" in nm or "selfattn" in nm:
            pat_counts["self_attn"] += 1
        elif "cross_attn" in nm or "crossattn" in nm:
            pat_counts["cross_attn"] += 1
        elif "attn" in nm:
            pat_counts["other_attn"] += 1
        else:
            pat_counts["other"] += 1
    for k, v in pat_counts.most_common():
        print(f"  {k:<15s} {v}")

    # Cross-check: count unique input tensors among dynamic-dynamic.
    inputs = set()
    for n in dynamic_dynamic:
        inputs.add(n.input[0])
        inputs.add(n.input[1])
    print()
    print(f"Unique distinct dynamic input tensors across these MatMuls: {len(inputs)}")
    print(f"(Each will need its own activation Q->DQ chain.)")

    # Save the list for later use.
    out_path = Path(__file__).parent / "attn_matmuls.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for n in dynamic_dynamic:
            f.write(f"{n.name}\t{n.input[0]}\t{n.input[1]}\n")
    print()
    print(f"Saved list to {out_path}")


if __name__ == "__main__":
    main()
