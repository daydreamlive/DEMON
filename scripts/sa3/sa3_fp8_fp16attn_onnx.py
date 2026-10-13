"""Run the attention cores of Stability's dit_fp8.onnx in fp16 (CLI over
:mod:`acestep.engine.trt.sa3_fp8_fp16attn`). ``sa3_build --fp8-refit``
applies the patch itself; this script is for inspecting it.

    python scripts/sa3/sa3_fp8_fp16attn_onnx.py --onnx <dit_fp8.onnx> --out <dit_fp8_fp16attn.onnx> [--ref <dit_fp16.onnx>]

``--ref`` prints the layer-0 attention-core placement next to
``dit_fp16.onnx``'s. The PV rows differ by design: the fp16 graph's v and
PV output are already fp16.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import onnx  # noqa: E402
from onnx import TensorProto  # noqa: E402

from acestep.engine.trt.sa3_fp8_fp16attn import (  # noqa: E402
    patch_fp8_attention_fp16,
    patch_model,
    softmax_input_types,
)


def core_signature(model, block: str) -> list:
    """One row per attention-core MatMul / Softmax under ``block``: its
    inputs (with any feeding Cast shown) and its consumers' op types."""
    prod = {o: n for n in model.graph.node for o in n.output}
    rows = []
    for n in model.graph.node:
        if not n.name.startswith(block) or n.op_type not in ("MatMul", "Softmax") or "/to_" in n.name:
            continue
        ins = []
        for i in n.input:
            p = prod.get(i)
            if p is not None and p.op_type == "Cast":
                to = [a.i for a in p.attribute if a.name == "to"][0]
                ins.append(f"Cast->{TensorProto.DataType.Name(to)}({p.input[0].split('/')[-1]})")
            else:
                ins.append(i.split("/")[-1])
        outs = [c.op_type for c in model.graph.node if n.output[0] in c.input]
        rows.append(f"{n.op_type} {n.name.split('/')[-1]} {ins} -> {outs}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="dit_fp16.onnx, for the layer-0 placement diff")
    a = ap.parse_args()
    if a.ref:
        proto = onnx.load(a.onnx, load_external_data=False)
        patch_model(proto)
        ref = onnx.load(a.ref, load_external_data=False)
        for blk in ("/transformer/layers.0/self_attn/", "/transformer/layers.0/cross_attn/"):
            print(f"== {blk}  (patched fp8 | dit_fp16)")
            for x, y in zip(core_signature(proto, blk), core_signature(ref, blk)):
                print(f"  {'SAME' if x == y else 'DIFF'} {x}" + ("" if x == y else f"\n       ref {y}"))
        print("softmax input types:", dict(softmax_input_types(proto)))
    print(patch_fp8_attention_fp16(a.onnx, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
