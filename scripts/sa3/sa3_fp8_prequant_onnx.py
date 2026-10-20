"""Pre-quantize the fp8 weights of Stability's dit_fp8.onnx.

For each weight site (QuantizeLinear on an fp32 initializer), quantize the
initializer with its baked scale (saturating E4M3), store it as a
FLOAT8E4M3FN initializer under the SAME name, drop the weight
QuantizeLinear and feed its DequantizeLinear straight from the initializer.
Activation Q/DQ pairs are untouched.

    python scripts/sa3/sa3_fp8_prequant_onnx.py --onnx <dit_fp8.onnx> --out <dit_fp8_prequant.onnx>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import torch  # noqa: E402
from onnx import numpy_helper  # noqa: E402

from acestep.engine.trt.sa3_export import load_onnx, save_onnx_checked  # noqa: E402
from acestep.engine.trt.sa3_fp8_refit import fp8_weight_sites, quantize_e4m3_fixed  # noqa: E402


def prequantize(model) -> int:
    g = model.graph
    sites = fp8_weight_sites(model)
    inits = {i.name: i for i in g.initializer}
    q_by_input = {n.input[0]: n for n in g.node if n.op_type == "QuantizeLinear"}
    drop = set()
    for site in sites:
        q = q_by_input[site.initializer]
        w = torch.from_numpy(numpy_helper.to_array(inits[site.initializer]).astype(np.float32))
        s = torch.from_numpy(numpy_helper.to_array(inits[site.scale_init]).astype(np.float32))
        u8 = quantize_e4m3_fixed(w, s, site.axis).contiguous().numpy()
        t = inits[site.initializer]
        t.ClearField("float_data")
        t.data_type = onnx.TensorProto.FLOAT8E4M3FN
        t.raw_data = u8.tobytes()
        for n in g.node:
            if n.op_type == "DequantizeLinear" and n.input[0] == q.output[0]:
                n.input[0] = site.initializer
        drop.add(id(q))
    keep = [n for n in g.node if id(n) not in drop]
    del g.node[:]
    g.node.extend(keep)
    used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
    unused = [i for i in g.initializer if i.name not in used]
    for i in unused:
        g.initializer.remove(i)
    print(f"sites={len(sites)} dropped_q={len(drop)} pruned_inits={len(unused)}")
    return len(drop)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = load_onnx(a.onnx)
    n0 = len(model.graph.node)
    dropped = prequantize(model)
    n1 = len(model.graph.node)
    if n1 != n0 - dropped:
        raise RuntimeError(f"node count {n1} != {n0} - {dropped}")
    print(f"nodes {n0} -> {n1}")
    sha = save_onnx_checked(model, a.out)
    print(f"saved {a.out} sha256={sha}")
    try:
        onnx.checker.check_model(a.out)
        print("onnx.checker OK")
    except onnx.checker.ValidationError as exc:  # Stability's source graph fails the same way
        print(f"onnx.checker: {str(exc).splitlines()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
