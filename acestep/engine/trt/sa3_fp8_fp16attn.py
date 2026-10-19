"""fp16 attention cores for Stability's ``dit_fp8.onnx``.

Stability's fp8 DiT graph runs attention in fp32: there is no Cast to
FLOAT16 anywhere in it, so QK^T, Softmax and PV all run on fp32 tensors.
A non-refit build rescues that with a fused fp32 MHA tactic
(``_gemm_mha_v2``), but as soon as any weight is refittable TensorRT loses
that tactic and attention runs unfused (separate QK^T / softmax / PV
layers, ~2x the step time at a 54 s window). ``dit_fp16.onnx`` casts q and
k (after the RoPE / scale Mul) to FLOAT16 right before the core MatMuls;
its attention fuses in both refit and non-refit builds.

:func:`patch_model` applies the same placement to the fp8 graph, for every
Softmax under ``/transformer/layers.N/{self,cross}_attn/``:

* both inputs of the QK^T MatMul feeding the Softmax get a Cast to FLOAT16
  (one Cast per tensor, shared when two cores read the same tensor);
* the PV MatMul (input A = the Softmax output) gets a Cast to FLOAT16 on
  its v input, and a Cast back to FLOAT on its output, so everything
  downstream keeps its original fp32 type.

Weights, Q/DQ pairs and every other node are untouched.
"""

from __future__ import annotations

import collections
import os
import re

_BLOCK = re.compile(r"^/transformer/layers\.\d+/(self|cross)_attn/")


def patch_model(model) -> dict:
    """Patch ``model`` (an ``onnx.ModelProto``, with or without weight
    data) in place. Returns ``{"cast_fp16", "cast_fp32", "softmax"}``
    counts. Raises when an attention core does not have the expected
    QK^T MatMul -> Softmax -> PV MatMul shape."""
    from onnx import TensorProto, helper

    g = model.graph
    inits = {i.name for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    before: dict[int, list] = collections.defaultdict(list)
    after: dict[int, list] = collections.defaultdict(list)
    cast_of: dict[str, str] = {}
    stats = {"cast_fp16": 0, "cast_fp32": 0, "softmax": 0}

    def to_fp16(tensor: str, user) -> str:
        if tensor in inits:
            raise RuntimeError(f"{user.name}: attention input {tensor} is an initializer")
        if tensor not in cast_of:
            out = f"{tensor}__attnfp16"
            before[id(user)].append(helper.make_node(
                "Cast", [tensor], [out], name=f"{tensor}__attnfp16_cast",
                to=TensorProto.FLOAT16,
            ))
            cast_of[tensor] = out
            stats["cast_fp16"] += 1
        return cast_of[tensor]

    for sm in [n for n in g.node if n.op_type == "Softmax" and _BLOCK.match(n.name)]:
        qk = prod.get(sm.input[0])
        if qk is None or qk.op_type != "MatMul":
            raise RuntimeError(f"{sm.name}: input is not a MatMul")
        users = cons[sm.output[0]]
        if len(users) != 1 or users[0].op_type != "MatMul" or users[0].input[0] != sm.output[0]:
            raise RuntimeError(f"{sm.name}: expected exactly one PV MatMul consumer")
        pv = users[0]
        qk.input[0] = to_fp16(qk.input[0], qk)
        qk.input[1] = to_fp16(qk.input[1], qk)
        pv.input[1] = to_fp16(pv.input[1], pv)
        orig = pv.output[0]
        pv.output[0] = f"{orig}__fp16"
        after[id(pv)].append(helper.make_node(
            "Cast", [pv.output[0]], [orig], name=f"{orig}__tofp32_cast",
            to=TensorProto.FLOAT,
        ))
        stats["cast_fp32"] += 1
        stats["softmax"] += 1
    if not stats["softmax"]:
        raise RuntimeError("no attention Softmax found under /transformer/layers.N/*_attn/")
    nodes = []
    for n in g.node:
        nodes.extend(before.get(id(n), []))
        nodes.append(n)
        nodes.extend(after.get(id(n), []))
    del g.node[:]
    g.node.extend(nodes)
    return stats


def softmax_input_types(model) -> collections.Counter:
    """ONNX element-type names of every attention Softmax input, by shape
    inference on ``model`` (no weight data needed)."""
    import onnx
    from onnx import TensorProto

    inf = onnx.shape_inference.infer_shapes(model)
    types = {v.name: v.type.tensor_type.elem_type
             for v in list(inf.graph.value_info) + list(inf.graph.input)}
    return collections.Counter(
        TensorProto.DataType.Name(types.get(n.input[0], 0))
        for n in inf.graph.node if n.op_type == "Softmax" and _BLOCK.match(n.name)
    )


def patch_fp8_attention_fp16(
    src_onnx: str | os.PathLike[str], dst_onnx: str | os.PathLike[str],
) -> dict:
    """Write ``src_onnx`` with fp16 attention cores to ``dst_onnx``
    (+ ``.data``, via the binary-safe :func:`save_onnx_checked`). Returns
    :func:`patch_model`'s stats plus ``nodes_before`` / ``nodes_after``
    and the output ``sha256``."""
    from .sa3_export import load_onnx, save_onnx_checked

    model = load_onnx(src_onnx)
    n0 = len(model.graph.node)
    stats = patch_model(model)
    stats["nodes_before"] = n0
    stats["nodes_after"] = len(model.graph.node)
    stats["sha256"] = save_onnx_checked(model, dst_onnx)
    return stats
