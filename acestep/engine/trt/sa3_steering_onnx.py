"""Add the activation-steering input to the upstream SA3 DiT ONNX graph.

The SA3 DiT engines compile Stability's official ONNX export
(``dit_fp16.onnx``) rather than a local export, so steering is added by
graph surgery instead of in a torch export wrapper. The result follows the
convention the ACE decoder engine established
(``acestep/engine/trt/export.py``): a graph input

    steering  float32  [1, num_blocks, hidden]

whose row ``i`` is added to the output residual of transformer block ``i``
(broadcast over every token, memory tokens included). The host zeros rows
for unused blocks, so an all-zero tensor is an exact no-op add.

Surgery is proto-only: the weights stay in the upstream external-data
sidecar, untouched. :func:`steered_dit_onnx_bytes` returns the serialized
modified proto, which the builder parses together with the ORIGINAL file
path so TensorRT resolves the external weights next to it (no 2.8 GB
copy). In the fp16mixed graph the residual stream between blocks is
FLOAT16, so the steering tensor is cast once to the block-output dtype.

Block boundaries are found structurally, not by node order (the fp16
autocast pass appends its Cast nodes at the end of the node list): the
output of block ``i`` is the one non-constant tensor produced inside
``/transformer/layers.{i}/`` and consumed outside it.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

#: Bump when the surgery changes; part of the engine build identity.
STEERING_SURGERY_VERSION = 1

STEERING_INPUT = "steering"
_LAYER_RE = re.compile(r"/transformer/layers\.(\d+)/")


def _layer_of(node) -> int | None:
    m = _LAYER_RE.search(node.name)
    return int(m.group(1)) if m else None


def find_block_outputs(graph) -> list[str]:
    """Ordered block-output tensor names (index = block index).

    Raises ValueError when a block does not have exactly one outgoing
    residual tensor (the graph is not the layout this surgery knows).
    """
    producer_layer = {}
    const_outputs = set()
    for n in graph.node:
        li = _layer_of(n)
        for o in n.output:
            producer_layer[o] = li
            if n.op_type == "Constant":
                const_outputs.add(o)
    crossing = defaultdict(set)
    for n in graph.node:
        li = _layer_of(n)
        for i in n.input:
            src = producer_layer.get(i)
            if src is None or src == li or i in const_outputs:
                continue
            crossing[src].add(i)
    if not crossing:
        raise ValueError("no /transformer/layers.N/ blocks found in the graph")
    n_blocks = max(crossing) + 1
    out = []
    for b in range(n_blocks):
        # The residual leaves block b towards block b+1 (or the trunk tail).
        cands = sorted(crossing.get(b, ()))
        # Drop side tensors consumed only by the next block's attention
        # bookkeeping (e.g. rotary caches): the residual is the one whose
        # producer is an Add.
        adds = [
            t for t in cands
            if any(
                n.op_type == "Add" and t in n.output for n in graph.node
                if _layer_of(n) == b
            )
        ]
        if len(adds) != 1:
            raise ValueError(
                f"block {b}: expected one residual Add output leaving the "
                f"block, found {adds or cands}"
            )
        out.append(adds[0])
    return out


def add_steering_input(model, *, hidden_size: int | None = None):
    """Return ``(model, num_blocks, hidden)`` with the steering input added.

    Mutates ``model`` (an ``onnx.ModelProto`` loaded WITHOUT external data)
    in place. ``hidden_size`` defaults to the static last dim of the
    trunk width read off the ``project_in`` weight.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    g = model.graph
    if any(i.name == STEERING_INPUT for i in g.input):
        raise ValueError("graph already has a steering input")
    outs = find_block_outputs(g)
    n_blocks = len(outs)
    hidden = int(hidden_size) if hidden_size is not None else infer_hidden_size(g)

    # Block-output dtype: the dtype of the residual Add's inputs, read off
    # the Cast that feeds it (fp16mixed: FLOAT16).
    before = _order_violations(g)
    producers = {o: n for n in g.node for o in n.output}
    res_dtype = _infer_residual_dtype(outs[0], producers)

    g.input.append(helper.make_tensor_value_info(
        STEERING_INPUT, TensorProto.FLOAT, [1, n_blocks, hidden],
    ))
    head = []
    steer_src = STEERING_INPUT
    if res_dtype != TensorProto.FLOAT:
        steer_src = "/steering/Cast_output_0"
        head.append(helper.make_node(
            "Cast", [STEERING_INPUT], [steer_src], name="/steering/Cast",
            to=res_dtype,
        ))
    gathered = []
    for b in range(n_blocks):
        idx = f"/steering/idx_{b}"
        g.initializer.append(numpy_helper.from_array(np.array([b], dtype=np.int64), idx))
        t = f"/steering/row_{b}"
        head.append(helper.make_node(
            "Gather", [steer_src, idx], [t], name=f"/steering/Gather_{b}", axis=1,
        ))
        gathered.append(t)  # [1, 1, hidden]

    # Rewire: every consumer of block b's output now reads out_b + row_b.
    new_nodes = list(head)
    for n in g.node:
        new_nodes.append(n)
        for b, src in enumerate(outs):
            if src in n.output:
                steered = f"{src}__steered"
                new_nodes.append(helper.make_node(
                    "Add", [src, gathered[b]], [steered],
                    name=f"/steering/Add_{b}",
                ))
    rename = {src: f"{src}__steered" for src in outs}
    for n in new_nodes:
        if n.name.startswith("/steering/Add_"):
            continue
        for k, i in enumerate(n.input):
            if i in rename:
                n.input[k] = rename[i]
    for o in g.output:
        if o.name in rename:
            raise ValueError("a block output is a graph output; unsupported")
    del g.node[:]
    g.node.extend(new_nodes)
    # onnx.checker would try to open the external-data sidecar relative
    # to the cwd; the property the surgery can break is node order, so
    # check exactly that.
    after = _order_violations(g)
    if not after <= before:
        raise ValueError(f"surgery broke node order at {sorted(after - before)[:5]}")
    return model, n_blocks, hidden


def _order_violations(graph) -> set:
    """Names of nodes that read a tensor produced later in the node list.

    The upstream fp16 graph is not strictly sorted (its autocast pass
    appends Cast nodes at the end; TensorRT's parser tolerates that), so
    the surgery's invariant is "introduces no NEW violation", checked by
    comparing this set before and after.
    """
    produced_at = {}
    for k, n in enumerate(graph.node):
        for o in n.output:
            produced_at[o] = k
    bad = set()
    for k, n in enumerate(graph.node):
        for i in n.input:
            if produced_at.get(i, -1) > k:
                bad.add(n.name)
    return bad


def infer_hidden_size(graph) -> int:
    """Trunk width from the ``project_in`` weight (``[dim, io_channels]``)."""
    for t in graph.initializer:
        if t.name.endswith("transformer.project_in.weight") and len(t.dims) == 2:
            return int(t.dims[0])
    raise ValueError("cannot infer hidden size: no transformer.project_in.weight")


def _infer_residual_dtype(tensor: str, producers: dict) -> int:
    """Element type of ``tensor`` (a block's residual Add output), found
    by walking back to the nearest Cast (or defaulting to FLOAT)."""
    from onnx import TensorProto

    seen = set()
    frontier = [tensor]
    while frontier:
        t = frontier.pop()
        if t in seen:
            continue
        seen.add(t)
        n = producers.get(t)
        if n is None:
            continue
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to":
                    return int(a.i)
        if n.op_type in ("Add", "Mul", "Sub"):
            frontier.extend(n.input)
    return TensorProto.FLOAT


def steered_dit_onnx_bytes(onnx_path: str | Path, *, hidden_size: int | None = None) -> tuple:
    """``(serialized_proto, num_blocks, hidden)`` for the steered graph.

    Loads the proto WITHOUT its external data, so the bytes stay small
    and the external-data references keep pointing at the sidecar next
    to ``onnx_path`` (parse them with that path).
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    model, n_blocks, hidden = add_steering_input(model, hidden_size=hidden_size)
    return model.SerializeToString(), n_blocks, hidden
