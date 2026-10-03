"""SA3 cross-attention-output steering hook (TADA's intervention site).

TADA (Staniszewski et al., arXiv 2602.11910) steers the OUTPUT of the
cross-attention module of the functional blocks, before it joins the
residual stream, so the shift passes through the block's feed-forward.
These CPU tests cover the ONNX surgery for that site (on a synthetic graph
with the upstream naming: ``/transformer/layers.N/cross_attn/to_out``
feeding the block's residual ``Add``), the engine-family selection, and
the eager delivery through the real StreamPipeline hook path onto
``cross_attn`` submodules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from acestep.engine import sa3_trt
from acestep.engine.trt.sa3_build import (
    SA3DiTSteerBuildConfig,
    SA3DiTSteerCrossAttnBuildConfig,
)
from acestep.engine.trt.sa3_steering_onnx import (
    CROSS_ATTN_SURGERY_VERSION,
    STEERING_CROSS_ATTN_INPUT,
    STEERING_INPUT,
    add_steering_input,
    find_block_outputs,
    find_cross_attn_outputs,
)
from acestep.steering.layout import HOOK_CROSS_ATTN_OUTPUT, HOOK_POST_BLOCK_RESIDUAL

H = 4
NB = 3


def _wx(b: int) -> np.ndarray:
    return (np.eye(H) * (0.5 + 0.25 * b)).astype(np.float16)


def _wf(b: int) -> np.ndarray:
    return np.full((H,), 1.0 + 0.5 * b, dtype=np.float16)


def _graph() -> onnx.ModelProto:
    """x fp32 -> fp16 trunk of NB blocks -> velocity fp32.

    Block b: r = cast16(cast32(h)) + cross_attn(h) with
    cross_attn(h) = h @ Wx_b (inside ``.../cross_attn/``); then
    out = r * wf_b + r (the feed-forward stand-in, residual Add leaving
    the block). The autocast Cast is appended at the end, as upstream.
    """
    nodes, tail = [], []
    inits = [numpy_helper.from_array(np.eye(H, dtype=np.float16), "dit.transformer.project_in.weight")]
    nodes.append(helper.make_node("Cast", ["x"], ["/transformer/Cast_output_0"], name="/transformer/Cast", to=TensorProto.FLOAT16))
    nodes.append(helper.make_node(
        "MatMul", ["/transformer/Cast_output_0", "dit.transformer.project_in.weight"],
        ["/transformer/project_in/MatMul_output_0"], name="/transformer/project_in/MatMul",
    ))
    h = "/transformer/project_in/MatMul_output_0"
    for b in range(NB):
        pre = f"/transformer/layers.{b}"
        inits.append(numpy_helper.from_array(_wx(b), f"wx{b}"))
        inits.append(numpy_helper.from_array(_wf(b), f"wf{b}"))
        xa = f"{pre}/cross_attn/to_out/MatMul_output_0"
        nodes.append(helper.make_node("Identity", [h], [f"{pre}/cross_attn/Reshape_5_output_0"], name=f"{pre}/cross_attn/Reshape_5"))
        nodes.append(helper.make_node("MatMul", [f"{pre}/cross_attn/Reshape_5_output_0", f"wx{b}"], [xa], name=f"{pre}/cross_attn/to_out/MatMul"))
        nodes.append(helper.make_node("Cast", [h], [f"{pre}/cross_attend_norm/Cast_output_0"], name=f"{pre}/cross_attend_norm/Cast", to=TensorProto.FLOAT))
        auto = f"{pre}/cross_attend_norm/Cast_output_0__autocast_13"
        nodes.append(helper.make_node("Add", [auto, xa], [f"{pre}/Add_5_output_0"], name=f"{pre}/Add_5"))
        tail.append(helper.make_node("Cast", [f"{pre}/cross_attend_norm/Cast_output_0"], [auto], name=f"{auto}_node", to=TensorProto.FLOAT16))
        r = f"{pre}/Add_5_output_0"
        nodes.append(helper.make_node("Mul", [r, f"wf{b}"], [f"{pre}/Mul_10_output_0"], name=f"{pre}/Mul_10"))
        nodes.append(helper.make_node("Add", [f"{pre}/Mul_10_output_0", r], [f"{pre}/Add_9_output_0"], name=f"{pre}/Add_9"))
        h = f"{pre}/Add_9_output_0"
    nodes.append(helper.make_node("Cast", [h], ["velocity"], name="/transformer/Cast_out", to=TensorProto.FLOAT))
    nodes.extend(tail)
    g = helper.make_graph(
        nodes, "dit",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, H])],
        [helper.make_tensor_value_info("velocity", TensorProto.FLOAT, [1, 2, H])],
        initializer=inits,
    )
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])


def _run(model: onnx.ModelProto, feeds: dict) -> np.ndarray:
    env = dict(feeds)
    for t in model.graph.initializer:
        env[t.name] = numpy_helper.to_array(t)
    dt = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16}
    pending = list(model.graph.node)
    while pending:
        progressed = False
        for n in list(pending):
            if not all(i in env for i in n.input):
                continue
            a = [env[i] for i in n.input]
            attrs = {x.name: helper.get_attribute_value(x) for x in n.attribute}
            op = n.op_type
            if op == "Cast":
                out = a[0].astype(dt[attrs["to"]])
            elif op == "MatMul":
                out = a[0] @ a[1]
            elif op == "Mul":
                out = a[0] * a[1]
            elif op == "Add":
                out = a[0] + a[1]
            elif op == "Identity":
                out = a[0]
            elif op == "Gather":
                out = np.take(a[0], a[1], axis=attrs.get("axis", 0))
            else:
                raise NotImplementedError(op)
            env[n.output[0]] = out
            pending.remove(n)
            progressed = True
        assert progressed, "graph has an unresolvable input"
    return env["velocity"]


def _reference(x: np.ndarray, shift: dict) -> np.ndarray:
    h = x.astype(np.float16)
    for b in range(NB):
        xa = h @ _wx(b)
        if b in shift:
            xa = xa + shift[b].astype(np.float16)
        r = h + xa
        h = r * _wf(b) + r
    return h.astype(np.float32)


def test_cross_attn_outputs_found_structurally():
    outs = find_cross_attn_outputs(_graph().graph)
    assert outs == [f"/transformer/layers.{b}/cross_attn/to_out/MatMul_output_0" for b in range(NB)]


def test_cross_attn_surgery_zero_is_exact_noop():
    x = np.random.default_rng(0).standard_normal((1, 2, H)).astype(np.float32)
    ref = _run(_graph(), {"x": x})
    model, nb, hidden = add_steering_input(_graph(), site=HOOK_CROSS_ATTN_OUTPUT)
    assert (nb, hidden) == (NB, H)
    names = [i.name for i in model.graph.input]
    assert STEERING_CROSS_ATTN_INPUT in names and STEERING_INPUT not in names
    cast = [n for n in model.graph.node if n.name == "/steering_xattn/Cast"][0]
    assert helper.get_attribute_value(cast.attribute[0]) == TensorProto.FLOAT16
    zero = _run(model, {"x": x, STEERING_CROSS_ATTN_INPUT: np.zeros((1, NB, H), np.float32)})
    assert np.array_equal(zero, ref)


def test_cross_attn_row_lands_before_the_feed_forward():
    x = np.random.default_rng(1).standard_normal((1, 2, H)).astype(np.float32)
    model, _, _ = add_steering_input(_graph(), site=HOOK_CROSS_ATTN_OUTPUT)
    s = np.zeros((1, NB, H), np.float32)
    s[0, 1] = np.array([0.5, -0.25, 1.0, 0.0], np.float32)
    got = _run(model, {"x": x, STEERING_CROSS_ATTN_INPUT: s})
    assert np.allclose(got, _reference(x, {1: s[0, 1]}), atol=1e-3)
    # Not the post-block site: the same row there gives a different output.
    post, _, _ = add_steering_input(_graph())
    other = _run(post, {"x": x, STEERING_INPUT: s})
    assert not np.allclose(got, other, atol=1e-3)


def test_both_sites_compose_and_post_block_still_finds_its_outputs():
    model, _, _ = add_steering_input(_graph(), site=HOOK_CROSS_ATTN_OUTPUT)
    # The post-block finder still sees the residual leaving each block.
    assert find_block_outputs(model.graph) == [
        f"/transformer/layers.{b}/Add_9_output_0" for b in range(NB)
    ]
    with pytest.raises(ValueError):
        add_steering_input(model, site=HOOK_CROSS_ATTN_OUTPUT)
    with pytest.raises(ValueError):
        add_steering_input(_graph(), site="nope")


def test_cross_attn_build_identity_is_distinct():
    a = SA3DiTSteerBuildConfig(1, 646, 646)
    b = SA3DiTSteerCrossAttnBuildConfig(1, 646, 646)
    assert b.engine_name() == "sa3_m_dit_steerxa_l1_646_646"
    assert a.engine_name() != b.engine_name()
    assert b.steering_site == HOOK_CROSS_ATTN_OUTPUT
    assert b.steering_surgery == CROSS_ATTN_SURGERY_VERSION


def _engine(root: Path, name: str) -> Path:
    path = root / "sa3" / "trt_engines" / name / f"{name}.trt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"engine")
    return path


def test_discovery_picks_the_engine_for_the_packs_hook(monkeypatch, tmp_path):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    fp8 = _engine(tmp_path, "sa3_m_dit_fp8_l1_646_646")
    steer = _engine(tmp_path, "sa3_m_dit_steer_l1_646_646")
    xa = _engine(tmp_path, "sa3_m_dit_steerxa_l1_646_646")
    assert sa3_trt.find_dit_engine("medium", 646) == fp8
    assert sa3_trt.find_dit_engine("medium", 646, want_steering=True) == steer
    assert sa3_trt.find_dit_engine(
        "medium", 646, want_steering=HOOK_POST_BLOCK_RESIDUAL,
    ) == steer
    assert sa3_trt.find_dit_engine(
        "medium", 646, want_steering=HOOK_CROSS_ATTN_OUTPUT,
    ) == xa
    xa.unlink()
    # No cross-attn engine: never the wrong site; the default engine runs.
    assert sa3_trt.find_dit_engine(
        "medium", 646, want_steering=HOOK_CROSS_ATTN_OUTPUT,
    ) == fp8


# ---- eager delivery through the real pipeline hook path -------------------


class _XAttn(torch.nn.Module):
    def forward(self, x):
        return x * 2.0


class _Block(torch.nn.Module):
    """x + cross_attn(x), then a nonlinear feed-forward stand-in."""

    def __init__(self):
        super().__init__()
        self.dim = H
        self.cross_attn = _XAttn()

    def forward(self, x):
        r = x + self.cross_attn(x)
        return r + torch.tanh(r)


class _Trunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block() for _ in range(NB)])


class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = _Trunk()


class _Wrapper(torch.nn.Module):
    """Shape of ``sam.model.model``: wrapper.model.transformer.layers."""

    def __init__(self):
        super().__init__()
        self.model = _Inner()

    def run(self, x):
        for blk in self.model.transformer.layers:
            x = blk(x)
        return x


def test_eager_adapter_hooks_cross_attn_modules(monkeypatch):
    from acestep.engine.sa3_adapter import SA3Adapter
    from acestep.engine.stream import StreamPipeline

    w = _Wrapper()
    ad = SA3Adapter(w, schedule_builder=lambda d: torch.linspace(d, 0, 3),
                    device="cpu", dtype=torch.float32)
    lay = ad.steering_layout()
    assert lay.hook == HOOK_POST_BLOCK_RESIDUAL and not lay.engine_input
    ad.steering_hook = HOOK_CROSS_ATTN_OUTPUT
    lay = ad.steering_layout()
    assert lay.hook == HOOK_CROSS_ATTN_OUTPUT and lay.num_blocks == NB
    blocks = ad.steering_blocks()
    assert [type(m) for m in blocks] == [_XAttn] * NB
    assert ad.steering_blocks() is blocks  # identity-stable across ticks

    # Drive the pipeline's own hook install + fill with a bare instance
    # (the hook machinery reads only the steering attributes).
    pipe = StreamPipeline.__new__(StreamPipeline)
    pipe.adapter = ad
    pipe._steering_by_layer = {}
    pipe._steering_hooks_installed = False
    pipe._steering_hooked_blocks = None
    pipe._steering_hook_handles = []
    pipe._current_step_per_row = [0]

    x = torch.randn(1, 5, H)
    ref = w.run(x.clone())
    v = torch.tensor([1.0, 0.0, -1.0, 0.5])
    pipe.set_steering([{"layer": 1, "step": 0, "vector": v, "magnitude": 1.0, "alpha": 3.0}])
    got = w.run(x.clone())
    # Reference: block 1's cross-attention output shifted by 3 * v.
    h = x.clone()
    for b, blk in enumerate(w.model.transformer.layers):
        xa = h * 2.0
        if b == 1:
            xa = xa + 3.0 * v
        r = h + xa
        h = r + torch.tanh(r)
    assert torch.allclose(got, h, atol=1e-6)
    pipe.set_steering([])
    assert torch.equal(w.run(x.clone()), ref)
    pipe.remove_steering_hooks()
    assert all(not m._forward_hooks for m in blocks)
