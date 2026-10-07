"""SA3 steering engine: ONNX surgery, engine discovery, adapter delivery.

CPU only. The surgery runs on a small synthetic graph that copies the
upstream fp16mixed DiT's naming and structure where the surgery depends on
them (``/transformer/layers.N/`` scopes, a residual ``Add`` leaving each
block, fp16 trunk fed by an autocast ``Cast``, a non-residual side tensor
crossing blocks, autocast nodes appended out of order). The result is
evaluated with a tiny numpy interpreter, so no onnxruntime is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from acestep.engine import sa3_trt
from acestep.engine.trt.sa3_build import SA3DiTBuildConfig, SA3DiTSteerBuildConfig
from acestep.engine.trt.sa3_steering_onnx import (
    STEERING_INPUT,
    STEERING_SURGERY_VERSION,
    add_steering_input,
    find_block_outputs,
)

H = 4
NB = 3


def _graph() -> onnx.ModelProto:
    """x[1,2,H] fp32 -> project_in (cast fp16) -> NB blocks -> out fp32.

    Block b: h = Add(Mul(h, w_b), Cast_fp16(Cast_fp32(h))) i.e. h*w + h,
    plus a rotary-like side tensor (Cast) consumed by the next block.
    """
    nodes = []
    inits = [
        numpy_helper.from_array(np.eye(H, dtype=np.float16), "dit.transformer.project_in.weight"),
    ]
    nodes.append(helper.make_node("Cast", ["x"], ["/transformer/Cast_output_0"], name="/transformer/Cast", to=TensorProto.FLOAT16))
    nodes.append(helper.make_node(
        "MatMul", ["/transformer/Cast_output_0", "dit.transformer.project_in.weight"],
        ["/transformer/project_in/MatMul_output_0"], name="/transformer/project_in/MatMul",
    ))
    h = "/transformer/project_in/MatMul_output_0"
    tail_casts = []
    for b in range(NB):
        pre = f"/transformer/layers.{b}"
        w = f"w{b}"
        inits.append(numpy_helper.from_array(np.full((H,), 1.0 + 0.5 * b, dtype=np.float16), w))
        nodes.append(helper.make_node("Cast", [h], [f"{pre}/ff_norm/Cast_output_0"], name=f"{pre}/ff_norm/Cast", to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Mul", [h, w], [f"{pre}/Mul_10_output_0"], name=f"{pre}/Mul_10"))
        # Residual Add reads the autocast Cast that is appended LATER.
        nodes.append(helper.make_node(
            "Add", [f"{pre}/Mul_10_output_0", f"{pre}/ff_norm/Cast_output_0__autocast"],
            [f"{pre}/Add_9_output_0"], name=f"{pre}/Add_9",
        ))
        tail_casts.append(helper.make_node(
            "Cast", [f"{pre}/ff_norm/Cast_output_0"], [f"{pre}/ff_norm/Cast_output_0__autocast"],
            name=f"{pre}/ff_norm/Cast_output_0__autocast_node", to=TensorProto.FLOAT16,
        ))
        # Side tensor crossing to the next block (not the residual).
        nodes.append(helper.make_node("Cast", [w], [f"{pre}/self_attn/Cast_17_output_0"], name=f"{pre}/self_attn/Cast_17", to=TensorProto.FLOAT16))
        if b > 0:
            prev = f"/transformer/layers.{b - 1}/self_attn/Cast_17_output_0"
            nodes.append(helper.make_node("Identity", [prev], [f"{pre}/self_attn/Cast_14_output_0"], name=f"{pre}/self_attn/Cast_14"))
        h = f"{pre}/Add_9_output_0"
    nodes.append(helper.make_node("Cast", [h], ["velocity"], name="/transformer/Cast_out", to=TensorProto.FLOAT))
    nodes.extend(tail_casts)
    g = helper.make_graph(
        nodes, "dit",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, H])],
        [helper.make_tensor_value_info("velocity", TensorProto.FLOAT, [1, 2, H])],
        initializer=inits,
    )
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])


def _run(model: onnx.ModelProto, feeds: dict) -> np.ndarray:
    """Minimal interpreter for the ops the synthetic graph uses."""
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
            if n.op_type == "Cast":
                out = a[0].astype(dt[attrs["to"]])
            elif n.op_type == "MatMul":
                out = a[0] @ a[1]
            elif n.op_type == "Mul":
                out = a[0] * a[1]
            elif n.op_type == "Add":
                out = a[0] + a[1]
            elif n.op_type == "Identity":
                out = a[0]
            elif n.op_type == "Gather":
                out = np.take(a[0], a[1], axis=attrs.get("axis", 0))
            else:
                raise NotImplementedError(n.op_type)
            env[n.output[0]] = out
            pending.remove(n)
            progressed = True
        assert progressed, "graph has an unresolvable input"
    return env["velocity"]


def test_block_outputs_found_structurally():
    outs = find_block_outputs(_graph().graph)
    assert outs == [f"/transformer/layers.{b}/Add_9_output_0" for b in range(NB)]


def test_surgery_adds_input_and_zero_is_exact_noop():
    x = np.random.default_rng(0).standard_normal((1, 2, H)).astype(np.float32)
    ref = _run(_graph(), {"x": x})
    model, nb, hidden = add_steering_input(_graph())
    assert (nb, hidden) == (NB, H)
    inp = [i for i in model.graph.input if i.name == STEERING_INPUT][0]
    assert inp.type.tensor_type.elem_type == TensorProto.FLOAT
    assert [d.dim_value for d in inp.type.tensor_type.shape.dim] == [1, NB, H]
    zero = _run(model, {"x": x, STEERING_INPUT: np.zeros((1, NB, H), np.float32)})
    assert np.array_equal(zero, ref)


def test_steering_row_lands_after_its_block():
    x = np.random.default_rng(1).standard_normal((1, 2, H)).astype(np.float32)
    model, _, _ = add_steering_input(_graph())
    s = np.zeros((1, NB, H), np.float32)
    s[0, 1] = np.array([0.5, -0.25, 1.0, 0.0], np.float32)
    got = _run(model, {"x": x, STEERING_INPUT: s})
    # Reference: same math, the shift added to block 1's output.
    h = x.astype(np.float16)
    for b in range(NB):
        h = h * np.float16(1.0 + 0.5 * b) + h
        if b == 1:
            h = h + s[0, 1].astype(np.float16)
    assert np.allclose(got, h.astype(np.float32), atol=1e-3)


def test_surgery_refuses_a_second_pass():
    model, _, _ = add_steering_input(_graph())
    with pytest.raises(ValueError):
        add_steering_input(model)


def test_steer_build_identity_is_distinct():
    a = SA3DiTBuildConfig(1, 646, 646)
    b = SA3DiTSteerBuildConfig(1, 646, 646)
    assert b.engine_name() == "sa3_m_dit_steer_l1_646_646"
    assert a.engine_name() != b.engine_name()
    assert b.steering_surgery == STEERING_SURGERY_VERSION


def _engine(root: Path, name: str) -> Path:
    path = root / "sa3" / "trt_engines" / name / f"{name}.trt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"engine")
    return path


def test_discovery_prefers_steer_engine_only_when_asked(monkeypatch, tmp_path):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    plain = _engine(tmp_path, "sa3_m_dit_l1_646_646")
    fp8 = _engine(tmp_path, "sa3_m_dit_fp8_l1_646_646")
    steer = _engine(tmp_path, "sa3_m_dit_steer_l1_646_646")
    refit = _engine(tmp_path, "sa3_m_dit_refit_l1_646_646")
    assert sa3_trt.find_dit_engine("medium", 646) == fp8
    assert sa3_trt.find_dit_engine("medium", 646, want_steering=True) == steer
    # LoRA keeps precedence (refit engines carry no steering input).
    assert sa3_trt.find_dit_engine(
        "medium", 646, want_refittable=True, want_steering=True,
    ) == refit
    assert sa3_trt.max_dit_engine_latents("medium") == 646
    steer.unlink()
    assert sa3_trt.find_dit_engine("medium", 646, want_steering=True) == fp8
    fp8.unlink()
    assert sa3_trt.find_dit_engine("medium", 646, want_steering=True) == plain


class _FakeTrtDit:
    """SA3TRTDit stand-in: batch-1 stepping with a steering input."""

    trt_batch1 = True
    steering_shape = (3, 4)

    def __init__(self):
        self.calls = []

    def step_bundle(self, x_1ct, t, bundle, steering=None):
        self.calls.append(None if steering is None else steering.clone())
        return torch.zeros_like(x_1ct)


def test_sa3_adapter_routes_rows_to_the_engine():
    from acestep.engine.sa3_adapter import SA3Adapter

    dit = _FakeTrtDit()
    ad = SA3Adapter(dit, schedule_builder=lambda d: torch.linspace(d, 0, 3),
                    device="cpu", dtype=torch.float32)
    lay = ad.steering_layout()
    assert lay.engine_input and (lay.num_blocks, lay.hidden_size) == (3, 4)
    assert ad.steering_blocks() is None
    x = torch.zeros(2, 5, 256)
    steering = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    ad.batched_forward(x, [1.0, 0.5], [None] * 2, [None] * 2, [None] * 2,
                       [{}, {}], steering=steering)
    assert torch.equal(dit.calls[0], steering[0:1])
    assert torch.equal(dit.calls[1], steering[1:2])
    ad.batched_forward(x, [1.0, 0.5], [None] * 2, [None] * 2, [None] * 2,
                       [{}, {}], steering=None)
    assert dit.calls[2] is None and dit.calls[3] is None


def test_sa3_adapter_without_engine_input_has_no_layout():
    from acestep.engine.sa3_adapter import SA3Adapter

    dit = _FakeTrtDit()
    dit.steering_shape = None
    ad = SA3Adapter(dit, schedule_builder=lambda d: torch.linspace(d, 0, 3),
                    device="cpu", dtype=torch.float32)
    assert ad.steering_layout() is None
