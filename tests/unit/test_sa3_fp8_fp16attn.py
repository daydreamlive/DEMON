"""fp16 attention cores for the fp8 DiT graph, and the fp8 refit build
that uses them (CPU: synthetic graph, fake ``tensorrt``)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("torch")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from acestep.engine.trt import sa3_build as B  # noqa: E402
from acestep.engine.trt import sa3_fp8_fp16attn as A  # noqa: E402
from acestep.engine.trt._engine_metadata import metadata_path  # noqa: E402
from test_sa3_build_from_onnx import ENV, _Rec, make_fake_trt  # noqa: E402

W = "dit.transformer.layers.0.self_attn.to_qkv.weight"
BLK = "/transformer/layers.0/self_attn/"


def make_graph():
    """x -> fp8 Q/DQ Linear -> h; QK^T = MatMul(h, h^T) -> Softmax -> PV =
    MatMul(softmax, h) -> y, named like Stability's dit_fp8.onnx."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 8)).astype(np.float32)
    scale = (np.abs(w).max(1) / 448.0).astype(np.float32)
    zp = helper.make_tensor("w_zp", TensorProto.FLOAT8E4M3FN, [8], [0.0] * 8)
    nodes = [
        helper.make_node("QuantizeLinear", [W, "w_scale", "w_zp"], ["w_q"], name=f"{BLK}to_qkv/QuantizeLinear", axis=0),
        helper.make_node("DequantizeLinear", ["w_q", "w_scale", "w_zp"], ["w_dq"], name=f"{BLK}to_qkv/DequantizeLinear", axis=0),
        helper.make_node("Transpose", ["w_dq"], ["w_t"], name=f"{BLK}to_qkv/Transpose", perm=[1, 0]),
        helper.make_node("MatMul", ["x", "w_t"], ["h"], name=f"{BLK}to_qkv/MatMul"),
        helper.make_node("Transpose", ["h"], ["h_t"], name=f"{BLK}Transpose_5", perm=[0, 2, 1]),
        helper.make_node("MatMul", ["h", "h_t"], ["qk"], name=f"{BLK}MatMul"),
        helper.make_node("Softmax", ["qk"], ["sm"], name=f"{BLK}Softmax", axis=-1),
        helper.make_node("MatMul", ["sm", "h"], ["pv"], name=f"{BLK}MatMul_1"),
        helper.make_node("Add", ["pv", "x"], ["y"], name=f"{BLK}Add_out"),
    ]
    g = helper.make_graph(
        nodes, "tiny_fp8_attn",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8])],
        initializer=[numpy_helper.from_array(w, W), numpy_helper.from_array(scale, "w_scale"), zp],
    )
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)])


def _producer(model, tensor):
    return next(n for n in model.graph.node if tensor in n.output)


def _cast_to(node):
    return next(a.i for a in node.attribute if a.name == "to")


def test_patch_casts_q_k_v_and_pv_output():
    m = make_graph()
    n0 = len(m.graph.node)
    stats = A.patch_model(m)
    # q = h, k = h^T, v = h: q and v share one fp16 Cast, k has its own.
    assert stats == {"cast_fp16": 2, "cast_fp32": 1, "softmax": 1}
    assert len(m.graph.node) == n0 + 3
    qk = next(n for n in m.graph.node if n.name == f"{BLK}MatMul")
    pv = next(n for n in m.graph.node if n.name == f"{BLK}MatMul_1")
    for t in (qk.input[0], pv.input[1]):
        c = _producer(m, t)
        assert c.op_type == "Cast" and _cast_to(c) == TensorProto.FLOAT16 and c.input[0] == "h"
    k = _producer(m, qk.input[1])
    assert k.op_type == "Cast" and _cast_to(k) == TensorProto.FLOAT16 and k.input[0] == "h_t"
    back = _producer(m, "pv")
    assert back.op_type == "Cast" and _cast_to(back) == TensorProto.FLOAT and back.input[0] == pv.output[0]
    assert A.softmax_input_types(m) == {"FLOAT16": 1}
    onnx.checker.check_model(m)
    inf = onnx.shape_inference.infer_shapes(m)
    types = {v.name: v.type.tensor_type.elem_type for v in inf.graph.value_info}
    assert types["pv"] == TensorProto.FLOAT


def test_patch_rejects_graph_without_attention():
    m = make_graph()
    for n in m.graph.node:
        if n.op_type == "Softmax":
            n.name = "/elsewhere/Softmax"
    with pytest.raises(RuntimeError, match="no attention Softmax"):
        A.patch_model(m)


def test_patch_fp8_attention_fp16_writes_file(tmp_path):
    src = tmp_path / "dit_fp8.onnx"
    onnx.save(make_graph(), str(src))
    stats = A.patch_fp8_attention_fp16(src, tmp_path / "out" / "patched.onnx")
    assert stats["softmax"] == 1 and stats["nodes_after"] == stats["nodes_before"] + 3
    out = onnx.load(str(tmp_path / "out" / "patched.onnx"))
    assert A.softmax_input_types(out) == {"FLOAT16": 1}


class _Net:
    def __init__(self, rec, refuse=()):
        self.rec, self.refuse = rec, set(refuse)

    def mark_weights_refittable(self, name):
        self.rec.calls.append(("mark", name))
        return name not in self.refuse


@pytest.fixture
def fake_trt_individual(monkeypatch):
    rec = _Rec()
    trt = make_fake_trt(rec)
    trt.BuilderFlag.REFIT_INDIVIDUAL = 9
    trt.Builder.create_network = lambda self, flags: (rec.calls.append(("network_flags", flags)), _Net(rec))[1]
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    return rec


def test_fp8_refit_build_marks_only_weights(tmp_path, fake_trt_individual):
    rec = fake_trt_individual
    src = tmp_path / "src" / "dit_fp8.onnx"
    src.parent.mkdir()
    onnx.save(make_graph(), str(src))
    cfg = B.SA3DiTFp8RefitBuildConfig(1, 140, 140)
    out = tmp_path / "engines"
    _, path, _, status = B._build_dit_engine(
        output_dir=str(out), config=cfg, env=ENV, component="sa3_m_dit_fp8_refit",
        precision_label="fp8 refit", local_onnx=str(src), refit=True,
    )
    assert status == "OK"
    assert [c[1] for c in rec.calls if c[0] == "mark"] == [W]
    flags = [c[1] for c in rec.calls if c[0] == "flag"]
    assert flags == [9]  # REFIT_INDIVIDUAL only, never whole-graph REFIT
    derived = out / "_onnx" / B.FP8_FP16ATTN_ONNX
    parsed = [c[1] for c in rec.calls if c[0] == "parse"]
    assert parsed == [str(derived)] and derived.is_file()
    assert A.softmax_input_types(onnx.load(str(derived))) == {"FLOAT16": 1}
    meta = json.loads(metadata_path(path).read_text())
    c = meta["config"]
    assert c["refit_mode"] == "individual" and c["attention_dtype"] == "fp16"
    from acestep.engine.trt.sa3_export import onnx_sha256
    assert c["source_onnx_sha"] == onnx_sha256(src)
    side = json.loads((out / "_onnx" / (B.FP8_FP16ATTN_ONNX + ".source.json")).read_text())
    assert side["source_onnx_sha"] == c["source_onnx_sha"]

    # Second build: the derived graph is reused (same bytes), engine skipped.
    before = derived.stat().st_mtime_ns
    _, _, _, status2 = B._build_dit_engine(
        output_dir=str(out), config=B.SA3DiTFp8RefitBuildConfig(1, 140, 140), env=ENV,
        component="sa3_m_dit_fp8_refit", precision_label="fp8 refit", local_onnx=str(src), refit=True,
    )
    assert status2 == "SKIPPED" and derived.stat().st_mtime_ns == before


def test_refit_individual_raises_when_mark_refused(tmp_path, monkeypatch):
    rec = _Rec()
    trt = make_fake_trt(rec)
    trt.BuilderFlag.REFIT_INDIVIDUAL = 9
    trt.Builder.create_network = lambda self, flags: _Net(rec, refuse={"w"})
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    src = tmp_path / "m.onnx"
    onnx.save(make_graph(), str(src))
    with pytest.raises(RuntimeError, match="mark_weights_refittable failed"):
        B._build_strongly_typed_engine(
            onnx_path=str(src), engine_path=str(tmp_path / "e.trt"), workspace_gb=1,
            profile_shapes={"x": ((1, 4, 8),) * 3}, refit_weight_names=["w"],
        )
    with pytest.raises(RuntimeError, match="empty"):
        B._build_strongly_typed_engine(
            onnx_path=str(src), engine_path=str(tmp_path / "e.trt"), workspace_gb=1,
            profile_shapes={"x": ((1, 4, 8),) * 3}, refit_weight_names=[],
        )


def test_fp16mixed_refit_build_keeps_whole_graph_refit(tmp_path, fake_trt_individual):
    rec = fake_trt_individual
    src = tmp_path / "dit_fp16.onnx"
    src.write_bytes(b"proto")
    B._build_dit_engine(output_dir=str(tmp_path), config=B.SA3DiTRefitBuildConfig(1, 140, 140),
                        env=ENV, component="sa3_m_dit_refit", precision_label="fp16mixed refit",
                        local_onnx=str(src), refit=True)
    assert [c[1] for c in rec.calls if c[0] == "flag"] == [1]
    assert not [c for c in rec.calls if c[0] == "mark"]


def test_all_fp8_matrix_lists_fp8_refit_variants():
    import argparse

    args = argparse.Namespace(model="medium", duration=None, same_l_only=False, dit_only=True,
                              fp8=True, refit=False)
    names = [n for _, n in B._matrix_jobs(args)]
    for lo, opt, hi in B.CANONICAL_DIT_PROFILES:
        assert B.SA3DiTFp8RefitBuildConfig(lo, opt, hi).engine_name() in names
