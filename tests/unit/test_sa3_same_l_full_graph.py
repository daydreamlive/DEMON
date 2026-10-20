"""SAME-L full-song decoder graph: the FP16 attention-projection rewrite and
the builder's routing between it and upstream's window graph.

CPU-only: a small synthetic graph with upstream's per-block attention shape
(Cast(FP32) -> to_qkv MatMul ... Reshape -> to_out MatMul -> Cast(FP16)).
"""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from acestep.engine.trt import sa3_build  # noqa: E402

D = 8


def _block(b: int, src: str, rng, *, shape_reader: bool, out_cast_to=TensorProto.FLOAT16):
    """One attention block: returns (nodes, initializers, output name)."""
    p = f"/dec/blocks.{b}/attn/"
    wq = numpy_helper.from_array(
        rng.standard_normal((3 * D, D)).astype(np.float32) * 0.1,
        f"dec.blocks.{b}.attn.to_qkv.weight")
    wo = numpy_helper.from_array(
        rng.standard_normal((D, D)).astype(np.float32) * 0.1,
        f"dec.blocks.{b}.attn.to_out.weight")
    shape_c = helper.make_tensor(f"{p}shape_c", TensorProto.INT64, [3], [1, -1, D])
    nodes = [
        helper.make_node("Cast", [src], [p + "Cast_output_0"], name=p + "Cast",
                         to=TensorProto.FLOAT),
        helper.make_node("Transpose", [wq.name], [p + "to_qkv/Transpose_output_0"],
                         name=p + "to_qkv/Transpose", perm=[1, 0]),
        helper.make_node("MatMul", [p + "Cast_output_0", p + "to_qkv/Transpose_output_0"],
                         [p + "to_qkv/MatMul_output_0"], name=p + "to_qkv/MatMul"),
        # stands in for the Slices + attention plugin (FP32 in, FP32 out)
        helper.make_node("Split", [p + "to_qkv/MatMul_output_0"],
                         [p + "q", p + "k", p + "v"], name=p + "Split", axis=-1,
                         num_outputs=3),
        helper.make_node("Reshape", [p + "v", f"{p}shape_c"], [p + "Reshape_output_0"],
                         name=p + "Reshape"),
        helper.make_node("Transpose", [wo.name], [p + "to_out/Transpose_output_0"],
                         name=p + "to_out/Transpose", perm=[1, 0]),
        helper.make_node("MatMul", [p + "Reshape_output_0", p + "to_out/Transpose_output_0"],
                         [p + "to_out/MatMul_output_0"], name=p + "to_out/MatMul"),
        helper.make_node("Cast", [p + "to_out/MatMul_output_0"], [p + "Cast_10_output_0"],
                         name=p + "Cast_10", to=out_cast_to),
    ]
    if shape_reader:
        nodes.insert(1, helper.make_node("Shape", [p + "Cast_output_0"],
                                         [p + "Shape_output_0"], name=p + "Shape"))
    return nodes, [wq, wo, shape_c], p + "Cast_10_output_0"


def _model(n_blocks: int = 2, *, out_cast_to=TensorProto.FLOAT16, wscale: float = 1.0):
    rng = np.random.default_rng(0)
    nodes, inits, src = [], [], "latent"
    for b in range(n_blocks):
        bn, bi, src = _block(b, src, rng, shape_reader=(b == 0), out_cast_to=out_cast_to)
        nodes += bn
        inits += bi
    if wscale != 1.0:
        a = numpy_helper.to_array(inits[0]) * wscale
        inits[0].CopyFrom(numpy_helper.from_array(a.astype(np.float32), inits[0].name))
    graph = helper.make_graph(
        nodes, "same_l_toy",
        [helper.make_tensor_value_info("latent", TensorProto.FLOAT16, [1, "T", D])],
        [helper.make_tensor_value_info(src, out_cast_to, [1, "T", D])],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _dtypes(model) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    g = inferred.graph
    return {v.name: v.type.tensor_type.elem_type
            for v in list(g.value_info) + list(g.input) + list(g.output)}


def _node(model, name):
    return next(n for n in model.graph.node if n.name == name)


def test_rewrite_runs_both_projections_in_fp16():
    m = _model()
    assert sa3_build._same_l_attn_proj_fp16(m) == 2
    onnx.checker.check_model(m)
    dt = _dtypes(m)
    inits = {i.name: i.data_type for i in m.graph.initializer}
    for b in range(2):
        p = f"/dec/blocks.{b}/attn/"
        qkv, out = _node(m, p + "to_qkv/MatMul"), _node(m, p + "to_out/MatMul")
        assert dt[qkv.input[0]] == TensorProto.FLOAT16
        assert dt[qkv.output[0]] == TensorProto.FLOAT16
        assert dt[out.input[0]] == TensorProto.FLOAT16
        assert dt[out.output[0]] == TensorProto.FLOAT16
        assert inits[f"dec.blocks.{b}.attn.to_qkv.weight"] == TensorProto.FLOAT16
        assert inits[f"dec.blocks.{b}.attn.to_out.weight"] == TensorProto.FLOAT16
        # what downstream reads keeps its upstream name and FP32 dtype
        assert dt[p + "to_qkv/MatMul_output_0"] == TensorProto.FLOAT
        assert dt[p + "Reshape_output_0"] == TensorProto.FLOAT
    # the graph is still topologically ordered and its IO unchanged
    assert [i.name for i in m.graph.input] == ["latent"]
    assert m.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16


def test_input_cast_kept_only_while_something_else_reads_it():
    m = _model()
    sa3_build._same_l_attn_proj_fp16(m)
    names = {n.name for n in m.graph.node}
    assert "/dec/blocks.0/attn/Cast" in names  # its Shape still reads it
    assert "/dec/blocks.1/attn/Cast" not in names  # to_qkv was the only reader


def test_rewrite_matches_the_fp32_graph_numerically():
    ort = pytest.importorskip("onnxruntime")
    ref, new = _model(), _model()
    sa3_build._same_l_attn_proj_fp16(new)
    x = np.random.default_rng(1).standard_normal((1, 64, D)).astype(np.float16)

    def run(m):
        sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        return sess.run(None, {"latent": x})[0].astype(np.float64)

    a, b = run(ref), run(new)
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.9999


def test_rewrite_refuses_a_changed_upstream_graph():
    with pytest.raises(RuntimeError, match="Cast\\(FP16\\)"):
        sa3_build._same_l_attn_proj_fp16(_model(out_cast_to=TensorProto.FLOAT))
    empty = helper.make_model(helper.make_graph([], "e", [], []))
    with pytest.raises(RuntimeError, match="upstream graph changed"):
        sa3_build._same_l_attn_proj_fp16(empty)


def test_rewrite_refuses_weights_outside_fp16_range():
    with pytest.raises(RuntimeError, match="does not fit FP16"):
        sa3_build._same_l_attn_proj_fp16(_model(wscale=1e6))


def test_derived_graph_is_cached_by_recipe_and_source_hash(tmp_path, monkeypatch):
    src = tmp_path / "dec_fp16.onnx"
    onnx.save(_model(), str(src))
    sha = "ab" * 32
    out = sa3_build._derive_same_l_full_onnx(str(src), str(tmp_path / "eng"), sha)
    assert out.endswith(f"same_l_dec_{sa3_build.SAME_L_FULL_RECIPE}_{sha[:12]}.onnx")
    derived = onnx.load(out)
    assert any(n.name.endswith("demon_cast_fp32") for n in derived.graph.node)
    assert sorted(p.name for p in (tmp_path / "eng" / "_onnx").iterdir()) == [
        f"same_l_dec_{sa3_build.SAME_L_FULL_RECIPE}_{sha[:12]}.onnx"]

    def boom(_model):
        raise AssertionError("cached graph must not be re-derived")

    monkeypatch.setattr(sa3_build, "_same_l_attn_proj_fp16", boom)
    assert sa3_build._derive_same_l_full_onnx(str(src), str(tmp_path / "eng"), sha) == out


@pytest.mark.parametrize("profile, full", [((32, 56, 96), False), ((32, 1292, 3230), True)])
def test_builder_routes_only_full_song_profiles_through_the_rewrite(
    tmp_path, monkeypatch, profile, full,
):
    upstream = tmp_path / "dec_fp16.onnx"
    upstream.write_bytes(b"upstream graph bytes")
    calls = {}
    monkeypatch.setattr(sa3_build, "_fetch_onnx", lambda files: str(upstream))
    monkeypatch.setattr(sa3_build, "_register_same_plugin", lambda: None)
    monkeypatch.setattr(
        sa3_build, "_derive_same_l_full_onnx",
        lambda src, out_dir, sha: calls.setdefault("derived", (src, sha)) and "derived.onnx",
    )
    monkeypatch.setattr(
        sa3_build, "_build_strongly_typed_engine",
        lambda **kw: calls.setdefault("build", kw),
    )
    monkeypatch.setattr(
        sa3_build, "_write_metadata",
        lambda **kw: calls.setdefault("meta", kw["expected"]),
    )
    env = {"packages": {"tensorrt": "10.x"}, "active_gpu": {}}
    cfg = sa3_build.SameLWindowBuildConfig(*profile)
    label, path, _, status = sa3_build._build_same_l_window_engine(
        output_dir=str(tmp_path / "eng"), config=cfg, env=env,
    )
    assert status == "OK"
    # the directory name the runtime finders match is unchanged
    assert path.endswith(f"{cfg.engine_name()}.trt")
    # identity hashes upstream's graph in both cases
    assert calls["meta"]["onnx_path"] == str(upstream.resolve())
    if full:
        assert calls["build"]["onnx_path"] == "derived.onnx"
        assert calls["derived"][0] == str(upstream)
        assert calls["derived"][1] == calls["meta"]["onnx_sha256"]
        assert calls["meta"]["component"] == (
            f"same_l_decode_full_{sa3_build.SAME_L_FULL_RECIPE}")
        assert "full-song" in label
    else:
        assert calls["build"]["onnx_path"] == str(upstream)
        assert "derived" not in calls
        assert calls["meta"]["component"] == "same_l_decode_window"


def test_full_engine_built_before_the_rewrite_is_rebuilt(tmp_path, monkeypatch):
    """An engine whose sidecar names the plain window component (built from
    upstream's graph as published) no longer matches at a full profile."""
    upstream = tmp_path / "dec_fp16.onnx"
    upstream.write_bytes(b"upstream graph bytes")
    env = {"packages": {"tensorrt": "10.x"}, "active_gpu": {}}
    cfg = sa3_build.SameLWindowBuildConfig(32, 1292, 3230)
    name = cfg.engine_name()
    eng = tmp_path / name / f"{name}.trt"
    eng.parent.mkdir(parents=True)
    eng.write_bytes(b"")
    old = sa3_build._expected_metadata(
        component="same_l_decode_window", onnx_path=str(upstream), config=cfg, env=env)
    sa3_build._write_metadata(engine_path=str(eng), expected=old, env=env)
    built = []
    monkeypatch.setattr(sa3_build, "_fetch_onnx", lambda files: str(upstream))
    monkeypatch.setattr(sa3_build, "_register_same_plugin", lambda: None)
    monkeypatch.setattr(sa3_build, "_derive_same_l_full_onnx", lambda *a: "derived.onnx")
    monkeypatch.setattr(sa3_build, "_build_strongly_typed_engine", lambda **kw: built.append(kw))
    *_, status = sa3_build._build_same_l_window_engine(
        output_dir=str(tmp_path), config=cfg, env=env)
    assert status == "OK" and len(built) == 1
    # and the rebuilt sidecar is current on the next run
    *_, status = sa3_build._build_same_l_window_engine(
        output_dir=str(tmp_path), config=cfg, env=env)
    assert status == "SKIPPED" and len(built) == 1
