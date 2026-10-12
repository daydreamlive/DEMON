"""CPU tests for acestep.engine.trt.sa3_fp8 (calibration capture + fp8 graft)
on the tiny fake DiT from test_sa3_export.py."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")

import numpy as np  # noqa: E402
from onnx import TensorProto, numpy_helper  # noqa: E402

from acestep.engine.trt import sa3_export as X  # noqa: E402
from acestep.engine.trt import sa3_fp8 as Q  # noqa: E402
from test_sa3_export import ELIGIBLE, export_tiny, make_tiny, tiny_inputs  # noqa: E402


def _capture(model, n=3):
    def run():
        for s in range(n):
            model(**tiny_inputs(seed=s))

    return Q.capture_activation_scales(model, run)


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def test_capture_keys_values_and_json():
    model = make_tiny()
    scales = _capture(model)
    assert sorted(scales) == ELIGIBLE                     # min(in, out) >= 128 only
    json.dumps(scales)
    assert all(isinstance(v, float) and v > 0 for v in scales.values())
    # value = max |input| over all calls, checked on to_cond_embed.0 (input = cond)
    want = max(float(tiny_inputs(seed=s)["cond"].abs().max()) for s in range(3))
    assert scales["to_cond_embed.0"] == pytest.approx(want, rel=1e-6)


def test_capture_removes_hooks_and_min_features():
    model = make_tiny()
    Q.capture_activation_scales(model, lambda: model(**tiny_inputs()))
    assert all(not m._forward_pre_hooks for m in model.modules())
    small = Q.capture_activation_scales(model, lambda: model(**tiny_inputs()), min_features=32)
    assert "project_in" in small and "project_out" in small


def test_capture_skips_never_called_and_rejects_nonfinite():
    model = make_tiny()
    assert Q.capture_activation_scales(model, lambda: None) == {}
    inp = tiny_inputs()
    inp["cond"][0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        Q.capture_activation_scales(model, lambda: model(**inp))


def test_act_scales_io(tmp_path):
    scales = {"b": 2.0, "a": 1.5}
    sha = Q.write_act_scales(tmp_path / "s.json", scales)
    assert len(sha) == 64
    assert Q.read_act_scales(tmp_path / "s.json") == scales
    (tmp_path / "w.json").write_text(json.dumps({"scales": scales}))
    assert Q.read_act_scales(tmp_path / "w.json") == scales


# ---------------------------------------------------------------------------
# graft
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("graft")
    model, rep = export_tiny(tmp)
    scales = _capture(model)
    return tmp, rep, scales


def _graph(path):
    return onnx.load(str(path), load_external_data=True)


def _qdq_for(model, site_node_name):
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    node = next(n for n in g.node if n.name == site_node_name)
    a_dq, w_dq = prod[node.input[0]], prod[node.input[1]]
    assert a_dq.op_type == "DequantizeLinear" and w_dq.op_type == "DequantizeLinear"
    return node, a_dq, w_dq, {i.name: i for i in g.initializer}


def test_graft_counts_and_structure(exported):
    tmp, rep, scales = exported
    out = tmp / "tiny_fp8.onnx"
    r = Q.graft_fp8(rep.onnx_path, out, scales)
    assert (r.linears_total, r.linears_fp8, r.unmatched, r.unused_scales) == (8, 8, [], [])
    m = _graph(out)
    ops = [n.op_type for n in m.graph.node]
    assert ops.count("QuantizeLinear") == 16 and ops.count("DequantizeLinear") == 16
    assert next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")) == 19
    assert X.is_topologically_sorted(m.graph)
    onnx.checker.check_model(str(out))
    # the opset 17 -> 19 Reduce* rewrite: axes are inputs now
    for n in m.graph.node:
        if n.op_type == "ReduceMean":
            assert not any(a.name == "axes" for a in n.attribute) and len(n.input) == 2
    assert r.sha256 == X.onnx_sha256(out)
    assert r.act_scales_sha256 == Q.act_scales_sha256(scales)
    json.dumps(r._asdict())


def test_graft_per_channel_scale_shapes_and_dtypes(exported):
    tmp, rep, scales = exported
    out = tmp / "tiny_fp8_pc.onnx"
    r = Q.graft_fp8(rep.onnx_path, out, scales)
    m = _graph(out)
    rows = {row["module"]: row for row in r.sites}
    expect_out = {"attn.to_qkv": 384, "attn.to_out": 128, "ff.ff.0": 512, "ff.ff.2": 128,
                  "to_cond_embed.0": 128, "to_timestep_embed.0": 128}
    for module, channels in expect_out.items():
        node, a_dq, w_dq, inits = _qdq_for(m, rows[module]["node"])
        w_scale = inits[w_dq.input[1]]
        assert list(w_scale.dims) == [channels], module
        assert list(inits[w_dq.input[2]].dims) == [channels]
        assert inits[w_dq.input[2]].data_type == TensorProto.FLOAT8E4M3FN
        assert list(inits[a_dq.input[1]].dims) == []
        want_dt = TensorProto.FLOAT if module.startswith("to_cond_embed") else TensorProto.FLOAT16
        assert w_scale.data_type == want_dt and inits[a_dq.input[1]].data_type == want_dt, module
        axis = next(a.i for a in w_dq.attribute if a.name == "axis")
        assert axis == (0 if node.op_type == "Gemm" else 1)
        a_val = float(numpy_helper.to_array(inits[a_dq.input[1]]))
        assert a_val == pytest.approx(max(scales[module] / 448.0, 1e-4), rel=2e-3)
    # weight scale = per-output-channel absmax / 448 of the source weight
    model = make_tiny().half()
    w = model.attn.to_qkv.weight.float().numpy()          # [out, in]
    node, _, w_dq, inits = _qdq_for(m, rows["attn.to_qkv"]["node"])
    got = numpy_helper.to_array(inits[w_dq.input[1]]).astype(np.float32)
    np.testing.assert_allclose(got, np.maximum(np.abs(w).max(axis=1) / 448.0, 1e-4), rtol=2e-3)


def test_graft_per_tensor_and_margin(exported):
    tmp, rep, scales = exported
    r = Q.graft_fp8(rep.onnx_path, tmp / "tiny_fp8_pt.onnx", scales, weight_scales="per_tensor", margin=1.5)
    m = _graph(tmp / "tiny_fp8_pt.onnx")
    row = next(row for row in r.sites if row["module"] == "ff.ff.2")
    _, a_dq, w_dq, inits = _qdq_for(m, row["node"])
    assert list(inits[w_dq.input[1]].dims) == [] and not w_dq.attribute
    a_val = float(numpy_helper.to_array(inits[a_dq.input[1]]))
    assert a_val == pytest.approx(scales["ff.ff.2"] * 1.5 / 448.0, rel=2e-3)


def test_graft_scale_floor(exported):
    tmp, rep, scales = exported
    zeroed = dict(scales, **{"attn.to_out": 0.0})
    r = Q.graft_fp8(rep.onnx_path, tmp / "tiny_fp8_floor.onnx", zeroed)
    row = next(row for row in r.sites if row["module"] == "attn.to_out")
    assert row["act_scale"] == pytest.approx(1e-4)
    assert r.scale_stats["act"]["floored"] == 1
    m = _graph(tmp / "tiny_fp8_floor.onnx")
    _, a_dq, _, inits = _qdq_for(m, row["node"])
    assert float(numpy_helper.to_array(inits[a_dq.input[1]])) > 0       # representable in fp16


def test_graft_unmatched_linear_is_an_error(exported):
    tmp, rep, scales = exported
    partial = {k: v for k, v in scales.items() if k != "ff.ff.0"}
    with pytest.raises(Q.UnmatchedLinearError, match="ff.ff.0"):
        Q.graft_fp8(rep.onnx_path, tmp / "x.onnx", partial)
    r = Q.graft_fp8(rep.onnx_path, tmp / "tiny_fp8_ex.onnx", partial, exclude=["ff.ff.0"])
    assert (r.linears_total, r.linears_fp8, r.unmatched) == (8, 7, ["ff.ff.0"])
    r = Q.graft_fp8(rep.onnx_path, tmp / "tiny_fp8_ex2.onnx", scales, exclude=["*to_timestep_embed.*"])
    assert r.linears_fp8 == 6 and sorted(r.unmatched) == ["to_timestep_embed.0", "to_timestep_embed.2"]
    assert r.unused_scales == ["to_timestep_embed.0", "to_timestep_embed.2"]


def test_scale_key_matching_rules():
    res = Q._ScaleResolver({"model.transformer.layers.0.to_q": 1.0, "conditioner.x.lin": 2.0})
    assert res.lookup("model.transformer.layers.0.to_q") == "model.transformer.layers.0.to_q"
    assert res.lookup("dit.model.transformer.layers.0.to_q") == "model.transformer.layers.0.to_q"
    assert res.lookup("transformer.layers.0.to_q") == "model.transformer.layers.0.to_q"
    assert res.lookup("wrapper.x.lin") == "conditioner.x.lin"          # first component dropped
    assert res.lookup("transformer.layers.1.to_q") is None


def test_graft_cli(exported, tmp_path):
    tmp, rep, scales = exported
    s = tmp_path / "scales.json"
    Q.write_act_scales(s, scales)
    out = tmp_path / "cli_fp8.onnx"
    assert Q.main(["graft", "--fp16-onnx", rep.onnx_path, "--act-scales", str(s), "--out", str(out),
                   "--expect-linears", "8"]) == 0
    side = json.loads(out.with_suffix(".graft.json").read_text())
    assert side["linears_fp8"] == 8
    assert Q.main(["graft", "--fp16-onnx", rep.onnx_path, "--act-scales", str(s), "--out", str(out),
                   "--expect-linears", "222"]) == 2


def test_parse_profile():
    p = Q._parse_profile(["x=1x256x140,t=1", "y=1x4:1x8:1x12"])
    assert p["x"] == ((1, 256, 140),) * 3 and p["t"] == ((1,),) * 3
    assert p["y"] == ((1, 4), (1, 8), (1, 12))


def test_graft_numerics_onnxruntime_reference(exported):
    """ORT's CPU fp8 Q/DQ reference kernels (graph optimizations off: ORT's
    QDQ propagation mis-types fp8 through Transpose) stay close to fp32
    eager on the tiny DiT."""
    ort = pytest.importorskip("onnxruntime")
    import torch.nn.functional as F

    tmp, rep, scales = exported
    out = tmp / "tiny_fp8_ort.onnx"
    Q.graft_fp8(rep.onnx_path, out, scales)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    except Exception as exc:
        pytest.skip(f"onnxruntime cannot load the fp8 graph: {exc}")
    inp = tiny_inputs(batch=1, seed=7)
    ref = make_tiny().float()(**inp)
    got = torch.from_numpy(sess.run(None, {k: v.numpy() for k, v in inp.items()})[0])
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    assert cos > 0.99, cos
