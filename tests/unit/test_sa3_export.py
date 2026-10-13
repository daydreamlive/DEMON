"""CPU tests for acestep.engine.trt.sa3_export on a tiny fake DiT.

The fake mirrors the vendored SA3 structure the island patches key on
(RMSNorm via F.rms_norm, an Attention class with ``apply_attn``, a
RotaryEmbedding with ``inv_freq``, FourierFeatures, ``to_cond_embed``,
a FeedForward with ``.ff``). ``TinyDiT``/``tiny_inputs`` are reused by
test_sa3_fp8.py.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")

from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from acestep.engine.trt import sa3_export as X  # noqa: E402

D = 128
IO = 32


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) + 0.1 * torch.randn(dim))

    def forward(self, x):
        return F.rms_norm(x, x.shape[-1:], weight=self.gamma, eps=1e-5)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, t):
        t = t.to(torch.float32)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def apply_rope(t, freqs):
    out_dtype = t.dtype
    t = t.float()
    x1, x2 = t.chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    return (t * freqs.cos() + rot * freqs.sin()).to(out_dtype)


class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.num_heads = self.kv_heads = heads
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def apply_attn(self, q, k, v, causal=None, varlen_metadata=None, **kw):
        return F.scaled_dot_product_attention(q, k, v)

    def forward(self, x, freqs):
        b, n, d = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(b, n, self.num_heads, d // self.num_heads).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        out = self.apply_attn(q, k, v)
        return self.to_out(out.transpose(1, 2).reshape(b, n, d))


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, varlen_metadata=None):
        return self.ff(x)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=16.0):
        super().__init__()
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class TinyDiT(nn.Module):
    """8 eligible Linears (min dim >= 128): to_timestep_embed.{0,2} (Gemm),
    to_cond_embed.{0,2} (fp32 island), attn.to_qkv/to_out, ff.ff.{0,2};
    project_in/out are below min_features."""

    def __init__(self):
        super().__init__()
        self.timestep_features = FourierFeatures(1, D)
        self.to_timestep_embed = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))
        self.to_cond_embed = nn.Sequential(nn.Linear(D, D, bias=False), nn.SiLU(), nn.Linear(D, D, bias=False))
        self.project_in = nn.Linear(IO, D, bias=False)
        self.rope = RotaryEmbedding(D // 4)
        self.norm1 = RMSNorm(D)
        self.attn = Attention(D)
        self.norm2 = RMSNorm(D)
        self.ff = FeedForward(D)
        self.project_out = nn.Linear(D, IO, bias=False)

    def forward(self, x, t, cond):
        dtype = next(self.parameters()).dtype
        h = self.project_in(x.transpose(1, 2).to(dtype))
        temb = self.to_timestep_embed(self.timestep_features(t.to(dtype)[:, None]))
        c = self.to_cond_embed(cond.to(dtype))
        h = h + temb[:, None, :]
        freqs = self.rope(torch.arange(h.shape[1], device=h.device))
        h = h + self.attn(self.norm1(h), freqs)
        h = h + c.mean(dim=1, keepdim=True)
        h = h + self.ff(self.norm2(h))
        return self.project_out(h).transpose(1, 2).float()


ELIGIBLE = sorted([
    "to_timestep_embed.0", "to_timestep_embed.2", "to_cond_embed.0", "to_cond_embed.2",
    "attn.to_qkv", "attn.to_out", "ff.ff.0", "ff.ff.2",
])
INPUT_NAMES = ["x", "t", "cond"]
DYNAMIC = {"x": {0: "B", 2: "L"}, "t": {0: "B"}, "cond": {0: "B"}, "velocity": {0: "B", 2: "L"}}


def tiny_inputs(batch=2, length=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(batch, IO, length, generator=g),
        "t": torch.rand(batch, generator=g),
        "cond": torch.randn(batch, 6, D, generator=g),
    }


def make_tiny(seed=0):
    torch.manual_seed(seed)
    return TinyDiT().eval()


def export_tiny(tmp_path, islands="upstream", name="tiny_fp16.onnx", model=None):
    model = model if model is not None else make_tiny()
    report = X.export_fp16mixed(model, tiny_inputs(), input_names=INPUT_NAMES, output_names=["velocity"],
                                dynamic_axes=DYNAMIC, out_onnx=tmp_path / name, islands=islands)
    return model, report


def _nodes_by_name(path):
    m = onnx.load(str(path), load_external_data=False)
    return m, {n.name: n for n in m.graph.node}


def test_export_report_and_names(tmp_path):
    model, rep = export_tiny(tmp_path)
    assert rep.linear_count == 8
    assert rep.input_names == INPUT_NAMES
    assert sorted(w[: -len(".weight")] for w in rep.linear_weights) == ELIGIBLE
    assert len(rep.sha256) == 64 and rep.sha256 == X.onnx_sha256(rep.onnx_path)
    assert (tmp_path / "tiny_fp16.onnx.data").is_file()
    assert set(rep.islands) == set(X.ISLANDS)
    assert next(model.parameters()).dtype == torch.float16
    assert rep._asdict()["node_count"] == rep.node_count


def test_every_linear_weight_findable_by_name(tmp_path):
    """D2 contract: each Linear weight is reachable by its module-qualified
    initializer name, directly or via Transpose/Cast of the initializer."""
    model, rep = export_tiny(tmp_path)
    m = onnx.load(rep.onnx_path, load_external_data=False)
    sites = X.linear_sites(m.graph)
    by_init = {s.init_name for s in sites}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            assert f"{name}.weight" in by_init, name
    assert not any(i.name.startswith("onnx::MatMul") for i in m.graph.initializer)


def test_islands_present_after_export(tmp_path):
    _, rep = export_tiny(tmp_path)
    ops = rep.island_ops
    for op in ("Softmax", "ReduceMean", "Cos", "Sin"):
        assert ops.get(op, 0) >= 1, (op, ops)
    m, nodes = _nodes_by_name(rep.onnx_path)
    island = set(rep.island_nodes)
    sites = {s.module_path: s for s in X.linear_sites(m.graph)}
    for p in ("to_cond_embed.0", "to_cond_embed.2"):
        assert sites[p].node.name in island, p
    for p in ("attn.to_qkv", "attn.to_out", "ff.ff.0", "ff.ff.2", "to_timestep_embed.0"):
        assert sites[p].node.name not in island, p
    # no fp32 accumulation pins: block weights reach their GEMM without a Cast
    assert all(n.op_type != "Cast" for n in sites["attn.to_qkv"].chain)
    assert any(n.op_type == "Cast" for n in sites["to_cond_embed.0"].chain)
    assert sites["to_timestep_embed.0"].node.op_type == "Gemm"


def test_islands_none_keeps_softmax_fp16(tmp_path):
    _, rep = export_tiny(tmp_path, islands="none")
    assert rep.islands == ["rope"]
    assert rep.island_ops.get("Softmax", 0) == 0
    assert rep.island_ops.get("Cos", 0) >= 1        # rope stays fp32


def test_resolve_islands():
    assert X.resolve_islands("upstream") == X.ISLANDS
    assert X.resolve_islands("rmsnorm,softmax") == ("rmsnorm", "softmax", "rope")
    with pytest.raises(ValueError):
        X.resolve_islands("bogus")


def test_patches_restored(tmp_path):
    model = make_tiny()
    rms, ln = F.rms_norm, F.layer_norm
    export_tiny(tmp_path, model=model)
    assert F.rms_norm is rms and F.layer_norm is ln
    assert "apply_attn" not in model.attn.__dict__
    assert "forward" not in model.to_cond_embed.__dict__
    assert "forward" not in model.rope.__dict__


def test_export_numerics_onnxruntime(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model, rep = export_tiny(tmp_path)
    inp = tiny_inputs(batch=1, length=20, seed=3)
    ref = make_tiny().float()(**inp)
    try:
        sess = ort.InferenceSession(rep.onnx_path, providers=["CPUExecutionProvider"])
    except Exception as exc:  # CPU EP may lack an fp16 kernel
        pytest.skip(f"onnxruntime cannot load the fp16 graph: {exc}")
    out = torch.from_numpy(sess.run(None, {k: v.numpy() for k, v in inp.items()})[0])
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    assert out.shape == ref.shape and cos > 0.999, cos


def test_topo_sort_kahn():
    from onnx import TensorProto, helper

    a = helper.make_node("Relu", ["x"], ["a"], name="a")
    b = helper.make_node("Neg", ["a"], ["b"], name="b")
    c = helper.make_node("Add", ["a", "b"], ["y"], name="c")
    g = helper.make_graph([c, b, a], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    assert not X.is_topologically_sorted(g)
    X.topo_sort(g)
    assert [n.name for n in g.node] == ["a", "b", "c"]
    assert X.is_topologically_sorted(g)
    loop = helper.make_graph([helper.make_node("Neg", ["p"], ["q"], name="p"),
                              helper.make_node("Neg", ["q"], ["p"], name="q")], "g2", [], [])
    with pytest.raises(RuntimeError):
        X.topo_sort(loop)
    dangling = helper.make_graph([helper.make_node("Neg", ["nope"], ["q"], name="n")], "g3", [], [])
    with pytest.raises(RuntimeError, match="undefined"):
        X.topo_sort(dangling)


def test_save_reload_assert_catches_corruption(tmp_path, monkeypatch):
    _, rep = export_tiny(tmp_path)
    model = X.load_onnx(rep.onnx_path)
    real_write = X._write_external_binary

    def corrupting_write(m, data_path, location, *a, **kw):
        real_write(m, data_path, location, *a, **kw)
        raw = bytearray(data_path.read_bytes())
        raw[-1] ^= 0xFF
        data_path.write_bytes(bytes(raw))

    monkeypatch.setattr(X, "_write_external_binary", corrupting_write)
    with pytest.raises(RuntimeError, match="changed on disk"):
        X.save_onnx_checked(model, tmp_path / "corrupt.onnx")


def _ctrl_z_model():
    """Two external-size fp16 initializers; the first ends in byte 0x1A
    (Ctrl-Z) and is followed by another tensor: the exact layout onnx's
    Windows text-mode writer corrupts."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    a = np.random.default_rng(0).standard_normal(1536).astype(np.float16)
    a.view(np.uint8)[-1] = 0x1A                   # last byte = Ctrl-Z
    a.view(np.uint8)[-2] = 0xEB
    b = np.random.default_rng(1).standard_normal(1536).astype(np.float16)
    ta, tb = numpy_helper.from_array(a, "a.bias"), numpy_helper.from_array(b, "b.bias")
    node = helper.make_node("Add", ["a.bias", "b.bias"], ["y"], name="add")
    g = helper.make_graph([node], "g", [], [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1536])],
                          initializer=[ta, tb])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), a, b


def test_save_keeps_trailing_ctrl_z_byte(tmp_path):
    """Regression (C3, 2026-10-12): a tensor whose last byte is 0x1A must
    survive save + reload bit-exact, and the next tensor must not shift."""
    import numpy as np
    from onnx import numpy_helper

    model, a, b = _ctrl_z_model()
    X.save_onnx_checked(model, tmp_path / "ctrlz.onnx")
    back = {t.name: numpy_helper.to_array(t) for t in X.load_onnx(tmp_path / "ctrlz.onnx").graph.initializer}
    np.testing.assert_array_equal(back["a.bias"].view(np.uint16), a.view(np.uint16))
    np.testing.assert_array_equal(back["b.bias"].view(np.uint16), b.view(np.uint16))
    # also through onnx's own loader (what TensorRT-side tooling and the graft see)
    ref = {t.name: numpy_helper.to_array(t)
           for t in onnx.load(str(tmp_path / "ctrlz.onnx"), load_external_data=True).graph.initializer}
    np.testing.assert_array_equal(ref["a.bias"].view(np.uint16), a.view(np.uint16))


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="CRT text-mode bug is Windows-only")
def test_onnx_native_writer_drops_trailing_ctrl_z_on_windows(tmp_path):
    """Documents the root cause: onnx's own external-data writer loses the
    0x1A. If this starts failing, onnx fixed it upstream."""
    import numpy as np
    from onnx import numpy_helper

    model, a, _ = _ctrl_z_model()
    onnx.save_model(model, str(tmp_path / "native.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="native.onnx.data", size_threshold=1024)
    back = {t.name: numpy_helper.to_array(t) for t in X.load_onnx(tmp_path / "native.onnx").graph.initializer}
    assert not np.array_equal(back["a.bias"].view(np.uint16), a.view(np.uint16))


def test_save_replaces_stale_data_file(tmp_path):
    _, rep = export_tiny(tmp_path)
    model = X.load_onnx(rep.onnx_path)
    out = tmp_path / "again.onnx"
    (tmp_path / "again.onnx.data").write_bytes(b"\x01" * 4096 * 1024)   # stale, larger
    X.save_onnx_checked(model, out)
    assert (tmp_path / "again.onnx.data").stat().st_size < 4096 * 1024
