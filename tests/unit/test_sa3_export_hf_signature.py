"""CPU tests for the HF-signature export (sa3_export.HFSignatureDiT) on a tiny
fake DiT + conditioner shaped like the vendored SA3 pieces it reads:
``DiTWrapper.model``, ``MultiConditioner.conditioners["prompt"]`` (learned
``padding_embedding``) and ``["seconds_total"]`` (NumberConditioner:
``embedder``, ``min_val``, ``max_val``)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")

from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from acestep.engine.trt import sa3_export as X  # noqa: E402
from acestep.engine.trt import sa3_fp8 as Q  # noqa: E402
from test_sa3_export import IO, Attention, FeedForward, FourierFeatures, RMSNorm  # noqa: E402

D = 128          # cond dim == embed dim
T5 = 8           # t5 tokens
LOCAL = 9        # local_add_cond channels
MAX_S = 384.0


class ExpoFourierFeatures(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        in_dtype = t.dtype
        t = t.float()
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        ramp = torch.linspace(0, 1, self.dim // 2, dtype=torch.float32)
        freqs = torch.exp(ramp * (math.log(10000.0) - math.log(0.5)) + math.log(0.5))
        f = 2 * math.pi * t * freqs
        return torch.cat([f.cos(), f.sin()], dim=-1).to(in_dtype)


class PromptCond(nn.Module):
    padding_mode = "learned"
    output_dim = D

    def __init__(self):
        super().__init__()
        self.padding_embedding = nn.Parameter(torch.randn(D) * 0.5)


class NumberEmbedder(nn.Module):
    """Vendored NumberEmbedder.forward: flatten, embed, view(*shape, features)."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(ExpoFourierFeatures(D), nn.Linear(D, D))

    def forward(self, x):
        shape = x.shape
        return self.embedding(x.reshape(-1)).view(*shape, D)


class NumberCond(nn.Module):
    min_val, max_val = 0.0, MAX_S

    def __init__(self):
        super().__init__()
        self.embedder = NumberEmbedder()

    def embed(self, seconds):
        s = (torch.as_tensor(seconds).float().clamp(self.min_val, self.max_val) - self.min_val)
        s = s / (self.max_val - self.min_val)
        return self.embedder(s.reshape(-1))                      # [B, D]


class MultiCond(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioners = nn.ModuleDict({"prompt": PromptCond(), "seconds_total": NumberCond()})


class Inner(nn.Module):
    """to_cond_embed (fp32 island), to_global_embed, timestep Fourier, local add,
    one attention + ff block. Eligible Linears (min dim >= 128):
    to_timestep_embed.0, to_cond_embed.{0,2}, to_global_embed.0, attn.to_qkv,
    attn.to_out, ff.ff.{0,2}."""

    def __init__(self):
        super().__init__()
        self.timestep_features = FourierFeatures(1, D)
        self.to_timestep_embed = nn.Sequential(nn.Linear(D, D), nn.SiLU())
        self.to_cond_embed = nn.Sequential(nn.Linear(D, D, bias=False), nn.SiLU(), nn.Linear(D, D, bias=False))
        self.to_global_embed = nn.Sequential(nn.Linear(D, D, bias=False), nn.SiLU())
        self.project_in = nn.Linear(IO, D, bias=False)
        self.local = nn.Linear(LOCAL, D, bias=False)
        self.norm1 = RMSNorm(D)
        self.attn = Attention(D)
        self.norm2 = RMSNorm(D)
        self.ff = FeedForward(D)
        self.project_out = nn.Linear(D, IO, bias=False)

    def forward(self, x, t, cross_attn_cond=None, cross_attn_cond_mask=None, global_embed=None,
                local_add_cond=None, padding_mask=None, cfg_scale=1.0, **kw):
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        h = self.project_in(x.transpose(1, 2)) + self.local(local_add_cond.to(dtype).transpose(1, 2))
        temb = self.to_timestep_embed(self.timestep_features(t.float()[:, None].to(dtype)))
        g = self.to_global_embed(global_embed.to(dtype))
        c = self.to_cond_embed(cross_attn_cond.to(dtype))
        h = h + (temb + g)[:, None, :]
        freqs = torch.zeros(h.shape[1], D // 4, device=h.device)
        h = h + self.attn(self.norm1(h), freqs)
        h = h + c.mean(dim=1, keepdim=True) + c[:, -1:, :]
        h = h + self.ff(self.norm2(h))
        return self.project_out(h).transpose(1, 2)


class FakeDiTWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()

    def forward(self, x, t, cross_attn_cond=None, cross_attn_mask=None, local_add_cond=None,
                global_cond=None, cfg_scale=1.0, batch_cfg=True, **kw):
        assert batch_cfg
        return self.model(x, t, cross_attn_cond=cross_attn_cond, cross_attn_cond_mask=cross_attn_mask,
                          global_embed=global_cond, local_add_cond=local_add_cond, cfg_scale=cfg_scale, **kw)


def make(seed=0):
    torch.manual_seed(seed)
    return FakeDiTWrapper().eval(), MultiCond().eval()


def bundle_for(cond, seconds, n_valid=5, L=12, seed=1):
    """A torch cond bundle as the conditioner builds it: prompt block (padding
    embedding already at the masked tokens) + seconds token; global = seconds."""
    g = torch.Generator().manual_seed(seed)
    pc = cond.conditioners["prompt"]
    mask = torch.zeros(1, T5, dtype=torch.bool)
    mask[:, :n_valid] = True
    hid = torch.randn(1, T5, D, generator=g)
    prompt = torch.where(mask[..., None], hid, pc.padding_embedding.detach()[None, None])
    sec = cond.conditioners["seconds_total"].embed([seconds]).detach()          # [1, D]
    return {
        "cross_attn_cond": torch.cat([prompt, sec[:, None]], dim=1),
        "cross_attn_mask": torch.cat([mask, torch.ones(1, 1, dtype=torch.bool)], dim=1),
        "global_cond": sec,
        "local_add_cond": torch.randn(1, LOCAL, L, generator=g),
        "padding_mask": torch.ones(1, L, dtype=torch.bool),
        "cfg_scale": 1.0, "batch_cfg": True,
    }


def inputs_for(bundle, seconds, L=12, seed=2):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, IO, L, generator=g)
    return X.hf_inputs_from_bundle(bundle, seconds, x, torch.tensor([0.37]), t5_tokens=T5)


# ---------------------------------------------------------------------------


def test_hf_inputs_follow_sa3trtdit_staging():
    dit, cond = make()
    b = bundle_for(cond, 13.0)
    inp = inputs_for(b, 13.0)
    assert list(inp) == list(X.HF_SIGNATURE_INPUTS)
    assert all(v.dtype == torch.float32 for v in inp.values())
    assert inp["t5_hidden"].shape == (1, T5, D) and inp["t5_mask"].shape == (1, T5)
    assert torch.equal(inp["t5_hidden"], b["cross_attn_cond"][:, :T5])        # seconds token dropped
    assert torch.equal(inp["t5_mask"], b["cross_attn_mask"][:, :T5].float())
    assert inp["seconds_total"].tolist() == [13.0] and inp["t"].shape == (1,)
    # a shorter prompt block is zero-padded; no local_add_cond -> zeros [1,257,L]
    short = dict(b, cross_attn_cond=b["cross_attn_cond"][:, :3], cross_attn_mask=b["cross_attn_mask"][:, :3])
    short.pop("local_add_cond")
    s = inputs_for(short, 13.0)
    assert s["t5_mask"][0, 3:].abs().sum() == 0 and s["t5_hidden"][0, 3:].abs().sum() == 0
    assert s["local_add_cond"].shape == (1, 257, 12) and s["local_add_cond"].abs().sum() == 0


def test_tail_rebuilds_the_conditioner_output():
    dit, cond = make()
    tail = X.conditioner_tail_from(cond)
    b = bundle_for(cond, 21.5)
    inp = inputs_for(b, 21.5)
    cross, glob = tail(inp["t5_hidden"], inp["t5_mask"], inp["seconds_total"])
    assert torch.allclose(cross, b["cross_attn_cond"], atol=1e-6)
    assert torch.allclose(glob, b["global_cond"], atol=1e-6)
    # masked tokens take the learned padding embedding whatever t5_hidden holds there
    junk = inp["t5_hidden"].clone()
    junk[:, 5:] = 7.0
    cross2, _ = tail(junk, inp["t5_mask"], inp["seconds_total"])
    assert torch.equal(cross2, cross)
    # seconds clamp to the conditioner range
    _, g_hi = tail(inp["t5_hidden"], inp["t5_mask"], torch.tensor([1e6]))
    _, g_max = tail(inp["t5_hidden"], inp["t5_mask"], torch.tensor([MAX_S]))
    assert torch.equal(g_hi, g_max)


def test_tail_zero_padding_mode_and_none_refused():
    dit, cond = make()
    cond.conditioners["prompt"].padding_mode = "zero"
    tail = X.conditioner_tail_from(cond)
    assert tail.padding_embedding.abs().sum() == 0 and tail.padding_embedding.shape == (D,)
    cond.conditioners["prompt"].padding_mode = "none"
    with pytest.raises(ValueError):
        X.conditioner_tail_from(cond)


def test_wrapper_matches_eager_dit_on_the_bundle_and_keeps_calibration_names():
    dit, cond = make()
    b = bundle_for(cond, 13.0)
    inp = inputs_for(b, 13.0)
    w = X.HFSignatureDiT(dit, X.conditioner_tail_from(cond)).eval()
    names = {n for n, m in w.named_modules() if isinstance(m, nn.Linear)}
    trunk = {n for n, m in dit.named_modules() if isinstance(m, nn.Linear)}
    assert trunk <= names and names - trunk == {f"{X.HF_TAIL_ATTR}.seconds.embedding.1"}
    with torch.no_grad():
        out = w(*[inp[n] for n in X.HF_SIGNATURE_INPUTS])
        ref = dit(inp["x"], inp["t"], cross_attn_cond=b["cross_attn_cond"], cross_attn_mask=b["cross_attn_mask"],
                  global_cond=b["global_cond"], local_add_cond=b["local_add_cond"])
    assert out.dtype == torch.float32 and out.shape == (1, IO, 12)
    assert torch.allclose(out, ref, atol=1e-5)


def _export(tmp_path, dit, cond, L=12):
    b = bundle_for(cond, 13.0, L=L)
    inp = inputs_for(b, 13.0, L=L)
    rep = X.export_hf_signature(dit, cond, inp, out_onnx=tmp_path / "hf_fp16.onnx")
    return rep, inp


def _dims(vi):
    return [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim]


def test_export_has_the_hf_signature(tmp_path):
    dit, cond = make()
    rep, _ = _export(tmp_path, dit, cond)
    m = onnx.load(rep.onnx_path, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    ins = [i for i in m.graph.input if i.name not in inits]
    assert [i.name for i in ins] == list(X.HF_SIGNATURE_INPUTS)
    assert all(i.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for i in ins)
    shapes = {i.name: _dims(i) for i in ins}
    assert shapes == {"x": [1, IO, "L"], "t": [1], "t5_hidden": [1, T5, D], "t5_mask": [1, T5],
                      "seconds_total": [1], "local_add_cond": [1, LOCAL, "L"]}
    (out,) = m.graph.output
    assert out.name == "velocity" and out.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert _dims(out)[1:] == [IO, "L"]
    tail_w = f"{X.HF_TAIL_ATTR}.seconds.embedding.1.weight"
    assert tail_w in rep.linear_weights
    assert f"{X.HF_TAIL_ATTR}.padding_embedding" in inits
    # trunk weights keep calibration names
    assert "model.to_cond_embed.0.weight" in rep.linear_weights
    assert "model.attn.to_qkv.weight" in rep.linear_weights


def test_tail_gemm_stays_fp32(tmp_path):
    dit, cond = make()
    rep, _ = _export(tmp_path, dit, cond)
    m = X.load_onnx(rep.onnx_path)
    types_ = X.elem_types(m)
    sites = {s.module_path: s for s in X.linear_sites(m.graph)}
    tail = sites[f"{X.HF_TAIL_ATTR}.seconds.embedding.1"]
    assert types_[tail.node.input[0]] == onnx.TensorProto.FLOAT
    assert types_[tail.node.output[0]] == onnx.TensorProto.FLOAT
    trunk = sites["model.attn.to_qkv"]
    assert types_[trunk.node.input[0]] == onnx.TensorProto.FLOAT16


def test_graft_leaves_the_tail_unquantized(tmp_path):
    dit, cond = make()
    rep, inp = _export(tmp_path, dit, cond)
    scales = Q.capture_activation_scales(
        dit.float(), lambda: X.HFSignatureDiT(dit, X.conditioner_tail_from(cond))(
            *[inp[n] for n in X.HF_SIGNATURE_INPUTS]))
    assert not any(k.startswith(X.HF_TAIL_ATTR) for k in scales)          # calibration is on the dit
    # without the tail exclude the graft refuses (no scale for the tail Linear)
    with pytest.raises(Q.UnmatchedLinearError):
        Q.graft_fp8(rep.onnx_path, tmp_path / "bad.onnx", scales, exclude=())
    g = Q.graft_fp8(rep.onnx_path, tmp_path / "hf_fp8.onnx", scales, exclude=Q.HF_MEDIUM_EXCLUDE)
    assert f"{X.HF_TAIL_ATTR}.seconds.embedding.1" in g.unmatched
    assert g.linears_fp8 == g.linears_total - len(g.unmatched)
    assert not any(s["module"].startswith(X.HF_TAIL_ATTR) for s in g.sites if "module" in s)


def test_tail_pattern_matches_nothing_in_the_trunk():
    from acestep.engine.trt.sa3_export import match_patterns

    assert "cond_tail.*" in Q.HF_MEDIUM_EXCLUDE and "cond_tail.*" in Q.DEMON_MEDIUM_EXCLUDE
    assert match_patterns("cond_tail.seconds.embedding.1", Q.HF_MEDIUM_EXCLUDE)
    for path in ("model.to_cond_embed.0", "model.to_global_embed.0", "model.transformer.layers.3.attn.to_q",
                 "model.transformer.layers.3.cross_attn.to_kv"):
        assert not match_patterns(path, Q.DEMON_MEDIUM_EXCLUDE)


@pytest.mark.parametrize("L", [12, 1, 29])
def test_export_numerics_onnxruntime_dynamic_L(tmp_path, L):
    ort = pytest.importorskip("onnxruntime")
    dit, cond = make()
    ref_dit, ref_cond = make()
    rep, _ = _export(tmp_path, dit, cond, L=12)
    b = bundle_for(ref_cond, 13.0, L=L, seed=5)
    inp = inputs_for(b, 13.0, L=L, seed=6)
    with torch.no_grad():
        ref = ref_dit(inp["x"], inp["t"], cross_attn_cond=b["cross_attn_cond"], global_cond=b["global_cond"],
                      local_add_cond=b["local_add_cond"])
    try:
        sess = ort.InferenceSession(rep.onnx_path, providers=["CPUExecutionProvider"])
    except Exception as exc:  # CPU EP may lack an fp16 kernel
        pytest.skip(f"onnxruntime cannot load the fp16 graph: {exc}")
    out = torch.from_numpy(sess.run(None, {k: v.numpy() for k, v in inp.items()})[0])
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    assert out.shape == ref.shape == (1, IO, L) and cos > 0.999, cos
