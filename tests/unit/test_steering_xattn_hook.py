"""The ``cross_attn_output`` steering hook point (TADA's intervention site).

CPU only. TADA (Staniszewski et al., arXiv 2602.11910) adds its steering
vectors to the OUTPUT of a block's cross-attention, before the residual
add, only on the conditional CFG pass, with one vector per diffusion step
and (in its benchmark) each token rescaled back to its original norm.
These tests pin that delivery through the real StreamPipeline (eager
hooks on ``decoder.layers[i].cross_attn``), the ACE TensorRT buffer fill,
the export-side island the engine is built from, and the pack format.
"""

from __future__ import annotations

import torch

from acestep.engine.diffusion import DiffusionConfig, DiffusionEngine
from acestep.engine.stream import SlotRequest, StreamPipeline
from acestep.engine.trt.export import _patch_cross_attn_steering
from acestep.steering.layout import (
    HOOK_CROSS_ATTN_OUTPUT,
    HOOK_POST_BLOCK_RESIDUAL,
    SteeringLayout,
)
from acestep.steering.packs import (
    PackSteering,
    SteeringPack,
    discover_packs,
    load_pack,
    save_pack,
)

CA = 64  # ACE latent channels
T = 5
NL = 3


class _XAttn(torch.nn.Module):
    """Cross-attention stand-in: returns ``(attn_output, weights)``."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, hidden_states, encoder_hidden_states=None):
        return (hidden_states * self.k + 0.05, None)


class _AceLayer(torch.nn.Module):
    """Same order as AceStepDiTLayer: self-attn, cross-attn residual, MLP."""

    def __init__(self, i):
        super().__init__()
        self.hidden_size = CA
        self.cross_attn = _XAttn(0.5 + 0.1 * i)
        self.k = 1.0 + 0.1 * i

    def forward(self, h):
        h = h * self.k                            # "self-attention"
        a, _ = self.cross_attn(h)
        h = h + a                                 # cross-attn residual
        h = h + torch.tanh(h)                     # "MLP" (non-linear)
        return (h,)


class _AceDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(_AceLayer(i) for i in range(NL))
        self.config = type("Cfg", (), {"hidden_size": CA})()

    def forward(self, *, hidden_states, timestep, **kw):
        h = hidden_states
        for layer in self.layers:
            h = layer(h)[0]
        return (0.1 * h - 0.1 * timestep.view(-1, 1, 1),)


class _AceEngine:
    _DENOISE_MIN = DiffusionEngine._DENOISE_MIN
    _build_timestep_schedule = DiffusionEngine._build_timestep_schedule

    def __init__(self):
        self.decoder = _AceDecoder()
        self.model = None
        self._trt_ctx = None
        self._trt_stream = None
        self._trt_engine = None
        self._trt_io_dtype = torch.float32
        self._trt_input_dtypes = {}
        self._trt_output_dtype = torch.float32
        self._compile_loops = False


def _cfg(steps=3):
    return DiffusionConfig(infer_steps=steps, shift=3.0, infer_method="ode",
                           noise_on_cpu=True, dcw_enabled=False)


def _req(seed):
    g = torch.Generator().manual_seed(seed)
    return SlotRequest(
        encoder_hidden_states=torch.randn(1, 4, 6, generator=g),
        encoder_attention_mask=torch.ones(1, 4),
        context_latents=torch.randn(1, T, 2 * CA, generator=g),
        seed=seed, denoise=1.0,
    )


def _vec(seed=0, n=CA):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=g)
    return v / v.norm()


def _drain(pipe, seed=1, ticks=4):
    pipe.submit(_req(seed))
    return [x for x in (pipe.tick() for _ in range(ticks)) if x is not None]


# ---- layout ---------------------------------------------------------------

def test_layout_extra_hooks_accept_cross_attn_output():
    lay = SteeringLayout(num_blocks=4, hidden_size=8, extra_hooks=(HOOK_CROSS_ATTN_OUTPUT,))
    assert lay.hooks == (HOOK_POST_BLOCK_RESIDUAL, HOOK_CROSS_ATTN_OUTPUT)
    assert lay.accepts(3, 8, HOOK_CROSS_ATTN_OUTPUT)
    assert lay.accepts(3, 8, HOOK_POST_BLOCK_RESIDUAL)
    assert not SteeringLayout(num_blocks=4, hidden_size=8).accepts(0, 8, HOOK_CROSS_ATTN_OUTPUT)


def test_ace_eager_layout_declares_cross_attn_output():
    pipe = StreamPipeline(_AceEngine(), _cfg(), pipeline_depth=1)
    lay = pipe.steering_layout()
    assert lay.hooks == (HOOK_POST_BLOCK_RESIDUAL, HOOK_CROSS_ATTN_OUTPUT)
    assert lay.num_blocks == NL and lay.hidden_size == CA


# ---- eager delivery -------------------------------------------------------

def test_cross_attn_shift_lands_before_the_residual_add_not_after_the_block():
    eng = _AceEngine()
    pipe = StreamPipeline(eng, _cfg(), pipeline_depth=1)
    v = _vec(1)
    pipe.set_steering([{
        "layer": 1, "step": 0, "vector": v, "magnitude": 1.0, "alpha": 2.0,
        "hook": HOOK_CROSS_ATTN_OUTPUT,
    }])
    layer = eng.decoder.layers[1]
    assert len(layer.cross_attn._forward_hooks) == 1
    assert len(layer._forward_hooks) == 0
    probe = torch.randn(2, T, CA)
    pipe._current_step_per_row = [0, 1]
    got = layer(probe)[0]
    pipe._current_step_per_row = []
    h = probe * layer.k
    a = h * layer.cross_attn.k + 0.05
    want0 = (h[0] + a[0] + 2.0 * v)
    want0 = want0 + torch.tanh(want0)
    want1 = h[1] + a[1]
    want1 = want1 + torch.tanh(want1)
    assert torch.allclose(got[0], want0, atol=1e-6)
    assert torch.equal(got[1], want1)


def test_per_step_vector_uses_the_rows_step():
    eng = _AceEngine()
    pipe = StreamPipeline(eng, _cfg(), pipeline_depth=1)
    vs = torch.stack([_vec(10), _vec(11), _vec(12)])
    pipe.set_steering([{
        "layer": 0, "step": -1, "weights": (1.0, 1.0, 0.0), "vector": vs,
        "magnitude": 1.0, "alpha": 1.0, "hook": HOOK_CROSS_ATTN_OUTPUT,
    }])
    ca = eng.decoder.layers[0].cross_attn
    probe = torch.zeros(3, T, CA)
    plain = ca(probe)[0]  # no row mapping -> untouched
    pipe._current_step_per_row = [1, 0, 2]
    got = ca(probe)[0] - plain
    pipe._current_step_per_row = []
    assert torch.allclose(got[0], vs[1].expand(T, CA), atol=1e-6)
    assert torch.allclose(got[1], vs[0].expand(T, CA), atol=1e-6)
    assert torch.count_nonzero(got[2]) == 0  # weight 0 at step 2


def test_cond_only_skips_the_negative_pass():
    eng = _AceEngine()
    pipe = StreamPipeline(eng, _cfg(), pipeline_depth=1)
    pipe.set_steering([{
        "layer": 2, "step": 0, "vector": _vec(3), "alpha": 1.0,
        "hook": HOOK_CROSS_ATTN_OUTPUT, "cond_only": True,
    }])
    ca = eng.decoder.layers[2].cross_attn
    probe = torch.ones(1, T, CA)
    plain = ca(probe)[0]
    pipe._current_step_per_row = [0]
    pos = ca(probe)[0]
    pipe._steering_neg_pass = True
    neg = ca(probe)[0]
    pipe._steering_neg_pass = False
    pipe._current_step_per_row = []
    assert not torch.equal(pos, plain)
    assert torch.equal(neg, plain)


def test_renorm_restores_each_tokens_norm():
    eng = _AceEngine()
    pipe = StreamPipeline(eng, _cfg(), pipeline_depth=1)
    v = _vec(4)
    pipe.set_steering([{
        "layer": 0, "step": 0, "vector": v, "alpha": 5.0,
        "hook": HOOK_CROSS_ATTN_OUTPUT, "renorm": True,
    }])
    ca = eng.decoder.layers[0].cross_attn
    probe = torch.randn(1, T, CA)
    plain = ca(probe)[0]
    pipe._current_step_per_row = [0]
    got = ca(probe)[0]
    pipe._current_step_per_row = []
    want = plain + 5.0 * v
    want = want * (plain.norm(dim=-1, keepdim=True) / want.norm(dim=-1, keepdim=True))
    assert torch.allclose(got, want, atol=1e-5)
    assert torch.allclose(got.norm(dim=-1), plain.norm(dim=-1), atol=1e-4)


def test_zero_strength_is_bit_identical_through_the_pipeline():
    base = _drain(StreamPipeline(_AceEngine(), _cfg(), pipeline_depth=1))
    pipe = StreamPipeline(_AceEngine(), _cfg(), pipeline_depth=1)
    pipe.set_steering([{
        "layer": 1, "step": -1, "weights": (1.0, 1.0, 1.0), "vector": _vec(5),
        "alpha": 0.0, "hook": HOOK_CROSS_ATTN_OUTPUT, "renorm": True,
    }])
    got = _drain(pipe)
    assert len(base) == len(got) == 1
    assert torch.equal(base[0], got[0])
    pipe2 = StreamPipeline(_AceEngine(), _cfg(), pipeline_depth=1)
    pipe2.set_steering([{
        "layer": 1, "step": -1, "weights": (1.0, 1.0, 1.0), "vector": _vec(5),
        "alpha": 3.0, "hook": HOOK_CROSS_ATTN_OUTPUT,
    }])
    assert not torch.equal(base[0], _drain(pipe2)[0])


def test_post_block_and_cross_attn_hooks_coexist_and_detach():
    eng = _AceEngine()
    pipe = StreamPipeline(eng, _cfg(), pipeline_depth=1)
    pipe.set_steering([
        {"layer": 0, "step": 0, "vector": _vec(6), "alpha": 1.0},
        {"layer": 0, "step": 0, "vector": _vec(7), "alpha": 1.0,
         "hook": HOOK_CROSS_ATTN_OUTPUT},
    ])
    assert all(len(l._forward_hooks) == 1 for l in eng.decoder.layers)
    assert all(len(l.cross_attn._forward_hooks) == 1 for l in eng.decoder.layers)
    pipe.set_steering([
        {"layer": 0, "step": 0, "vector": _vec(6), "alpha": 2.0},
        {"layer": 0, "step": 0, "vector": _vec(7), "alpha": 2.0,
         "hook": HOOK_CROSS_ATTN_OUTPUT},
    ])  # same modules: no re-install
    assert all(len(l.cross_attn._forward_hooks) == 1 for l in eng.decoder.layers)
    pipe.close()
    assert all(len(l._forward_hooks) == 0 for l in eng.decoder.layers)
    assert all(len(l.cross_attn._forward_hooks) == 0 for l in eng.decoder.layers)


# ---- TRT buffer fill ------------------------------------------------------

def test_trt_fill_routes_by_hook_and_sets_renorm_flags():
    pipe = StreamPipeline(_AceEngine(), _cfg(4), pipeline_depth=2)
    v0, v1 = _vec(8), _vec(9)
    pipe.set_steering([
        {"layer": 0, "step": 1, "vector": v0, "alpha": 2.0},
        {"layer": 2, "step": 1, "vector": v1, "alpha": -3.0,
         "hook": HOOK_CROSS_ATTN_OUTPUT, "renorm": True, "cond_only": True},
    ])
    post = torch.full((2, NL, CA), 9.0)
    xat = torch.full((2, NL, CA), 9.0)
    ren = torch.full((2, NL), 9.0)
    pipe._trt_bufs = {}
    pipe._current_step_per_row = [1, 0]
    pipe._fill_trt_steering_buffer(post, 2)
    pipe._fill_trt_steering_buffer(xat, 2, hook=HOOK_CROSS_ATTN_OUTPUT, renorm_buf=ren)
    e_post = torch.zeros(2, NL, CA)
    e_post[0, 0] = 2.0 * v0
    e_x = torch.zeros(2, NL, CA)
    e_x[0, 2] = -3.0 * v1
    e_r = torch.zeros(2, NL)
    e_r[0, 2] = 1.0
    assert torch.equal(post, e_post) and torch.equal(xat, e_x) and torch.equal(ren, e_r)
    # Negative CFG pass: the cond_only cross-attn shift drops out.
    pipe._steering_neg_pass = True
    pipe._fill_trt_steering_buffer(xat, 2, hook=HOOK_CROSS_ATTN_OUTPUT, renorm_buf=ren)
    pipe._steering_neg_pass = False
    assert torch.count_nonzero(xat) == 0 and torch.count_nonzero(ren) == 0
    # Clearing zeroes a dirty buffer.
    pipe._fill_trt_steering_buffer(xat, 2, hook=HOOK_CROSS_ATTN_OUTPUT, renorm_buf=ren)
    pipe.set_steering([])
    pipe._fill_trt_steering_buffer(xat, 2, hook=HOOK_CROSS_ATTN_OUTPUT, renorm_buf=ren)
    assert torch.count_nonzero(xat) == 0


def test_trt_layout_reports_cross_attn_only_when_engine_has_the_input():
    eng = _AceEngine()
    pipe = StreamPipeline(eng, _cfg(), pipeline_depth=1)
    pipe._trt_engine = object()
    pipe._steering_num_layers = NL
    pipe._steering_hidden_size = CA
    pipe._steering_xattn = False
    assert pipe.steering_layout().hooks == (HOOK_POST_BLOCK_RESIDUAL,)
    pipe._steering_xattn = True
    lay = pipe.steering_layout()
    assert lay.engine_input and lay.hooks == (HOOK_POST_BLOCK_RESIDUAL, HOOK_CROSS_ATTN_OUTPUT)
    assert pipe._steering_blocks(HOOK_CROSS_ATTN_OUTPUT) is None


# ---- export island parity with the eager hook ----------------------------

def test_export_island_matches_eager_hook_and_is_exact_at_zero():
    eng_ref = _AceEngine()
    eng_exp = _AceEngine()
    holder: dict = {"s": None, "r": None}
    _patch_cross_attn_steering(eng_exp.decoder, holder)
    probe = torch.randn(2, T, CA)

    # Zero shift, renorm off: bit-identical to the unpatched module.
    holder["s"] = torch.zeros(2, NL, CA)
    holder["r"] = torch.zeros(2, NL)
    for i in range(NL):
        assert torch.equal(
            eng_exp.decoder.layers[i].cross_attn(probe)[0],
            eng_ref.decoder.layers[i].cross_attn(probe)[0],
        )

    # Non-zero with renorm on one row: equals the pipeline's eager hook.
    v = _vec(13)
    pipe = StreamPipeline(eng_ref, _cfg(), pipeline_depth=1)
    pipe.set_steering([{
        "layer": 1, "step": 0, "vector": v, "alpha": 4.0,
        "hook": HOOK_CROSS_ATTN_OUTPUT, "renorm": True,
    }])
    s = torch.zeros(2, NL, CA)
    r = torch.zeros(2, NL)
    pipe._trt_bufs = {}
    pipe._current_step_per_row = [0, 2]
    pipe._fill_steering_rows(s, 2, HOOK_CROSS_ATTN_OUTPUT, r)
    eager = eng_ref.decoder.layers[1].cross_attn(probe)[0]
    pipe._current_step_per_row = []
    holder["s"], holder["r"] = s, r
    exported = eng_exp.decoder.layers[1].cross_attn(probe)[0]
    assert torch.allclose(exported, eager, atol=1e-6)
    assert torch.equal(exported[1], eng_exp.decoder.layers[1].cross_attn.__class__.forward(
        eng_exp.decoder.layers[1].cross_attn, probe)[0][1])


# ---- pack format ----------------------------------------------------------

def _tada_pack(**kw):
    vec = torch.stack([torch.stack([_vec(20 + 3 * k + s) for s in range(3)]) for k in range(2)])
    base = dict(
        family="acestep", checkpoint="acestep-v15-turbo", block=1, blocks=[1, 2],
        hidden_size=CA, name="piano", vector=vec, hook=HOOK_CROSS_ATTN_OUTPUT,
        method="tada_caa", magnitude=1.0, cond_only=True, renorm=True,
    )
    base.update(kw)
    return SteeringPack(**base)


def test_multi_block_per_step_pack_round_trips_and_builds_configs(tmp_path):
    pack = _tada_pack()
    path = save_pack(pack, tmp_path / "piano.safetensors")
    back = load_pack(path)
    assert back.blocks == [1, 2] and back.cond_only and back.renorm
    assert back.hook == HOOK_CROSS_ATTN_OUTPUT
    assert torch.equal(back.vector, pack.vector)

    cfgs = PackSteering([back]).build_configs({"steer_piano": 2.0}, 6)
    assert [c["layer"] for c in cfgs] == [1, 2]
    for k, c in enumerate(cfgs):
        assert c["hook"] == HOOK_CROSS_ATTN_OUTPUT and c["cond_only"] and c["renorm"]
        # 3 stored steps resampled to 6 live steps by schedule position.
        assert c["vector"].shape == (6, CA)
        assert torch.equal(c["vector"][0], pack.vector[k][0])
        assert torch.equal(c["vector"][5], pack.vector[k][2])
    assert PackSteering([back]).build_configs({"steer_piano": 0.0}, 6) == []


def test_discover_filters_cross_attn_packs_by_layout(tmp_path):
    save_pack(_tada_pack(), tmp_path / "piano.safetensors")
    plain = SteeringLayout(num_blocks=NL, hidden_size=CA)
    xattn = SteeringLayout(num_blocks=NL, hidden_size=CA, extra_hooks=(HOOK_CROSS_ATTN_OUTPUT,))
    short = SteeringLayout(num_blocks=2, hidden_size=CA, extra_hooks=(HOOK_CROSS_ATTN_OUTPUT,))
    kw = dict(family="acestep", checkpoint="acestep-v15-turbo")
    assert discover_packs(tmp_path, layout=plain, **kw) == []
    assert [p.name for p in discover_packs(tmp_path, layout=xattn, **kw)] == ["piano"]
    assert discover_packs(tmp_path, layout=short, **kw) == []  # block 2 out of range


def test_pack_knob_spec_names_the_cross_attn_site():
    spec = PackSteering([_tada_pack()]).knob_specs()[0]
    assert spec.name == "steer_piano"
    assert "cross-attention output of blocks 1,2" in spec.description
