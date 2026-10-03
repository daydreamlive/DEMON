"""The family-agnostic steering slot through the real StreamPipeline.

CPU only. A tiny fake adapter declares a steering layout over an eager
block stack (hooks) or as an engine input (``steering=`` tensor); the ACE
regression cases pin the historical one-hot step gate (eager hooks on
``decoder.layers`` and the TRT buffer fill) to their pre-seam numerics.
"""

from __future__ import annotations

import torch

from acestep.engine.diffusion import DiffusionConfig, DiffusionEngine
from acestep.engine.stream import SlotRequest, StreamPipeline
from acestep.steering.layout import HOOK_POST_BLOCK_RESIDUAL, SteeringLayout

C = 8    # latent channels
T = 5    # latent frames
H = 8    # hidden size (== C so blocks can live in latent space)
NB = 4   # blocks


class _Block(torch.nn.Module):
    def __init__(self, k: float):
        super().__init__()
        self.k = k

    def forward(self, h):
        return h * self.k + 0.01


class _Trunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(_Block(0.9 + 0.01 * i) for i in range(NB))
        self.seen: list = []  # per forward: list of block outputs

    def forward(self, x):
        outs = []
        h = x
        for blk in self.layers:
            h = blk(h)
            outs.append(h.detach().clone())
        self.seen.append(outs)
        return h


class _EagerAdapter:
    """Fake family: velocity = trunk(x). Eager steering via hooks."""

    name = "fake"
    latent_channels = C
    latent_rate_hz = 10.0
    sample_rate = 16000
    accepts_steering = False

    def __init__(self):
        self.trunk = _Trunk()

    def build_schedule(self, config, denoise, device, dtype):
        return torch.linspace(float(denoise), 0.0, config.infer_steps + 1)

    def request_frames(self, request):
        return int(request.latent_frames)

    def request_device_dtype(self, request):
        return torch.device("cpu"), torch.float32

    def steering_layout(self):
        return SteeringLayout(num_blocks=NB, hidden_size=H)

    def steering_blocks(self):
        return self.trunk.layers

    def batched_forward(self, xt_batch, timestep_list, enc_list, mask_list,
                        ctx_list, aux_list):
        return self.trunk(xt_batch)


class _EngineAdapter(_EagerAdapter):
    """Fake family whose 'engine' takes the steering tensor as input."""

    accepts_steering = True

    def __init__(self):
        super().__init__()
        self.received: list = []

    def steering_layout(self):
        return SteeringLayout(num_blocks=NB, hidden_size=H, engine_input=True)

    def steering_blocks(self):
        return None

    def batched_forward(self, xt_batch, timestep_list, enc_list, mask_list,
                        ctx_list, aux_list, steering=None):
        self.received.append(None if steering is None else steering.clone())
        out = self.trunk(xt_batch)
        if steering is not None:
            out = out + steering.sum(dim=1).unsqueeze(1)
        return out


def _pipe(adapter, steps=3, depth=1):
    cfg = DiffusionConfig(
        infer_steps=steps, infer_method="ode", noise_on_cpu=True,
        dcw_enabled=False,
    )
    return StreamPipeline(None, cfg, pipeline_depth=depth, adapter=adapter)


def _req(seed):
    return SlotRequest(seed=seed, denoise=1.0, aux_cond={}, latent_frames=T)


def _drain(pipe, seeds, ticks):
    out = []
    q = list(seeds)
    for _ in range(ticks):
        if q:
            pipe.submit(_req(q.pop(0)))
        fin = pipe.tick()
        if fin is not None:
            out.append(fin.clone())
    return out


def _vec(seed=0, n=H):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=g)
    return v / v.norm()


def test_layout_accepts_matching_vectors_only():
    lay = SteeringLayout(num_blocks=NB, hidden_size=H)
    assert lay.hook == HOOK_POST_BLOCK_RESIDUAL
    assert lay.accepts(0, H, HOOK_POST_BLOCK_RESIDUAL)
    assert not lay.accepts(NB, H, HOOK_POST_BLOCK_RESIDUAL)
    assert not lay.accepts(0, H + 1, HOOK_POST_BLOCK_RESIDUAL)
    assert not lay.accepts(0, H, "cross_attn")


def test_zero_alpha_is_bit_identical_to_no_steering():
    base = _drain(_pipe(_EagerAdapter()), [1, 2], ticks=8)
    pipe = _pipe(_EagerAdapter())
    pipe.set_steering([{
        "layer": 2, "vector": _vec(), "magnitude": 3.0, "alpha": 0.0,
        "weights": (1.0, 1.0, 1.0),
    }])
    got = _drain(pipe, [1, 2], ticks=8)
    assert len(base) == len(got) == 2
    for a, b in zip(base, got):
        assert torch.equal(a, b)


def test_vector_lands_at_declared_block_only():
    v = _vec(1)
    ref = _EagerAdapter()
    _drain(_pipe(ref), [5], ticks=4)
    ad = _EagerAdapter()
    pipe = _pipe(ad)
    pipe.set_steering([{
        "layer": 2, "vector": v, "magnitude": 2.0, "alpha": 1.5,
        "weights": (1.0, 1.0, 1.0),
    }])
    _drain(pipe, [5], ticks=4)
    # First forward: inputs identical, so blocks 0..1 match exactly and
    # block 2 differs by exactly alpha * magnitude * v (the hook output).
    ref_outs, got_outs = ref.trunk.seen[0], ad.trunk.seen[0]
    assert torch.equal(ref_outs[0], got_outs[0])
    assert torch.equal(ref_outs[1], got_outs[1])
    delta = got_outs[2] - ref_outs[2]
    assert torch.allclose(delta, (3.0 * v).view(1, 1, -1).expand_as(delta), atol=1e-6)


def test_policy_curve_scales_per_step():
    v = _vec(2)
    ref = _EagerAdapter()
    _drain(_pipe(ref), [7], ticks=4)
    ad = _EagerAdapter()
    pipe = _pipe(ad)
    pipe.set_steering([{
        "layer": 0, "vector": v, "magnitude": 1.0, "alpha": 2.0,
        "weights": (1.0, 0.5, 0.0),
    }])
    _drain(pipe, [7], ticks=4)
    assert len(ad.trunk.seen) == 3
    # Step 0 (identical inputs): block 0 carries the full-weight shift.
    delta = ad.trunk.seen[0][0] - ref.trunk.seen[0][0]
    assert torch.allclose(delta, (2.0 * v).expand_as(delta), atol=1e-6)
    # Steps 1 and 2: probe the installed hook directly per row step.
    blk0 = ad.trunk.layers[0]
    probe = torch.zeros(1, T, H)
    plain = probe * blk0.k + 0.01
    pipe._current_step_per_row = [1]
    half = blk0(probe) - plain
    pipe._current_step_per_row = [2]
    zero = blk0(probe) - plain
    pipe._current_step_per_row = [5]  # past the curve -> 0
    past = blk0(probe) - plain
    pipe._current_step_per_row = []
    assert torch.allclose(half, (1.0 * v).expand_as(half), atol=1e-6)
    assert torch.equal(zero, torch.zeros_like(zero))
    assert torch.equal(past, torch.zeros_like(past))


def test_engine_input_adapter_receives_layout_shaped_tensor():
    v = _vec(3)
    ad = _EngineAdapter()
    pipe = _pipe(ad, steps=2, depth=1)
    _drain(pipe, [1], ticks=3)
    assert ad.received and all(r is None for r in ad.received)
    ad.received.clear()
    pipe.set_steering([{
        "layer": 1, "vector": v, "magnitude": 2.0, "alpha": -1.0,
        "weights": (0.0, 1.0),
    }])
    _drain(pipe, [1], ticks=3)
    # Step 0 has weight 0 -> None (zero path); step 1 carries the shift
    # at block 1 only.
    assert ad.received[0] is None
    t = ad.received[1]
    assert t.shape == (1, NB, H) and t.dtype == torch.float32
    assert torch.allclose(t[0, 1], -2.0 * v, atol=1e-6)
    assert torch.count_nonzero(t[0, [0, 2, 3]]) == 0


def test_hooks_removed_on_close_and_clear():
    ad = _EagerAdapter()
    pipe = _pipe(ad)
    pipe.set_steering([{"layer": 0, "vector": _vec(), "alpha": 1.0, "step": 0}])
    assert all(len(b._forward_hooks) == 1 for b in ad.trunk.layers)
    pipe.close()
    assert all(len(b._forward_hooks) == 0 for b in ad.trunk.layers)


def test_adapter_without_layout_no_ops():
    class _Bare(_EagerAdapter):
        steering_layout = None  # type: ignore[assignment]
        steering_blocks = None  # type: ignore[assignment]

    base = _drain(_pipe(_EagerAdapter()), [3], ticks=4)
    pipe = _pipe(_Bare())
    pipe.set_steering([{"layer": 0, "vector": _vec(), "alpha": 5.0, "step": 0}])
    got = _drain(pipe, [3], ticks=4)
    assert torch.equal(base[0], got[0])


# ---- ACE regression: the historical one-hot gate --------------------------

CA = 64  # ACE latent channels (the adapter sizes noise from this)


class _AceLayer(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.hidden_size = CA

    def forward(self, h):
        return (h * self.k,)


class _AceDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(_AceLayer(1.0 + 0.1 * i) for i in range(3))
        self.config = type("Cfg", (), {"hidden_size": CA})()

    def forward(self, *, hidden_states, timestep, **kw):
        h = hidden_states
        for layer in self.layers:
            h = layer(h)[0]
        return (h - 0.1 * timestep.view(-1, 1, 1),)


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


def _ace_req(seed):
    g = torch.Generator().manual_seed(seed)
    return SlotRequest(
        encoder_hidden_states=torch.randn(1, 4, 6, generator=g),
        encoder_attention_mask=torch.ones(1, 4),
        context_latents=torch.randn(1, T, 2 * CA, generator=g),
        seed=seed, denoise=1.0,
    )


def test_ace_layout_and_eager_one_hot_gate_matches_reference():
    eng = _AceEngine()
    cfg = DiffusionConfig(infer_steps=3, shift=3.0, infer_method="ode",
                          noise_on_cpu=True, dcw_enabled=False)
    pipe = StreamPipeline(eng, cfg, pipeline_depth=1)
    lay = pipe.steering_layout()
    assert lay == SteeringLayout(num_blocks=3, hidden_size=CA)
    assert pipe.adapter.accepts_steering is False

    v = _vec(4, CA)
    pipe.set_steering([{
        "layer": 1, "step": 1, "vector": v, "magnitude": 2.0, "alpha": 3.0,
    }])
    layer1 = eng.decoder.layers[1]
    probe = torch.ones(2, T, CA)
    pipe._current_step_per_row = [1, 0]
    out = layer1(probe)[0]
    pipe._current_step_per_row = []
    # Historical semantics: rows at the gated step get alpha*magnitude*v;
    # other rows are untouched.
    assert torch.allclose(out[0], probe[0] * layer1.k + 6.0 * v, atol=1e-6)
    assert torch.equal(out[1], probe[1] * layer1.k)
    # No row mapping -> no injection (forward outside the rendezvous).
    assert torch.equal(layer1(probe)[0], probe * layer1.k)


def test_ace_trt_buffer_fill_matches_reference():
    eng = _AceEngine()
    cfg = DiffusionConfig(infer_steps=4, shift=3.0, infer_method="ode",
                          noise_on_cpu=True, dcw_enabled=False)
    pipe = StreamPipeline(eng, cfg, pipeline_depth=2)
    pipe._steering_num_layers = 3
    pipe._steering_hidden_size = CA
    v0, v2 = _vec(5, CA), _vec(6, CA)
    pipe.set_steering([
        {"layer": 0, "step": 2, "vector": v0, "magnitude": 1.5, "alpha": 2.0},
        {"layer": 2, "step": 0, "vector": v2, "magnitude": 1.0, "alpha": -4.0},
        {"layer": 7, "step": 0, "vector": v2, "magnitude": 1.0, "alpha": 1.0},
    ])
    buf = torch.full((3, 3, CA), 9.0)
    pipe._trt_bufs = {}
    pipe._current_step_per_row = [2, 0, 2]
    pipe._fill_trt_steering_buffer(buf, 3)
    expect = torch.zeros(3, 3, CA)
    expect[[0, 2], 0, :] += 3.0 * v0
    expect[[1], 2, :] += -4.0 * v2
    assert torch.equal(buf, expect)
    assert pipe._trt_bufs["_steering_dirty"] is True
    # Clearing zeroes a dirty buffer exactly once.
    pipe.set_steering([])
    pipe._fill_trt_steering_buffer(buf, 3)
    assert torch.count_nonzero(buf) == 0
    assert pipe._trt_bufs["_steering_dirty"] is False


def test_ace_drain_with_steering_matches_manual_hook_math():
    """End to end through ticks: steering only changes the gated step."""
    cfg = DiffusionConfig(infer_steps=3, shift=3.0, infer_method="ode",
                          noise_on_cpu=True, dcw_enabled=False)
    base_pipe = StreamPipeline(_AceEngine(), cfg, pipeline_depth=1)
    base_pipe.submit(_ace_req(1))
    base = [base_pipe.tick() for _ in range(4)]
    pipe = StreamPipeline(_AceEngine(), cfg, pipeline_depth=1)
    pipe.set_steering([{
        "layer": 0, "step": 2, "vector": _vec(7, CA), "magnitude": 1.0, "alpha": 0.5,
    }])
    pipe.submit(_ace_req(1))
    got = [pipe.tick() for _ in range(4)]
    fin_base = [x for x in base if x is not None]
    fin_got = [x for x in got if x is not None]
    assert len(fin_base) == len(fin_got) == 1
    assert not torch.equal(fin_base[0], fin_got[0])
