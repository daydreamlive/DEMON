"""MiniMaxAdapter through the SHARED StreamPipeline (Tier-2 seam).

MiniMax-Music3 runs flow matching in the opposite time direction from
every other family in this tree (t=0 is noise, t=1 is data, Euler steps
forward), so :class:`~acestep.engine.minimax_adapter.MiniMaxAdapter` has
to flip both the timestep and the sign of the velocity on top of the
usual native-layout transpose. Those three conversions are the whole
risk surface of the adapter, and a sign error there denoises *away* from
the data manifold — which is why the headline test here is an exact
round trip rather than a shape assertion.

CPU + mock DiT throughout; no weights, no GPU.
"""

from __future__ import annotations

import pytest
import torch

from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.minimax_adapter import (
    MINIMAX_COND_DIM,
    MiniMaxAdapter,
    stack_minimax_cond_bundles,
)
from acestep.engine.stream import SlotRequest, StreamPipeline

C = 128  # MiniMax latent channels
T = 6    # latent frames


def _cond(fill: float = 1.0) -> dict:
    return {
        "encoder_hidden_states": torch.full((1, T, MINIMAX_COND_DIM), fill),
    }


class _RecordDit(torch.nn.Module):
    """Records call shapes and timesteps; returns zero velocity."""

    def __init__(self):
        super().__init__()
        self.calls: list = []

    def forward(self, x, t, cond):
        self.calls.append({
            "shape": tuple(x.shape),
            "t": t.clone(),
            "cond_shape": tuple(cond.shape),
        })
        return torch.zeros_like(x)


class _ExactFlowDit(torch.nn.Module):
    """The analytically exact velocity field for a known target.

    On the MiniMax interpolant ``x_t = (1-t)*noise + t*data`` the true
    velocity is ``dx/dt = data - noise``. Eliminating ``noise`` gives a
    form that depends only on the current state, so this mock is exact
    from *any* point on the path and self-correcting off it:

        noise = (x - t*data) / (1-t)
        v     = data - noise = (data - x) / (1-t)

    With ``s = 1-t`` that is ``(data - x)/s``. If the adapter's
    conversions are right, DEMON's Euler integrator must therefore walk
    a pure-noise start exactly onto ``data``.
    """

    def __init__(self, data_bct: torch.Tensor):
        super().__init__()
        self.register_buffer("data", data_bct)
        self.seen_t: list = []

    def forward(self, x, t, cond):
        self.seen_t.append(t.clone())
        s = (1.0 - t).clamp_min(1e-6).view(-1, 1, 1)
        return (self.data - x) / s


def _schedule_builder(steps: int):
    def _build(denoise: float) -> torch.Tensor:
        # MiniMax's own sampler is uniform in t, so uniform in s too.
        return torch.linspace(float(denoise), 0.0, steps + 1)
    return _build


def _pipeline(dit, *, steps=4, depth=1, method="ode"):
    adapter = MiniMaxAdapter(
        dit,
        schedule_builder=_schedule_builder(steps),
        device="cpu",
        dtype=torch.float32,
    )
    config = DiffusionConfig(
        infer_steps=steps,
        infer_method=method,
        noise_on_cpu=True,
        dcw_enabled=False,
    )
    return StreamPipeline(None, config, pipeline_depth=depth, adapter=adapter)


def _request(seed: int, *, denoise=1.0, source=None, fill=1.0) -> SlotRequest:
    return SlotRequest(
        seed=seed,
        denoise=denoise,
        source_latents=source,
        aux_cond=_cond(fill),
        latent_frames=T,
    )


def _drain(pipe, requests, ticks):
    out = []
    queue = list(requests)
    for _ in range(ticks):
        if queue:
            pipe.submit(queue.pop(0))
        fin = pipe.tick()
        if fin is not None:
            out.append(fin)
    return out


# ---- the convention bridge ------------------------------------------------


def test_exact_flow_recovers_target_latent():
    """The headline test: with an exact velocity field the pipeline must
    land on the target. Fails loudly if the t-flip or the velocity
    negation is wrong in either direction."""
    torch.manual_seed(0)
    data = torch.randn(1, C, T)
    dit = _ExactFlowDit(data)
    pipe = _pipeline(dit, steps=4)

    finished = _drain(pipe, [_request(11)], ticks=8)

    assert len(finished) == 1
    # Pipeline returns engine layout [1, T, C]; the target is native.
    got = finished[0]
    assert got.shape == (1, T, C)
    torch.testing.assert_close(got, data.movedim(1, 2), rtol=1e-4, atol=1e-4)


def test_timestep_is_flipped_for_the_dit():
    """DEMON walks s from 1 down to 0; the DiT must see t = 1 - s."""
    dit = _RecordDit()
    pipe = _pipeline(dit, steps=4)

    _drain(pipe, [_request(3)], ticks=6)

    assert dit.calls, "DiT was never called"
    seen = [float(c["t"][0]) for c in dit.calls]
    # s schedule is linspace(1, 0, 5) -> evaluated at s = 1.0 .. 0.25,
    # so the DiT must see t = 0.0 .. 0.75, strictly increasing.
    assert seen == pytest.approx([0.0, 0.25, 0.5, 0.75], abs=1e-6)
    assert seen == sorted(seen)


def test_velocity_is_negated():
    """v_demon = -v_minimax, checked directly at the adapter."""
    const = 3.0

    class _ConstDit(torch.nn.Module):
        def forward(self, x, t, cond):
            return torch.full_like(x, const)

    adapter = MiniMaxAdapter(
        _ConstDit(),
        schedule_builder=_schedule_builder(4),
        device="cpu",
        dtype=torch.float32,
    )
    xt = torch.zeros(2, T, C)
    out = adapter.batched_forward(
        xt, [1.0, 0.5], [None, None], [None, None], [None, None],
        [_cond(), _cond()],
    )
    assert out.shape == (2, T, C)
    torch.testing.assert_close(out, torch.full_like(xt, -const))


def test_native_layout_transpose_at_the_boundary():
    dit = _RecordDit()
    adapter = MiniMaxAdapter(
        dit,
        schedule_builder=_schedule_builder(4),
        device="cpu",
        dtype=torch.float32,
    )
    out = adapter.batched_forward(
        torch.zeros(3, T, C), [0.9, 0.5, 0.1],
        [None] * 3, [None] * 3, [None] * 3, [_cond()] * 3,
    )
    # DiT sees MiniMax-native [B, C, T]; the seam gets [B, T, C] back.
    assert dit.calls[0]["shape"] == (3, C, T)
    assert out.shape == (3, T, C)


def test_batch_rows_carry_independent_timesteps():
    """The ring buffer batches slots at different denoise stages, so a
    per-row scalar t is load-bearing, not incidental."""
    dit = _RecordDit()
    adapter = MiniMaxAdapter(
        dit,
        schedule_builder=_schedule_builder(4),
        device="cpu",
        dtype=torch.float32,
    )
    adapter.batched_forward(
        torch.zeros(4, T, C), [1.0, 0.75, 0.5, 0.25],
        [None] * 4, [None] * 4, [None] * 4, [_cond()] * 4,
    )
    t = dit.calls[0]["t"]
    assert t.shape == (4,)
    torch.testing.assert_close(t, torch.tensor([0.0, 0.25, 0.5, 0.75]))


# ---- audio-to-audio -------------------------------------------------------


def test_partial_denoise_init_mixes_source_and_noise():
    """The cover path: at denoise < 1 a slot starts as a partially
    noised copy of the anchor, which is what makes consecutive
    generations mutually coherent rather than independent draws."""
    torch.manual_seed(0)
    data = torch.randn(1, C, T)
    source = torch.randn(1, T, C)
    dit = _ExactFlowDit(data)
    pipe = _pipeline(dit, steps=4)

    finished = _drain(pipe, [_request(5, denoise=0.5, source=source)], ticks=8)

    assert len(finished) == 1
    # An exact field still converges on the target from a mixed start.
    torch.testing.assert_close(
        finished[0], data.movedim(1, 2), rtol=1e-4, atol=1e-4,
    )


# ---- conditioning ---------------------------------------------------------


def test_cond_bundles_stack_on_the_batch_axis():
    stacked = stack_minimax_cond_bundles([_cond(1.0), _cond(2.0)])
    cond = stacked["encoder_hidden_states"]
    assert cond.shape == (2, T, MINIMAX_COND_DIM)
    assert float(cond[0, 0, 0]) == 1.0
    assert float(cond[1, 0, 0]) == 2.0


def test_missing_cond_bundle_is_rejected():
    adapter = MiniMaxAdapter(
        _RecordDit(),
        schedule_builder=_schedule_builder(4),
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="aux_cond"):
        adapter.batched_forward(
            torch.zeros(1, T, C), [1.0], [None], [None], [None], [None],
        )


def test_malformed_cond_width_is_rejected():
    with pytest.raises(ValueError, match="encoder_hidden_states"):
        stack_minimax_cond_bundles([{"encoder_hidden_states": torch.zeros(1, T, 7)}])


# ---- schedule -------------------------------------------------------------


def test_schedule_length_must_match_step_count():
    adapter = MiniMaxAdapter(
        _RecordDit(),
        schedule_builder=_schedule_builder(4),
        device="cpu",
        dtype=torch.float32,
    )
    config = DiffusionConfig(infer_steps=8, noise_on_cpu=True, dcw_enabled=False)
    with pytest.raises(ValueError, match="length mismatch"):
        adapter.build_schedule(config, 1.0, "cpu", torch.float32)


def test_shift_warp_preserves_endpoints():
    adapter = MiniMaxAdapter(
        _RecordDit(),
        schedule_builder=_schedule_builder(8),
        device="cpu",
        dtype=torch.float32,
    )
    config = DiffusionConfig(infer_steps=8, noise_on_cpu=True, dcw_enabled=False)
    for alpha in (0.5, 1.0, 2.5):
        adapter.shift_alpha = alpha
        sched = adapter.build_schedule(config, 0.6, "cpu", torch.float32)
        assert float(sched[0]) == pytest.approx(0.6, abs=1e-6)
        assert float(sched[-1]) == pytest.approx(0.0, abs=1e-6)
        # Monotone descending: an inverted dt would break the solver.
        assert torch.all(sched[1:] <= sched[:-1] + 1e-6)


def test_shift_warp_rejects_nonpositive_alpha():
    adapter = MiniMaxAdapter(
        _RecordDit(),
        schedule_builder=_schedule_builder(4),
        device="cpu",
        dtype=torch.float32,
    )
    config = DiffusionConfig(infer_steps=4, noise_on_cpu=True, dcw_enabled=False)
    adapter.shift_alpha = 0.0
    with pytest.raises(ValueError, match="shift_alpha"):
        adapter.build_schedule(config, 1.0, "cpu", torch.float32)


def test_request_frames_requires_latent_frames():
    adapter = MiniMaxAdapter(
        _RecordDit(),
        schedule_builder=_schedule_builder(4),
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="latent_frames"):
        adapter.request_frames(SlotRequest(seed=1, aux_cond=_cond()))
