"""Classifier-free guidance for Tier-2 (``aux_cond``) families.

``neg_conditions`` is ACE-shaped: a list of ``SlotCondition`` carrying
``encoder_hidden_states`` and a mask. A family that puts all of its
conditioning in the opaque ``aux_cond`` bundle cannot populate it, so
before ``neg_aux_cond`` existed such a family had no way to run CFG at
all — and the failure was silent in the worst way. ``has_cfg`` returned
False, no negative pass was scheduled, nothing raised, and the only
symptom was that the audio was worse than the reference model's.

Worse still is the near miss: schedule the negative pass but let it read
``aux_cond``, and every negative row sees the POSITIVE bundle. Then
``v_neg == v_pos``, APG returns ``v_pos`` unchanged, and guidance costs
a full extra forward per step while doing nothing at all.

Both are asserted here against the real :class:`StreamPipeline` with a
fake adapter, on CPU, no weights.
"""

from __future__ import annotations

import pytest
import torch

from acestep.engine import ode_steps
from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.stream import SlotRequest, StreamPipeline

T = 6
C = 4


class _AuxAdapter:
    """Minimal Tier-2 adapter that records the bundle of every row."""

    name = "fake-aux"
    latent_channels = C

    def __init__(self):
        self.seen: list = []

    def build_schedule(self, config, denoise, device, dtype):
        return torch.linspace(
            float(denoise), 0.0, config.infer_steps + 1,
        ).to(device=device, dtype=dtype)

    def request_frames(self, request):
        return int(request.latent_frames)

    def request_device_dtype(self, request):
        return torch.device("cpu"), torch.float32

    def batched_forward(self, xt_batch, timestep_list, enc_list, mask_list,
                        ctx_list, aux_list):
        for bundle in aux_list:
            self.seen.append(None if bundle is None else bundle["tag"])
        # Depend on the bundle, so a negative pass that re-sent the
        # positive one would be indistinguishable in value as well as
        # invisible in the record.
        gains = torch.tensor(
            [[[float(b["gain"])]] for b in aux_list], dtype=xt_batch.dtype,
        )
        return xt_batch * gains


def _pipeline(adapter, *, steps=3, depth=1):
    return StreamPipeline(
        None,
        DiffusionConfig(infer_steps=steps, infer_method="ode",
                        noise_on_cpu=True, dcw_enabled=False),
        pipeline_depth=depth,
        adapter=adapter,
    )


def _request(**kw) -> SlotRequest:
    params = dict(
        seed=11,
        latent_frames=T,
        aux_cond={"tag": "positive", "gain": 1.0},
    )
    params.update(kw)
    return SlotRequest(**params)


def _drain(pipe, req, ticks=8):
    pipe.submit(req)
    for _ in range(ticks):
        out = pipe.tick()
        if out is not None:
            return out
    return None


# ---- has_cfg --------------------------------------------------------------


def test_aux_family_enables_cfg_without_ace_neg_conditions():
    req = _request(neg_aux_cond={"tag": "negative", "gain": 0.0},
                   guidance_curve=2.0)
    assert req.has_cfg is True
    assert req.neg_conditions == []


def test_guidance_curve_alone_is_not_enough():
    assert _request(guidance_curve=2.0).has_cfg is False


def test_neg_bundle_alone_is_not_enough():
    assert _request(neg_aux_cond={"tag": "n", "gain": 0.0}).has_cfg is False


# ---- the negative pass ----------------------------------------------------


def test_negative_pass_receives_the_negative_bundle():
    """The near miss: a scheduled negative pass reading ``aux_cond``.

    ``v_neg`` would equal ``v_pos``, APG would return ``v_pos``, and the
    extra forward would buy nothing.
    """
    adapter = _AuxAdapter()
    pipe = _pipeline(adapter)
    _drain(pipe, _request(neg_aux_cond={"tag": "negative", "gain": 0.0},
                          guidance_curve=2.0))

    assert "positive" in adapter.seen
    assert "negative" in adapter.seen, (
        "the negative pass re-sent the positive bundle; guidance is a "
        "no-op that still costs a forward per step"
    )
    assert adapter.seen.count("negative") == adapter.seen.count("positive")


def test_no_negative_pass_without_guidance():
    adapter = _AuxAdapter()
    pipe = _pipeline(adapter)
    _drain(pipe, _request())
    assert set(adapter.seen) == {"positive"}


def test_ace_requests_still_get_their_own_aux_cond_on_the_negative_pass():
    """``neg_aux_cond`` is opt-in. A family that leaves it unset must
    keep seeing ``aux_cond`` on both passes, exactly as before."""
    adapter = _AuxAdapter()
    pipe = _pipeline(adapter)
    from acestep.engine.stream import SlotCondition

    _drain(pipe, _request(
        neg_conditions=[SlotCondition(
            encoder_hidden_states=torch.zeros(1, 2, 3),
            encoder_attention_mask=torch.ones(1, 2),
        )],
        guidance_curve=2.0,
    ))
    assert set(adapter.seen) == {"positive"}


# ---- the combine operator -------------------------------------------------


def test_apg_reduces_to_textbook_cfg_at_eta_one_without_a_norm_cap():
    """``apg_eta=1`` + ``norm_threshold<=0`` + ``momentum=0`` is the
    reference operator ``v_u + w*(v_c - v_u)``.

    A family whose upstream sampler uses plain CFG needs this exactly;
    stock APG's norm cap is calibrated for ACE's latent scale and
    throttles a long-sequence guidance delta hard.
    """
    torch.manual_seed(0)
    v_c = torch.randn(1, 32, 8)
    v_u = torch.randn(1, 32, 8)
    w = 1.7

    got = ode_steps.apg_forward(
        v_c, v_u, guidance_scale=w,
        momentum_buffer=ode_steps.MomentumBuffer(), momentum=0.0,
        eta=1.0, norm_threshold=0.0,
    )
    want = v_u + w * (v_c - v_u)
    assert torch.allclose(got, want, atol=1e-6)


def test_stock_apg_defaults_are_untouched():
    """The new parameters must not move ACE or SA3. Same call, default
    request values, has to equal the historical hardcoded behavior."""
    torch.manual_seed(0)
    v_c, v_u = torch.randn(1, 32, 8), torch.randn(1, 32, 8)
    default = SlotRequest()
    got = ode_steps.apg_forward(
        v_c, v_u, guidance_scale=1.7,
        momentum_buffer=ode_steps.MomentumBuffer(), momentum=-0.75,
        eta=default.apg_eta, norm_threshold=default.apg_norm_threshold,
    )
    legacy = ode_steps.apg_forward(
        v_c, v_u, guidance_scale=1.7,
        momentum_buffer=ode_steps.MomentumBuffer(), momentum=-0.75,
    )
    assert torch.equal(got, legacy)
    assert not torch.allclose(got, v_u + 1.7 * (v_c - v_u), atol=1e-4), (
        "stock APG is meaningfully different from textbook CFG; if this "
        "ever stops holding, the vanilla-mode plumbing is redundant"
    )


def test_guidance_actually_changes_the_result():
    """End-to-end: the same request with and without guidance must not
    drain to the same latent. Guards the whole chain at once."""
    plain = _pipeline(_AuxAdapter())
    guided = _pipeline(_AuxAdapter())
    a = _drain(plain, _request())
    b = _drain(guided, _request(
        neg_aux_cond={"tag": "negative", "gain": 0.0},
        guidance_curve=2.0, apg_momentum=0.0, apg_eta=1.0,
        apg_norm_threshold=0.0,
    ))
    assert a is not None and b is not None
    assert not torch.allclose(a, b), "guidance had no effect on the output"
