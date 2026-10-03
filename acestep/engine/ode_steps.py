"""Pure step primitives for the flow-matching diffusion loop.

Every function here is a small pure tensor op with no ``self`` reads
and no calls into the DiT model — the two tiny flow-matching formulas
that used to live on the model class (``get_x0_from_noise`` and
``renoise``) are inlined as :func:`x0_from_vel` and inside
:func:`step_sde_renoise`. That makes the bricks composable and
``torch.compile``-friendly.

The unified streaming tick composes these primitives in Python:

    x0_pred = x0_from_vel(xt, vt * vs, t_curr)
    if latent_mask:    x0_pred = mask_post_blend_x0(x0_pred, mask, ...)
    if x0_target:      x0_pred = blend_x0_target(x0_pred, target, curve)
    if sde_curve:      xt = step_sde_curve(xt, x0_pred, t_next, sdc, source)
    elif sde:          xt = step_sde_renoise(xt, x0_pred, t_next, noise)
    else:              xt = step_ode_euler(xt, v_blended, t_curr, t_next, 1, onc)

The fast path (no blends, ODE) skips straight to
``step_ode_euler(xt, vt, t_curr, t_next, vs, onc)`` — byte-identical to
the pre-refactor ``_step_simple_ode`` helper.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


# ---------------------------------------------------------------------------
# Curve broadcasting
# ---------------------------------------------------------------------------


def normalize_curve(value: "float | int | bool | torch.Tensor") -> torch.Tensor:
    """Canonicalize a scalar-or-curve value to ``[B, T, 1]``.

    This is the single boundary that turns scalar-or-curve API inputs
    into a uniform tensor shape.  Downstream code multiplies the result
    against ``[B, T, *]`` operands and lets broadcasting handle both
    "scalar" (``[1, 1, 1]``) and per-frame (``[1, T, 1]``) cases — there
    is no need for ``isinstance`` checks anywhere else.

    Accepts:
        - Python ``int`` / ``float`` / ``bool`` -> ``[1, 1, 1]``
        - 0-d tensor -> ``[1, 1, 1]``
        - 1-d ``[T]`` -> ``[1, T, 1]``
        - 2-d ``[B, T]`` -> ``[B, T, 1]``
        - 3-d ``[B, T, 1]`` -> unchanged
    """
    if isinstance(value, (int, float, bool)):
        return torch.tensor([[[float(value)]]], dtype=torch.float32)
    if value.ndim == 0:
        return value.reshape(1, 1, 1).to(dtype=torch.float32)
    if value.ndim == 1:
        return value.unsqueeze(0).unsqueeze(-1)
    if value.ndim == 2:
        return value.unsqueeze(-1)
    if value.ndim == 3:
        return value
    raise ValueError(
        f"normalize_curve: expected scalar / 0-d / 1-d / 2-d / 3-d, got {value.ndim}-d",
    )


# ---------------------------------------------------------------------------
# Flow-matching conversions (inlined from the DiT model class)
# ---------------------------------------------------------------------------


def x0_from_vel(
    xt: torch.Tensor, vt: torch.Tensor, t_curr: float,
) -> torch.Tensor:
    """Predict clean latent from velocity: ``x0 = xt - vt * t``.

    Inlined replacement for ``model.get_x0_from_noise`` so callers
    don't have to hold a model reference. ``t_curr`` is a Python
    scalar; the broadcast is implicit against ``[B, T, D]``.
    """
    return xt - vt * t_curr


# ---------------------------------------------------------------------------
# Latent-mask (inpainting) blends — two-sided
# ---------------------------------------------------------------------------


def mask_pre_blend(
    xt: torch.Tensor,
    t_curr: float,
    latent_mask,  # LatentNoiseMask
    step_idx: int,
    infer_steps: int,
) -> torch.Tensor:
    """Pre-decoder blend: preserved regions get properly-noised original.

    ``x_input = mask * xt + (1 - mask) * (t * noise + (1 - t) * original)``

    Gives the model correct context at the current noise level in
    preserved regions, preventing boundary artifacts.
    """
    mask = latent_mask.get_mask(step_idx, infer_steps)
    noise = latent_mask.ensure_noise(xt.device, xt.dtype)
    noised_original = t_curr * noise + (1.0 - t_curr) * latent_mask.original_latents
    return mask * xt + (1.0 - mask) * noised_original


def mask_post_blend_x0(
    x0_pred: torch.Tensor,
    latent_mask,  # LatentNoiseMask
    step_idx: int,
    infer_steps: int,
) -> torch.Tensor:
    """Post-decoder blend on the x0 prediction: preserved regions get
    the clean original.

    ``x0_blended = mask * x0_pred + (1 - mask) * original``
    """
    mask = latent_mask.get_mask(step_idx, infer_steps)
    return mask * x0_pred + (1.0 - mask) * latent_mask.original_latents


# ---------------------------------------------------------------------------
# x0-target blend (per-frame morph toward a pre-computed target latent)
# ---------------------------------------------------------------------------


def blend_x0_target(
    x0_pred: torch.Tensor,
    x0_target: torch.Tensor,
    x0_target_curve: torch.Tensor,
) -> torch.Tensor:
    """Per-frame blend of the x0 prediction toward a target latent.

    ``x0_blended = (1 - curve) * x0_pred + curve * target``

    ``x0_target_curve`` is a broadcastable ``[1, T, 1]`` tensor
    (already shaped by the caller, e.g. via :func:`normalize_curve`
    and multiplied by any gate). ``x0_target`` is a full latent.
    """
    return (1.0 - x0_target_curve) * x0_pred + x0_target_curve * x0_target


# ---------------------------------------------------------------------------
# Multi-condition weighted velocity compositing
# ---------------------------------------------------------------------------


def blend_velocities(
    velocity_cond_pairs: List[Tuple[torch.Tensor, "Any"]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Blend velocity outputs using per-condition temporal weights.

    ``vt_composite = sum(vt_i * w_i) / sum(w_i)``

    Each pair is ``(vt, condition)`` where condition has a
    ``temporal_weight`` attribute (``[T]``, ``[B, T]``, or ``[B, T, 1]``).
    Conditions with ``temporal_weight=None`` use uniform weight of 1.
    """
    numerator = None
    denominator = None

    for vt, cond in velocity_cond_pairs:
        if cond.temporal_weight is not None:
            w = cond.temporal_weight
            if w.ndim == 1:
                w = w.unsqueeze(0).unsqueeze(-1)
            elif w.ndim == 2:
                w = w.unsqueeze(-1)
        else:
            w = torch.ones(1, 1, 1, device=device, dtype=dtype)

        if numerator is None:
            numerator = vt * w
            denominator = w
        else:
            numerator = numerator + vt * w
            denominator = denominator + w

    return numerator / denominator.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Integration steps — pure, branch-free, torch.compile-friendly
# ---------------------------------------------------------------------------


def step_ode_euler(
    xt: torch.Tensor,
    vt: torch.Tensor,
    t_curr: float,
    t_next: float,
    vs: torch.Tensor,   # [1,1,1] sentinel (ones) or [1,T,1]/[B,T,1] curve
    onc: torch.Tensor,  # [1,1,1] sentinel (zeros) or normalized ode_noise_curve
) -> torch.Tensor:
    """Deterministic Euler ODE step with optional per-step noise injection.

    ``xt_next = xt + (t_next - t_curr) * (vt * vs)``
    ``xt_next = xt_next + randn_like(xt) * onc * t_next``

    The ``vs`` / ``onc`` sentinels (ones / zeros) let the compiled
    graph stay branch-free: multiplying by the sentinel is a no-op in
    value but lets ``torch.compile`` fuse one straight-line graph.

    Callers that already blended ``x0_pred`` (mask, x0_target) can
    synthesize ``v_blended = (xt - x0_pred) / t_curr`` and pass it in
    place of ``vt`` with ``vs=ones_3d`` to reuse this same kernel.
    """
    vt = vt * vs
    dt = t_next - t_curr
    xt = xt + dt * vt
    xt = xt + torch.randn_like(xt) * onc * t_next
    return xt


def step_sde_curve(
    xt: torch.Tensor,
    x0_pred: torch.Tensor,
    t_next: float,
    sdc: torch.Tensor,                 # normalized sde_denoise_curve
    source_latents: torch.Tensor,      # [1, T, D]
) -> torch.Tensor:
    """SDE step with per-frame source blending (paper §3.5).

    Blends two re-noised candidates per frame::

        noise     = randn_like(xt)
        xt_full   = t_next * noise + (1 - t_next) * x0_pred
        xt_source = t_next * noise + (1 - t_next) * source_latents
        xt_next   = sdc * xt_full + (1 - sdc) * xt_source

    ``x0_pred`` is supplied by the caller; callers may pre-blend it
    via :func:`mask_post_blend_x0` and/or :func:`blend_x0_target`
    before calling in.
    """
    sde_noise = torch.randn_like(xt)
    xt_full = t_next * sde_noise + (1.0 - t_next) * x0_pred
    xt_source = t_next * sde_noise + (1.0 - t_next) * source_latents
    return sdc * xt_full + (1.0 - sdc) * xt_source


def step_sde_renoise(
    xt: torch.Tensor,
    x0_pred: torch.Tensor,
    t_next: float,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Bare SDE re-noise (no curve, no source). Upstream-compatible.

    ``xt_next = t_next * noise + (1 - t_next) * x0_pred``

    Inlines the formula that used to live on ``model.renoise``. Noise
    is supplied by the caller so that ``latent_mask``'s fixed-noise
    semantics (:meth:`latent_mask.ensure_noise`) are preserved when a
    mask is active; otherwise the caller passes ``torch.randn_like(xt)``.

    The ``t_next <= 0`` branch returns ``x0_pred`` directly, matching
    the pre-refactor ``_step_simple_sde_renoise`` behavior on the
    final step.
    """
    if t_next <= 0:
        return x0_pred
    return t_next * noise + (1.0 - t_next) * x0_pred


# ---------------------------------------------------------------------------
# APG (Adaptive Prompt Guidance) — classifier-free guidance variant used by
# ACE-Step. Ported from upstream ``apg_guidance.py``. Used by streaming
# callers that mix per-frame ``guidance_curve`` with negative conditioning.
# ---------------------------------------------------------------------------


class MomentumBuffer:
    """Running-average accumulator over velocity deltas for APG.

    The momentum coefficient can be supplied either at construction
    (``MomentumBuffer(momentum=-0.75)``, the legacy form) or per
    :meth:`update` call (the hot-mutable form used by streaming, where
    :func:`apg_forward` looks up the effective momentum from the
    shared-curve registry every step).  The per-call value wins when
    both are supplied; the constructor default makes the legacy
    no-argument ``buffer.update(diff)`` form keep working.

    One buffer per in-flight generation: callers construct it when a
    slot is created with CFG enabled and pass the same instance to
    :func:`apg_forward` every step so the accumulator persists across
    the slot's schedule.
    """

    def __init__(self, momentum: "float | torch.Tensor" = -0.75) -> None:
        self.momentum = momentum
        self.running_average: "torch.Tensor | float" = 0

    def update(
        self,
        update_value: torch.Tensor,
        momentum: "float | torch.Tensor | None" = None,
    ) -> None:
        """Advance the running average by ``update_value``.

        ``momentum`` overrides the constructor default for this call.
        Either form (scalar or per-frame curve) broadcasts against
        ``update_value`` since the multiplication is the only operation.
        """
        m = momentum if momentum is not None else self.momentum
        self.running_average = update_value + m * self.running_average


def apg_project(
    v0: torch.Tensor, v1: torch.Tensor, dims: Tuple[int, ...] = (-1,),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``v0`` into components parallel and orthogonal to ``v1``.

    Runs in fp64 for numerical stability, then casts back to ``v0``'s
    original dtype.
    """
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def apg_forward(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale,
    momentum_buffer: MomentumBuffer,
    momentum: "float | int | torch.Tensor" = -0.75,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
    dims: Tuple[int, ...] = (1,),
) -> torch.Tensor:
    """Apply APG classifier-free guidance to a velocity prediction.

    Both ``guidance_scale`` and ``momentum`` accept the scalar-or-curve
    form: scalars and per-frame ``[T]`` / ``[1, T]`` / ``[1, T, 1]``
    curves are normalized internally and broadcast against
    ``[B, T, D]`` velocities.  The ``momentum_buffer`` accumulates the
    cond/uncond delta across steps and must belong to the generation
    the call is part of.
    """
    momentum_t = normalize_curve(momentum).to(
        device=pred_cond.device, dtype=pred_cond.dtype,
    )
    diff = pred_cond - pred_uncond
    momentum_buffer.update(diff, momentum_t)
    diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    _parallel, diff_orthogonal = apg_project(diff, pred_cond, dims)
    normalized_update = diff_orthogonal + eta * _parallel
    return pred_cond + (guidance_scale - 1) * normalized_update


def cfg_rescale(
    v_guided: torch.Tensor,
    v_pos: torch.Tensor,
    rescale: "float | int | torch.Tensor",
    dims: Tuple[int, ...] = (1,),
    eps: float = 1e-12,
) -> torch.Tensor:
    """Rescale the APG output's per-frame norm toward ``v_pos``'s norm.

    Fix for the magnitude blow-up that high guidance scales cause: the
    APG output's direction is good but its magnitude can be 2x+ of
    ``v_pos``, producing the saturated/clipped sound characteristic of
    over-guided diffusion outputs. Reference: Lin et al., "Common
    Diffusion Noise Schedules and Sample Steps are Flawed."

    ``rescale`` is a scalar or per-frame ``[1, T, 1]`` curve in [0, 1].
    0 returns ``v_guided`` unchanged; 1 fully snaps its per-frame norm
    to ``v_pos``'s while keeping the APG-derived direction. The norm
    computation runs in fp32 for stability, then casts back.
    """
    rescale_t = normalize_curve(rescale).to(
        device=v_guided.device, dtype=v_guided.dtype,
    )
    norm_pos = v_pos.float().norm(p=2, dim=dims, keepdim=True)
    norm_g = v_guided.float().norm(p=2, dim=dims, keepdim=True).clamp_min(eps)
    scale = (norm_pos / norm_g).to(v_guided.dtype)
    rescaled = v_guided * scale
    return rescale_t * rescaled + (1.0 - rescale_t) * v_guided
