"""Named interpolation strategies for the live blend paths.

Single source of truth for the ``{name: fn(a, b, t)}`` interpolation
registry shared by the conditioning blend (``cond_nodes``), the latent
blend (``vae_nodes``), and the streaming session's per-path method
selector (``streaming.session``). Add a new interpolation type by
defining one function here and adding it to ``INTERPOLATIONS``; every
consumer (the two node ``method`` params and the session's method-name
validation) picks it up automatically. Unknown names raise ``KeyError``
at the call site rather than silently falling back to linear.

``t`` may be a Python scalar (the slider / strength case) or a tensor
broadcastable to the operands' shape along the last dim (the per-frame
``CURVE`` case in ``LatentBlend``).
"""

from __future__ import annotations

import torch


def linear(a: torch.Tensor, b: torch.Tensor, t) -> torch.Tensor:
    """Straight average. ``t=0`` -> a, ``t=1`` -> b."""
    return (1.0 - t) * a + t * b


def slerp(a: torch.Tensor, b: torch.Tensor, t, eps: float = 1e-6) -> torch.Tensor:
    """Spherical-linear interpolation along the last dim, per position.

    Linear interpolation collapses each vector's norm toward the midpoint
    when ``a`` and ``b`` diverge in direction (``|(1-t)a + tb|`` is
    smallest near ``t=0.5``), which thins the blend halfway through a
    sweep: an under-conditioned, washed-out middle for the prompt A->B
    conditioning crossfade, a normed-down latent for latents. Slerp walks
    the geodesic instead, so each position's magnitude is preserved across
    the whole sweep.

    Falls back to linear per position where the geodesic is undefined: the
    two directions parallel (``sin theta -> 0``) or either operand ~0 (the
    silence latent the structure blend mixes against, or a zero-padded
    prompt tail), so those positions just scale the non-zero side instead
    of producing NaNs.

    No scalar ``t<=0`` / ``t>=1`` early-out: at exactly ``t=0`` the
    non-degenerate weights resolve to ``(1, 0)`` (and degenerate positions
    take the linear branch, also ``a``), so the endpoints are returned
    exactly without a guard that would break the tensor-``t`` case.
    """
    na = a.norm(dim=-1, keepdim=True)
    nb = b.norm(dim=-1, keepdim=True)
    ua = a / na.clamp_min(eps)
    ub = b / nb.clamp_min(eps)

    dot = (ua * ub).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp_min(eps)

    w_a = torch.sin((1.0 - t) * theta) / sin_theta
    w_b = torch.sin(t * theta) / sin_theta
    slerped = w_a * a + w_b * b

    lin = (1.0 - t) * a + t * b
    degenerate = (theta < eps) | (na < eps) | (nb < eps)
    return torch.where(degenerate, lin, slerped)


# Registry: name -> fn(a, b, t). Insertion order is the canonical method
# order surfaced to the node-editor dropdowns and the session validator.
# slerp leads because it's the default for the conditioning crossfade and
# the streaming blend paths.
INTERPOLATIONS = {
    "slerp": slerp,
    "linear": linear,
}

# Canonical tuple of method names. Imported by the NodeParam option tuples
# (cond_nodes / vae_nodes) and the streaming session's method validation so
# the set lives in exactly one place on the Python side.
INTERP_METHOD_NAMES = tuple(INTERPOLATIONS)
