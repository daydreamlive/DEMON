"""YuE2 acoustic velocity: the per-song conditioning bundle, the raw-time
transform, the 2nd-order midpoint step, and the eager velocity backend.

YuE2's acoustic stage (the NAR) is a flow-matching solve over a whole
song's ``[T, 64]`` latent, conditioned on an AR-prefix KV cache that is
fixed for the song (upstream ``yue2.nar.CachedNAR``). The released
sampler is a uniform 32-step MIDPOINT grid from noise (s=1) to data
(s=0): ``x <- x - v(x - v(x,s)*h/2, s - h/2) * h`` with ``h = 1/32``.

The ring advances one midpoint step per tick. The pipeline integrates
Euler (``x + (t_next - t_curr) * v``), so the adapter hands it the
MIDPOINT velocity ``v_eff`` (:func:`midpoint_velocity`); Euler on
``v_eff`` is then exactly upstream's update, and every pipeline feature
downstream of the forward (x0_target, feedback, the final-step x0)
composes with it because those only ever see "a velocity".

A velocity backend is any callable
``velocity(bundle, state[B,T,64], raw_times: list[float]) -> [B,T,64]``
whose rows all share ``bundle``. It may return a persistent output
buffer (the TRT backend does); :func:`midpoint_velocity` materializes
the midpoint state before the second call overwrites it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

#: The released acoustic sampler's step count (upstream ``nar.solve``).
RELEASED_STEPS = 32

#: Upstream clamps the logit time to this magnitude, so s=1 enters the
#: model as sigmoid(20), not exactly 1.
RAW_TIME_CLAMP = 20.0

VelocityFn = Callable[[object, torch.Tensor, List[float]], torch.Tensor]


def raw_time(s: float) -> float:
    """Upstream's model time input for schedule time ``s``: ``logit(s)``
    computed in float64 on the CPU and clamped to +-20
    (``yue2.nar.CachedNAR.solve``). The model applies
    ``sigmoid`` (and the checkpoint's shift, 1.0) itself."""
    value = torch.logit(torch.tensor(float(s), dtype=torch.float64))
    return value.clamp(-RAW_TIME_CLAMP, RAW_TIME_CLAMP).item()


def midpoint_velocity(
    velocity: VelocityFn,
    bundle,
    state: torch.Tensor,
    times: List[float],
    h: float,
) -> torch.Tensor:
    """The effective velocity of one midpoint step from ``times`` (one
    schedule time per row) with step size ``h``.

    ``first = v(x, s)``, ``mid = x - first*h/2``, returns
    ``v(mid, s - h/2)``: the exact operations of upstream's solver, so
    the caller's ``x - h*v_eff`` reproduces ``nar.solve`` bit for bit
    on the same backend. The result is a fresh tensor."""
    first = velocity(bundle, state, [raw_time(s) for s in times])
    # Materialize before the second call can overwrite a persistent
    # output buffer.
    mid = state - first * (h / 2)
    second = velocity(bundle, mid, [raw_time(s - h / 2) for s in times])
    return second.clone()


@dataclass(eq=False)
class YuE2Bundle:
    """One song's acoustic conditioning, shared by every slot that
    renders it (rides ``SlotRequest.aux_cond`` by reference; identity is
    what the adapter groups rows by and what the backend attributes a
    finished latent to).

    ``nar`` is the upstream ``CachedNAR`` (its per-layer cache entries
    are views into ``keys`` / ``values``); ``keys`` / ``values`` are the
    stacked ``[layers, L, kv_heads, head_dim]`` AR-prefix KV that the
    TRT engine binds. ``frames`` is the song's latent length T, ``seed``
    the song seed (semantic sampling and the acoustic noise draw).
    """

    nar: object
    keys: Optional[torch.Tensor]
    values: Optional[torch.Tensor]
    frames: int
    seed: int
    prefix: List[int] = field(default_factory=list)
    codec: List[int] = field(default_factory=list)
    tags: str = ""
    lyrics: str = ""
    truncated: bool = False
    epoch: int = 0

    @property
    def cond_tokens(self) -> int:
        """Length L of the AR-prefix KV (prefix + codec + end marker)."""
        return int(self.keys.shape[1]) if self.keys is not None else 0

    def close(self) -> None:
        close = getattr(self.nar, "close", None)
        if close is not None:
            close()
        self.keys = self.values = None


def bundle_from_cached_nar(nar, *, seed: int, **meta) -> YuE2Bundle:
    """Wrap a prefilled ``CachedNAR``: stack its per-layer KV once and
    point the engine's cache entries at views of the stacked tensors,
    so the eager and TRT backends read one copy."""
    keys = torch.stack([k for k, _ in nar.cache])
    values = torch.stack([v for _, v in nar.cache])
    nar.cache = [(keys[i], values[i]) for i in range(keys.shape[0])]
    return YuE2Bundle(
        nar=nar, keys=keys, values=values,
        frames=int(nar.chunk.noise.shape[0]), seed=int(seed), **meta,
    )


def upstream_noise(seed: int, frames: int) -> torch.Tensor:
    """The acoustic noise upstream draws for a song: frame-major CPU
    fp32 ``randn((T, 64))`` from a generator seeded with the song seed
    (``yue2.nar.song_chunks``)."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((int(frames), 64), dtype=torch.float32, generator=generator)


class EagerVelocity:
    """Upstream ``CachedNAR.velocity``, one row at a time.

    Rows loop through the unmodified upstream equations so the eager
    path is numerically upstream's, not a re-derivation of it. Eager is
    the fallback when no TRT engine covers a song; at depth 1 there is
    one row per call anyway.
    """

    def __call__(self, bundle: YuE2Bundle, state: torch.Tensor,
                 raw_times: List[float]) -> torch.Tensor:
        rows = [
            bundle.nar.velocity(state[i], float(raw_times[i]))
            for i in range(state.shape[0])
        ]
        return torch.stack(rows)


def on_grid_index(s: float, steps: int, *, tol: float = 1e-6) -> int:
    """``k`` with ``s == k/steps``; raises when ``s`` is off the uniform
    grid or not strictly inside (0, 1]. The midpoint offset ``h/2`` is
    derived from the grid, so an off-grid time would integrate wrong
    silently."""
    k = s * steps
    k_round = int(round(k))
    if abs(k - k_round) > tol or not 1 <= k_round <= steps:
        raise ValueError(
            f"yue2 timestep {s!r} is not on the {steps}-step grid; the "
            "adapter's step count must match the pipeline's"
        )
    return k_round


def truncated_grid(steps: int, denoise: float) -> torch.Tensor:
    """The released uniform grid truncated by ``denoise``, NOT rescaled:
    ``s_k = 1 - k/steps`` for ``k >= k0``, ``k0 = floor(steps*(1-d))``,
    so a slot takes ``ceil(steps*d)`` steps (at least one) and every
    schedule is a suffix of the same grid (constant ``h = 1/steps``).
    float64 on the CPU so ``k/steps`` survives exactly."""
    d = min(1.0, max(0.0, float(denoise)))
    k0 = min(steps - 1, int(math.floor(steps * (1.0 - d) + 1e-9)))
    k = torch.arange(k0, steps + 1, dtype=torch.float64)
    return 1.0 - k / steps
