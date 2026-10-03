"""YuE2Adapter: YuE2's acoustic NAR behind the Tier-2 ModelAdapter seam.

The shared :class:`~acestep.engine.stream.StreamPipeline` drives the
NAR the way it drives SA3: one ring slot per in-flight generation of
the whole song, one midpoint step per tick. What is YuE2-shaped lives
here:

* geometry: ACE's exactly (64 channels, 25 Hz, 48 kHz stereo), so the
  decode boundary needs no resample;
* schedule: the released 32-step uniform grid, truncated (never
  rescaled) by ``yue2_denoise`` (:func:`~acestep.engine.yue2_velocity.
  truncated_grid`), so every schedule shares ``h = 1/steps``;
* forward: rows grouped by conditioning bundle (the flexible TRT engine
  binds ONE KV tensor per execution), each group returning the midpoint
  velocity ``v_eff`` (:func:`~acestep.engine.yue2_velocity.
  midpoint_velocity`); the pipeline's Euler step on ``v_eff`` is
  upstream's midpoint update;
* noise: upstream's frame-major per-seed draw (:meth:`make_noise`), so a
  seed renders the same performance it renders upstream.

No CFG: YuE2's guidance acts only in the AR stages. Requests carry no
``guidance_curve`` and the adapter reads only ``aux_list``.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from acestep.engine.yue2_velocity import (
    RELEASED_STEPS,
    VelocityFn,
    midpoint_velocity,
    on_grid_index,
    truncated_grid,
    upstream_noise,
)


class YuE2Adapter:
    """See module docstring. ``velocity`` is a velocity backend
    (``yue2_velocity.EagerVelocity`` or the TRT one); ``steps`` must
    equal the pipeline's ``config.infer_steps`` (rebuild both together
    on a steps change, as SA3 does)."""

    name = "yue2"
    latent_channels = 64
    latent_rate_hz = 25.0
    sample_rate = 48000

    def __init__(self, velocity: VelocityFn, *, steps: int = RELEASED_STEPS,
                 device="cpu", dtype=torch.float32):
        self.velocity = velocity
        self.steps = int(steps)
        self._device = torch.device(device)
        self._dtype = dtype
        # One cached noise draw: (seed, frames) -> [T, 64] fp32 CPU. A
        # session re-renders the same seed every tick.
        self._noise_key: Optional[tuple] = None
        self._noise: Optional[torch.Tensor] = None

    # ---- ModelAdapter ------------------------------------------------------

    def build_schedule(self, config, denoise: float, device, dtype) -> torch.Tensor:
        if int(config.infer_steps) != self.steps:
            raise ValueError(
                f"yue2 adapter built for {self.steps} steps, pipeline has "
                f"{config.infer_steps}; rebuild both together"
            )
        return truncated_grid(self.steps, denoise)

    def request_frames(self, request) -> int:
        if request.latent_frames is None:
            raise ValueError("YuE2 SlotRequest must carry latent_frames")
        return int(request.latent_frames)

    def request_device_dtype(self, request):
        return self._device, self._dtype

    def make_noise(self, request) -> torch.Tensor:
        """Upstream's acoustic noise for ``request.seed``, ``[1, T, 64]``
        on the pipeline device/dtype."""
        if not isinstance(request.seed, int) or isinstance(request.seed, bool):
            raise ValueError(f"yue2 requests need an int seed, got {request.seed!r}")
        key = (int(request.seed), self.request_frames(request))
        if key != self._noise_key:
            self._noise = upstream_noise(*key)
            self._noise_key = key
        return self._noise.to(device=self._device, dtype=self._dtype)[None]

    def batched_forward(
        self,
        xt_batch: torch.Tensor,
        timestep_list: List[float],
        enc_list: List[Optional[torch.Tensor]],
        mask_list: List[Optional[torch.Tensor]],
        ctx_list: List[Optional[torch.Tensor]],
        aux_list: List[Optional[object]],
    ) -> torch.Tensor:
        if any(b is None for b in aux_list):
            raise ValueError("YuE2 SlotRequest must carry aux_cond")
        for s in timestep_list:
            on_grid_index(float(s), self.steps)
        h = 1.0 / self.steps
        out = torch.empty_like(xt_batch)
        for bundle, rows in group_rows_by_bundle(aux_list):
            v = midpoint_velocity(
                self.velocity, bundle, xt_batch[rows],
                [float(timestep_list[i]) for i in rows], h,
            )
            out[rows] = v.to(dtype=xt_batch.dtype)
        return out


def group_rows_by_bundle(aux_list) -> list:
    """``[(bundle, [row, ...]), ...]`` in first-seen order, grouped by
    bundle IDENTITY: after a conditioning swap, in-flight slots carry
    the old bundle and new ones the new bundle, and one execution can
    bind only one KV tensor."""
    groups: dict = {}
    order: list = []
    for i, bundle in enumerate(aux_list):
        key = id(bundle)
        if key not in groups:
            groups[key] = (bundle, [])
            order.append(key)
        groups[key][1].append(i)
    return [groups[k] for k in order]
