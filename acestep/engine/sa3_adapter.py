"""SA3Adapter: Stable Audio 3 behind the Tier-2 ModelAdapter seam.

Drives the shared :class:`~acestep.engine.stream.StreamPipeline` with
the SA3 DiT exactly the way the validated spike pipeline
(``scripts/sa3/sa3_stream_pipeline.SA3StreamPipeline``) drives it: one
batched ``dit(x, t, **stacked_cond)`` call per tick, conditioning
bundles stacked on dim 0 with cross-attn padding, per-slot schedules
from SA3's own ``build_schedule``. What the seam adds on top of the
spike is everything the spike reimplemented or skipped: ring-buffer
slot management, shared curves, the produce/render split, and the rest
of the engine the ACE family already exercises.

Layout: the shared pipeline is engine-layout ``[B, T, C]``; SA3 is
native ``[B, C, T]``. This adapter is the single transpose boundary —
``xt`` is transposed on the way into the DiT and the velocity back on
the way out. Solver math (``ode_steps``) runs in engine layout, which
is equivalent: every step primitive is elementwise over (T, C).

Conditioning rides ``SlotRequest.aux_cond`` (one opaque bundle per
request — the spike's ``cond_bundle`` from ``prepare_sa3_conditioning``,
i.e. the DiT kwargs plus ``cfg_scale=1.0`` / ``padding_mask`` /
``apg_scale``); the ACE-shaped enc/mask/ctx lists are ignored. The
pipeline's own CFG/APG machinery stays dormant for SA3 v1 (post-trained
checkpoints run ``cfg_scale=1.0``; requests carry no
``guidance_curve``).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from acestep.engine.sa3_helpers import import_stream_helpers


class SA3Adapter:
    """See module docstring. ``schedule_builder`` maps a ``denoise``
    (SA3 ``sigma_max`` / ``init_noise_level``) to a 1-D ``(steps+1,)``
    schedule — production uses the spike's ``build_schedule`` closure
    over the captured ``sched_args`` (see
    :class:`~acestep.engine.sa3_context.SA3Context`)."""

    name = "sa3"
    latent_channels = 256
    sample_rate = 44100
    latent_rate_hz = 44100.0 / 4096.0

    def __init__(
        self,
        dit,
        *,
        schedule_builder: Callable[[float], torch.Tensor],
        device,
        dtype,
    ):
        self.dit = dit
        self.schedule_builder = schedule_builder
        # Relative schedule warp on top of the checkpoint's own
        # dist_shift (the ``sa3_shift`` knob): the Flux/SD3 map
        # ``t -> a*t / (1 + (a-1)*t)`` upstream itself uses for
        # constant-alpha shifts (``FluxDistributionShift``). 1.0 is the
        # untouched checkpoint schedule; >1 pushes steps toward noise
        # (more structure work), <1 toward refinement. The backend
        # mutates this and must invalidate the pipeline's schedule
        # cache — the cache is keyed by denoise alone.
        self.shift_alpha: float = 1.0
        self._device = torch.device(device)
        self._dtype = dtype
        # The spike's cond stacker (pads cross-attn tensors to the
        # batch max length, concats dim 0, passes scalars through).
        self._stack = import_stream_helpers().stack_sa3_cond_bundles

    # ---- ModelAdapter ------------------------------------------------------

    def build_schedule(self, config, denoise: float, device, dtype) -> torch.Tensor:
        schedule = self.schedule_builder(float(denoise)).detach().float()
        if schedule.ndim != 1:
            raise ValueError(
                f"SA3 schedule must be 1-D, got {tuple(schedule.shape)}"
            )
        if schedule.numel() != config.infer_steps + 1:
            raise ValueError(
                "SA3 schedule length mismatch: expected "
                f"{config.infer_steps + 1}, got {schedule.numel()} — "
                "rebuild the pipeline when the step count changes"
            )
        alpha = float(self.shift_alpha)
        if alpha <= 0.0:
            raise ValueError(f"sa3 shift_alpha must be > 0, got {alpha}")
        if abs(alpha - 1.0) > 1e-6:
            # Composed AFTER the builder's dist_shift warp. The Flux/SD3
            # map t -> a*t/(1+(a-1)*t) is a monotone [0,1] bijection with
            # fixed points 0 and 1, so it must act on the schedule
            # NORMALIZED to [0,1]. Warping the already-sigma_max-scaled
            # values directly is wrong whenever sigma_max < 1 (the
            # audio-to-audio / cover path, where sigma_max = denoise):
            # the map neither keeps sigma_max fixed nor stays monotone —
            # for a>1 an interior value can overshoot sigma_max and
            # invert the first step's dt. Normalize, warp, rescale so
            # t[0] lands back exactly on sigma_max and t[-1]=0 stays 0;
            # at full denoise (sigma_max=1) this reduces to the bare map.
            sigma_max = schedule[0].clone()
            u = schedule / sigma_max.clamp_min(1e-9)
            u = alpha * u / (1.0 + (alpha - 1.0) * u)
            schedule = u * sigma_max
            schedule[0] = sigma_max
        return schedule.to(device=device, dtype=dtype)

    def request_frames(self, request) -> int:
        if request.latent_frames is None:
            raise ValueError("SA3 SlotRequest must carry latent_frames")
        return int(request.latent_frames)

    def request_device_dtype(self, request):
        return self._device, self._dtype

    def batched_forward(
        self,
        xt_batch: torch.Tensor,
        timestep_list: List[float],
        enc_list: List[Optional[torch.Tensor]],
        mask_list: List[Optional[torch.Tensor]],
        ctx_list: List[Optional[torch.Tensor]],
        aux_list: List[Optional[dict]],
    ) -> torch.Tensor:
        if any(b is None for b in aux_list):
            raise ValueError("SA3 SlotRequest must carry aux_cond")
        if getattr(self.dit, "trt_batch1", False):
            # TRT DiT engines are batch-1 (every profile fixes dim 0 at
            # 1 — see acestep/engine/sa3_trt.py), so the ring buffer's
            # batched tick LOOPS slots through the engine. Per-slot
            # bundles are passed individually (no stacking/padding):
            # after a prompt swap, in-flight slots legitimately carry
            # the old bundle while new submissions carry the new one.
            # Measured ~11-17 ms per call on the 5090, so depth 4 stays
            # well inside the tick budget where one eager batched
            # forward (~54 ms/slot-equivalent) would not.
            outs = []
            for i in range(xt_batch.shape[0]):
                v_1ct = self.dit.step_bundle(
                    xt_batch[i:i + 1].movedim(1, 2),
                    float(timestep_list[i]),
                    aux_list[i],
                )
                # The wrapper returns its persistent output buffer —
                # materialize before the next iteration overwrites it.
                outs.append(
                    v_1ct.movedim(1, 2).to(dtype=xt_batch.dtype, copy=True)
                )
            return torch.cat(outs, dim=0)
        cond = self._stack(list(aux_list))
        x_bct = xt_batch.movedim(1, 2)  # [B,T,C] -> SA3-native [B,C,T]
        t_b = torch.tensor(
            timestep_list, device=xt_batch.device, dtype=xt_batch.dtype,
        )
        v_bct = self.dit(x_bct, t_b, **cond)
        return v_bct.movedim(1, 2)
