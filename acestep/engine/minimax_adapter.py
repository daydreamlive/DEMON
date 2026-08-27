"""MiniMaxAdapter: the MiniMax-Music3 renderer behind the Tier-2 seam.

MiniMax-Music3 is a three-stage model: an 8.58B Qwen3 autoregressive LM
emits one RVQ frame per 40 ms, a depth decoder fills the 7 residual
codebooks, and the fused per-frame hidden states drive a 2.43B
flow-matching DiT over a continuous 128-channel latent at 86.133 Hz. A
deterministic DAC-style decoder ("DAV") takes that latent to 44.1 kHz
stereo.

Only the last two stages live behind this seam. The DiT's sole
conditioning input is ``encoder_hidden_states`` ``[B, T, 2048]`` — there
is no cross-attention and no text tensor anywhere in it, so a prompt
reaches the renderer only after an 8.58B LM has metabolized it into
per-frame hidden states. That makes the conditioning a *captured*
artifact rather than something recomputed per tick: the AR stage runs
once per composition (see :class:`~acestep.engine.minimax_context.
MiniMaxContext`), and the stream then covers that fixed idea forever.
The bundle rides ``SlotRequest.aux_cond``; the ACE-shaped enc/mask/ctx
lists are ignored.

This adapter is TWO boundaries at once, and both matter:

Layout — the shared pipeline is engine-layout ``[B, T, C]``; MiniMax is
native ``[B, C, T]``. ``xt`` is transposed on the way in and the
velocity on the way back out, exactly as :class:`SA3Adapter` does.

Time direction — MiniMax runs flow matching the OTHER WAY ROUND from
every other family in this tree. Its ``t`` goes 0 (noise) to 1 (data),
its interpolant is ``x_t = (1-t)*noise + t*data``, and its Euler step
is ``x += (+1/N) * v``. DEMON's solver (``acestep.engine.ode_steps``)
assumes the rectified-flow convention: ``s`` from 1 (noise) down to 0
(data), ``x_s = s*noise + (1-s)*x0``, and ``x0 = xt - v*s``.

Substituting ``s = 1 - t`` makes the two interpolants identical, so the
latents themselves need no conversion — only the two scalars around
them do:

    t_minimax = 1 - s_demon
    v_demon   = -v_minimax

The velocity flips because ``v_minimax = dx/dt = data - noise`` while
``v_demon = dx/ds = noise - data``. Upstream's own ComfyUI integration
negates the DiT output for the same reason. Getting either half of this
wrong denoises backwards and produces noise, not a wrong-sounding song,
so the failure mode is at least loud.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

# 44100 / 512: the DAV decoder upsamples each latent frame by 8*8*4*2.
MINIMAX_SAMPLE_RATE = 44100
MINIMAX_UPSAMPLE = 512
MINIMAX_LATENT_CHANNELS = 128
MINIMAX_COND_DIM = 2048


def stack_minimax_cond_bundles(bundles: List[dict]) -> dict:
    """Concat per-slot conditioning onto the batch axis.

    Every slot in a tick shares T (``StreamPipeline`` drops slots whose
    frame count disagrees with the newest request), so unlike SA3's
    cross-attention stacker there is nothing to pad — the tensors are
    already the same shape. They are not the same *object*, though:
    after a prompt swap, in-flight slots legitimately carry the old
    composition while fresh submissions carry the new one, which is
    precisely the blend the ring buffer exists to make audible.
    """
    conds = []
    for i, bundle in enumerate(bundles):
        cond = bundle.get("encoder_hidden_states")
        if cond is None:
            raise ValueError(
                f"minimax cond bundle {i} is missing encoder_hidden_states"
            )
        if cond.ndim != 3 or cond.shape[-1] != MINIMAX_COND_DIM:
            raise ValueError(
                "minimax encoder_hidden_states must be [B, T, "
                f"{MINIMAX_COND_DIM}], got {tuple(cond.shape)}"
            )
        conds.append(cond if cond.shape[0] == 1 else cond[:1])
    return {"encoder_hidden_states": torch.cat(conds, dim=0)}


class MiniMaxAdapter:
    """See module docstring. ``schedule_builder`` maps a ``denoise`` to
    a 1-D ``(steps+1,)`` schedule in DEMON convention (descending to
    0); :class:`~acestep.engine.minimax_context.MiniMaxContext` supplies
    the production closure."""

    name = "minimax"
    latent_channels = MINIMAX_LATENT_CHANNELS
    sample_rate = MINIMAX_SAMPLE_RATE
    latent_rate_hz = float(MINIMAX_SAMPLE_RATE) / float(MINIMAX_UPSAMPLE)

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
        # Relative schedule warp (the ``minimax_shift`` knob), the same
        # Flux/SD3 map SA3 uses. 1.0 is the untouched schedule; >1
        # pushes steps toward noise (more structure work), <1 toward
        # refinement. The backend mutates this and MUST invalidate the
        # pipeline's schedule cache — that cache is keyed by denoise
        # alone, so a warp change is invisible to it otherwise.
        self.shift_alpha: float = 1.0
        self._device = torch.device(device)
        self._dtype = dtype

    # ---- ModelAdapter ------------------------------------------------------

    def build_schedule(self, config, denoise: float, device, dtype) -> torch.Tensor:
        schedule = self.schedule_builder(float(denoise)).detach().float()
        if schedule.ndim != 1:
            raise ValueError(
                f"minimax schedule must be 1-D, got {tuple(schedule.shape)}"
            )
        if schedule.numel() != config.infer_steps + 1:
            raise ValueError(
                "minimax schedule length mismatch: expected "
                f"{config.infer_steps + 1}, got {schedule.numel()} — "
                "rebuild the pipeline when the step count changes"
            )
        alpha = float(self.shift_alpha)
        if alpha <= 0.0:
            raise ValueError(f"minimax shift_alpha must be > 0, got {alpha}")
        if abs(alpha - 1.0) > 1e-6:
            # Normalize, warp, rescale — see the long note in
            # SA3Adapter.build_schedule. The map is a monotone [0,1]
            # bijection with fixed points 0 and 1, so it has to act on
            # the schedule normalized to [0,1]; warping already-scaled
            # values inverts the first step's dt whenever the entry
            # sigma is below 1, which is exactly the cover path.
            s_max = schedule[0].clone()
            u = schedule / s_max.clamp_min(1e-9)
            u = alpha * u / (1.0 + (alpha - 1.0) * u)
            schedule = u * s_max
            schedule[0] = s_max
        return schedule.to(device=device, dtype=dtype)

    def request_frames(self, request) -> int:
        if request.latent_frames is None:
            raise ValueError("minimax SlotRequest must carry latent_frames")
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
            raise ValueError("minimax SlotRequest must carry aux_cond")

        # DEMON s -> MiniMax t. See the module docstring: this and the
        # negation below are the whole of the convention bridge.
        t_mm = [1.0 - float(s) for s in timestep_list]

        if getattr(self.dit, "trt_batch1", False):
            # A batch-1 TRT engine makes the ring buffer's batched tick
            # loop instead. Same shape as the SA3 path, including the
            # copy: the wrapper hands back a persistent output buffer
            # that the next iteration overwrites.
            outs = []
            for i in range(xt_batch.shape[0]):
                v_1ct = self.dit.step_bundle(
                    xt_batch[i:i + 1].movedim(1, 2),
                    t_mm[i],
                    aux_list[i],
                )
                outs.append(
                    v_1ct.movedim(1, 2).to(dtype=xt_batch.dtype, copy=True)
                )
            return torch.cat(outs, dim=0).neg_()

        cond = stack_minimax_cond_bundles(list(aux_list))
        x_bct = xt_batch.movedim(1, 2)  # [B,T,C] -> MiniMax-native [B,C,T]
        t_b = torch.tensor(
            t_mm, device=xt_batch.device, dtype=xt_batch.dtype,
        )
        v_bct = self.dit(
            x_bct,
            t_b,
            cond["encoder_hidden_states"].to(
                device=xt_batch.device, dtype=xt_batch.dtype
            ),
        )
        # neg() not neg_(): the DiT's output may alias a buffer we do
        # not own, and the transposed view shares its storage.
        return v_bct.movedim(1, 2).neg()
