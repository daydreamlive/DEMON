"""ModelAdapter: the Tier-2 diffusion-family seam inside StreamPipeline.

The backend-seam plan (``round_3_BACKEND_PLAN_FINAL.md`` §2/Tier 2;
engine detail in ``notes/SA3/CLAUDE_SA3_INTEGRATION_PLAN.md`` §4) nests
a model-family seam inside the shared :class:`~acestep.engine.stream.
StreamPipeline`: anything with the rectified-flow shape (per-step
velocity forward, t-schedule, frame-rate codec) plugs in here and
inherits the ring buffer, slot batching, shared curves, CFG/APG, and
refinement semantics unchanged. The pipeline owns the solver math
(:mod:`acestep.engine.ode_steps` — provably the same equations as
SA3's sampler); the adapter owns everything model-shaped:

* the batched decoder forward, INCLUDING conditioning batching (ACE
  pads/concats ``encoder_hidden_states``; SA3 stacks its opaque
  ``aux_cond`` bundle and transposes ``[B,T,C]`` ↔ ``[B,C,T]``),
* timestep-schedule construction (ACE flow-matching ``shift`` warp vs
  SA3's ``build_schedule`` LogSNR warp),
* noise sizing (``latent_channels``: ACE 64, SA3 256),
* per-request frame count / device discovery (ACE reads its
  ``context_latents``; families without one read their own fields).

Realized interface notes (deviations from the plan sketch, recorded so
the doc can be updated from working code):

* ``batched_forward`` receives the SAME per-pair lists the historical
  ``_decoder_forward`` consumed (enc/mask/ctx) plus ``aux_list`` (one
  opaque ``SlotRequest.aux_cond`` per pair). ACE ignores ``aux_list``;
  SA3 ignores the ACE-shaped lists. This keeps the ACE path
  byte-identical (the parity rail: ``tests/unit/
  test_ace_adapter_parity.py`` + ``scripts/ace_drain_parity.py``).
* ``request_frames`` / ``request_device_dtype`` exist because the
  pipeline needs T and device/dtype before the first forward, and the
  historical source for both (``context_latents`` /
  ``encoder_hidden_states``) is ACE-shaped.
* :class:`ACEAdapter` holds a back-reference to its pipeline: the ACE
  TRT dispatch state (engine snapshot, shape-keyed I/O buffer cache)
  is pipeline-owned and stays there — relocating it is Phase-4
  acceleration-contract work, not seam work.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class ModelAdapter(Protocol):
    """The diffusion-family seam. See module docstring."""

    name: str
    latent_channels: int
    latent_rate_hz: float
    sample_rate: int

    def build_schedule(
        self, config, denoise: float, device, dtype,
    ) -> torch.Tensor:
        """Build the (steps+1,) timestep schedule for ``denoise``.

        ``config`` is the pipeline's ``DiffusionConfig`` (step count +
        family-interpreted warp parameters). The pipeline caches the
        result per ``denoise`` value.
        """
        ...

    def batched_forward(
        self,
        xt_batch: torch.Tensor,
        timestep_list: List[float],
        enc_list: List[Optional[torch.Tensor]],
        mask_list: List[Optional[torch.Tensor]],
        ctx_list: List[Optional[torch.Tensor]],
        aux_list: List[Optional[dict]],
    ) -> torch.Tensor:
        """One batched velocity forward for N (slot, condition) pairs.

        ``xt_batch`` is engine-layout ``[B, T, C]``; the returned
        velocity must be too. Each parallel list has one entry per
        batch row; which lists a family reads is its own business.
        """
        ...

    def request_frames(self, request) -> int:
        """Latent frame count T of one ``SlotRequest``."""
        ...

    def request_device_dtype(self, request) -> Tuple[torch.device, torch.dtype]:
        """Device/dtype the pipeline should adopt from one request."""
        ...


class ACEAdapter:
    """The ACE-Step v1.5 family behind the seam — today's math, moved
    verbatim from ``StreamPipeline._decoder_forward`` and friends. The
    default adapter when ``StreamPipeline`` is built without one, so
    every existing call site is behind the seam automatically."""

    name = "acestep"
    latent_channels = 64
    latent_rate_hz = 25.0
    sample_rate = 48000

    def __init__(self, pipeline):
        self._pipeline = pipeline

    def build_schedule(self, config, denoise: float, device, dtype) -> torch.Tensor:
        from .diffusion import DiffusionConfig

        cfg = DiffusionConfig(
            infer_steps=config.infer_steps,
            shift=config.shift,
            denoise=denoise,
        )
        return self._pipeline.engine._build_timestep_schedule(cfg, device, dtype)

    def request_frames(self, request) -> int:
        return request.context_latents.shape[1]

    def request_device_dtype(self, request):
        enc = request.encoder_hidden_states
        return enc.device, enc.dtype

    def batched_forward(
        self,
        xt_batch: torch.Tensor,
        timestep_list: List[float],
        enc_list: List[torch.Tensor],
        mask_list: List[torch.Tensor],
        ctx_list: List[torch.Tensor],
        aux_list: List[Optional[dict]],
    ) -> torch.Tensor:
        """Run one batched decoder forward pass, dispatching TRT or PyTorch.

        Pads encoder tensors to max sequence length and concats along
        the batch dim. Callers apply any channel-gain scaling to
        ``xt_batch`` before this call. List lengths must match
        ``xt_batch.shape[0]``. ``aux_list`` is unused by ACE.

        The TRT engine doesn't consume ``attention_mask`` or
        ``encoder_attention_mask`` — it handles padding via the
        zero-value convention on ``encoder_hidden_states``. Those
        tensors are built only on the PyTorch path.
        """
        p = self._pipeline

        mL = max(e.shape[1] for e in enc_list)
        for i, (e, m) in enumerate(zip(enc_list, mask_list)):
            if e.shape[1] < mL:
                pad = mL - e.shape[1]
                enc_list[i] = torch.nn.functional.pad(e, (0, 0, 0, pad))
                mask_list[i] = torch.nn.functional.pad(m, (0, pad), value=0)

        enc_b = torch.cat(enc_list, dim=0)
        ctx_b = torch.cat(ctx_list, dim=0)

        if p._trt_engine is not None:
            return p._trt_forward(
                xt_batch=xt_batch,
                timestep_list=timestep_list,
                enc_batch=enc_b,
                ctx_batch=ctx_b,
            )

        t_b = torch.tensor(
            timestep_list, device=p._device, dtype=p._dtype,
        )
        mask_b = torch.cat(mask_list, dim=0)
        attn_b = torch.ones(
            xt_batch.shape[0], xt_batch.shape[1],
            device=p._device, dtype=p._dtype,
        )

        out = p.decoder(
            hidden_states=xt_batch,
            timestep=t_b,
            timestep_r=t_b,
            attention_mask=attn_b,
            encoder_hidden_states=enc_b,
            encoder_attention_mask=mask_b,
            context_latents=ctx_b,
            use_cache=False,
            past_key_values=None,
        )
        return out[0]
