"""Stable Audio 3 as a TADA target (offline, eager).

TADA (Staniszewski et al., "TADA! Tuning Audio Diffusion Models through
Activation Steering", arXiv 2602.11910; code github.com/luk-st/steer-audio,
MIT) reads, steers and patches the cross-attention layers of a diffusion
model. :mod:`acestep.tada` holds the method; this module only tells it
where those layers are on SA3:

* ``cross_attn_output``: the output of ``blocks[i].cross_attn`` (the
  vendored ``TransformerBlock`` adds it to the residual stream as
  ``x + cross_attn_scale(cross_attn(norm(x), context=context))``, with
  ``cross_attn_scale`` the identity on SA3 medium). This is the module the
  reference Stable Audio Open controller hooks (``attn2``).
* ``cross_attn_cond``: the ``context`` keyword of the same modules. K and
  V are linear maps of it, and SA3 masks padded prompt tokens inside the
  context itself (the cross-attention gets no separate mask), so
  substituting ``context`` is exactly TADA's K/V activation patch.

SA3 medium is post-trained for ``cfg_scale=1``: each sampler step is one
conditional forward, so TADA's ``cond_only`` is every row and every
forward is one step (``forward_counter(1)``).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Mapping, Sequence

import torch

from acestep.steering.layout import HOOK_CROSS_ATTN_OUTPUT


def sa3_blocks(sam) -> Sequence:
    """The SA3 trunk blocks of a loaded ``StableAudioModel``."""
    from acestep.engine.sa3_internals import trunk_blocks

    return trunk_blocks(sam)


def cross_attn_modules(sam) -> list:
    """``blocks[i].cross_attn`` for every trunk block (index = block)."""
    mods = [getattr(b, "cross_attn", None) for b in sa3_blocks(sam)]
    if not mods or any(m is None for m in mods):
        raise RuntimeError("an SA3 trunk block has no cross_attn module")
    return mods


def sa3_target(sam):
    """The :class:`acestep.tada.ModuleTarget` for a loaded SA3 model."""
    from acestep.tada import ModuleTarget

    return ModuleTarget(
        {HOOK_CROSS_ATTN_OUTPUT: cross_attn_modules(sam)}, cond_keys=("context",),
    )


@contextmanager
def steer_offline(
    sam, vectors: Mapping[int, Mapping[int, torch.Tensor]], alpha: float,
    *, renorm: bool = False,
):
    """TADA's CAA application during an offline SA3 render:
    ``h_l <- h_l + alpha * v[step][l]`` on the cross-attention output
    (:func:`acestep.tada.caa.steer_activation`), every row (one
    conditional pass per step). ``vectors`` is ``{step: {block: [H]}}``;
    absent entries leave that call unchanged. Forwards are counted at
    block 0, so the step index is the sampler step.
    """
    from acestep.tada.caa import steer_activation

    mods = cross_attn_modules(sam)
    blocks = sorted({int(b) for per in vectors.values() for b in per})
    state = {"step": -1}

    def tick(_m, _i):
        state["step"] += 1

    handles = [sa3_blocks(sam)[0].register_forward_pre_hook(tick)]

    def make(block: int):
        def hook(_m, _inp, out):
            v = vectors.get(state["step"], {}).get(block)
            if v is None:
                return out
            return steer_activation(out, v, alpha, renorm=renorm)
        return hook

    for b in blocks:
        handles.append(mods[b].register_forward_hook(make(b)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def generate(sam, prompts: Sequence[str], *, seed: int, duration: float = 10.0,
             steps: int = 8) -> torch.Tensor:
    """One batched SA3 text-to-audio render: ``[B, channels, samples]``
    float32 on cpu, ``cfg_scale=1``. The same ``seed`` and batch size give
    the same initial latents row by row, which patching (clean vs
    corrupted) and the strength sweep (alpha vs alpha = 0) rely on."""
    prompts = list(prompts)
    audio = sam.generate(
        prompt=prompts, duration=float(duration), steps=int(steps),
        seed=int(seed), cfg_scale=1.0, batch_size=len(prompts),
    )
    return audio.float().cpu()
