"""Stable Audio 3 hook points for TADA (offline, eager).

TADA (Staniszewski et al., "TADA! Tuning Audio Diffusion Models through
Activation Steering", arXiv 2602.11910; code github.com/luk-st/steer-audio,
MIT) works on the cross-attention layers of an audio diffusion model:

* **capture / steer**: the OUTPUT of block ``l``'s cross-attention module,
  before it is added to the residual stream (the paper's Eq. 1; the repo's
  ``attn2`` forward hook for Stable Audio Open). Here: a forward hook on
  ``blocks[l].cross_attn``.
* **patch** (activation patching for layer localisation): block ``l``'s
  cross-attention keys and values come from the clean prompt while every
  other layer sees the corrupted one. ``K = c W_K`` and ``V = c W_V`` depend
  only on the text context ``c``, and SA3 masks padded prompt tokens by
  replacing them inside the context (the cross-attention gets no separate
  mask), so substituting block ``l``'s ``context`` argument with the clean
  run's context is exactly the K/V patch. Here: a forward pre-hook on
  ``blocks[l].cross_attn`` that swaps the ``context`` keyword.

SA3 medium runs ``cfg_scale=1`` (one conditional pass per step), so TADA's
``cond_only`` application is "every row". Mean pooling follows the repo's
Stable Audio controller: over batch and every sequence position the
cross-attention sees (SA3 prepends 64 memory tokens; the diffusers SAO DiT
prepends its global token the same way).

Everything here is eager-only and offline; the live path steers through the
pipeline's steering slot (``cross_attn_output`` hook, eager hooks or the
``steering_cross_attn`` TensorRT input).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence

import torch

#: The hook point names this module implements (``acestep.steering.layout``).
HOOK_CROSS_ATTN_OUTPUT = "cross_attn_output"


def sa3_blocks(sam) -> Sequence:
    """The SA3 trunk blocks of a loaded ``StableAudioModel``."""
    from acestep.engine.sa3_internals import trunk_blocks

    return trunk_blocks(sam)


def cross_attn_modules(sam) -> List[torch.nn.Module]:
    """``blocks[i].cross_attn`` for every trunk block (index = block)."""
    mods = [getattr(b, "cross_attn", None) for b in sa3_blocks(sam)]
    if any(m is None for m in mods):
        raise RuntimeError("an SA3 trunk block has no cross_attn module")
    return mods


class StepCounter:
    """Counts DiT forwards so per-step hooks know the diffusion step.

    Installed as a forward pre-hook on block 0 (one call per model
    forward; SA3 at ``cfg_scale=1`` runs one forward per sampler step).
    """

    def __init__(self, sam):
        self.step = -1
        self._h = sa3_blocks(sam)[0].register_forward_pre_hook(self._tick)

    def _tick(self, _m, _inp):
        self.step += 1

    def remove(self) -> None:
        self._h.remove()


@contextmanager
def capture_cross_attn_means(sam, layers: Optional[Iterable[int]] = None):
    """Record the mean cross-attention output per (step, layer).

    Yields a dict ``{step: {layer: tensor[hidden] float32 cpu}}`` that
    fills as generation runs. Mean over batch and sequence (the repo's
    ``_save_activation``).
    """
    mods = cross_attn_modules(sam)
    want = set(range(len(mods))) if layers is None else set(int(i) for i in layers)
    store: Dict[int, Dict[int, torch.Tensor]] = {}
    counter = StepCounter(sam)
    handles = []

    def make(i):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            store.setdefault(counter.step, {})[i] = (
                o.detach().float().mean(dim=(0, 1)).cpu()
            )
        return hook

    for i in sorted(want):
        handles.append(mods[i].register_forward_hook(make(i)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()
        counter.remove()


@contextmanager
def capture_context(sam):
    """Record the text context block 0's cross-attention receives.

    Yields a one-element list that holds the context tensor (the same
    tensor reaches every block; it does not depend on the step)."""
    mods = cross_attn_modules(sam)
    box: list = []

    def pre(_m, args, kwargs):
        ctx = kwargs.get("context")
        if ctx is not None and not box:
            box.append(ctx.detach().clone())
        return None

    h = mods[0].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield box
    finally:
        h.remove()


@contextmanager
def patch_context(sam, layers: Iterable[int], context: torch.Tensor):
    """Blocks in ``layers`` attend to ``context`` (the clean prompt's)
    instead of the running prompt's: TADA's K/V activation patch."""
    mods = cross_attn_modules(sam)

    def pre(_m, args, kwargs):
        if kwargs.get("context") is None:
            return None
        cur = kwargs["context"]
        new = dict(kwargs)
        new["context"] = context.to(device=cur.device, dtype=cur.dtype).expand_as(cur)
        return args, new

    handles = [
        mods[int(i)].register_forward_pre_hook(pre, with_kwargs=True)
        for i in layers
    ]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def steer_cross_attn(
    sam, vectors: Dict[int, Dict[int, torch.Tensor]], alpha: float,
):
    """Offline TADA CAA application: ``h_l += alpha * v[step][l]`` on
    the cross-attention output (``cond_only``; SA3 has one pass).

    ``vectors`` is ``{step: {layer: tensor[hidden]}}``; a missing
    ``(step, layer)`` leaves that call untouched.
    """
    mods = cross_attn_modules(sam)
    counter = StepCounter(sam)
    layers = sorted({li for per in vectors.values() for li in per})
    handles = []

    def make(i):
        def hook(_m, _inp, out):
            v = vectors.get(counter.step, {}).get(i)
            if v is None or alpha == 0.0:
                return out
            return out + float(alpha) * v.to(device=out.device, dtype=out.dtype).view(1, 1, -1)
        return hook

    for i in layers:
        handles.append(mods[i].register_forward_hook(make(i)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
        counter.remove()


@torch.no_grad()
def generate(sam, prompt: str, *, seed: int, duration: float = 10.0, steps: int = 8):
    """One SA3 text-to-audio render: ``[channels, samples]`` float32 cpu.
    ``cfg_scale=1`` (SA3 medium is post-trained for it); the same seed
    gives the same initial latents, which TADA's patching requires."""
    audio = sam.generate(
        prompt=prompt, duration=float(duration), steps=int(steps),
        seed=int(seed), cfg_scale=1.0,
    )
    return audio[0].float().cpu()
