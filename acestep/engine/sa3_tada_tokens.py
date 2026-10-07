"""Audio-token selection on Stable Audio 3 for TADA's token means.

The SA3 trunk prepends ``num_memory_tokens`` learned memory tokens (64 on
SA3 medium) to the audio latent sequence before the first block
(vendored ``ContinuousTransformer.forward``: ``x = cat(memory_tokens,
x)``), and every cross-attention output carries those rows too. Stable
Audio Open and ACE-Step have no such rows, so TADA's mean over "the
sequence" is a mean over audio frames there. :class:`SA3AudioTokens`
gives the TADA recorders (``tokens=``) a mask that keeps audio rows only:
memory rows dropped, and audio padding dropped when the trunk is called
with a ``padding_mask`` (True = valid).
"""

from __future__ import annotations

from typing import Optional

import torch


class SA3AudioTokens:
    """``tokens(hs) -> bool [B, seq]`` for hook outputs of the SA3 trunk.

    ``transformer`` is the module whose forward receives ``padding_mask``
    (the trunk's ``transformer``); its last ``padding_mask`` is read by a
    forward pre-hook. Call :meth:`remove` when done.
    """

    def __init__(self, transformer: torch.nn.Module, num_memory_tokens: Optional[int] = None):
        n = num_memory_tokens
        if n is None:
            n = int(getattr(transformer, "num_memory_tokens", 0) or 0)
        self.num_memory_tokens = int(n)
        self.padding_mask: Optional[torch.Tensor] = None
        self._h = transformer.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, _m, _args, kwargs):
        pm = kwargs.get("padding_mask")
        self.padding_mask = None if pm is None else pm.detach().to(torch.bool)
        return None

    def __call__(self, hs: torch.Tensor) -> torch.Tensor:
        b, seq = int(hs.shape[0]), int(hs.shape[1])
        n_audio = seq - self.num_memory_tokens
        if n_audio <= 0:
            raise ValueError(f"sequence of {seq} has no audio rows after "
                             f"{self.num_memory_tokens} memory tokens")
        audio = torch.ones(b, n_audio, dtype=torch.bool, device=hs.device)
        pm = self.padding_mask
        if pm is not None:
            if pm.shape[-1] != n_audio:
                raise ValueError(f"padding_mask covers {pm.shape[-1]} rows, "
                                 f"expected {n_audio} audio rows")
            pm = pm.to(hs.device)
            if pm.shape[0] == 1:
                audio = pm.expand(b, -1)
            elif pm.shape[0] == b:
                audio = pm
            else:
                raise ValueError(f"padding_mask has {pm.shape[0]} rows for a batch of {b}")
        mem = torch.zeros(b, self.num_memory_tokens, dtype=torch.bool, device=hs.device)
        return torch.cat([mem, audio], dim=1)

    def remove(self) -> None:
        self._h.remove()


def sa3_audio_tokens(sam) -> SA3AudioTokens:
    """The selector for a loaded SA3 model (``trunk_module(sam).transformer``)."""
    from acestep.engine.sa3_internals import trunk_module

    return SA3AudioTokens(trunk_module(sam).transformer)
