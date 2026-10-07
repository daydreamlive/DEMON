"""Capture and patching surface for TADA, independent of any model.

TADA (Staniszewski et al., arXiv 2602.11910) works on one site in every
model it studies: the output of each block's cross-attention module,
before that output joins the residual stream (paper Eq. 1). Its two
operations are

* recording that output (time-averaged) for contrastive activation
  addition (:mod:`.caa`), and
* activation patching (:mod:`.patching`): re-running a generation with
  one layer's cross-attention keys and values taken from another
  prompt's run. Keys and values are linear maps of the text conditioning
  (``K = c W_K``, ``V = c W_V``), so the reference code patches the
  cross-attention module's conditioning inputs; so does this module.

A family supplies an :class:`ActivationTarget`: a block count, named hook
points, the modules behind each hook point (one per block, in layout
order), and which call arguments carry the text conditioning.
:class:`ModuleTarget` is the plain implementation over module lists. No
code here names a model; the family adapters build the target.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Callable, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence,
    Tuple, Union, runtime_checkable,
)

import torch

from acestep.steering.layout import HOOK_CROSS_ATTN_OUTPUT

#: The patch point: the conditioning inputs (the source of K and V) of the
#: same cross-attention modules whose outputs ``cross_attn_output`` names.
HOOK_CROSS_ATTN_COND = "cross_attn_cond"

#: A call argument: a keyword name, or an int for a positional argument.
ArgKey = Union[str, int]

#: Per-call recording context: ``None`` skips the call; otherwise a list of
#: ``(rows, step)`` groups, ``rows`` indexing the batch dimension.
CallGroups = Optional[List[Tuple[Union[slice, Sequence[int]], int]]]


@runtime_checkable
class ActivationTarget(Protocol):
    """What TADA needs from a model."""

    @property
    def num_blocks(self) -> int: ...

    def hook_points(self) -> Tuple[str, ...]: ...

    def modules(self, hook: str) -> Sequence[torch.nn.Module]: ...

    def patch_keys(self, hook: str) -> Tuple[ArgKey, ...]: ...


@dataclass
class ModuleTarget:
    """An :class:`ActivationTarget` over explicit module lists.

    ``modules_by_hook[HOOK_CROSS_ATTN_OUTPUT]`` is the list of cross-attention
    modules, one per block. ``cond_keys`` names the call arguments of those
    modules that carry the text conditioning (for example the encoder
    hidden states, their mask, and any rotary table applied to them).
    """

    modules_by_hook: Mapping[str, Sequence[torch.nn.Module]]
    cond_keys: Tuple[ArgKey, ...] = ("encoder_hidden_states",)

    @property
    def num_blocks(self) -> int:
        return len(self.modules_by_hook[HOOK_CROSS_ATTN_OUTPUT])

    def hook_points(self) -> Tuple[str, ...]:
        hooks = tuple(self.modules_by_hook)
        if HOOK_CROSS_ATTN_OUTPUT in hooks and HOOK_CROSS_ATTN_COND not in hooks:
            hooks += (HOOK_CROSS_ATTN_COND,)
        return hooks

    def modules(self, hook: str) -> Sequence[torch.nn.Module]:
        if hook == HOOK_CROSS_ATTN_COND and hook not in self.modules_by_hook:
            hook = HOOK_CROSS_ATTN_OUTPUT
        return self.modules_by_hook[hook]

    def patch_keys(self, hook: str) -> Tuple[ArgKey, ...]:
        return tuple(self.cond_keys)


@dataclass
class _ForwardCounter:
    """Recording context for families that run each denoise step as
    ``passes_per_step`` sequential forwards in a fixed order (classifier
    free guidance as separate passes, the reference ACE-Step layout:
    conditional first). Only ``cond_pass`` is recorded, as TADA does."""

    passes_per_step: int = 2
    cond_pass: int = 0
    forwards: int = -1

    def tick(self) -> None:
        self.forwards += 1

    def __call__(self, batch: int) -> CallGroups:
        k = max(0, self.forwards)
        if k % self.passes_per_step != self.cond_pass:
            return None
        return [(slice(None), k // self.passes_per_step)]


def forward_counter(passes_per_step: int = 2, cond_pass: int = 0) -> _ForwardCounter:
    """A :class:`ActivationRecorder` context counting forwards (see
    :class:`_ForwardCounter`). ``passes_per_step=1`` records every forward
    as its own step (no CFG, or CFG batched into the rows)."""
    return _ForwardCounter(passes_per_step=int(passes_per_step), cond_pass=int(cond_pass))


def time_mean(h: torch.Tensor) -> torch.Tensor:
    """Average over every axis but the last (batch rows and time frames),
    as the reference ``VectorStore`` stores cross-attention outputs."""
    return h.detach().float().reshape(-1, h.shape[-1]).mean(dim=0).cpu()


def frames(h: torch.Tensor) -> torch.Tensor:
    """Every time frame of every batch row as a sample, ``[rows * frames,
    hidden]`` float32 (AUSteer pools pairs and frames; reference
    ``collect_raw_activations``)."""
    return h.detach().float().reshape(-1, h.shape[-1]).cpu()


class ActivationRecorder:
    """Records time-averaged hook-point outputs per (step, block).

    ``context(batch)`` is called at every hooked call and returns the
    row groups to record with their denoise step, or None to skip (for
    example the unconditional CFG pass). A context with a ``tick()``
    method is ticked once per forward (at the first hooked block).

    ``store[step][block]`` is a list with one ``[hidden]`` float32 tensor
    per recorded call, the reference code's ``{step: {layer: [vec]}}``.
    ``reduce`` maps the recorded rows to what is stored: :func:`time_mean`
    (CAA, the default) or :func:`frames` (AUSteer).
    """

    def __init__(
        self,
        target: ActivationTarget,
        hook: str = HOOK_CROSS_ATTN_OUTPUT,
        blocks: Optional[Sequence[int]] = None,
        context: Optional[Callable[[int], CallGroups]] = None,
        reduce: Callable[[torch.Tensor], torch.Tensor] = time_mean,
    ):
        self.reduce = reduce
        self.target = target
        self.hook = hook
        self.blocks = (
            list(range(target.num_blocks)) if blocks is None else [int(b) for b in blocks]
        )
        self.context = context if context is not None else forward_counter(1)
        self.store: Dict[int, Dict[int, List[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._handles: list = []

    def _make_hook(self, block: int, first: bool):
        def _hook(_module, _inputs, output):
            if first and hasattr(self.context, "tick"):
                self.context.tick()
            hs = output[0] if isinstance(output, tuple) else output
            groups = self.context(int(hs.shape[0]))
            if not groups:
                return None
            for rows, step in groups:
                self.store[int(step)][block].append(self.reduce(hs[rows]))
            return None
        return _hook

    def __enter__(self) -> "ActivationRecorder":
        mods = self.target.modules(self.hook)
        first = min(self.blocks)
        for b in self.blocks:
            self._handles.append(
                mods[b].register_forward_hook(self._make_hook(b, b == first))
            )
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def steps(self) -> Dict[int, Dict[int, List[torch.Tensor]]]:
        """Plain-dict copy of ``store``."""
        return {s: dict(b) for s, b in self.store.items()}


@dataclass
class ConditioningPatcher:
    """Activation patching at the cross-attention conditioning.

    :meth:`record` (the clean run) stores, per patched block, the
    conditioning arguments of every call in order; :meth:`patch` (the
    corrupted run, same initial noise and settings) substitutes them call
    by call, so block ``l`` attends to the clean prompt while every other
    block attends to the corrupted one. Every call is patched, the
    unconditional CFG pass included, as the reference implementation does.
    """

    target: ActivationTarget
    blocks: Sequence[int]
    hook: str = HOOK_CROSS_ATTN_COND
    cache: Dict[int, List[Dict[ArgKey, object]]] = field(default_factory=dict)
    leftover: Dict[int, int] = field(default_factory=dict)

    def _keys(self) -> Tuple[ArgKey, ...]:
        return tuple(self.target.patch_keys(self.hook))

    @staticmethod
    def _get(args, kwargs, key: ArgKey):
        if isinstance(key, int):
            return args[key] if key < len(args) else None
        return kwargs.get(key)

    @staticmethod
    def _detach(v):
        if isinstance(v, torch.Tensor):
            return v.detach().clone()
        if isinstance(v, tuple):
            return tuple(ConditioningPatcher._detach(x) for x in v)
        return v

    @contextmanager
    def record(self) -> Iterator["ConditioningPatcher"]:
        mods = self.target.modules(self.hook)
        keys = self._keys()
        self.cache = {int(b): [] for b in self.blocks}
        handles = []

        def make(block: int):
            def pre(_module, args, kwargs):
                self.cache[block].append(
                    {k: self._detach(self._get(args, kwargs, k)) for k in keys}
                )
                return None
            return pre

        for b in self.blocks:
            handles.append(mods[int(b)].register_forward_pre_hook(make(int(b)), with_kwargs=True))
        try:
            yield self
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def patch(self) -> Iterator["ConditioningPatcher"]:
        mods = self.target.modules(self.hook)
        queues = {b: list(calls) for b, calls in self.cache.items()}
        handles = []

        def make(block: int):
            def pre(_module, args, kwargs):
                if not queues[block]:
                    raise RuntimeError(
                        f"patching block {block}: more calls than the clean run recorded"
                    )
                saved = queues[block].pop(0)
                args = list(args)
                for k, v in saved.items():
                    if isinstance(k, int):
                        if k < len(args):
                            args[k] = v
                    elif k in kwargs or v is not None:
                        kwargs[k] = v
                return tuple(args), kwargs
            return pre

        for b in self.blocks:
            handles.append(mods[int(b)].register_forward_pre_hook(make(int(b)), with_kwargs=True))
        try:
            yield self
        finally:
            for h in handles:
                h.remove()
            # Calls the clean run made that the patched run did not
            # consume; nonzero means the two runs were not aligned.
            self.leftover = {b: len(q) for b, q in queues.items() if q}
