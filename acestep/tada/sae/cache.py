"""Generation-time activation caching for TADA's SAEs.

TADA (arXiv 2602.11910, App. I.2; reference
``cache_activations_runner*.py``) trains its SAEs on cross-attention
outputs recorded WHILE the model generates from a set of varied captions
(MusicCaps): the conditional CFG pass only, every token of the layer's
output kept (one training sample = one ``[sequence, d_in]`` matrix), at
every ``k``-th denoise step (ACE-Step: every 6th of 30 steps, CFG 5, 10 s;
Stable Audio Open: every 10th of 100 steps, CFG 7, 10 s, each caption four
times). Nothing is pooled at capture time.

:class:`TokenRecorder` is that capture on any :class:`~acestep.tada.target.
ActivationTarget`: forward hooks on the hook point's modules, a call
context (the core's :func:`~acestep.tada.target.forward_counter`) that
says which batch rows are the conditional pass and which denoise step a
call belongs to, and an optional ``sigma_fn`` that reports the noise level
of the current forward (recorded so the trainer can report FVU per noise
bucket). :class:`ActivationStore` writes one fp16 shard per generation
batch per block and reads the whole cache back for training.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from acestep.tada.target import (
    HOOK_CROSS_ATTN_OUTPUT, ActivationTarget, CallGroups, forward_counter,
)


#: ``tokens(hs) -> bool mask`` over the sequence axis of a hook output
#: ``hs [B, seq, d]``: ``[seq]`` (same for every row) or ``[B, seq]``.
#: True = an audio token to keep. Families use it to drop tokens that are
#: not audio (learned memory or prepended conditioning tokens, padding),
#: which the reference models do not have inside the cross-attention
#: output sequence.
TokenSelect = Callable[[torch.Tensor], torch.Tensor]


def select_tokens(hs: torch.Tensor, tokens: Optional[TokenSelect]) -> torch.Tensor:
    """``hs [B, seq, d]`` restricted to the kept tokens: ``[B, kept, d]``
    when the mask is shared by every row, else ``[n_kept_total, d]``."""
    if tokens is None:
        return hs
    m = tokens(hs).to(device=hs.device, dtype=torch.bool)
    if m.ndim == 1:
        return hs[:, m]
    if m.ndim == 2 and bool((m == m[:1]).all()):
        return hs[:, m[0]]
    return hs[m]


class TokenRecorder:
    """Records full-token hook-point outputs every ``every``-th step.

    ``store[block]`` is a list of ``(step, sigma, tensor[rows, seq, d])``
    in call order (fp16 on cpu). ``context`` is as for the core's
    :class:`~acestep.tada.target.ActivationRecorder`: called once per
    hooked call with the batch size, returns ``[(rows, step)]`` to record
    or ``None`` (for example the unconditional CFG pass); a context with
    ``tick()`` is ticked once per forward at the first recorded block.
    """

    def __init__(
        self,
        target: ActivationTarget,
        blocks: Sequence[int],
        *,
        every: int = 1,
        hook: str = HOOK_CROSS_ATTN_OUTPUT,
        context: Optional[Callable[[int], CallGroups]] = None,
        sigma_fn: Optional[Callable[[], float]] = None,
        tokens: Optional[TokenSelect] = None,
        dtype: torch.dtype = torch.float16,
    ):
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self.target = target
        self.blocks = [int(b) for b in blocks]
        self.every = int(every)
        self.hook = hook
        self.context = context if context is not None else forward_counter(1)
        self.sigma_fn = sigma_fn
        self.tokens = tokens
        self.dtype = dtype
        self.store: Dict[int, List[Tuple[int, float, torch.Tensor]]] = defaultdict(list)
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
                if int(step) % self.every:
                    continue
                sigma = float(self.sigma_fn()) if self.sigma_fn is not None else float("nan")
                x = select_tokens(hs[rows], self.tokens)
                if x.ndim != 3:
                    raise ValueError("rows of one recorded call keep different token sets; "
                                     "a training sample needs a fixed token count")
                self.store[block].append((int(step), sigma, x.detach().to(self.dtype).cpu()))
            return None
        return _hook

    def __enter__(self) -> "TokenRecorder":
        mods = self.target.modules(self.hook)
        first = min(self.blocks)
        for b in self.blocks:
            self._handles.append(mods[b].register_forward_hook(self._make_hook(b, b == first)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def samples(self, block: int) -> Tuple[torch.Tensor, List[int], List[float], List[int]]:
        """``(acts[n, seq, d], steps, sigmas, rows)``; each recorded row is
        one sample (a batch of ``B`` rows at one step gives ``B`` samples;
        ``rows`` is the row's position within its recorded group)."""
        acts, steps, sigmas, rows = [], [], [], []
        for step, sigma, t in self.store[int(block)]:
            for r in range(t.shape[0]):
                acts.append(t[r])
                steps.append(step)
                sigmas.append(sigma)
                rows.append(r)
        if not acts:
            raise ValueError(f"no activations recorded for block {block}")
        return torch.stack(acts), steps, sigmas, rows


class ActivationStore:
    """On-disk cache: ``<root>/block_<b>/shard_<i>.npy`` (fp16
    ``[n, seq, d]``) plus ``shard_<i>.json`` (per-sample ``prompt``,
    ``step``, ``sigma``) and ``<root>/config.json`` (capture settings)."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def write_config(self, config: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def config(self) -> dict:
        p = self.root / "config.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _dir(self, block: int) -> Path:
        return self.root / f"block_{int(block)}"

    def shard_count(self, block: int) -> int:
        d = self._dir(block)
        return len(list(d.glob("shard_*.npy"))) if d.exists() else 0

    def add(self, block: int, acts: torch.Tensor, prompts: Sequence[int],
            steps: Sequence[int], sigmas: Sequence[float]) -> Path:
        n = int(acts.shape[0])
        if not (len(prompts) == len(steps) == len(sigmas) == n):
            raise ValueError("prompts, steps and sigmas must have one entry per sample")
        d = self._dir(block)
        d.mkdir(parents=True, exist_ok=True)
        i = self.shard_count(block)
        path = d / f"shard_{i:05d}.npy"
        tmp = d / f"shard_{i:05d}.tmp.npy"
        np.save(tmp, acts.to(torch.float16).numpy())
        (d / f"shard_{i:05d}.json").write_text(json.dumps({
            "prompt": [int(p) for p in prompts], "step": [int(s) for s in steps],
            "sigma": [float(s) for s in sigmas],
        }), encoding="utf-8")
        tmp.replace(path)
        return path

    def blocks(self) -> List[int]:
        return sorted(int(p.name.split("_")[1]) for p in self.root.glob("block_*") if p.is_dir())

    def load(self, block: int, prompts: Optional[Iterable[int]] = None,
             ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        """All samples of one block as ``(acts fp16 [N, seq, d], meta)``;
        ``prompts`` keeps only samples of those prompt indices."""
        keep = None if prompts is None else set(int(p) for p in prompts)
        acts, meta = [], {"prompt": [], "step": [], "sigma": []}
        for path in sorted(self._dir(block).glob("shard_*.npy")):
            if path.name.endswith(".tmp.npy"):
                continue
            m = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            a = np.load(path, mmap_mode="r")
            idx = [i for i, p in enumerate(m["prompt"]) if keep is None or p in keep]
            if not idx:
                continue
            acts.append(torch.from_numpy(np.ascontiguousarray(a[idx])))
            for k in meta:
                meta[k].extend(m[k][i] for i in idx)
        if not acts:
            raise FileNotFoundError(f"no cached samples for block {block} under {self.root}")
        return torch.cat(acts), {k: np.asarray(v) for k, v in meta.items()}


def cache_generations(
    generate: Callable[[Sequence[str], int], object],
    prompts: Sequence[str],
    recorder_factory: Callable[[], TokenRecorder],
    store: ActivationStore,
    *,
    seeds: Sequence[int],
    batch_size: int = 1,
    start: int = 0,
    on_batch: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Generate ``prompts`` in batches of ``batch_size`` (``generate(batch,
    seed)``, batch ``j`` uses ``seeds[j]``; recorded row ``r`` of a batch
    is that batch's ``r``-th prompt) under a fresh recorder and append
    every recorded sample to ``store``, one shard per batch and block.
    Resumable: ``start`` (a multiple of ``batch_size``) skips prompts
    already cached. Returns the number of prompts cached by this call."""
    bs = max(1, int(batch_size))
    if int(start) % bs:
        raise ValueError(f"start {start} is not a multiple of batch_size {bs}")
    done = 0
    for i in range(int(start), len(prompts), bs):
        batch = list(prompts[i:i + bs])
        rec = recorder_factory()
        with rec:
            generate(batch, int(seeds[i // bs]))
        for b in rec.blocks:
            acts, steps, sigmas, rows = rec.samples(b)
            if max(rows) >= len(batch):
                raise RuntimeError(f"recorded {max(rows) + 1} rows for a batch of {len(batch)}")
            store.add(b, acts, [i + r for r in rows], steps, sigmas)
        done += len(batch)
        if on_batch is not None:
            on_batch(i, len(batch))
    return done
