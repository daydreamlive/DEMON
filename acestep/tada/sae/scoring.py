"""Concept features and SAE steering vectors (TADA Eq. 12-14).

Paper App. I.1.4 and I.2; reference ``SAEScoresScorer`` (``src/steering/
methods/sae/scorer.py``) and the ``steering_vector`` intervention
(``sae_intervention`` in ``acestep_hooks.py``).

* Feature means: generate the concept's positive and negative prompts
  (TADA's Table 8 templates, 50 each; reference: seed 10, 30 steps, CFG 5,
  30 s, the conditional pass), pass the localised layer's cross-attention
  output through that layer's SAE and average the encoder activations
  ``relu(W_enc (h - b_dec) + b_enc)`` (no TopK, as the reference scores)
  over prompts and tokens, per denoise step: ``mu_j(P)``.
* Score (Eq. 12): ``score(j, c) = mu_j(P_c) * log(1 + 1 / (mu_j(P_~c) +
  eps))``, ``eps = 1e-6``. The reference stores it per step
  (``[num_timesteps, num_features]``) next to ``diff`` and ``mean_pos``.
* Vector (Eq. 13): ``v_SAE = sum_{j in F_c} W_dec[:, j]``, ``F_c`` the
  ``k_c`` highest scores; unit weights (their Table 19 winner).
  Per-step form (the reference): top-``k_c`` per step, one vector per
  step. Single-vector form (the paper's equation): scores from the means
  pooled over every recorded step.
* Application (Eq. 14): ``h_l <- h_l + alpha * v_SAE`` at the
  cross-attention output, conditional pass only, renorm (the reference
  eval configs), negative direction = negative ``alpha`` (their Table 20).
* ``k_c`` is chosen per concept on held-out prompts by the downstream
  steering metric (Fig. 13 grid ``{5, 10, 20, 50, 100, 500}`` per layer):
  :func:`select_k`.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from acestep.tada.target import (
    HOOK_CROSS_ATTN_OUTPUT, ActivationTarget, CallGroups, forward_counter,
)

from .cache import TokenSelect, select_tokens
from .model import Sae

#: Reference epsilon in Eq. 12.
TFIDF_EPS = 1e-6
#: Paper Fig. 13 grid for ``k_c``.
K_GRID = (5, 10, 20, 50, 100, 500)


class FeatureMeanRecorder:
    """Per-step mean SAE encoder activation of each block's hook output.

    ``means[block][step]`` accumulates ``(sum over tokens of relu
    pre-acts, token count)`` over the kept tokens (``tokens``, as for
    :class:`~acestep.tada.sae.cache.TokenRecorder`) so that :meth:`result` returns
    ``{block: tensor[n_steps, num_latents]}`` (steps in ascending order),
    the mean over every recorded token of every generation run under this
    recorder (prompts of one set have equal lengths, so this is the
    reference's mean over prompts and tokens).
    """

    def __init__(
        self,
        target: ActivationTarget,
        saes: Mapping[int, Sae],
        *,
        hook: str = HOOK_CROSS_ATTN_OUTPUT,
        context: Optional[Callable[[int], CallGroups]] = None,
        tokens: Optional[TokenSelect] = None,
        sigma_fn: Optional[Callable[[], float]] = None,
        step_scale: Optional[Mapping[int, Mapping[int, float]]] = None,
    ):
        self.target = target
        self.tokens = tokens
        #: ``{block: {step: rms}}``: inputs are divided by it before the
        #: encoder (for SAEs trained on per-step RMS normalized activations).
        self.step_scale = {int(b): {int(k): float(v) for k, v in m.items()}
                           for b, m in (step_scale or {}).items()}
        self.sigma_fn = sigma_fn
        #: ``{step: sigma}`` seen while recording (noise level per step).
        self.sigmas: Dict[int, float] = {}
        self.saes = {int(b): s for b, s in saes.items()}
        self.hook = hook
        self.context = context if context is not None else forward_counter(1)
        self._sum: Dict[int, Dict[int, Tensor]] = defaultdict(dict)
        self._cnt: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._handles: list = []

    def reset_steps(self) -> None:
        """Restart step counting (call between generations when the
        context counts forwards)."""
        if hasattr(self.context, "forwards"):
            self.context.forwards = -1

    def _make_hook(self, block: int, first: bool):
        sae = self.saes[block]

        @torch.no_grad()
        def _hook(_module, _inputs, output):
            if first and hasattr(self.context, "tick"):
                self.context.tick()
            hs = output[0] if isinstance(output, tuple) else output
            groups = self.context(int(hs.shape[0]))
            if not groups:
                return None
            for rows, step in groups:
                x = select_tokens(hs[rows].detach(), self.tokens)
                x = x.reshape(-1, x.shape[-1]).to(device=sae.device)
                scale = self.step_scale.get(block, {}).get(int(step))
                if scale is not None:
                    x = x / scale
                acts = sae.pre_acts(x).float()
                s = acts.sum(0).cpu()
                d = self._sum[block]
                d[int(step)] = d[int(step)] + s if int(step) in d else s
                self._cnt[block][int(step)] += int(acts.shape[0])
                if self.sigma_fn is not None:
                    self.sigmas.setdefault(int(step), float(self.sigma_fn()))
            return None
        return _hook

    def __enter__(self) -> "FeatureMeanRecorder":
        mods = self.target.modules(self.hook)
        first = min(self.saes)
        for b in sorted(self.saes):
            self._handles.append(mods[b].register_forward_hook(self._make_hook(b, b == first)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def step_sigmas(self) -> List[float]:
        """Noise level of each recorded step, ascending step order (nan
        when no ``sigma_fn`` was given)."""
        steps = sorted({s for per in self._sum.values() for s in per})
        return [self.sigmas.get(s, float("nan")) for s in steps]

    def result(self) -> Dict[int, Tensor]:
        out = {}
        for b, per in self._sum.items():
            steps = sorted(per)
            out[b] = torch.stack([per[s] / max(1, self._cnt[b][s]) for s in steps])
        return out


def tfidf(mean_pos: Tensor, mean_neg: Tensor, eps: float = TFIDF_EPS) -> Tensor:
    """Eq. 12 elementwise (any shape, last axis = features)."""
    return mean_pos * torch.log(1 + 1 / (mean_neg + eps))


def score_tables(mean_pos: Tensor, mean_neg: Tensor, eps: float = TFIDF_EPS) -> Dict[str, Tensor]:
    """The reference score file content for one layer: per-step
    ``tfidf``, ``diff`` and ``mean_pos`` (``[n_steps, num_features]``)."""
    return {"tfidf": tfidf(mean_pos, mean_neg, eps), "diff": mean_pos - mean_neg,
            "mean_pos": mean_pos.clone()}


def pooled_scores(mean_pos: Tensor, mean_neg: Tensor, eps: float = TFIDF_EPS) -> Tensor:
    """Eq. 12 with ``mu_j(P)`` over every recorded step: ``[num_features]``
    (the single-vector form)."""
    return tfidf(mean_pos.mean(0), mean_neg.mean(0), eps)


def top_features(scores: Tensor, k: int) -> Union[List[int], Dict[int, List[int]]]:
    """Top-``k`` feature indices: a list for ``[F]`` scores, ``{step:
    list}`` for per-step ``[T, F]`` scores (the reference's
    ``argsort(descending)[:k]``)."""
    if scores.ndim == 1:
        return torch.argsort(scores, descending=True)[: int(k)].tolist()
    return {t: torch.argsort(scores[t], descending=True)[: int(k)].tolist()
            for t in range(scores.shape[0])}


def sae_vector(W_dec: Tensor, features: Union[Sequence[int], Mapping[int, Sequence[int]]]) -> Tensor:
    """Eq. 13 with unit weights: ``[d_in]`` for a feature list, ``[T,
    d_in]`` for per-step lists. ``W_dec`` is ``[num_latents, d_in]``."""
    W = W_dec.detach().float().cpu()
    if isinstance(features, Mapping):
        return torch.stack([W[list(features[t])].sum(0) for t in sorted(features)])
    return W[list(features)].sum(0)


def concept_vectors(
    saes: Mapping[int, Sae],
    mean_pos: Mapping[int, Tensor],
    mean_neg: Mapping[int, Tensor],
    k_per_block: Mapping[int, int],
    *,
    per_step: bool = True,
    eps: float = TFIDF_EPS,
) -> Dict[int, Tensor]:
    """``{block: v}`` for one concept: per-step ``[T, d]`` (reference) or
    pooled ``[d]`` (paper Eq. 13)."""
    out = {}
    for b, sae in saes.items():
        if per_step:
            feats = top_features(tfidf(mean_pos[b], mean_neg[b], eps), k_per_block[b])
        else:
            feats = top_features(pooled_scores(mean_pos[b], mean_neg[b], eps), k_per_block[b])
        out[int(b)] = sae_vector(sae.W_dec, feats)
    return out


def stack_blocks(vectors: Mapping[int, Tensor], blocks: Sequence[int]) -> Tensor:
    """Pack layout ``[len(blocks), n_steps, d]`` (a pooled ``[d]`` vector
    becomes one step)."""
    rows = []
    for b in blocks:
        v = vectors[int(b)].float()
        rows.append(v.unsqueeze(0) if v.ndim == 1 else v)
    return torch.stack(rows)


def select_k(
    blocks: Sequence[int],
    evaluate: Callable[[Dict[int, int]], float],
    grid: Sequence[int] = K_GRID,
) -> Tuple[Dict[int, int], Dict[Tuple[int, ...], float]]:
    """Choose ``k_c`` per block on held-out prompts: ``evaluate({block:
    k})`` returns the downstream steering metric (TADA: alignment AUC
    averaged over both directions); every combination of ``grid`` over the
    blocks is tried (their Fig. 13). Returns the best assignment and the
    whole table."""
    table: Dict[Tuple[int, ...], float] = {}
    best, best_val = None, float("-inf")
    for combo in itertools.product(grid, repeat=len(blocks)):
        ks = {int(b): int(k) for b, k in zip(blocks, combo)}
        val = float(evaluate(ks))
        table[tuple(combo)] = val
        if val > best_val:
            best, best_val = ks, val
    return best, table


__all__ = [
    "FeatureMeanRecorder", "K_GRID", "TFIDF_EPS", "concept_vectors", "pooled_scores",
    "sae_vector", "score_tables", "select_k", "stack_blocks", "tfidf", "top_features",
]
