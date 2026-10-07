"""Layer localisation by activation patching (TADA, arXiv 2602.11910, Sec. 3-4).

For a concept ``c`` and counterfactual prompt pairs ``(P_c, P_~c)``, each
cross-attention layer ``l`` is scored by generating with the corrupted
prompt everywhere except layer ``l``, which sees the clean prompt's keys
and values, and measuring how much of the concept comes back (Eq. 2)::

    I(l, c) = (s(l <- c, rest <- ~c) - s(all <- ~c))
              / (s(all <- c) - s(all <- ~c))

``s`` is the mean audio-text similarity between the generations and the
concept text (MuQ-MuLan for mood, tempo, instruments and genres; CLAP for
vocal gender). The two references are the unpatched corrupted run (no
layer patched) and the fully patched run (every layer sees the clean
prompt), so one concept costs ``L + 2`` generation sets (Eq. 3). Layer
impacts are averaged over concepts, ``I(l) = mean_c I(l, c)``, and the
functional layers are ``{l : I(l) >= tau}`` with ``tau = 0.10``.

The published per-concept impacts (reference repo
``results/localization_impact.csv``) are floored at 0 and not capped at
1 (its maximum is 1.3125); :func:`impact_score` does the same.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .target import HOOK_CROSS_ATTN_COND, ActivationTarget, ConditioningPatcher

#: Paper's selection threshold (Sec. 4; Table 7 sensitivity).
TAU = 0.10

NONE = "none"
ALL = "all"


def impact_score(s_layer: float, s_none: float, s_all: float) -> float:
    """Eq. 2 for one layer and concept, floored at 0. ``nan`` when the
    clean and corrupted references score the same (no signal)."""
    denom = float(s_all) - float(s_none)
    if denom == 0.0 or math.isnan(denom):
        return float("nan")
    return max(0.0, (float(s_layer) - float(s_none)) / denom)


def layer_impacts(scores: Mapping[object, float], num_blocks: int) -> List[float]:
    """``I(l, c)`` for every block from one concept's sweep scores.

    ``scores`` maps ``"none"``, ``"all"`` and each block index to the
    mean similarity of that run set.
    """
    s_none, s_all = scores[NONE], scores[ALL]
    return [impact_score(scores[l], s_none, s_all) for l in range(int(num_blocks))]


def aggregate_impacts(per_concept: Mapping[str, Sequence[float]]) -> List[float]:
    """``I(l)``: mean of ``I(l, c)`` over concepts (nan concepts skipped)."""
    rows = list(per_concept.values())
    if not rows:
        return []
    n = len(rows[0])
    out = []
    for l in range(n):
        vals = [float(r[l]) for r in rows if not math.isnan(float(r[l]))]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def select_layers(impacts: Sequence[float], tau: float = TAU) -> List[int]:
    """Functional layers: indices with ``I(l) >= tau``."""
    return [l for l, v in enumerate(impacts) if not math.isnan(v) and v >= tau]


def run_sets(num_blocks: int) -> List[Tuple[object, Tuple[int, ...]]]:
    """The ``L + 2`` patch sets of one concept sweep, as ``(key, blocks)``:
    ``none`` (corrupted reference), ``all`` (clean reference), then each
    single block."""
    blocks = tuple(range(int(num_blocks)))
    return [(NONE, ()), (ALL, blocks)] + [(l, (l,)) for l in blocks]


def run_patched(
    target: ActivationTarget,
    blocks: Sequence[int],
    run_clean: Callable[[], object],
    run_corrupted: Callable[[], object],
) -> Tuple[object, object]:
    """One patched generation: record ``blocks``' conditioning during the
    clean run, then substitute it during the corrupted run (same noise and
    settings are the caller's contract). Returns ``(clean, patched)``.
    With no blocks this is just the two plain runs."""
    patcher = ConditioningPatcher(target, list(blocks), hook=HOOK_CROSS_ATTN_COND)
    with patcher.record():
        clean = run_clean()
    with patcher.patch():
        patched = run_corrupted()
    if patcher.leftover:
        raise RuntimeError(
            f"patched run made fewer cross-attention calls than the clean run: "
            f"{patcher.leftover}"
        )
    return clean, patched


@dataclass
class SweepResult:
    """One concept's sweep: mean similarity per run set and ``I(l, c)``."""

    concept: str
    scores: Dict[object, float]
    impacts: List[float]
    n_samples: int = 0
    extra: Dict[str, object] = field(default_factory=dict)


def patching_sweep(
    concept: str,
    num_blocks: int,
    generate: Callable[[Tuple[int, ...]], Sequence[object]],
    similarity: Callable[[Sequence[object]], float],
    blocks: Optional[Sequence[int]] = None,
) -> SweepResult:
    """Run the ``L + 2`` patch sets for one concept and score them.

    ``generate(patched_blocks)`` returns the corrupted-prompt generations
    of every (prompt pair, seed) with ``patched_blocks`` seeing the clean
    prompt (``()`` = plain corrupted run; all blocks = the clean
    reference). ``similarity(outputs)`` returns their mean similarity to
    the concept text. ``blocks`` restricts the single-layer sets (the
    references always run); unswept blocks score nan.
    """
    sets = run_sets(num_blocks)
    if blocks is not None:
        keep = set(int(b) for b in blocks)
        sets = [(k, b) for k, b in sets if k in (NONE, ALL) or k in keep]
    scores: Dict[object, float] = {}
    n = 0
    for key, patch_blocks in sets:
        outs = generate(patch_blocks)
        n = max(n, len(outs))
        scores[key] = float(similarity(outs))
    full = {k: scores.get(k, float("nan")) for k in [NONE, ALL] + list(range(num_blocks))}
    return SweepResult(
        concept=concept, scores=full,
        impacts=layer_impacts(full, num_blocks), n_samples=n,
    )


def localize(
    sweeps: Sequence[SweepResult], tau: float = TAU,
) -> Tuple[List[float], List[int]]:
    """``(I(l), functional layers)`` from several concepts' sweeps."""
    agg = aggregate_impacts({s.concept: s.impacts for s in sweeps})
    return agg, select_layers(agg, tau)
