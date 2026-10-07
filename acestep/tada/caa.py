"""Contrastive activation addition, as TADA applies it (arXiv 2602.11910).

Vector construction (paper Eq. 6, App. I.1.4; reference
``compute_standard_steering_vectors``): for each denoise step ``t`` and
cross-attention layer ``l``, run ``N`` positive and ``N`` negative prompts
(paired, same seed), average each run's cross-attention output over batch
and time frames, and take::

    v_{t,l} = mean_i(h_pos[i]) - mean_i(h_neg[i]),   v <- v / ||v||

Only the conditional CFG pass is recorded. Application (paper Sec. 3 and
App. I.2; reference ``VectorStore.forward`` in ``cond_only`` mode)::

    h_l <- h_l + alpha * v_{t,l}            (broadcast over time frames)

on the conditional pass only, at the localised layers. The released
evaluation configs also set ``renorm``: each time frame is rescaled back
to its pre-steering L2 norm after the add. Negative ``alpha`` removes the
concept. Reference settings for ACE-Step: 50 prompt pairs per concept,
seed 10, 30 steps, guidance 5.0, 30 s; localised layers (6, 7).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import torch

#: ``{step: {block: [vec, ...]}}``, one dict per generation (recorder output).
RunActivations = Mapping[int, Mapping[int, Sequence[torch.Tensor]]]


def caa_vectors(
    pos_runs: Sequence[RunActivations],
    neg_runs: Sequence[RunActivations],
    *,
    normalize: bool = True,
    call_index: int = 0,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """Per-step, per-block CAA vectors from paired recordings.

    ``pos_runs[i]`` / ``neg_runs[i]`` are the recorder stores of the i-th
    positive / negative generation. ``call_index`` picks which recorded
    call of a block at a step is used (the reference takes the first).
    Returns ``{step: {block: [hidden] float32}}``; unit norm when
    ``normalize`` (zero vectors stay zero).
    """
    if len(pos_runs) != len(neg_runs) or not pos_runs:
        raise ValueError(
            f"need the same nonzero number of positive and negative runs, "
            f"got {len(pos_runs)} and {len(neg_runs)}"
        )
    steps = sorted(pos_runs[0].keys())
    out: Dict[int, Dict[int, torch.Tensor]] = {}
    for step in steps:
        out[step] = {}
        for block in sorted(pos_runs[0][step].keys()):
            pos = torch.stack([r[step][block][call_index].float() for r in pos_runs])
            neg = torch.stack([r[step][block][call_index].float() for r in neg_runs])
            v = pos.mean(dim=0) - neg.mean(dim=0)
            if normalize:
                n = torch.linalg.vector_norm(v)
                if n > 0:
                    v = v / n
            out[step][block] = v
    return out


def stack_vectors(
    vectors: Mapping[int, Mapping[int, torch.Tensor]],
    blocks: Sequence[int],
    steps: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """``[len(blocks), n_steps, hidden]`` tensor (pack layout) from
    :func:`caa_vectors` output."""
    steps = sorted(vectors.keys()) if steps is None else list(steps)
    return torch.stack([
        torch.stack([vectors[s][int(b)].float() for s in steps]) for b in blocks
    ])


def steer_activation(
    h: torch.Tensor, v: torch.Tensor, alpha: float, *, renorm: bool = False,
) -> torch.Tensor:
    """The reference CAA intervention on one cross-attention output.

    ``h`` is ``[..., hidden]``; ``v`` is ``[hidden]``. With ``alpha == 0``
    ``h`` is returned unchanged. ``renorm`` restores each frame's L2 norm.
    """
    if alpha == 0:
        return h
    norm = torch.linalg.vector_norm(h, dim=-1, keepdim=True)
    out = h + float(alpha) * v.to(device=h.device, dtype=h.dtype)
    if renorm:
        out = out / torch.linalg.vector_norm(out, dim=-1, keepdim=True) * norm
    return out


def combine(
    vectors: Sequence[torch.Tensor], negate: Optional[Sequence[bool]] = None,
) -> torch.Tensor:
    """Multi-concept vector (paper Sec. 5.5, App. O): the unit-weight sum
    of single-concept vectors of one method (CAA, AUSteer or SAE), each
    sign-flipped where ``negate`` is True (a concept-suppressing
    direction, e.g. male vocal from the female-vocal vector). Vectors must
    share a shape, e.g. the pack layout ``[blocks, steps, hidden]``."""
    if not vectors:
        raise ValueError("need at least one vector")
    negate = [False] * len(vectors) if negate is None else [bool(n) for n in negate]
    if len(negate) != len(vectors):
        raise ValueError(f"negate has {len(negate)} entries for {len(vectors)} vectors")
    shapes = {tuple(v.shape) for v in vectors}
    if len(shapes) != 1:
        raise ValueError(f"vectors differ in shape: {sorted(shapes)}")
    out = torch.zeros_like(vectors[0], dtype=torch.float32)
    for v, n in zip(vectors, negate):
        out = out - v.float() if n else out + v.float()
    return out


def alphas_from_range(min_range: float, max_range: float, steps_per_side: int = 15) -> List[float]:
    """The reference symmetric strength grid: ``steps_per_side`` negative
    values, 0, ``steps_per_side`` positive values, ``alpha_k = range * k /
    steps_per_side`` rounded to 4 decimals (``run_eval._alphas_from_range``).
    ``min_range == 0`` gives the one-sided grid."""
    neg = [] if min_range == 0 else [
        round(min_range * i / steps_per_side, 4) for i in range(steps_per_side, 0, -1)
    ]
    pos = [round(max_range * i / steps_per_side, 4) for i in range(1, steps_per_side + 1)]
    return neg + [0.0] + pos
