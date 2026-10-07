"""AUSteer, as TADA adapts it (arXiv 2602.11910 Sec. 3, App. I.1.4, Eq. 7-9).

AUSteer (Feng et al.) replaces CAA's dense mean difference with a sparse,
sign-consistency selection of dimensions. It uses the same paired
cross-attention activations as CAA (conditional pass, per denoise step and
layer), but every time frame of every pair is a sample. Reference:
``compute_sv_austeer.compute_austeer_scores`` and
``AUSteerSteeringController`` in github.com/luk-st/steer-audio.

For dimension ``j`` at step ``t`` and layer ``l``, with momentum
``m_j = h_pos[j] - h_neg[j]`` over the ``S`` pooled (pair, frame) samples::

    r+_j = #(m_j > 0) / S,   r-_j = #(m_j < 0) / S
    beta_j = +max(r+_j, r-_j)  if r+_j >= r-_j  else  -max(r+_j, r-_j)

``beta`` lies in [-1, 1]; its magnitude is the directional reliability of
the dimension. (The paper writes ``sign(r+ - r-)``; the reference code
sends a tie to ``+``, and this port follows the code.) At each step a
global top-``s`` over ``|beta|`` across every active layer keeps ``s``
dimensions and zeroes the rest, so layers whose dimensions never make the
cut are not steered. The steering vector is that sparse ``beta`` (not
normalised) and is applied additively, as in CAA (Eq. 9)::

    h_l <- h_l + alpha * v_{t,l}

on the conditional pass, with ``renorm`` in the released eval configs.
The original multiplicative form (Eq. 8) gave no gain in the paper and is
not ported. The reference default budget is ``s = 256``; the released
per-concept budgets are in ``concepts.austeer_eval_config(concept)
["method-kwargs"]["k"]``.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch

from .caa import RunActivations

#: The reference controller's default top-s budget.
DEFAULT_TOP_S = 256


def austeer_scores(
    pos_runs: Sequence[RunActivations],
    neg_runs: Sequence[RunActivations],
    *,
    call_index: int = 0,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """Signed sign-consistency scores ``beta`` per step and block.

    ``pos_runs[i]`` / ``neg_runs[i]`` are the recorder stores (recorded with
    ``reduce=frames``) of the i-th positive / negative generation; each
    entry is ``[frames, hidden]``. Returns ``{step: {block: [hidden]}}``.
    """
    if len(pos_runs) != len(neg_runs) or not pos_runs:
        raise ValueError(
            f"need the same nonzero number of positive and negative runs, "
            f"got {len(pos_runs)} and {len(neg_runs)}"
        )
    out: Dict[int, Dict[int, torch.Tensor]] = {}
    for step in sorted(pos_runs[0].keys()):
        out[step] = {}
        for block in sorted(pos_runs[0][step].keys()):
            m = torch.cat([
                p[step][block][call_index].float().reshape(-1, p[step][block][call_index].shape[-1])
                - n[step][block][call_index].float().reshape(-1, n[step][block][call_index].shape[-1])
                for p, n in zip(pos_runs, neg_runs)
            ])
            S = m.shape[0]
            r_pos = (m > 0).sum(dim=0).float() / S
            r_neg = (m < 0).sum(dim=0).float() / S
            score = torch.maximum(r_pos, r_neg)
            out[step][block] = torch.where(r_pos >= r_neg, score, -score)
    return out


def select_top_s(
    betas: Mapping[int, Mapping[int, torch.Tensor]],
    s: int = DEFAULT_TOP_S,
    blocks: Optional[Sequence[int]] = None,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """Global top-``s`` per step across ``blocks`` (default: every block
    scored), zeroing ``beta`` outside it. Ties keep block order, then
    dimension order (the reference's stable sort).

    Returns ``{step: {block: [hidden]}}``, the sparse steering vectors; pass
    to :func:`acestep.tada.caa.stack_vectors` for the pack layout.
    """
    out: Dict[int, Dict[int, torch.Tensor]] = {}
    for step, per_block in betas.items():
        bl = sorted(per_block.keys()) if blocks is None else [int(b) for b in blocks]
        flat = torch.cat([per_block[b].float().reshape(-1) for b in bl])
        k = min(int(s), flat.numel())
        order = torch.sort(flat.abs(), descending=True, stable=True).indices[:k]
        mask = torch.zeros_like(flat)
        mask[order] = 1.0
        sparse = flat * mask
        out[step] = {}
        off = 0
        for b in bl:
            n = per_block[b].numel()
            out[step][b] = sparse[off:off + n].clone()
            off += n
    return out


def austeer_vectors(
    pos_runs: Sequence[RunActivations],
    neg_runs: Sequence[RunActivations],
    *,
    s: int = DEFAULT_TOP_S,
    blocks: Optional[Sequence[int]] = None,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """:func:`austeer_scores` then :func:`select_top_s`."""
    return select_top_s(austeer_scores(pos_runs, neg_runs), s, blocks)
