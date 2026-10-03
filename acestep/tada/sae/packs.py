"""TADA SAE vectors as steering packs (one ``steer_<concept>`` knob each).

Same container as the CAA packs (:mod:`acestep.tada.packs`): hook
``cross_attn_output``, ``cond_only`` (TADA steers the conditional pass),
``renorm`` (the reference eval configs set it), ``blocks`` = the localised
layers, each with the vector built from its own SAE. ``vector`` is
``[len(blocks), n_steps, d]`` for the per-step form (the reference picks
features per denoise step) or ``[len(blocks), 1, d]`` for the paper's
single vector (Eq. 13). Unlike CAA vectors these are NOT unit norm: a sum
of ``k_c`` unit decoder rows, applied as ``alpha * v_SAE``. The knob value
times ``magnitude`` is TADA's ``alpha``.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

from acestep.steering.layout import HOOK_CROSS_ATTN_OUTPUT
from acestep.steering.packs import SteeringPack, save_pack
from acestep.tada.packs import CITATION

METHOD_TADA_SAE = "tada_sae"


def sae_pack(
    vectors: torch.Tensor,
    *,
    family: str,
    checkpoint: str,
    concept: str,
    blocks: Sequence[int],
    k_per_block: Mapping[int, int],
    label: str = "",
    blurb: str = "",
    magnitude: float = 1.0,
    renorm: bool = True,
    cond_only: bool = True,
    policy: Optional[Mapping] = None,
    provenance: Optional[Mapping] = None,
) -> SteeringPack:
    """Build a TADA SAE pack from ``[len(blocks), n_steps, d]`` vectors."""
    if vectors.ndim != 3 or vectors.shape[0] != len(blocks):
        raise ValueError(
            f"vectors must be [len(blocks)={len(blocks)}, n_steps, d], got {tuple(vectors.shape)}"
        )
    prov = {
        "method": "sum of top-k_c TF-IDF SAE decoder rows (Eq. 12-13), unit weights",
        "per_step": bool(vectors.shape[1] > 1),
        "k_c": {str(int(b)): int(k_per_block[int(b)]) for b in blocks},
        "citation": CITATION,
        "date": _dt.date.today().isoformat(),
    }
    prov.update(dict(provenance or {}))
    pack = SteeringPack(
        family=family,
        checkpoint=checkpoint,
        block=int(blocks[0]),
        blocks=[int(b) for b in blocks],
        hidden_size=int(vectors.shape[2]),
        name=concept,
        label=label or concept.replace("_", " ").title(),
        blurb=blurb,
        vector=vectors.detach().float().cpu().contiguous(),
        hook=HOOK_CROSS_ATTN_OUTPUT,
        method=METHOD_TADA_SAE,
        norm=float(torch.linalg.vector_norm(vectors.float(), dim=-1).mean()),
        magnitude=float(magnitude),
        policy=dict(policy) if policy is not None else {"kind": "range", "start": 0.0, "end": 1.0},
        provenance=prov,
        cond_only=bool(cond_only),
        renorm=bool(renorm),
    )
    pack.validate()
    return pack


def write_sae_pack(pack: SteeringPack, directory: Path | str) -> Path:
    """Save under ``<directory>/<family>/<checkpoint>/<name>.safetensors``."""
    root = Path(directory) / pack.family / pack.checkpoint
    return save_pack(pack, root / f"{pack.name}.safetensors")


__all__ = ["METHOD_TADA_SAE", "sae_pack", "write_sae_pack"]
