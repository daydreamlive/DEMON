"""TADA CAA vectors as steering packs (one ``steer_<concept>`` knob each).

A TADA pack is a format-2 :class:`~acestep.steering.packs.SteeringPack`:
``vector`` is ``[len(blocks), n_steps, hidden]`` (one unit vector per
localised block and denoise step), ``hook`` is ``cross_attn_output``,
``cond_only`` is set (the conditional CFG pass only) and ``renorm``
follows the reference evaluation configs (True). The knob value times
``magnitude`` is TADA's ``alpha``. The pipeline's steering slot maps the
pack's steps onto the live schedule by position (step ``i`` of ``n`` uses
row ``int(i * n_steps / n)``).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

from acestep.steering.layout import HOOK_CROSS_ATTN_OUTPUT
from acestep.steering.packs import SteeringPack, load_pack, save_pack

METHOD_TADA_CAA = "tada_caa"
CITATION = (
    "Staniszewski, Zaleska, Modrzejewski, Deja. TADA! Tuning Audio Diffusion "
    "Models through Activation Steering. arXiv 2602.11910"
)


def caa_pack(
    vectors: torch.Tensor,
    *,
    family: str,
    checkpoint: str,
    concept: str,
    blocks: Sequence[int],
    label: str = "",
    blurb: str = "",
    magnitude: float = 1.0,
    renorm: bool = True,
    cond_only: bool = True,
    hook: str = HOOK_CROSS_ATTN_OUTPUT,
    policy: Optional[Mapping] = None,
    provenance: Optional[Mapping] = None,
) -> SteeringPack:
    """Build a TADA pack from ``[len(blocks), n_steps, hidden]`` vectors."""
    if vectors.ndim != 3 or vectors.shape[0] != len(blocks):
        raise ValueError(
            f"vectors must be [len(blocks)={len(blocks)}, n_steps, hidden], "
            f"got {tuple(vectors.shape)}"
        )
    prov = {
        "method": "contrastive activation addition, per step and layer, unit norm",
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
        hook=hook,
        method=METHOD_TADA_CAA,
        norm=1.0,
        magnitude=float(magnitude),
        policy=dict(policy) if policy is not None else {"kind": "range", "start": 0.0, "end": 1.0},
        provenance=prov,
        cond_only=bool(cond_only),
        renorm=bool(renorm),
    )
    pack.validate()
    return pack


def write_caa_pack(pack: SteeringPack, directory: Path | str) -> Path:
    """Save under ``<directory>/<family>/<checkpoint>/<name>.safetensors``."""
    root = Path(directory) / pack.family / pack.checkpoint
    return save_pack(pack, root / f"{pack.name}.safetensors")


__all__ = ["CITATION", "METHOD_TADA_CAA", "caa_pack", "load_pack", "write_caa_pack"]
