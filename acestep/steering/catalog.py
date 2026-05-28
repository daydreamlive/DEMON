"""Vector catalog: walk the on-disk probe dir into stable indexed entries.

Split into a torch-free enumerator (filename metadata) and a
torch-paying loader. Stems look like ``brightness_l09_t3.pt`` — the
zero-padded layer lets alphabetical sort yield the intended
(l03..l18) x (t0, t3, ...) order naturally.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .policy import MANUAL_CATALOG_AXES
from .types import AutoAxis, CatalogEntry

if TYPE_CHECKING:
    import torch


def _parse_stem(stem: str, axis: str) -> tuple[int, int] | None:
    """Extract (layer, step) from ``brightness_l09_t3`` style stems.

    Returns ``None`` if the stem doesn't parse against ``axis``.
    """
    prefix = f"{axis}_l"
    if not stem.startswith(prefix):
        return None
    tail = stem[len(axis) + 1:]   # "l09_t3"
    try:
        layer = int(tail.split("_")[0][1:])
        step = int(tail.split("_")[1][1:])
    except (IndexError, ValueError):
        return None
    return layer, step


def enumerate_catalog(vector_dir: Path) -> tuple[CatalogEntry, ...]:
    """List every (axis, build_layer, build_step) cell on disk.

    Torch-free; reads filenames only. Empty tuple when the dir is
    missing.
    """
    if not vector_dir.exists():
        return ()
    per_axis: dict[str, list[Path]] = {a: [] for a in MANUAL_CATALOG_AXES}
    for path in sorted(vector_dir.glob("*.pt")):
        for axis in MANUAL_CATALOG_AXES:
            if path.stem.startswith(f"{axis}_l"):
                per_axis[axis].append(path)
                break
    out: list[CatalogEntry] = []
    for axis in MANUAL_CATALOG_AXES:
        for path in per_axis[axis]:
            parsed = _parse_stem(path.stem, axis)
            if parsed is None:
                continue
            layer, step = parsed
            out.append(CatalogEntry(
                index=len(out),
                axis=axis,
                build_layer=layer,
                build_step=step,
                filename=path.name,
            ))
    return tuple(out)


def load_vector(
    vector_dir: Path,
    entry: CatalogEntry,
) -> tuple["torch.Tensor", float]:
    """Load one catalog entry's (vector, magnitude) as CPU float32."""
    import torch as _t

    path = vector_dir / entry.filename
    blob = _t.load(path, map_location="cpu", weights_only=False)
    return blob["vector"].to(_t.float32), float(blob["magnitude"])


def load_auto_vectors(
    vector_dir: Path,
    auto_axes: tuple[AutoAxis, ...],
) -> dict[str, dict]:
    """Load the per-axis vector for each verified auto-path axis.

    Returns ``{axis.name: {"layer": int, "vector": Tensor,
    "magnitude": float}}``. Missing files are silently skipped — the
    knob for that axis just never produces a config.
    """
    import torch as _t

    out: dict[str, dict] = {}
    if not vector_dir.exists():
        return out
    for ax in auto_axes:
        path = vector_dir / f"{ax.axis}_l{ax.probe_layer:02d}_t{ax.probe_step}.pt"
        if not path.exists():
            continue
        try:
            blob = _t.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        out[ax.name] = {
            "layer": ax.probe_layer,
            "vector": blob["vector"].to(_t.float32),
            "magnitude": float(blob["magnitude"]),
        }
    return out
