"""Steering vector packs: one file per vector, any model family.

A pack is a single ``.safetensors`` file holding one tensor, ``vector``
(float32, unit norm, ``[hidden_size]``), and one metadata entry,
``steering_pack``, a JSON object describing where and how it applies::

    {
      "format": 1,
      "family": "sa3",                      # FamilySpec name
      "checkpoint": "medium",               # the booted checkpoint id
      "hook": "post_block_residual",
      "block": 7,                           # index into the steering layout
      "hidden_size": 1536,
      "name": "bright",                     # knob = steer_<name>
      "label": "Bright",
      "blurb": "positive brightens (spectral centroid up)",
      "method": "caa_diff_means",
      "norm": 41.2,                         # L2 of the raw mean difference
      "magnitude": 4.1,                     # shift per knob unit
      "policy": {"kind": "range", "start": 0.0, "end": 1.0},
      "provenance": {...}                   # prompts, pairs, steps, date...
    }

The effective shift at a step is ``knob * magnitude * policy_weight *
vector`` added to block ``block``'s output residual, the same additive
post-block convention the ACE decoder engine uses. The vector method is
contrastive activation addition (difference of means over paired
prompts), with the block chosen by activation patching, following TADA
(Staniszewski et al., arXiv 2602.11910). ``scripts/steering/discover.py``
writes packs.

The loader is additive to the ACE built-in axes (``policy.AUTO_AXES``),
which keep their own on-disk format: a pack whose knob name collides with
a built-in knob is skipped with a warning.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Sequence

from .layout import HOOK_POST_BLOCK_RESIDUAL

if TYPE_CHECKING:
    import torch

PACK_FORMAT = 1
PACK_SUFFIX = ".safetensors"
PACK_METADATA_KEY = "steering_pack"
PACK_KNOB_PREFIX = "steer_"
METHOD_CAA = "caa_diff_means"

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def policy_weights(policy: Mapping | None, n: int) -> tuple:
    """Per-step weight curve of length ``n`` for a pack policy.

    Kinds:

    * ``{"kind": "range", "start": a, "end": b}``: weight 1 on steps whose
      schedule position ``i / n`` lies in ``[a, b)``, else 0. Expressed as
      fractions so a pack built at one step count transfers to another
      (the ACE fractional-step idea).
    * ``{"kind": "step", "step": s, "of": N}``: one-hot at
      ``round(s / N * n)`` (ACE's ``fractional_inject_step``).
    * ``{"kind": "curve", "weights": [...]}``: explicit weights, resampled
      to ``n`` by nearest position.

    A missing policy means every step at weight 1.
    """
    n = max(1, int(n))
    if not policy:
        return tuple(1.0 for _ in range(n))
    kind = policy.get("kind", "range")
    if kind == "range":
        a = float(policy.get("start", 0.0))
        b = float(policy.get("end", 1.0))
        return tuple(1.0 if a <= i / n < b else 0.0 for i in range(n))
    if kind == "step":
        from .policy import fractional_inject_step

        s = fractional_inject_step(
            int(policy.get("step", 0)), n, probe_n=int(policy.get("of", n)),
        )
        return tuple(1.0 if i == s else 0.0 for i in range(n))
    if kind == "curve":
        w = [float(x) for x in policy.get("weights", ())]
        if not w:
            return tuple(0.0 for _ in range(n))
        m = len(w)
        return tuple(w[min(m - 1, int(i * m / n))] for i in range(n))
    raise ValueError(f"unknown steering policy kind {kind!r}")


@dataclass
class SteeringPack:
    """One steering vector plus everything needed to apply it."""

    family: str
    checkpoint: str
    block: int
    hidden_size: int
    name: str
    vector: "torch.Tensor"
    label: str = ""
    blurb: str = ""
    hook: str = HOOK_POST_BLOCK_RESIDUAL
    method: str = METHOD_CAA
    norm: float = 0.0
    magnitude: float = 1.0
    policy: dict = field(default_factory=lambda: {"kind": "range", "start": 0.0, "end": 1.0})
    provenance: dict = field(default_factory=dict)
    path: Optional[Path] = None

    @property
    def knob_name(self) -> str:
        return f"{PACK_KNOB_PREFIX}{self.name}"

    def metadata(self) -> dict:
        meta = asdict(self)
        meta.pop("vector")
        meta.pop("path")
        meta["format"] = PACK_FORMAT
        return meta

    def validate(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"pack name {self.name!r} must match {_NAME_RE.pattern} "
                "(it becomes the knob steer_<name>)"
            )
        if self.vector.ndim != 1 or int(self.vector.shape[0]) != int(self.hidden_size):
            raise ValueError(
                f"pack {self.name!r}: vector shape {tuple(self.vector.shape)} "
                f"!= [hidden_size={self.hidden_size}]"
            )
        if int(self.block) < 0:
            raise ValueError(f"pack {self.name!r}: negative block {self.block}")
        if not self.family or not self.checkpoint:
            raise ValueError(f"pack {self.name!r}: family and checkpoint are required")
        policy_weights(self.policy, 8)  # raises on an unknown kind


def save_pack(pack: SteeringPack, path: Path | str) -> Path:
    """Write ``pack`` to ``path`` (``.safetensors``); returns the path."""
    import torch
    from safetensors.torch import save_file

    pack.validate()
    path = Path(path)
    if path.suffix != PACK_SUFFIX:
        path = path.with_suffix(PACK_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    vec = pack.vector.detach().to(device="cpu", dtype=torch.float32).contiguous()
    save_file(
        {"vector": vec}, str(path),
        metadata={PACK_METADATA_KEY: json.dumps(pack.metadata(), sort_keys=True)},
    )
    return path


def load_pack(path: Path | str) -> SteeringPack:
    """Read one pack file. Raises ValueError on a malformed pack."""
    import torch
    from safetensors import safe_open

    path = Path(path)
    with safe_open(str(path), framework="pt") as f:
        meta_raw = (f.metadata() or {}).get(PACK_METADATA_KEY)
        if meta_raw is None:
            raise ValueError(f"{path.name}: no {PACK_METADATA_KEY!r} metadata")
        vec = f.get_tensor("vector").to(torch.float32)
    meta = json.loads(meta_raw)
    fmt = int(meta.pop("format", 0))
    if fmt != PACK_FORMAT:
        raise ValueError(f"{path.name}: pack format {fmt} != {PACK_FORMAT}")
    known = {
        "family", "checkpoint", "block", "hidden_size", "name", "label",
        "blurb", "hook", "method", "norm", "magnitude", "policy", "provenance",
    }
    pack = SteeringPack(
        vector=vec, path=path, **{k: v for k, v in meta.items() if k in known},
    )
    pack.validate()
    return pack


def discover_packs(
    directory: Path | str | None,
    *,
    family: str,
    checkpoint: str,
    layout=None,
    reserved_names: Iterable[str] = (),
) -> list:
    """Every valid pack under ``directory`` (recursive) for this boot.

    Keeps packs whose ``family`` and ``checkpoint`` match, and, when a
    :class:`~acestep.steering.layout.SteeringLayout` is given, whose
    hook/block/hidden size fit it. Knob names in ``reserved_names``
    (built-in steering knobs) and duplicate names are skipped with a
    warning; the first file in sorted path order wins a duplicate.
    """
    if directory is None:
        return []
    root = Path(directory)
    if not root.is_dir():
        return []
    from loguru import logger

    reserved = set(reserved_names)
    out: list = []
    seen: set = set()
    for path in sorted(root.rglob(f"*{PACK_SUFFIX}")):
        try:
            pack = load_pack(path)
        except Exception as exc:
            logger.warning("steering_pack_skipped path={} reason={}", path, exc)
            continue
        if pack.family != family or pack.checkpoint != checkpoint:
            continue
        if layout is not None and not layout.accepts(
            pack.block, pack.hidden_size, pack.hook,
        ):
            logger.warning(
                "steering_pack_skipped path={} reason=layout_mismatch "
                "block={} hidden={} hook={} layout={}",
                path, pack.block, pack.hidden_size, pack.hook, layout,
            )
            continue
        if pack.knob_name in reserved or pack.knob_name in seen:
            logger.warning(
                "steering_pack_skipped path={} reason=duplicate_knob knob={}",
                path, pack.knob_name,
            )
            continue
        seen.add(pack.knob_name)
        out.append(pack)
    return out


class PackSteering:
    """Per-session pack surface: knob specs and the knob -> config map.

    Mirrors :class:`~acestep.steering.controller.SteeringController`'s
    ``snapshot_key`` / ``build_configs`` contract so a backend can merge
    both into the pipeline's single steering slot.
    """

    def __init__(self, packs: Sequence[SteeringPack] = ()):
        self.packs: tuple = tuple(packs)

    @property
    def is_loaded(self) -> bool:
        return bool(self.packs)

    def knob_specs(self) -> list:
        from acestep.streaming.knobs import steering_pack_spec

        return [
            steering_pack_spec(
                p.knob_name,
                label=p.label or p.name,
                block=p.block,
                policy=p.policy,
                blurb=p.blurb,
            )
            for p in self.packs
        ]

    def snapshot_key(self, raw: Mapping[str, float], n: int) -> tuple:
        return tuple(float(raw.get(p.knob_name, 0.0)) for p in self.packs) + (
            float(max(1, int(n))),
        )

    def build_configs(self, raw: Mapping[str, float], n: int) -> list:
        n = max(1, int(n))
        configs: list = []
        for p in self.packs:
            alpha = float(raw.get(p.knob_name, 0.0))
            if alpha == 0.0:
                continue
            configs.append({
                "layer": int(p.block),
                "step": -1,
                "weights": policy_weights(p.policy, n),
                "vector": p.vector,
                "magnitude": float(p.magnitude),
                "alpha": alpha,
            })
        return configs


def packs_available(*, family: str, checkpoint: str) -> bool:
    """Whether any pack in the configured directory targets this boot
    (family + checkpoint only; layout filtering needs the loaded model).
    Engine selection reads this before the model forward exists."""
    from acestep.paths import steering_packs_dir

    try:
        return bool(discover_packs(
            steering_packs_dir(), family=family, checkpoint=checkpoint,
        ))
    except Exception:
        return False


def load_session_packs(
    *, family: str, checkpoint: str, layout=None, reserved_names: Iterable[str] = (),
) -> PackSteering:
    """The pack surface for one session, from the configured directory."""
    from acestep.paths import steering_packs_dir

    return PackSteering(discover_packs(
        steering_packs_dir(), family=family, checkpoint=checkpoint,
        layout=layout, reserved_names=reserved_names,
    ))
