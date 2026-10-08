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

Format 2 (additive; every format-1 pack loads and applies unchanged)
adds optional header keys ``category``, ``applies_to``, ``seeds``,
``variant``, ``description``, ``pos_anchor`` / ``neg_anchor`` and
``norms``, and two optional tensor shapes:

* ``vectors`` ``[K, hidden]`` + ``blocks`` int64 ``[K]`` in place of
  ``vector``: row k lands on block ``blocks[k]``, all rows driven by the
  same knob value. A unit-norm row uses the pack ``magnitude`` (or
  ``norms[k] * knob_unit`` when the header carries per-block ``norms``);
  a non-unit row is a raw mean difference and is applied as its unit
  direction times its own norm times ``knob_unit`` (``magnitude / norm``,
  i.e. 0.1).
* ``vectors_neg`` ``[K, hidden]`` (or ``vector_neg`` ``[hidden]`` beside
  a legacy ``vector``): a negative knob value applies these rows at
  ``|knob|`` instead of negating the positive rows.
  A unit row's magnitude is ``provenance.dn_magnitude_own`` when the
  header carries it (variant c: the neg gain is calibrated in the dn
  vector's own magnitude units), else as for ``vectors``.

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
PACK_FORMAT_V2 = 2
SUPPORTED_PACK_FORMATS = (PACK_FORMAT, PACK_FORMAT_V2)
# Default knob unit (magnitude / norm) when a pack does not imply one.
DEFAULT_KNOB_UNIT = 0.1
# A format-2 row within this of unit L2 norm counts as a unit direction.
_UNIT_TOL = 1e-3
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


@dataclass(frozen=True)
class SteeringTerm:
    """One block's additive shift: the shared internal form of every pack
    shape. ``neg`` is None when a negative knob simply negates ``pos``."""

    block: int
    pos: "torch.Tensor"
    pos_magnitude: float
    neg: Optional["torch.Tensor"] = None
    neg_magnitude: float = 0.0


@dataclass
class SteeringPack:
    """One steering pack plus everything needed to apply it.

    ``vector`` / ``block`` are the format-1 single-block form (for a
    multi-block pack they mirror row 0, so single-block readers keep
    working). ``vectors`` / ``blocks`` / ``vectors_neg`` / ``vector_neg``
    are the format-2 additions; :meth:`terms` folds every shape into one
    tuple of :class:`SteeringTerm`.
    """

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
    # ---- format 2 (all optional) ----
    vectors: Optional["torch.Tensor"] = None       # [K, hidden]
    blocks: tuple = ()                             # K block indices
    vectors_neg: Optional["torch.Tensor"] = None   # [K, hidden]
    vector_neg: Optional["torch.Tensor"] = None    # [hidden], beside `vector`
    norms: tuple = ()                              # optional per-block raw norms
    category: str = ""
    applies_to: object = None
    seeds: object = None
    variant: str = ""
    description: str = ""
    pos_anchor: str = ""
    neg_anchor: str = ""
    # Ship quality bar per sign ({"pos": bool, "neg": bool}); set only by
    # a bundle (``load_bundle``). None = not measured (loose packs).
    bar_pass: Optional[dict] = None

    @property
    def knob_name(self) -> str:
        return f"{PACK_KNOB_PREFIX}{self.name}"

    @property
    def is_v2(self) -> bool:
        return (
            self.vectors is not None or self.vectors_neg is not None
            or self.vector_neg is not None
        )

    @property
    def all_blocks(self) -> tuple:
        if self.vectors is not None:
            return tuple(int(b) for b in self.blocks)
        return (int(self.block),)

    @property
    def effective_category(self) -> str:
        """Top-level ``category`` when set, else ``provenance.category``."""
        if self.category:
            return str(self.category)
        prov = self.provenance if isinstance(self.provenance, Mapping) else {}
        return str(prov.get("category") or "")

    def knob_unit(self) -> float:
        if self.norm and self.magnitude:
            return float(self.magnitude) / float(self.norm)
        prov = self.provenance if isinstance(self.provenance, Mapping) else {}
        try:
            return float(prov.get("knob_unit", DEFAULT_KNOB_UNIT))
        except (TypeError, ValueError):
            return DEFAULT_KNOB_UNIT

    def _row(self, row: "torch.Tensor", k: int) -> tuple:
        """(direction, magnitude) for one format-2 row."""
        n = float(row.norm())
        if n > 0.0 and abs(n - 1.0) > _UNIT_TOL:
            return row / n, n * self.knob_unit()
        if self.norms and k < len(self.norms):
            return row, float(self.norms[k]) * self.knob_unit()
        return row, float(self.magnitude)

    def _neg_row(self, row: "torch.Tensor", k: int) -> tuple:
        """(direction, magnitude) for one negative row.

        A variant-c pack's unit ``vectors_neg`` row is the dn pack's own
        direction, and its calibrated neg gain is in that dn vector's own
        magnitude units (``provenance.dn_magnitude_own``, a float or one
        per row). Without that key the row follows :meth:`_row`."""
        prov = self.provenance if isinstance(self.provenance, Mapping) else {}
        dn = prov.get("dn_magnitude_own")
        if dn is not None:
            if isinstance(dn, (list, tuple)):
                dn = dn[k] if k < len(dn) else None
            if dn is not None:
                n = float(row.norm())
                return (row / n if n > 0.0 else row), float(dn)
        return self._row(row, k)

    def terms(self) -> tuple:
        """Every block shift this pack applies, one per target block.

        A format-1 pack yields exactly ``(block, vector, magnitude)``
        with the stored tensor untouched, so its numerics are unchanged.
        """
        out = []
        if self.vectors is None:
            neg, neg_mag = None, 0.0
            if self.vector_neg is not None:
                neg, neg_mag = self._neg_row(self.vector_neg, 0)
            out.append(SteeringTerm(
                int(self.block), self.vector, float(self.magnitude), neg, neg_mag,
            ))
        else:
            for k, b in enumerate(self.blocks):
                pos, pos_mag = self._row(self.vectors[k], k)
                neg, neg_mag = None, 0.0
                if self.vectors_neg is not None:
                    neg, neg_mag = self._neg_row(self.vectors_neg[k], k)
                out.append(SteeringTerm(int(b), pos, pos_mag, neg, neg_mag))
        return tuple(out)

    def metadata(self) -> dict:
        meta = asdict(self)
        for k in ("vector", "path", "vectors", "blocks", "vectors_neg", "vector_neg",
                  "bar_pass"):
            meta.pop(k)
        # Format-2 header keys are written only when set, so a format-1
        # pack's header is unchanged.
        for k in _V2_META_KEYS:
            if meta.get(k) in (None, "", (), []):
                meta.pop(k, None)
        if "norms" in meta:
            meta["norms"] = [float(x) for x in meta["norms"]]
        meta["format"] = PACK_FORMAT_V2 if self.is_v2 else PACK_FORMAT
        return meta

    def validate(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"pack name {self.name!r} must match {_NAME_RE.pattern} "
                "(it becomes the knob steer_<name>)"
            )
        H = int(self.hidden_size)
        for nm in ("vector", "vector_neg"):
            v = getattr(self, nm)
            if v is None and nm == "vector_neg":
                continue
            if v is None or v.ndim != 1 or int(v.shape[0]) != H:
                raise ValueError(
                    f"pack {self.name!r}: {nm} shape "
                    f"{None if v is None else tuple(v.shape)} != [hidden_size={H}]"
                )
        if self.vectors is not None:
            K = len(self.blocks)
            if K == 0:
                raise ValueError(f"pack {self.name!r}: vectors without blocks")
            for nm in ("vectors", "vectors_neg"):
                v = getattr(self, nm)
                if v is not None and (v.ndim != 2 or tuple(v.shape) != (K, H)):
                    raise ValueError(
                        f"pack {self.name!r}: {nm} shape {tuple(v.shape)} "
                        f"!= [K={K}, hidden_size={H}]"
                    )
            if self.vector_neg is not None:
                raise ValueError(
                    f"pack {self.name!r}: vector_neg beside vectors (use vectors_neg)"
                )
            if self.norms and len(self.norms) != K:
                raise ValueError(f"pack {self.name!r}: norms length != K={K}")
        elif self.vectors_neg is not None:
            raise ValueError(f"pack {self.name!r}: vectors_neg without vectors")
        for b in self.all_blocks:
            if int(b) < 0:
                raise ValueError(f"pack {self.name!r}: negative block {b}")
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
    def f32(t):
        return t.detach().to(device="cpu", dtype=torch.float32).contiguous()

    if pack.vectors is not None:
        tensors = {
            "vectors": f32(pack.vectors),
            "blocks": torch.tensor([int(b) for b in pack.blocks], dtype=torch.int64),
        }
        if pack.vectors_neg is not None:
            tensors["vectors_neg"] = f32(pack.vectors_neg)
    else:
        tensors = {"vector": f32(pack.vector)}
        if pack.vector_neg is not None:
            tensors["vector_neg"] = f32(pack.vector_neg)
    save_file(
        tensors, str(path),
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
        t = {k: f.get_tensor(k) for k in set(f.keys()) & _TENSOR_KEYS}
    meta = json.loads(meta_raw)
    fmt = int(meta.pop("format", 0))
    if fmt not in SUPPORTED_PACK_FORMATS:
        raise ValueError(
            f"{path.name}: pack format {fmt} not in {SUPPORTED_PACK_FORMATS}"
        )
    kw = {k: v for k, v in meta.items() if k in _KNOWN_META}
    if "norms" in kw:
        kw["norms"] = tuple(float(x) for x in (kw["norms"] or ()))
    if "vectors" in t:
        if "blocks" not in t:
            raise ValueError(f"{path.name}: 'vectors' without 'blocks'")
        vectors = t["vectors"].to(torch.float32)
        blocks = tuple(int(b) for b in t["blocks"].reshape(-1).tolist())
        if vectors.ndim != 2 or int(vectors.shape[0]) != len(blocks) or not blocks:
            raise ValueError(f"{path.name}: vectors/blocks shape mismatch")
        kw.setdefault("block", blocks[0])
        kw.setdefault("hidden_size", int(vectors.shape[1]))
        kw["vector"] = vectors[0]
        kw["vectors"] = vectors
        kw["blocks"] = blocks
        if "vectors_neg" in t:
            kw["vectors_neg"] = t["vectors_neg"].to(torch.float32)
    elif "vector" in t:
        kw["vector"] = t["vector"].to(torch.float32)
        if "vector_neg" in t:
            kw["vector_neg"] = t["vector_neg"].to(torch.float32)
    else:
        raise ValueError(f"{path.name}: no 'vector' or 'vectors' tensor")
    pack = SteeringPack(path=path, **kw)
    pack.validate()
    return pack


_TENSOR_KEYS = frozenset({"vector", "vector_neg", "vectors", "vectors_neg", "blocks"})
# Header keys added by format 2 (written only when set). The legacy
# header's own ``blocks`` list (always empty) is not one of them: the
# block list of a multi-block pack is the ``blocks`` tensor.
_V2_META_KEYS = (
    "norms", "category", "applies_to", "seeds", "variant", "description",
    "pos_anchor", "neg_anchor",
)
_KNOWN_META = frozenset({
    "family", "checkpoint", "block", "hidden_size", "name", "label",
    "blurb", "hook", "method", "norm", "magnitude", "policy", "provenance",
    *_V2_META_KEYS,
})


# ---------------------------------------------------------------------------
# Bundles: every shipped knob of one model in ONE file.
#
# ``<family>/<checkpoint>/bundle.safetensors`` holds each knob's tensors
# under ``<name>/<tensor>`` (the pack's own tensor names: ``vector`` /
# ``vector_neg`` for a single-block pack, ``vectors`` / ``blocks`` /
# ``vectors_neg`` for a multi-block one, stored byte for byte) and one
# metadata entry, ``manifest``, a JSON object::
#
#   {"version": 1, "model": "sa3/medium", "created": "...Z",
#    "knobs": [{"name", "label", "category", "applies_to", "variant",
#               "blocks", "description", "anchors": {"pos", "neg"},
#               "calibrated_gain": {"pos"|"neg": {median, min, max,
#                                   reached, seeds, cutoff}},
#               "bar_pass": {"pos": bool, "neg": bool},
#               "apply": {...header keys the engine needs...},
#               "provenance": {...}}, ...]}
#
# Every knob carries the same calibrated_gain and bar_pass shape;
# ``build_bundle`` refuses a knob missing either. Loose per-file packs
# keep working beside a bundle (user-authored vectors); a loose pack
# whose name a bundle already provides is skipped.
# ---------------------------------------------------------------------------

BUNDLE_NAME = "bundle" + PACK_SUFFIX
BUNDLE_METADATA_KEY = "manifest"
BUNDLE_VERSION = 1
SIGNS = ("pos", "neg")
GAIN_KEYS = ("median", "min", "max", "reached", "seeds", "cutoff")
# Header keys the engine path needs that are not top-level manifest fields.
_APPLY_KEYS = (
    "family", "checkpoint", "hook", "block", "hidden_size", "method", "norm",
    "magnitude", "policy", "norms", "seeds", "blurb",
)


def normalize_calibrated_gain(gain, name: str) -> dict:
    """The one calibrated-gain shape a bundle carries, or ValueError.

    Per sign: ``median`` / ``min`` / ``max`` (float) and per-seed lists
    ``reached`` (bool), ``seeds`` (int) and ``cutoff`` (float) of equal
    length. Anything else (a flat number, a missing sign or key) refuses.
    """
    if not isinstance(gain, Mapping):
        raise ValueError(f"knob {name!r}: no calibrated_gain")
    out: dict = {}
    for sign in SIGNS:
        g = gain.get(sign)
        if not isinstance(g, Mapping):
            raise ValueError(f"knob {name!r}: calibrated_gain.{sign} missing")
        missing = [k for k in GAIN_KEYS if k not in g]
        if missing:
            raise ValueError(
                f"knob {name!r}: calibrated_gain.{sign} lacks {missing}"
            )
        try:
            entry = {k: float(g[k]) for k in ("median", "min", "max")}
            entry["reached"] = [bool(x) for x in g["reached"]]
            entry["seeds"] = [int(x) for x in g["seeds"]]
            entry["cutoff"] = [float(x) for x in g["cutoff"]]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"knob {name!r}: calibrated_gain.{sign} malformed ({exc})"
            ) from None
        n = len(entry["seeds"])
        if n == 0 or len(entry["reached"]) != n or len(entry["cutoff"]) != n:
            raise ValueError(
                f"knob {name!r}: calibrated_gain.{sign} per-seed lists "
                "empty or of unequal length"
            )
        out[sign] = entry
    return out


def normalize_bar_pass(bar, name: str) -> dict:
    """``{"pos": bool, "neg": bool}`` or ValueError."""
    if not isinstance(bar, Mapping) or any(
        not isinstance(bar.get(s), bool) for s in SIGNS
    ):
        raise ValueError(f"knob {name!r}: bar_pass needs a bool per sign, got {bar!r}")
    return {s: bool(bar[s]) for s in SIGNS}


def _bundle_entry(pack: SteeringPack, bar_pass) -> dict:
    prov = dict(pack.provenance) if isinstance(pack.provenance, Mapping) else {}
    gain = normalize_calibrated_gain(prov.pop("calibrated_gain", None), pack.name)
    meta = pack.metadata()
    apply = {k: meta[k] for k in _APPLY_KEYS if k in meta}
    return {
        "name": pack.name,
        "label": pack.label,
        "category": pack.effective_category,
        "applies_to": pack.applies_to,
        "variant": pack.variant,
        "blocks": list(pack.all_blocks),
        "description": pack.description,
        "anchors": {"pos": pack.pos_anchor, "neg": pack.neg_anchor},
        "calibrated_gain": gain,
        "bar_pass": normalize_bar_pass(bar_pass, pack.name),
        "apply": apply,
        "provenance": prov,
    }


def _pack_tensors(pack: SteeringPack) -> dict:
    """The tensors ``save_pack`` would write, unprefixed."""
    import torch

    def f32(t):
        return t.detach().to(device="cpu", dtype=torch.float32).contiguous()

    if pack.vectors is not None:
        out = {
            "vectors": f32(pack.vectors),
            "blocks": torch.tensor([int(b) for b in pack.blocks], dtype=torch.int64),
        }
        if pack.vectors_neg is not None:
            out["vectors_neg"] = f32(pack.vectors_neg)
    else:
        out = {"vector": f32(pack.vector)}
        if pack.vector_neg is not None:
            out["vector_neg"] = f32(pack.vector_neg)
    return out


def build_bundle(
    packs: Sequence[SteeringPack],
    bar_pass: Mapping[str, Mapping],
    path: Path | str,
    *,
    created: Optional[str] = None,
) -> dict:
    """Write ``packs`` to one bundle file; returns the manifest.

    ``bar_pass`` maps pack name -> ``{"pos": bool, "neg": bool}``. All
    packs must share one family/checkpoint, have distinct names and the
    full calibrated-gain shape; any violation raises before writing.
    """
    from datetime import datetime, timezone

    from safetensors.torch import save_file

    if not packs:
        raise ValueError("no packs to bundle")
    models = {(p.family, p.checkpoint) for p in packs}
    if len(models) != 1:
        raise ValueError(f"packs span several models: {sorted(models)}")
    (family, checkpoint), = models
    tensors: dict = {}
    knobs: list = []
    for pack in sorted(packs, key=lambda p: p.name):
        pack.validate()
        if any(k["name"] == pack.name for k in knobs):
            raise ValueError(f"duplicate pack name {pack.name!r}")
        if pack.name not in bar_pass:
            raise ValueError(f"knob {pack.name!r}: no bar_pass data")
        knobs.append(_bundle_entry(pack, bar_pass[pack.name]))
        for k, t in _pack_tensors(pack).items():
            tensors[f"{pack.name}/{k}"] = t
    manifest = {
        "version": BUNDLE_VERSION,
        "model": f"{family}/{checkpoint}",
        "created": created or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "knobs": knobs,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={
        BUNDLE_METADATA_KEY: json.dumps(manifest, sort_keys=True),
    })
    return manifest


def read_bundle_manifest(path: Path | str) -> dict:
    """The ``manifest`` header of one bundle file, version-checked."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        raw = (f.metadata() or {}).get(BUNDLE_METADATA_KEY)
    if raw is None:
        raise ValueError(f"{Path(path).name}: no {BUNDLE_METADATA_KEY!r} metadata")
    manifest = json.loads(raw)
    if int(manifest.get("version", 0)) != BUNDLE_VERSION:
        raise ValueError(
            f"{Path(path).name}: bundle version {manifest.get('version')} "
            f"!= {BUNDLE_VERSION}"
        )
    return manifest


def load_bundle(path: Path | str) -> list:
    """Every pack in one bundle file, as :class:`SteeringPack` objects
    whose tensors and apply header match the per-file packs exactly."""
    import torch
    from safetensors import safe_open

    path = Path(path)
    manifest = read_bundle_manifest(path)
    out: list = []
    with safe_open(str(path), framework="pt") as f:
        names = set(f.keys())
        for k in manifest.get("knobs", ()):
            name = k["name"]
            t = {
                key: f.get_tensor(f"{name}/{key}")
                for key in _TENSOR_KEYS if f"{name}/{key}" in names
            }
            prov = dict(k.get("provenance") or {})
            prov["calibrated_gain"] = normalize_calibrated_gain(
                k.get("calibrated_gain"), name,
            )
            anchors = k.get("anchors") or {}
            kw = {kk: v for kk, v in (k.get("apply") or {}).items() if kk in _KNOWN_META}
            if "norms" in kw:
                kw["norms"] = tuple(float(x) for x in (kw["norms"] or ()))
            kw.update(
                name=name, label=k.get("label") or "",
                category=k.get("category") or "", applies_to=k.get("applies_to"),
                variant=k.get("variant") or "", description=k.get("description") or "",
                pos_anchor=anchors.get("pos") or "", neg_anchor=anchors.get("neg") or "",
                provenance=prov, bar_pass=normalize_bar_pass(k.get("bar_pass"), name),
                path=path,
            )
            if "vectors" in t:
                if "blocks" not in t:
                    raise ValueError(f"{path.name}: {name}/vectors without blocks")
                vectors = t["vectors"].to(torch.float32)
                kw["blocks"] = tuple(int(b) for b in t["blocks"].reshape(-1).tolist())
                kw["vector"] = vectors[0]
                kw["vectors"] = vectors
                if "vectors_neg" in t:
                    kw["vectors_neg"] = t["vectors_neg"].to(torch.float32)
            elif "vector" in t:
                kw["vector"] = t["vector"].to(torch.float32)
                if "vector_neg" in t:
                    kw["vector_neg"] = t["vector_neg"].to(torch.float32)
            else:
                raise ValueError(f"{path.name}: knob {name!r} has no tensors")
            pack = SteeringPack(**kw)
            pack.validate()
            out.append(pack)
    return out


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
    Every ``bundle.safetensors`` is read first; a loose pack whose knob a
    bundle already provides is skipped silently.
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
    # Bundles first: a bundled knob shadows a loose pack of the same name.
    candidates: list = []
    bundled: set = set()
    for path in sorted(root.rglob(BUNDLE_NAME)):
        try:
            packs = load_bundle(path)
        except Exception as exc:
            logger.warning("steering_bundle_skipped path={} reason={}", path, exc)
            continue
        for pack in packs:
            if pack.family == family and pack.checkpoint == checkpoint:
                bundled.add(pack.knob_name)
            candidates.append((path, pack))
    for path in sorted(root.rglob(f"*{PACK_SUFFIX}")):
        if path.name == BUNDLE_NAME:
            continue
        if bundled and f"{PACK_KNOB_PREFIX}{path.stem}" in bundled:
            continue  # cheap skip: the installed file name is the pack name
        try:
            pack = load_pack(path)
        except Exception as exc:
            logger.warning("steering_pack_skipped path={} reason={}", path, exc)
            continue
        if pack.knob_name in bundled:
            continue
        candidates.append((path, pack))
    for path, pack in candidates:
        if pack.family != family or pack.checkpoint != checkpoint:
            continue
        if layout is not None and not all(
            layout.accepts(b, pack.hidden_size, pack.hook) for b in pack.all_blocks
        ):
            logger.warning(
                "steering_pack_skipped path={} reason=layout_mismatch "
                "blocks={} hidden={} hook={} layout={}",
                path, list(pack.all_blocks), pack.hidden_size, pack.hook, layout,
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


# ---- stacking rule ----------------------------------------------------------

STACKING_ENV = "DEMON_STEERING_STACKING"
STACKING_INV_N = "inv_n"
STACKING_NONE = "none"
STACKING_RULES = (STACKING_INV_N, STACKING_NONE)


def stacking_rule(rule: Optional[str] = None) -> str:
    """The stacking rule for simultaneously active pack knobs.

    ``none`` (default) sums active knobs' shifts unscaled (the v2
    behaviour); ``inv_n`` (opt-in) scales every active knob's shift by
    1/k when k pack knobs are non-zero. ``rule`` None
    reads ``$DEMON_STEERING_STACKING``; an unknown value falls back to
    the default with a warning."""
    import os

    raw = rule if rule is not None else os.environ.get(STACKING_ENV, "")
    val = str(raw).strip().lower() or STACKING_NONE
    if val in ("0", "off", "false"):
        val = STACKING_NONE
    if val not in STACKING_RULES:
        from loguru import logger

        logger.warning("steering_stacking_unknown rule={} using={}", raw, STACKING_NONE)
        val = STACKING_NONE
    return val


# ---- knob response sidecar (perceptual knob map) ----------------------------

KNOB_RESPONSE_NAME = "knob_response.json"
KNOB_RESPONSE_KIND = "steering_knob_response"
KNOB_RESPONSE_VERSION = 1


@dataclass(frozen=True)
class KnobResponse:
    """One knob's perceptual map from ``knob_response.json``.

    The client value ``v`` is a perceptual position in [-1, 1] per sign.
    ``u = |v| x headroom`` puts the shipped (calibrated) gain at
    ``|v| = 1/headroom`` (0.8). For ``u <= 1`` the applied knob is
    ``sign x gain[sign] x interp(u_grid, r[sign], u)``; past it (the
    headroom) it stays linear, ``sign x gain[sign] x u``. A sign without
    a curve uses ``r(u) = u`` (the linear map on the same throw)."""

    u_grid: tuple
    headroom: float
    gain: dict                      # sign -> calibrated gain (median, > 0)
    r: dict                         # sign -> tuple of len(u_grid), or None

    def apply(self, v: float) -> float:
        if v == 0.0:
            return 0.0
        sign = "pos" if v > 0.0 else "neg"
        g = self.gain.get(sign)
        if not g:
            return 0.0
        u = min(abs(float(v)), 1.0) * self.headroom
        curve = self.r.get(sign)
        f = _interp(self.u_grid, curve, u) if (curve is not None and u <= 1.0) else u
        return g * f if v > 0.0 else -g * f


def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """Piecewise-linear interpolation, clamped to the end points."""
    if x <= xs[0]:
        return float(ys[0])
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            t = (x - x0) / (x1 - x0) if x1 > x0 else 1.0
            return float(ys[i - 1]) + t * (float(ys[i]) - float(ys[i - 1]))
    return float(ys[-1])


def _read_knob_response_file(path: Path, *, family: str, checkpoint: str) -> dict:
    """``{pack name: (u_grid, headroom, {sign: r or None})}`` from one
    sidecar, or {} when it is the wrong kind / model / shape."""
    from loguru import logger

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("knob_response_skipped path={} reason={}", path, exc)
        return {}
    if not isinstance(doc, Mapping) or doc.get("kind") != KNOB_RESPONSE_KIND:
        logger.warning("knob_response_skipped path={} reason=kind", path)
        return {}
    if int(doc.get("version", 0)) != KNOB_RESPONSE_VERSION:
        logger.warning("knob_response_skipped path={} reason=version", path)
        return {}
    model = doc.get("model")
    if model and model != f"{family}/{checkpoint}":
        return {}
    if doc.get("interp", "piecewise_linear") != "piecewise_linear":
        logger.warning("knob_response_skipped path={} reason=interp", path)
        return {}
    try:
        grid = tuple(float(x) for x in doc["u_grid"])
        headroom = float(doc.get("headroom", 1.25))
    except (KeyError, TypeError, ValueError):
        logger.warning("knob_response_skipped path={} reason=u_grid", path)
        return {}
    if (len(grid) < 2 or grid[0] != 0.0 or abs(grid[-1] - 1.0) > 1e-9
            or any(b <= a for a, b in zip(grid, grid[1:])) or headroom < 1.0):
        logger.warning("knob_response_skipped path={} reason=u_grid", path)
        return {}
    out: dict = {}
    for name, entry in (doc.get("knobs") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        curves: dict = {}
        for sign in SIGNS:
            side = entry.get(sign)
            r = side.get("r") if isinstance(side, Mapping) else None
            if r is None:
                curves[sign] = None
                continue
            try:
                r = tuple(float(x) for x in r)
            except (TypeError, ValueError):
                r = ()
            if len(r) != len(grid):
                logger.warning(
                    "knob_response_knob_skipped path={} knob={} sign={} reason=shape",
                    path, name, sign,
                )
                r = None
            curves[sign] = r
        out[str(name)] = (grid, headroom, curves)
    return out


def load_knob_responses(
    packs: Sequence[SteeringPack], *, family: str, checkpoint: str,
) -> dict:
    """``{knob name: KnobResponse}`` for every calibrated pack with an
    entry in a ``knob_response.json`` beside its bundle (or pack file).
    Packs without a sidecar entry or a calibrated gain are left out and
    keep the raw linear knob."""
    from acestep.streaming.knobs import _gain_value

    files: dict = {}
    out: dict = {}
    for p in packs:
        if p.path is None:
            continue
        sidecar = Path(p.path).parent / KNOB_RESPONSE_NAME
        if sidecar not in files:
            files[sidecar] = (
                _read_knob_response_file(sidecar, family=family, checkpoint=checkpoint)
                if sidecar.is_file() else {}
            )
        hit = files[sidecar].get(p.name)
        if hit is None:
            continue
        prov = p.provenance if isinstance(p.provenance, Mapping) else {}
        cal = prov.get("calibrated_gain")
        if not isinstance(cal, Mapping):
            continue
        pos = _gain_value(cal, "pos")
        if pos is None:
            continue
        neg = _gain_value(cal, "neg") or pos
        grid, headroom, curves = hit
        out[p.knob_name] = KnobResponse(
            u_grid=grid, headroom=headroom,
            gain={"pos": pos, "neg": neg}, r=dict(curves),
        )
    return out


class PackSteering:
    """Per-session pack surface: knob specs and the knob -> config map.

    Mirrors :class:`~acestep.steering.controller.SteeringController`'s
    ``snapshot_key`` / ``build_configs`` contract so a backend can merge
    both into the pipeline's single steering slot.
    """

    def __init__(
        self,
        packs: Sequence[SteeringPack] = (),
        *,
        stacking: Optional[str] = None,
        responses: Optional[Mapping[str, "KnobResponse"]] = None,
    ):
        self.packs: tuple = tuple(packs)
        # Stacking rule across simultaneously active pack knobs; None =
        # $DEMON_STEERING_STACKING, default none (see stacking_rule).
        self.stacking: str = stacking_rule(stacking)
        # knob name -> KnobResponse (perceptual knob map); a knob absent
        # here keeps the raw linear map.
        self.responses: dict = dict(responses or {})

    @property
    def is_loaded(self) -> bool:
        return bool(self.packs)

    def knob_specs(self) -> list:
        from acestep.streaming.knobs import steering_pack_spec

        specs = []
        for p in self.packs:
            prov = p.provenance if isinstance(p.provenance, Mapping) else {}
            screening = prov.get("screening")
            flags = screening.get("flags") if isinstance(screening, Mapping) else None
            specs.append(steering_pack_spec(
                p.knob_name,
                label=p.label or p.name,
                block=p.block,
                policy=p.policy,
                blurb=p.blurb,
                category=p.effective_category,
                gain=prov.get("calibrated_gain"),
                blocks=p.all_blocks,
                applies_to=p.applies_to,
                variant=p.variant,
                pack_description=p.description,
                pos_anchor=p.pos_anchor,
                neg_anchor=p.neg_anchor,
                bar_pass=p.bar_pass,
                perceptual=(
                    self.responses[p.knob_name].headroom
                    if p.knob_name in self.responses else None
                ),
                flags=(
                    tuple(flags) if isinstance(flags, (list, tuple))
                    else tuple(f for f in re.split(r"[;,\s]+", flags) if f)
                    if isinstance(flags, str) else ()
                ),
            ))
        return specs

    def snapshot_key(self, raw: Mapping[str, float], n: int) -> tuple:
        return tuple(float(raw.get(p.knob_name, 0.0)) for p in self.packs) + (
            float(max(1, int(n))),
        )

    def build_configs(self, raw: Mapping[str, float], n: int) -> list:
        n = max(1, int(n))
        configs: list = []
        values = {p.knob_name: float(raw.get(p.knob_name, 0.0)) for p in self.packs}
        k = sum(1 for v in values.values() if v != 0.0)
        # inv_n: with k pack knobs non-zero, each knob's shift is scaled
        # by 1/k (v3 smoke S3: 82% of pairs and every triple stay under
        # the strictest member's fidelity cutoff; unscaled sums leave 5%).
        share = 1.0 / k if (self.stacking == STACKING_INV_N and k > 1) else 1.0
        for p in self.packs:
            alpha = values[p.knob_name]
            if alpha == 0.0:
                continue
            resp = self.responses.get(p.knob_name)
            if resp is not None:
                alpha = resp.apply(alpha)
                if alpha == 0.0:
                    continue
            alpha *= share
            weights = policy_weights(p.policy, n)
            for t in p.terms():
                # A negative knob uses the pack's own negative direction
                # at |knob| when it has one, else negates the positive one.
                if alpha < 0.0 and t.neg is not None:
                    vec, mag, a = t.neg, t.neg_magnitude, -alpha
                else:
                    vec, mag, a = t.pos, t.pos_magnitude, alpha
                configs.append({
                    "layer": int(t.block),
                    "step": -1,
                    "weights": weights,
                    "vector": vec,
                    "magnitude": float(mag),
                    "alpha": a,
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

    packs = discover_packs(
        steering_packs_dir(), family=family, checkpoint=checkpoint,
        layout=layout, reserved_names=reserved_names,
    )
    return PackSteering(packs, responses=load_knob_responses(
        packs, family=family, checkpoint=checkpoint,
    ))
