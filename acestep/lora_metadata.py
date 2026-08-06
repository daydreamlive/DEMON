"""LoRA adapter metadata sidecar loader.

Each LoRA on disk may ship a ``<stem>.metadata.json`` sidecar next to
its ``.safetensors`` that describes how the adapter should be used at
inference: the activation token(s), recommended strength, training
classification, dataset summary, etc. The full schema is owned upstream
(see ``_adapter_metadata.schema.json`` in the model bundle on
HuggingFace); this module only consumes it.

This module owns three things and nothing else:

1. The runtime view of that schema (:class:`LoraMetadata`) — a small
   dataclass exposing exactly the fields the engine and the UI actually
   read. Extending the schema upstream doesn't break us; new fields are
   ignored until somebody adds them here.

2. Graceful degradation across three on-disk states:

   ============================================  ==========================
   On disk                                        Result
   ============================================  ==========================
   ``<stem>.metadata.json`` present and valid     Full record (has_metadata=True)
   ``metadata.json`` missing, ``.trigger.txt``    Synthesized minimal record:
   present                                         primary_trigger_word = file contents
   Neither                                         Sparse record: id/name only
   ============================================  ==========================

   Malformed ``metadata.json`` (bad JSON, IO error, unicode) logs a
   warning and falls back to the ``.trigger.txt`` path, so a broken
   sidecar never takes the WS catalog broadcast down with it.

3. A small memoization layer keyed on the mtimes of every file that
   feeds a record (weights header, ``metadata.json``, ``.trigger.txt``)
   so a catalog refresh over ~30 LoRAs costs a few ``stat`` calls per
   entry, not a JSON parse or a header read.

It also owns the *format* axis of LoRA identity: a header-only sniff of
the ``.safetensors`` key layout classifies each file into a
``lora_family`` ("ace" for PEFT-style ``lora_A.weight``/``lora_B.weight``
pairs, "sa3" for stable-audio-3 ``.parametrizations.weight.<idx>.``
entries), and for SA3 files the embedded ``lora_config`` header (written
by the trainer) supplies rank / adapter_type / base_model when no
sidecar documents them. No tensor data is ever read here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .paths import lora_sidecar, lora_trigger

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 1


@dataclass
class LoraMetadata:
    """Normalized LoRA metadata. Always returns a record — missing
    sidecars produce a sparse record with most fields ``None`` rather
    than raising.

    ``id`` is always the filename stem (the stable runtime identifier
    used by enable/disable RPCs). The sidecar's own ``id`` field is
    informational only; on mismatch we warn but keep using the stem so
    wire compat with the rest of the engine doesn't break.
    """

    id: str
    name: str
    description: Optional[str] = None
    primary_trigger_word: Optional[str] = None
    trigger_words: list[str] = field(default_factory=list)
    recommended_strength: Optional[float] = None
    recommended_steps: Optional[int] = None
    recommended_shift: Optional[float] = None
    recommended_guidance: Optional[float] = None
    primary_genre: Optional[str] = None
    secondary_genres: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)
    # The LoRA's training-time base model. ``base_model_scale`` is what
    # the runtime compares against the active checkpoint's scale (via
    # :func:`acestep.paths.checkpoint_scale`) to decide whether the
    # LoRA is loadable on the current session. ``None`` = unknown
    # (legacy LoRAs without sidecars, or sidecars that omit the model
    # block); callers treat this as "compatible with everything"
    # rather than "incompatible with everything" so we don't hide
    # undocumented LoRAs.
    base_model: Optional[str] = None
    base_model_scale: Optional[str] = None
    # Weight-format family sniffed from the safetensors header key
    # layout: "ace" (PEFT ``lora_A.weight``/``lora_B.weight`` pairs),
    # "sa3" (torch-parametrize ``.parametrizations.weight.<idx>.``
    # entries), or ``None`` (unknown / unreadable header). Code-derived
    # from the weights file itself — a sidecar cannot override it.
    lora_family: Optional[str] = None
    # SA3-only fields synthesized from the embedded ``lora_config``
    # header (the trainer writes rank/alpha/adapter_type/base_model
    # there). ``None`` on ACE files and on SA3 files without a config.
    adapter_type: Optional[str] = None
    rank: Optional[int] = None
    # True iff a valid metadata.json was loaded for this record. Lets
    # callers distinguish "rich metadata" from "synthesized fallback"
    # without inspecting individual field nullity.
    has_metadata: bool = False

    def to_wire(self) -> dict[str, Any]:
        """JSON-safe dict for shipping to the UI / MCP clients."""
        return asdict(self)


def lora_scale_compatible(
    lora_scale: Optional[str],
    checkpoint_scale: Optional[str],
) -> bool:
    """True iff a LoRA trained at ``lora_scale`` can load against a
    checkpoint of ``checkpoint_scale`` ("2B" / "5B" labels, see
    :func:`acestep.paths.checkpoint_scale`).

    Unknown on EITHER side is compatible — same permissive stance as
    the module-level ``base_model_scale`` comment above and the demo's
    ``isLoraCompatibleWithScale`` — so undocumented LoRAs and
    unrecognized checkpoints are never hidden or excluded. This is the
    scale AXIS only; the per-session predicate lives on the generator
    backend (``GeneratorBackend.lora_compatible``) so families can add
    axes beyond scale.
    """
    if not checkpoint_scale or not lora_scale:
        return True
    return lora_scale == checkpoint_scale


# Safetensors headers are capped at 100 MB by the format spec; anything
# claiming more is corrupt (or not a safetensors file at all).
_MAX_HEADER_BYTES = 100_000_000

_ACE_KEY_MARKERS = (".lora_A.weight", ".lora_B.weight")
_SA3_KEY_MARKER = ".parametrizations.weight."


def read_safetensors_header(path: Path | str) -> Optional[dict[str, Any]]:
    """Read just the JSON header of a ``.safetensors`` file (8-byte
    little-endian length prefix + JSON blob). No tensor data is read,
    so this is safe on multi-GB files and on every catalog refresh.

    Returns ``None`` on any IO/parse problem — sniffing must never take
    the catalog broadcast down with it.
    """
    try:
        with open(path, "rb") as f:
            prefix = f.read(8)
            if len(prefix) < 8:
                return None
            n = int.from_bytes(prefix, "little")
            if n <= 0 or n > _MAX_HEADER_BYTES:
                return None
            raw = f.read(n)
        header = json.loads(raw)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return header if isinstance(header, dict) else None


def _family_from_keys(keys: list[str]) -> Optional[str]:
    """Classify a LoRA's weight-format family from its tensor key names.

    SA3 keys look like
    ``model.…​.parametrizations.weight.0.lora_A`` (torch parametrize,
    no ``.weight`` suffix on the adapter tensors); ACE/PEFT keys look
    like ``base_model.model.….lora_A.weight``. The two markers cannot
    co-occur on one key, and a file mixing both families doesn't exist
    in the wild — SA3 wins the tie because its marker is the more
    specific one.
    """
    saw_ace = False
    for k in keys:
        if _SA3_KEY_MARKER in k:
            return "sa3"
        if not saw_ace and any(m in k for m in _ACE_KEY_MARKERS):
            saw_ace = True
    return "ace" if saw_ace else None


def sniff_lora_file(lora_path: Path | str) -> tuple[Optional[str], dict[str, Any]]:
    """Header-only sniff of a LoRA ``.safetensors``.

    Returns ``(family, config)`` where ``family`` is "ace" / "sa3" /
    ``None`` (see :func:`_family_from_keys`) and ``config`` is the
    parsed ``lora_config`` JSON embedded in the header's
    ``__metadata__`` block (SA3 trainer convention: rank, alpha,
    adapter_type, include/exclude, base_model, …). ``config`` is ``{}``
    when absent or malformed. For SA3 files without an explicit rank,
    the rank is inferred from the first adapter tensor's shape (same
    rule as upstream's ``infer_global_rank``).
    """
    header = read_safetensors_header(lora_path)
    if not header:
        return None, {}
    keys = [k for k in header if k != "__metadata__"]
    family = _family_from_keys(keys)

    cfg: dict[str, Any] = {}
    meta = header.get("__metadata__")
    if isinstance(meta, dict):
        raw_cfg = meta.get("lora_config")
        if isinstance(raw_cfg, str) and raw_cfg:
            try:
                parsed = json.loads(raw_cfg)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "LoRA %s has a malformed embedded lora_config (%s); "
                    "ignoring it",
                    lora_path,
                    exc,
                )
            else:
                if isinstance(parsed, dict):
                    cfg = parsed

    if family == "sa3" and "rank" not in cfg:
        for k in keys:
            spec = header.get(k)
            shape = spec.get("shape") if isinstance(spec, dict) else None
            if not shape:
                continue
            if k.endswith(".lora_A") or k.endswith(".M_xs"):
                cfg["rank"] = int(shape[0])
                break
            if k.endswith(".lora_B") and len(shape) > 1:
                cfg["rank"] = int(shape[1])
                break
    return family, cfg


# Runtime SA3 checkpoint ids as the loaders know them (SA3Context
# model_id). Training happens against the "-base" variants; both sides
# canonicalize to these.
_SA3_RUNTIME_LINEAGES = frozenset({"medium", "small-music", "small-sfx"})


def canonical_sa3_lineage(value: Optional[str]) -> Optional[str]:
    """Map an SA3 base-model identifier to its runtime checkpoint id
    ("medium" / "small-music" / "small-sfx").

    Accepts the trainer spellings ("medium-base"), the runtime ids
    themselves, and full HF repo ids (with or without the org prefix:
    "stabilityai/stable-audio-3-medium-base"). Case-insensitive.
    Returns ``None`` for anything unrecognized — callers treat ``None``
    as "unknown lineage, don't filter" per the permissive seam
    contract, so a new checkpoint spelling degrades to permissive
    rather than hiding LoRAs.
    """
    if not value:
        return None
    s = str(value).strip().lower()
    s = s.rsplit("/", 1)[-1]
    if s.startswith("stable-audio-3-"):
        s = s[len("stable-audio-3-"):]
    elif s.startswith("sa3-"):
        # underfit writes e.g. base_model="sa3-medium" (seen in the wild)
        s = s[len("sa3-"):]
    if s.endswith("-base"):
        s = s[: -len("-base")]
    return s if s in _SA3_RUNTIME_LINEAGES else None


# (lora_path_str, weights_mtime_ns, sidecar_mtime_ns, trigger_mtime_ns)
# -> parsed record. -1 stands for "file absent" so a sidecar/trigger
# appearing (or the weights file being replaced) changes the key.
_cache: dict[tuple[str, int, int, int], LoraMetadata] = {}


def _mtime_ns_or_absent(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return -1


def load_lora_metadata(lora_path: Path | str) -> LoraMetadata:
    """Load metadata for a LoRA ``.safetensors`` at ``lora_path``.

    Returns a normalized :class:`LoraMetadata` covering the three input
    states documented at the module level, plus the header-derived
    format fields (``lora_family``, and for SA3 files
    ``adapter_type``/``rank``/synthesized ``base_model``). Never raises
    on malformed sidecars — falls back through ``metadata.json`` →
    ``.trigger.txt`` → bare in that order, logging warnings as it goes.
    A present sidecar wins for the display/trigger/genre fields; the
    weight-format family always comes from the header sniff.
    """
    p = Path(lora_path)
    stem = p.stem
    sidecar = _metadata_sidecar(p)

    weights_mtime = _mtime_ns_or_absent(p)
    sidecar_mtime = _mtime_ns_or_absent(sidecar)
    trigger_mtime = _mtime_ns_or_absent(lora_sidecar(p, ".trigger.txt"))

    cache_key: Optional[tuple[str, int, int, int]] = None
    if weights_mtime != -1 or sidecar_mtime != -1 or trigger_mtime != -1:
        cache_key = (str(p), weights_mtime, sidecar_mtime, trigger_mtime)
        if cache_key in _cache:
            return _cache[cache_key]

    family: Optional[str] = None
    header_cfg: dict[str, Any] = {}
    if weights_mtime != -1:
        family, header_cfg = sniff_lora_file(p)

    md: Optional[LoraMetadata] = None
    if sidecar_mtime != -1:
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning(
                "LoRA metadata sidecar %s is unreadable (%s); falling back",
                sidecar,
                exc,
            )
        else:
            md = _from_schema(raw, stem)

    if md is None:
        # Legacy .trigger.txt fallback. lora_trigger() already returns ""
        # on miss/IO error so this is a single string check.
        legacy = lora_trigger(p)
        if legacy:
            md = LoraMetadata(
                id=stem,
                name=stem,
                primary_trigger_word=legacy,
                trigger_words=[legacy],
            )
        else:
            md = LoraMetadata(id=stem, name=stem)

    _apply_header_fields(md, family, header_cfg)

    if cache_key is not None:
        _cache[cache_key] = md
    return md


def _apply_header_fields(
    md: LoraMetadata, family: Optional[str], cfg: dict[str, Any],
) -> None:
    """Overlay header-derived fields onto a record.

    ``lora_family`` is always header-truth (a sidecar cannot claim a
    different weight format than the file actually has). The SA3
    ``lora_config`` fields only fill gaps the sidecar left: base_model
    from a sidecar wins over the trainer's embedded value.
    """
    md.lora_family = family
    if family != "sa3" or not cfg:
        return
    at = cfg.get("adapter_type")
    if isinstance(at, str) and at:
        md.adapter_type = at
    md.rank = _optional_int(cfg.get("rank"))
    if md.base_model is None:
        bm = cfg.get("base_model")
        if isinstance(bm, str) and bm:
            md.base_model = bm


def _metadata_sidecar(lora_path: Path) -> Path:
    """Resolve ``foo/bar.safetensors`` → ``foo/bar.metadata.json``.

    Delegates to :func:`acestep.paths.lora_sidecar` so stems with dots in
    them (e.g. ``alt_pop50-acestep1.5-dora-v2.safetensors``) resolve to
    the right sibling instead of being truncated at the internal dot.
    """
    return lora_sidecar(lora_path, ".metadata.json")


def _from_schema(raw: dict[str, Any], stem: str) -> LoraMetadata:
    """Parse a v1 schema dict into a :class:`LoraMetadata`.

    Resilient to missing optional fields. Logs warnings (but does not
    raise) on schema_version mismatch, id/stem mismatch, or
    primary_trigger_word that doesn't appear in trigger_words.
    """
    sv = raw.get("schema_version")
    if sv is not None and sv != CURRENT_SCHEMA_VERSION:
        logger.warning(
            "LoRA metadata for %s has schema_version=%s, runtime expects %s; "
            "reading optimistically",
            stem,
            sv,
            CURRENT_SCHEMA_VERSION,
        )

    sidecar_id = raw.get("id")
    if sidecar_id and sidecar_id != stem:
        # The runtime identifier is the filename stem (used by
        # enable/disable RPCs and the wire-side `id` field). The
        # sidecar's `id` is documentation. Warn and prefer the stem.
        logger.warning(
            "LoRA metadata for %s has sidecar id=%s; using stem as runtime id",
            stem,
            sidecar_id,
        )

    inference = raw.get("inference") or {}
    trigger_words = [t for t in (inference.get("trigger_words") or []) if t]
    primary = inference.get("primary_trigger_word")

    if primary is not None and trigger_words and primary not in trigger_words:
        logger.warning(
            "LoRA metadata for %s: primary_trigger_word %r is not in "
            "trigger_words %r; using it anyway",
            stem,
            primary,
            trigger_words,
        )

    # If the upstream record forgot to set primary but did list triggers,
    # treat the first as canonical so the UI still has something to copy
    # and prepend.
    if primary is None and trigger_words:
        primary = trigger_words[0]

    cls = raw.get("classification") or {}
    model = raw.get("model") or {}

    return LoraMetadata(
        id=stem,
        name=raw.get("name") or stem,
        description=raw.get("description"),
        primary_trigger_word=primary,
        trigger_words=trigger_words,
        recommended_strength=_optional_float(inference.get("recommended_strength")),
        recommended_steps=_optional_int(inference.get("recommended_steps")),
        recommended_shift=_optional_float(inference.get("recommended_shift")),
        recommended_guidance=_optional_float(inference.get("recommended_guidance")),
        primary_genre=cls.get("primary_genre"),
        secondary_genres=[g for g in (cls.get("secondary_genres") or []) if g],
        tags=[t for t in (cls.get("tags") or []) if t],
        moods=[m for m in (cls.get("moods") or []) if m],
        base_model=model.get("base_model"),
        base_model_scale=model.get("base_model_scale"),
        has_metadata=True,
    )


def _optional_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _optional_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def clear_cache() -> None:
    """Drop the in-memory metadata cache. Tests + manual reloads only."""
    _cache.clear()
