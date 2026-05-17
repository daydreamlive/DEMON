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

3. A small ``(path, mtime_ns)`` memoization layer so a catalog refresh
   over ~30 LoRAs costs one ``stat`` per entry, not a JSON parse.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .paths import lora_trigger

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
    # True iff a valid metadata.json was loaded for this record. Lets
    # callers distinguish "rich metadata" from "synthesized fallback"
    # without inspecting individual field nullity.
    has_metadata: bool = False

    def to_wire(self) -> dict[str, Any]:
        """JSON-safe dict for shipping to the UI / MCP clients."""
        return asdict(self)


# (sidecar_path_str, mtime_ns) -> parsed record
_cache: dict[tuple[str, int], LoraMetadata] = {}


def load_lora_metadata(lora_path: Path | str) -> LoraMetadata:
    """Load metadata for a LoRA ``.safetensors`` at ``lora_path``.

    Returns a normalized :class:`LoraMetadata` covering the three input
    states documented at the module level. Never raises on malformed
    sidecars — falls back through ``metadata.json`` → ``.trigger.txt``
    → bare in that order, logging warnings as it goes.
    """
    p = Path(lora_path)
    stem = p.stem
    sidecar = _metadata_sidecar(p)

    # Cache by mtime_ns when the sidecar exists. We don't cache the
    # "no metadata.json" path because there's nothing to invalidate
    # against (a file appearing wouldn't bust a cache keyed on nothing).
    try:
        st = sidecar.stat()
        cache_key: Optional[tuple[str, int]] = (str(sidecar), st.st_mtime_ns)
    except OSError:
        cache_key = None

    if cache_key is not None and cache_key in _cache:
        return _cache[cache_key]

    if cache_key is not None:
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
            _cache[cache_key] = md
            return md

    # Legacy .trigger.txt fallback. lora_trigger() already returns ""
    # on miss/IO error so this is a single string check.
    legacy = lora_trigger(p)
    if legacy:
        return LoraMetadata(
            id=stem,
            name=stem,
            primary_trigger_word=legacy,
            trigger_words=[legacy],
        )

    return LoraMetadata(id=stem, name=stem)


def _metadata_sidecar(lora_path: Path) -> Path:
    """Resolve ``foo/bar.safetensors`` → ``foo/bar.metadata.json``.

    Uses the same ``with_suffix("").with_suffix(...)`` trick as
    :func:`acestep.paths.lora_trigger` so stems with dots in them
    (e.g. ``my.cool.lora.safetensors``) resolve consistently.
    """
    return lora_path.with_suffix("").with_suffix(".metadata.json")


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
