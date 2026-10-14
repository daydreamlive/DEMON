"""Stable identity for the LoRA adapters (styles) a session ran.

Level 1 records *which* styles were used, and that record has to stay
meaningful after it leaves the pod: the cloud action log is joined
against the LoRA registry (``lora_artifacts``) to answer "whose style
was this, and how long was it enabled" — the artist-payout question.

The join key is the **weight file's SHA-256**, because it is the only
identifier the pod can be trusted to produce:

* the catalog ``id`` is the filename stem — chosen by whoever
  materialized the file, and not unique across users;
* the registry UUID never reaches the pod; an adapter arrives as a bare
  ``.safetensors`` plus an optional metadata sidecar;
* the sha256 is exactly what ``lora_artifacts.sha256`` already stores,
  so the publishing row — and with it the owner and the style name they
  chose — is one join away.

Hashing a miss means reading 10–400 MB, so it must never sit on a hot
path. Results are memoized on ``(path, size, mtime_ns)`` — the same key
the metadata sidecar loader uses — which makes a style toggled several
times in one session hash exactly once; :func:`pending` lets a caller
ask whether the answer is already cached, and defer to a worker thread
when it isn't (see :mod:`acestep.provenance.session_log`).

Everything fails open. An unreadable or vanished file yields
``sha256: None`` and the event still carries the id, name and trigger
word: a style with no hash is a style we cannot attribute, never a
dropped session.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Optional

from loguru import logger

__all__ = ["identities", "pending", "weight_sha256"]

# Read granularity for the weight hash. Large enough that a 400 MB
# adapter is ~400 reads, small enough not to spike RSS on a pod that is
# already holding a model in host memory.
_CHUNK_BYTES = 1 << 20

# Distinct weight files remembered. A pod's catalog is tens of entries;
# this only has to outlive one session's toggling.
_CACHE_MAX = 64

_cache: dict[tuple[str, int, int], Optional[str]] = {}
_cache_lock = threading.Lock()


def _cache_key(path: Path) -> Optional[tuple[str, int, int]]:
    """Identity of the bytes on disk: path + size + mtime. ``None`` when
    the file cannot be stat'd, which callers treat as "unknown hash"
    rather than caching a miss against a key that may never repeat."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (str(path), int(st.st_size), int(st.st_mtime_ns))


def weight_sha256(path: Path | str, *, compute: bool = True) -> Optional[str]:
    """SHA-256 of a LoRA weight file, memoized on its stat identity.

    With ``compute=False`` this is a pure cache lookup — it never reads
    the file — so it is safe to call from a latency-sensitive thread.
    Returns ``None`` when the hash is unknown (not cached and
    ``compute=False``, or the file could not be read).
    """
    p = Path(path)
    key = _cache_key(p)
    if key is None:
        return None
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    if not compute:
        return None

    digest: Optional[str]
    try:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            while chunk := fh.read(_CHUNK_BYTES):
                h.update(chunk)
        digest = h.hexdigest()
    except OSError as exc:
        logger.warning("lora weight hash failed path={}: {}", p, exc)
        digest = None

    with _cache_lock:
        # Bounded, insertion-ordered eviction: dicts preserve order, so
        # the oldest key is the first one.
        if len(_cache) >= _CACHE_MAX and key not in _cache:
            _cache.pop(next(iter(_cache)), None)
        _cache[key] = digest
    return digest


def _enabled(catalog: Any) -> list[dict]:
    """The enabled entries of a wire-shaped LoRA catalog. Anything that
    is not a dict, or is not enabled, is dropped — a malformed catalog
    must not raise on the session path."""
    if not isinstance(catalog, (list, tuple)):
        return []
    return [
        e for e in catalog
        if isinstance(e, dict) and e.get("state") == "enabled"
    ]


def pending(catalog: Any) -> bool:
    """True when at least one enabled entry's weight hash is not cached
    yet, i.e. building identities now would cost file reads."""
    for entry in _enabled(catalog):
        path = entry.get("path")
        if path and weight_sha256(path, compute=False) is None:
            return True
    return False


def identities(catalog: Any, *, compute: bool = False) -> list[dict]:
    """Identity payloads for the enabled entries of a LoRA catalog.

    Shape per style (all fields optional except ``id``)::

        {"id": "moody-piano",          # catalog id = filename stem
         "name": "Moody Piano",        # display name, sidecar or stem
         "sha256": "9f86d0…",          # weight hash: the registry join key
         "size_bytes": 268435456,
         "family": "ace",              # weight-format family
         "base_model": "acestep",
         "trigger_word": "moody piano",
         "strength": 0.8}

    The filesystem path is deliberately absent: it is pod-local state
    that would leak a server path into a user-visible log, and the hash
    identifies the bytes better than the path does.
    """
    out: list[dict] = []
    for entry in _enabled(catalog):
        metadata = entry.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        path = entry.get("path")
        item: dict[str, Any] = {
            "id": entry.get("id"),
            "name": metadata.get("name") or entry.get("name"),
            "sha256": (
                weight_sha256(path, compute=compute) if path else None
            ),
            "size_bytes": entry.get("materialized_bytes"),
            "family": metadata.get("lora_family"),
            "base_model": metadata.get("base_model"),
            "trigger_word": metadata.get("primary_trigger_word"),
            "strength": entry.get("strength"),
        }
        out.append(item)
    return out
