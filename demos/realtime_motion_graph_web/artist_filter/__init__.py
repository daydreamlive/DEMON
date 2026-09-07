"""Artist-name filter: deterministic, circumvention-resistant, hot-path safe.

Public API for the pod (see SPEC.md for the normative algorithm; the
TypeScript twin lives in demon-public-demo and runs the same golden vectors):

    scan(text)      -> ArtistMatch | None   (None = clean)
    filter_mode()   -> "on" | "log" | "off" (DEMON_ARTIST_FILTER; unknown=on)
    FILTER_VERSION  -> artifact version string for wire events / logs

The entry map (~tens of MB) loads lazily on first scan and is cached for the
process lifetime; the server pre-warms it at boot with ``scan("")`` so the
first real prompt never pays the load.
"""

from __future__ import annotations

import gzip
import json
import os
import threading
from pathlib import Path

from .matcher import ArtistMatch, Matcher  # noqa: F401 (re-export)
from .normalize import Normalizer

_DATA_DIR = Path(__file__).parent / "data"
_CONFIG_PATH = _DATA_DIR / "filter_config.v1.json"
_ARTISTS_PATH = _DATA_DIR / "artists.v1.json.gz"

_lock = threading.Lock()
_matcher: Matcher | None = None
_version: str = "unloaded"


def _load() -> Matcher:
    global _matcher, _version
    if _matcher is not None:
        return _matcher
    with _lock:
        if _matcher is not None:
            return _matcher
        config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        with gzip.open(_ARTISTS_PATH, "rt", encoding="utf-8") as fh:
            artifact = json.load(fh)
        normalizer = Normalizer(config)
        min_len = int(config["min_key_len"])
        # When two artists share a squash key, keep the strictest class:
        # multi and single_rare fire unconditionally, single_common is
        # gated — a permissive duplicate must never shadow a strict one.
        rank = {"multi": 2, "single_rare": 1, "single_common": 0}
        entries: dict[str, tuple[str, str, int]] = {}
        for key, display, cls, ntokens in artifact["entries"]:
            if len(key) < min_len:
                continue
            prev = entries.get(key)
            if prev is not None and rank.get(prev[1], 0) >= rank.get(cls, 0):
                continue
            entries[key] = (display, cls, int(ntokens))
        _version = str(artifact.get("version", "unknown"))
        _matcher = Matcher(normalizer, entries, config)
        return _matcher


def scan(text: str) -> ArtistMatch | None:
    """First artist-name match in ``text``, or None. Microseconds; no I/O
    after the first call."""
    if not text:
        _load()  # pre-warm path: scan("") at server boot
        return None
    return _load().scan(text)


def filter_version() -> str:
    _load()
    return _version


def filter_mode() -> str:
    """``DEMON_ARTIST_FILTER`` = on (default) | log | off. Unknown -> on:
    a typo in a deployment env must fail closed, not disable the filter."""
    mode = os.environ.get("DEMON_ARTIST_FILTER", "on").strip().lower()
    return mode if mode in ("on", "log", "off") else "on"
