"""One-shot renders: N whole versions of a session's window, outside the ring.

The ``oneshot`` command (``Capabilities.oneshot``) asks a backend for one
complete render per seed: ONE denoise of the session's window per version
(the ring's step math, no StreamPipeline ticks, no slots in flight), a full
decode of every stream, and compact audio back (int16). Each version's
latent is kept here by id for a bounded time, so a later ``redo`` can
re-denoise one stream against the others of that version without any
upload (the stream layout's ``hold_streams``).

This module holds the parts that are not family code: the request parse,
the kept-latent store and the int16 packing. The render itself is the
backend's ``render_oneshot`` (``SA3Backend`` today).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

#: How long a kept latent stays redo-able (seconds since it was stored).
KEEP_TTL_S = 15 * 60.0
#: Kept latents at most (oldest dropped first).
KEEP_MAX = 32
#: Versions one request may ask for.
MAX_VERSIONS = 16


class OneshotError(ValueError):
    """A one-shot request the backend cannot serve (bad fields, an unknown
    or expired id, a redo on a layout without ``hold_streams``)."""


@dataclass(frozen=True)
class Redo:
    """Re-denoise stream ``stream`` of kept version ``id`` from seed
    ``seed``, every other stream given from the kept latent."""

    id: str
    stream: int
    seed: int


@dataclass(frozen=True)
class OneshotRequest:
    request_id: str
    seeds: tuple
    knobs: dict
    redo: Optional[Redo] = None

    @property
    def versions(self) -> int:
        return 1 if self.redo is not None else len(self.seeds)


def _int(value, what: str) -> int:
    if isinstance(value, bool):
        raise OneshotError(f"{what} must be an integer, got {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise OneshotError(f"{what} must be an integer, got {value!r}") from None
    if f != int(f):
        raise OneshotError(f"{what} must be an integer, got {value!r}")
    return int(f)


def parse_request(data: Mapping[str, Any]) -> OneshotRequest:
    """The wire ``oneshot`` frame as a typed request. Raises
    :class:`OneshotError` with a client-facing message."""
    request_id = str(data.get("request_id") or "")
    knobs_raw = data.get("knobs") or {}
    if not isinstance(knobs_raw, Mapping):
        raise OneshotError("knobs must be an object of knob name -> number")
    knobs = {}
    for k, v in knobs_raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise OneshotError(f"knob {k!r} must be a number, got {v!r}")
        knobs[str(k)] = float(v)
    redo_raw = data.get("redo")
    if redo_raw is not None:
        if not isinstance(redo_raw, Mapping):
            raise OneshotError("redo must be an object {id, stream, seed}")
        rid = str(redo_raw.get("id") or "")
        if not rid:
            raise OneshotError("redo.id is required")
        stream = _int(redo_raw.get("stream"), "redo.stream")
        seed = _int(redo_raw.get("seed"), "redo.seed")
        if seed < 0:
            raise OneshotError("redo.seed must be >= 0")
        return OneshotRequest(request_id, (seed,), knobs, Redo(rid, stream, seed))
    seeds_raw = data.get("seeds")
    if not isinstance(seeds_raw, (list, tuple)) or not seeds_raw:
        raise OneshotError("seeds must be a non-empty list of integers")
    if len(seeds_raw) > MAX_VERSIONS:
        raise OneshotError(f"at most {MAX_VERSIONS} versions per request")
    seeds = tuple(_int(s, "seed") for s in seeds_raw)
    if any(s < 0 for s in seeds):
        raise OneshotError("seeds must be >= 0")
    return OneshotRequest(request_id, seeds, knobs, None)


@dataclass
class Kept:
    """A kept version: its SA3-native latent on the CPU and its geometry."""

    latent: Any            # torch.Tensor [1, C, T], CPU
    n_streams: int
    seed: int
    stored_at: float


class KeptLatents:
    """Process-wide store of one-shot latents by id (thread-safe, bounded by
    count and age). Process-wide rather than per session: a client may
    redo on a fresh session of the same geometry."""

    def __init__(self, ttl_s: float = KEEP_TTL_S, max_items: int = KEEP_MAX,
                 clock=time.monotonic):
        self._ttl = float(ttl_s)
        self._max = int(max_items)
        self._clock = clock
        self._lock = threading.Lock()
        self._items: "OrderedDict[str, Kept]" = OrderedDict()

    def _expire(self) -> None:
        now = self._clock()
        for k in [k for k, v in self._items.items() if now - v.stored_at > self._ttl]:
            del self._items[k]
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def put(self, latent, *, n_streams: int, seed: int) -> str:
        key = uuid.uuid4().hex[:16]
        cpu = latent.detach().to("cpu").clone()
        with self._lock:
            self._items[key] = Kept(cpu, int(n_streams), int(seed), self._clock())
            self._expire()
        return key

    def get(self, key: str) -> Kept:
        with self._lock:
            self._expire()
            item = self._items.get(str(key))
        if item is None:
            raise OneshotError(
                f"no kept version {key!r} (unknown, or older than "
                f"{self._ttl / 60:.0f} min)"
            )
        return item

    def __len__(self) -> int:
        with self._lock:
            self._expire()
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


#: The store every session uses.
KEPT = KeptLatents()


@dataclass(frozen=True)
class OneshotResult:
    """What a backend's ``render_oneshot`` returns: int16
    ``[versions, len(streams), frames, channels]`` plus the ids, seeds and
    per-phase wall times."""

    pcm: np.ndarray
    sample_rate: int
    ids: tuple
    seeds: tuple
    streams: tuple
    ms: dict
    seconds: float
    redo_of: Optional[str] = None
    #: What the denoise ran on, for the host log: the DiT's ``describe()``
    #: (a stream layout's step: its engine file, or eager and why), else its
    #: class name.
    dit: str = ""


def to_int16(audio) -> np.ndarray:
    """float audio (torch or numpy, any shape) in [-1, 1] -> int16 numpy,
    rounded to nearest. Torch tensors are converted on their device."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is always present in serving
        torch = None
    if torch is not None and isinstance(audio, torch.Tensor):
        q = (audio.float().clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)
        return q.cpu().numpy()
    a = np.asarray(audio, dtype=np.float32)
    return np.round(np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16)
