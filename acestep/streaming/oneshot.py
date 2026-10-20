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
from dataclasses import dataclass, replace
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
    """Re-denoise streams ``streams`` of a held version from seed ``seed``,
    every other stream given from the held latent.

    The held version is kept version ``id``, or ``latent``: client-held
    state, a float16 ``[n_streams, C, L]`` array (SA3-native per stream)
    the wire adapter reads from the binary frame that follows the header
    (``latent_shape`` is the shape the header declared). ``denoise[i]`` is
    stream ``streams[i]``'s denoise: 1.0 re-rolls it from noise, below 1.0
    adjusts it from its own held slice. ``streams`` is ascending."""

    id: Optional[str]
    stream: int
    seed: int
    streams: tuple = ()
    denoise: tuple = ()
    latent: Any = None
    latent_shape: Optional[tuple] = None

    def __post_init__(self):
        if not self.streams:
            object.__setattr__(self, "streams", (int(self.stream),))
        if not self.denoise:
            object.__setattr__(self, "denoise", (1.0,) * len(self.streams))


@dataclass(frozen=True)
class OneshotRequest:
    request_id: str
    seeds: tuple
    knobs: dict
    redo: Optional[Redo] = None
    #: Also return every version's full latent (float16 ``[V, S, C, L]``).
    return_latents: bool = False

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
    return_latents = data.get("return_latents", False)
    if return_latents is None:
        return_latents = False
    if not isinstance(return_latents, bool):
        raise OneshotError(f"return_latents must be a boolean, got {return_latents!r}")
    redo_raw = data.get("redo")
    if redo_raw is not None:
        redo = _parse_redo(redo_raw)
        return OneshotRequest(
            request_id, (redo.seed,), knobs, redo, return_latents,
        )
    seeds_raw = data.get("seeds")
    if not isinstance(seeds_raw, (list, tuple)) or not seeds_raw:
        raise OneshotError("seeds must be a non-empty list of integers")
    if len(seeds_raw) > MAX_VERSIONS:
        raise OneshotError(f"at most {MAX_VERSIONS} versions per request")
    seeds = tuple(_int(s, "seed") for s in seeds_raw)
    if any(s < 0 for s in seeds):
        raise OneshotError("seeds must be >= 0")
    return OneshotRequest(request_id, seeds, knobs, None, return_latents)


def latent_shape(spec) -> tuple:
    """The ``[n_streams, C, L]`` shape a ``redo.latent`` header declares.
    Raises :class:`OneshotError` on anything but float16 and three
    positive integers."""
    if not isinstance(spec, Mapping):
        raise OneshotError("redo.latent must be an object {dtype, shape}")
    if spec.get("dtype", "float16") != "float16":
        raise OneshotError(
            f"redo.latent.dtype must be 'float16', got {spec.get('dtype')!r}"
        )
    shape = spec.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise OneshotError("redo.latent.shape must be [n_streams, C, L]")
    dims = tuple(_int(d, "redo.latent.shape") for d in shape)
    if any(d < 1 for d in dims):
        raise OneshotError(f"redo.latent.shape must be positive, got {list(dims)}")
    return dims


def latent_nbytes(spec) -> int:
    """Byte length of the binary frame a ``redo.latent`` header announces."""
    n, c, length = latent_shape(spec)
    return n * c * length * 2


def latent_from_bytes(redo: Redo, payload) -> Redo:
    """``redo`` with its ``latent`` read from the binary frame ``payload``
    (raw little-endian float16, C-order ``latent_shape``)."""
    shape = redo.latent_shape
    if shape is None:
        raise OneshotError("redo has no latent header")
    want = shape[0] * shape[1] * shape[2] * 2
    if payload is None or len(payload) != want:
        got = 0 if payload is None else len(payload)
        raise OneshotError(
            f"redo.latent frame is {got} bytes, expected {want} "
            f"(float16 {list(shape)})"
        )
    arr = np.frombuffer(payload, dtype="<f2").reshape(shape).copy()
    return replace(redo, latent=arr)


def _parse_redo(redo_raw) -> Redo:
    if not isinstance(redo_raw, Mapping):
        raise OneshotError("redo must be an object {id, stream, seed}")
    rid = redo_raw.get("id")
    rid = str(rid) if rid not in (None, "") else None
    latent_raw = redo_raw.get("latent")
    if rid is not None and latent_raw is not None:
        raise OneshotError("redo takes id or latent, not both")
    if rid is None and latent_raw is None:
        raise OneshotError("redo.id (or redo.latent) is required")
    shape = latent_shape(latent_raw) if latent_raw is not None else None
    one = redo_raw.get("stream")
    many = redo_raw.get("streams")
    if many is not None:
        if not isinstance(many, (list, tuple)) or not many:
            raise OneshotError("redo.streams must be a non-empty list of integers")
        streams = tuple(_int(k, "redo.streams") for k in many)
        if len(set(streams)) != len(streams):
            raise OneshotError(f"redo.streams has a repeat: {list(streams)}")
        if one is not None and (_int(one, "redo.stream"),) != streams:
            raise OneshotError(
                f"redo.stream {one!r} and redo.streams {list(streams)} disagree"
            )
    elif one is not None:
        streams = (_int(one, "redo.stream"),)
    else:
        raise OneshotError("redo.stream (or redo.streams) is required")
    if any(k < 0 for k in streams):
        raise OneshotError("redo.streams must be >= 0")
    d_raw = redo_raw.get("denoise")
    if d_raw is None:
        denoise = (1.0,) * len(streams)
    elif isinstance(d_raw, (list, tuple)):
        if len(d_raw) != len(streams):
            raise OneshotError(
                f"redo.denoise has {len(d_raw)} values for {len(streams)} streams"
            )
        denoise = tuple(_denoise(d) for d in d_raw)
    else:
        denoise = (_denoise(d_raw),) * len(streams)
    seed = _int(redo_raw.get("seed"), "redo.seed")
    if seed < 0:
        raise OneshotError("redo.seed must be >= 0")
    pairs = sorted(zip(streams, denoise))
    return Redo(
        id=rid,
        stream=pairs[0][0],
        seed=seed,
        streams=tuple(k for k, _ in pairs),
        denoise=tuple(d for _, d in pairs),
        latent_shape=shape,
    )


def _denoise(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OneshotError(f"redo.denoise must be a number, got {value!r}")
    d = float(value)
    if not 0.0 < d <= 1.0:
        raise OneshotError(f"redo.denoise must be in (0, 1], got {value!r}")
    return d


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
    #: Every version's full latent, float16 ``[V, S, C, L]`` on the CPU
    #: (``return_latents``), else None.
    latents: Optional[np.ndarray] = None
    #: The geometry a held latent must match:
    #: ``{n_streams, channels, frames, seconds}``.
    latent_header: Optional[dict] = None


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
