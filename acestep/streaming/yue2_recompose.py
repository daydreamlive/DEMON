"""Songs and the background re-compose behind YuE2's ``set_prompt``.

A YuE2 style lives in the semantic tokens, not in the acoustic stage's
``[Tags]`` prefix: with the semantics frozen, a tag change moves the
audio less than a seed change does (experiment E3, docs/FAMILIES.md).
So a prompt change re-composes: score plan + semantic AR for the new
tags at the session's fixed length, then a KV prefill. That is seconds
of AR, so it runs on the YuE2Context worker thread while the ring keeps
playing the current song, and the finished :class:`Song` is published
to the backend atomically.

* :class:`Song`: one composition's conditioning bundle plus its anchor
  (the clean full solve). A freshly re-composed song has no anchor yet;
  the ring renders it with one full solve and adopts that latent.
* :class:`Recomposer`: latest-wins jobs per A/B slot. A job superseded
  before it starts is skipped; one superseded while it ran releases its
  bundle instead of publishing.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from acestep.engine.obs import logger


@dataclass(eq=False)
class Song:
    """One composition in the ring: its KV bundle and, once rendered,
    its anchor latent ``[1, T, 64]``. Identity matters (signatures and
    attribution compare songs by ``is``)."""

    bundle: Any
    anchor: Optional[torch.Tensor]
    tags: str
    epoch: int = 0

    @property
    def frames(self) -> int:
        return int(self.bundle.frames)

    @property
    def seed(self) -> int:
        return int(getattr(self.bundle, "seed", 0))


def run_inline(fn: Callable, *args, **kwargs) -> Future:
    """A ``submit`` that runs the job on the calling thread (tests)."""
    future: Future = Future()
    try:
        future.set_result(fn(*args, **kwargs))
    except BaseException as exc:  # noqa: BLE001 - surfaced through the future
        future.set_exception(exc)
    return future


class Recomposer:
    """Latest-wins background re-compose for the A and B song slots.

    ``build(tags, epoch) -> Song`` does the AR work (on the worker that
    ``submit`` targets); ``publish(slot, song, build_ms)`` hands the
    result to the backend; ``release(song)`` frees a result nobody will
    play.
    """

    def __init__(self, *, build: Callable[[str, int], Song], submit: Callable,
                 publish: Callable[[str, Song, float], None], release: Callable[[Song], None]):
        self._build = build
        self._submit = submit
        self._publish = publish
        self._release = release
        self._lock = threading.Lock()
        self._latest: dict = {}

    def request(self, slot: str, tags: str, epoch: int) -> Future:
        token = object()
        with self._lock:
            self._latest[slot] = token
        future = self._submit(self._job, slot, tags, epoch, token)
        future.add_done_callback(lambda f: self._log_failure(f, slot, tags))
        return future

    def cancel_all(self) -> None:
        with self._lock:
            self._latest.clear()

    def _current(self, slot: str, token) -> bool:
        with self._lock:
            return self._latest.get(slot) is token

    def _job(self, slot: str, tags: str, epoch: int, token) -> Optional[Song]:
        if not self._current(slot, token):
            logger.info("yue2_recompose_skipped slot={} tags={!r} reason=superseded", slot, tags)
            return None
        t0 = time.perf_counter()
        song = self._build(tags, epoch)
        build_ms = (time.perf_counter() - t0) * 1000
        if not self._current(slot, token):
            logger.info("yue2_recompose_dropped slot={} tags={!r} reason=superseded", slot, tags)
            self._release(song)
            return None
        self._publish(slot, song, build_ms)
        return song

    @staticmethod
    def _log_failure(future: Future, slot: str, tags: str) -> None:
        exc = future.exception()
        if exc is not None:
            logger.error("yue2_recompose_failed slot={} tags={!r} error={!r}", slot, tags, exc)
