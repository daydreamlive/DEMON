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
  (or cancelled: the user went back to the playing prompt) before it
  starts is skipped; one superseded while it ran releases its bundle
  instead of publishing. A failed job is logged at WARNING and handed to
  ``on_failure`` (the session's error event). :meth:`Recomposer.inflight`
  names the tags a slot is building, so the backend can treat a repeated
  request as the job already running.
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
    ``submit`` targets); ``publish(slot, song, build_ms, is_current)``
    hands the result to the backend, which must re-check ``is_current()``
    under its own lock before swapping the song in (a cancel can land
    between the job's check and the publish); ``release(song)`` frees a
    result nobody will play; ``on_failure(slot, tags, exc)`` reports a
    build that raised.
    """

    def __init__(self, *, build: Callable[[str, int], Song], submit: Callable,
                 publish: Callable[[str, Song, float, Callable[[], bool]], None],
                 release: Callable[[Song], None],
                 on_failure: Optional[Callable[[str, str, BaseException], None]] = None):
        self._build = build
        self._submit = submit
        self._publish = publish
        self._release = release
        self._on_failure = on_failure
        self._lock = threading.Lock()
        self._latest: dict = {}
        self._inflight: dict = {}
        self._pending: set = set()

    def inflight(self, slot: str) -> Optional[str]:
        """The tags the slot's current (queued or running) job builds,
        or None when the slot has no job that may still publish."""
        with self._lock:
            return self._inflight.get(slot) if slot in self._latest else None

    def request(self, slot: str, tags: str, epoch: int) -> Future:
        token = object()
        with self._lock:
            self._latest[slot] = token
            self._inflight[slot] = tags
        future = self._submit(self._job, slot, tags, epoch, token)
        with self._lock:
            self._pending.add(future)
        future.add_done_callback(lambda f: self._finished(f, slot, tags))
        return future

    def cancel(self, slot: str) -> None:
        """Abandon the slot's queued or running job (its result, if any,
        is released instead of published)."""
        with self._lock:
            self._latest.pop(slot, None)
            self._inflight.pop(slot, None)

    def cancel_all(self) -> None:
        """Abandon every job; a job that has not started yet is also
        removed from the worker's queue."""
        with self._lock:
            self._latest.clear()
            self._inflight.clear()
            pending = list(self._pending)
        for future in pending:
            future.cancel()

    def _current(self, slot: str, token) -> bool:
        with self._lock:
            return self._latest.get(slot) is token

    def _job(self, slot: str, tags: str, epoch: int, token) -> Optional[Song]:
        try:
            return self._run(slot, tags, epoch, token)
        finally:
            # Done (published, dropped or failed): a repeat of these tags
            # is a new request from here on (a retry after a failure).
            with self._lock:
                if self._latest.get(slot) is token:
                    del self._latest[slot]
                    self._inflight.pop(slot, None)

    def _run(self, slot: str, tags: str, epoch: int, token) -> Optional[Song]:
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
        self._publish(slot, song, build_ms, lambda: self._current(slot, token))
        return song

    def _finished(self, future: Future, slot: str, tags: str) -> None:
        with self._lock:
            self._pending.discard(future)
        if future.cancelled():
            return
        exc = future.exception()
        if exc is None:
            return
        logger.warning("yue2_recompose_failed slot={} tags={!r} error={!r}", slot, tags, exc)
        if self._on_failure is not None:
            try:
                self._on_failure(slot, tags, exc)
            except Exception as report_exc:  # noqa: BLE001 - never kill the worker
                logger.warning("yue2_recompose_failure_report_raised error={!r}", report_exc)
