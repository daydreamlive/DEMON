"""Shared TensorRT launch: CUDA graph capture, stream ordering and
refit-epoch invalidation for one execution context with a fixed binding
set.

Any engine wrapper, in this tree or out of it, gets graph capture,
caller-stream ordering and refit-epoch invalidation by composing
:class:`TrtLaunch`: create the execution context, ``bind`` persistent
input/output buffers once, write inputs with ``copy_()`` and call
``launch()``. This module is the only place that logic lives; the SA3
DiT wrapper (:class:`acestep.engine.sa3_trt.SA3TRTDit`) composes it, and
the refit mirrors call :func:`note_engine_refit` after every committed
refit so a graph captured over the old weights is re-captured.

Engine-agnostic by design: no TensorRT import (the engine and context are
duck-typed), no knowledge of any model family.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Optional

import torch
from loguru import logger

#: CUDA graph capture of TensorRT launches. Default ON; ``0`` is the
#: escape hatch (plain ``execute_async_v3`` every call). Read when a
#: wrapper (or a :class:`TrtLaunch`) is constructed.
TRT_CUDA_GRAPH_ENV = "DEMON_SA3_TRT_CUDA_GRAPH"

# Per-engine refit counter. A refit rewrites the engine's weights in
# place (refit_cuda_engine()); a graph captured before it is treated as
# stale and re-captured on the next launch, so replay never depends on how
# TensorRT stores refitted weights. Keyed by id(engine): TensorRT engines
# are not weak-referenceable, and a reused id only means the new engine's
# wrappers start from whatever count is there (they read it at capture
# time, so the key stays consistent).
_REFIT_EPOCH: dict = {}
_REFIT_EPOCH_LOCK = threading.Lock()


def cuda_graph_enabled() -> bool:
    """Whether TensorRT launches are captured
    (``DEMON_SA3_TRT_CUDA_GRAPH``, default on; ``0`` / ``false`` / ``off``
    / ``no`` turn it off)."""
    val = os.environ.get(TRT_CUDA_GRAPH_ENV, "").strip().lower()
    return val not in ("0", "false", "off", "no")


def note_engine_refit(engine) -> int:
    """Record that ``engine``'s weights were refit; returns the new count.
    Called by a refit mirror after every committed refit."""
    with _REFIT_EPOCH_LOCK:
        n = _REFIT_EPOCH.get(id(engine), 0) + 1
        _REFIT_EPOCH[id(engine)] = n
        return n


def engine_refit_epoch(engine) -> int:
    """How many refits have been committed on ``engine`` (0 = none)."""
    return _REFIT_EPOCH.get(id(engine), 0)


def launch_key(engine, ctx, bindings) -> tuple:
    """What a captured launch is valid for: the engine (and its refit
    count), the execution context, and each bound tensor's name, shape
    and device address. ``bindings`` is ``(name, tensor)`` pairs, exactly
    what was passed to ``set_tensor_address``."""
    return (
        id(engine),
        engine_refit_epoch(engine),
        id(ctx),
        tuple((name, tuple(buf.shape), buf.data_ptr()) for name, buf in bindings),
    )


_launch_key = launch_key


class LaunchGraphCache:
    """Captured launches per wrapper, keyed by :func:`launch_key`.

    ``capacity=1`` (one fixed binding set per wrapper) keeps a single
    graph, and a lookup under any other key drops it, so the next launch
    re-captures. ``capacity>1`` (one graph per input shape) keeps up to
    that many graphs, least recently used evicted first; a key that is not
    cached simply misses. Capture failure disables capture for the wrapper
    (plain launches from then on)."""

    def __init__(self, enabled: bool, capacity: int = 1) -> None:
        self.enabled = bool(enabled)
        self.capacity = max(1, int(capacity))
        self._entries: "OrderedDict[tuple, object]" = OrderedDict()
        self.captures = 0
        self.invalidations = 0
        self.evictions = 0
        self.disabled_reason: Optional[str] = None

    @property
    def key(self):
        """Most recently used key (None when empty)."""
        return next(reversed(self._entries)) if self._entries else None

    @property
    def graph(self):
        """Most recently used graph (None when empty)."""
        return self._entries[self.key] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, key):
        """The graph captured under ``key``, or None. With capacity 1 a
        miss also drops the stale graph (anything in the key changed)."""
        graph = self._entries.get(key) if key is not None else None
        if graph is None:
            if self.capacity == 1 and self._entries:
                self._entries.clear()
                self.invalidations += 1
            return None
        self._entries.move_to_end(key)
        return graph

    def store(self, key, graph) -> None:
        self._entries[key] = graph
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
            self.evictions += 1
        self.captures += 1

    def discard(self, key) -> None:
        """Drop the graph under ``key`` if there is one."""
        if self._entries.pop(key, None) is not None:
            self.invalidations += 1

    def disable(self, reason: str) -> None:
        self.enabled = False
        self._entries.clear()
        self.disabled_reason = reason


_LaunchGraphCache = LaunchGraphCache


class TrtLaunch:
    """One execution context with a fixed binding set, launched on a
    dedicated non-blocking stream after a wait_stream on the caller's
    stream, and captured into a CUDA graph (DEMON_SA3_TRT_CUDA_GRAPH,
    default on). Bindings are persistent buffers bound once; the caller
    writes inputs with copy_() and reads outputs from the bound buffers
    after launch()."""

    def __init__(
        self,
        engine,
        ctx,
        *,
        label: str = "trt",
        cuda_graph: Optional[bool] = None,
        stream: Optional["torch.cuda.Stream"] = None,
        capacity: int = 1,
    ) -> None:
        self.engine = engine
        self.ctx = ctx
        self.label = label
        # Torch streams are created cudaStreamNonBlocking, so they do NOT
        # implicitly order after the caller's stream; launch() issues an
        # explicit wait_stream before every enqueue.
        self.stream = stream if stream is not None else torch.cuda.Stream()
        self.graphs = LaunchGraphCache(
            cuda_graph_enabled() if cuda_graph is None else cuda_graph, capacity
        )
        self._bindings: list = []

    def bind(self, name: str, tensor: torch.Tensor, *, is_input: bool = True) -> None:
        """Bind ``tensor`` as the persistent buffer for ``name``. Inputs
        also set the context's input shape. Binding a name again replaces
        its entry and drops the captured launch."""
        if is_input:
            self.ctx.set_input_shape(name, tuple(tensor.shape))
        if not self.ctx.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"{self.label}: TensorRT rejected the address for {name!r}")
        for i, (bound, _) in enumerate(self._bindings):
            if bound == name:
                self._bindings[i] = (name, tensor)
                self.invalidate_graph()
                return
        self._bindings.append((name, tensor))

    @property
    def bindings(self) -> tuple:
        """``((name, tensor), ...)`` in bind order."""
        return tuple(self._bindings)

    def key(self) -> tuple:
        """What a captured launch is valid for (see :func:`launch_key`)."""
        return launch_key(self.engine, self.ctx, self.bindings)

    def invalidate_graph(self) -> None:
        """Drop the captured launch; the next launch re-captures."""
        self.graphs.lookup(None)

    def launch(self) -> None:
        """Run the engine once on the bound buffers and wait for it."""
        # The input copies ran on the caller's current stream; the dedicated
        # non-blocking stream would otherwise race them, so make the launch
        # wait for that stream first. Captured BEFORE the context switch:
        # inside it current_stream() is already self.stream, which would
        # make the wait a no-op.
        caller_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(caller_stream)
            if self.graphs.enabled:
                self._launch_graph()
            elif not self.ctx.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError(f"{self.label} launch failed")
        self.stream.synchronize()

    def _launch_graph(self) -> None:
        """Replay the captured launch, capturing first when there is none
        for the current key. Runs on (and must be called inside) the
        dedicated stream context. Capture is preceded by one plain execute:
        TensorRT does its lazy per-shape setup on the first enqueue, which
        must not land inside a capture. That execute's output is a full
        launch on the current inputs, and the replay right after recomputes
        the same values from the same buffers, so it is just overwritten.
        thread_local capture mode keeps CUDA calls on other server threads
        from being flagged as capture violations."""
        key = self.key()
        graph = self.graphs.lookup(key)
        if graph is None:
            if not self.ctx.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError(f"{self.label} launch failed")
            self.stream.synchronize()
            # capture_begin/end directly rather than the torch.cuda.graph
            # context manager, which adds a device-wide synchronize,
            # gc.collect() and empty_cache() around every capture. We are
            # already inside the dedicated stream context, and the capture
            # allocates nothing through torch (TensorRT's scratch belongs to
            # the execution context).
            g = torch.cuda.CUDAGraph()
            try:
                g.capture_begin(capture_error_mode="thread_local")
                try:
                    ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
                finally:
                    g.capture_end()
                if not ok:
                    raise RuntimeError(f"{self.label} capture enqueue failed")
            except Exception as exc:  # noqa: BLE001 - fall back, never fail the launch
                self.graphs.disable(f"capture_failed: {exc}")
                logger.warning(
                    "trt_launch_graph_disabled label={} error={}", self.label, exc,
                )
                # The warm-up execute above already produced this launch.
                return
            self.graphs.store(key, g)
            graph = g
            logger.info(
                "trt_launch_graph_captured label={} refit_epoch={} captures={}",
                self.label, key[1], self.graphs.captures,
            )
        graph.replay()
