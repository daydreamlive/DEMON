"""Typed events + bounded fan-out event bus for the streaming session.

The :class:`~acestep.streaming.session.StreamingSession` publishes typed
events; transport adapters subscribe via an :class:`EventBus` that fans
them out through bounded per-subscriber queues.

Field shapes follow ``notes/api_layer_protocol_matrix.md`` Section 3:
the dataclass is the wire's superset (the adapter may serialize only a
subset to JSON). Numpy arrays carried inside events are produced once
on the runner thread; subscribers must not mutate them in place.

Backpressure model:

- Each :class:`Subscription` owns a bounded buffer (default 256) and a
  daemon drainer thread that invokes the listener callback.
- :meth:`EventBus.publish` is non-blocking. The runner thread never
  waits on a subscriber. A slow remote subscriber cannot stall GPU
  ticks; it sees drops, coalesces, or close-on-overflow per its
  configured :class:`BackpressurePolicy`.
- :class:`AudioReady` events drop oldest-first when full. Best-effort
  recovery: under the current runner's sliding-window decode (stride
  well below ``vae_window``) a dropped buffer region is typically
  re-covered by the next overlapping slice. Persistent saturation
  compounds drops, so stale regions can survive until a later
  overlapping slice lands or the WS reconnects. A runner with stride
  near or above the window would need close-on-overflow for audio
  instead. :class:`ParamsUpdate` and :class:`ParamsEcho` coalesce
  (only the latest value matters). Control-plane events
  (``SwapReady``, ``LoraCatalogUpdate``, ``SessionReady``, etc.)
  never drop; an overfull subscription is closed instead, after a
  single :class:`SubscriberDropped` is delivered to the listener.

Today the only subscriber is the demo's WS adapter and the buffer
rarely fills, so this is structural-readiness for the future
remote-transport case.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

import numpy as np


__all__ = [
    "BackpressurePolicy",
    "EventBus",
    "Subscription",
    "SubscriptionClosed",
    "AudioReady",
    "CommandFailed",
    "ParamsUpdate",
    "ParamsEcho",
    "PromptApplied",
    "PromptBlendEcho",
    "LoraCatalogUpdate",
    "DepthApplied",
    "SwapReady",
    "SwapFailed",
    "StemAssets",
    "StemFailed",
    "TimbreSet",
    "TimbreCleared",
    "TimbreFailed",
    "StructureSet",
    "StructureCleared",
    "StructureFailed",
    "SessionReady",
    "SessionError",
    "SubscriberDropped",
]


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------
#
# Sibling dataclasses, no shared base class. Dispatch via ``isinstance``
# at the subscriber. Each dataclass carries exactly the fields the
# corresponding wire message (or the corresponding API contract) needs,
# per ``notes/api_layer_protocol_matrix.md``.
#
# Numpy fields are read-only by convention: subscribers MUST treat the
# arrays as immutable. Producing a fresh array is the publisher's
# responsibility when a mutation would be needed.


@dataclass(frozen=True)
class AudioReady:
    """One decoded slice from the streaming pipeline.

    Carries the raw audio plus the runtime telemetry that the wire's
    ``params_update`` JSON ships immediately after the binary slice.
    The session has already mutated server-side ring state
    (``audio_eng.swap`` for full-buffer decodes) before publishing, so
    subscribers see one event per delivered slice with no implied
    side effect.
    """

    audio: np.ndarray              # float32 [N, C]
    start_sample: int              # offset into the full playback buffer
    num_samples: int               # == audio.shape[0]
    channels: int
    tick_ms: float
    dec_ms: float
    num_gens: int
    params: dict                   # snapshot of state.params at slice time


@dataclass(frozen=True)
class ParamsUpdate:
    """Runtime telemetry snapshot. Today wired in alongside ``AudioReady``;
    kept as a separate event so future transports can subscribe to one
    without the other."""

    params: dict


@dataclass(frozen=True)
class ParamsEcho:
    """Echo of a knob update that came from a non-primary origin.

    Emitted ONLY when ``set_knobs(origin=CommandOrigin.EXTERNAL)``.
    The primary transport's UI layer (today: the browser's smoothing
    tween) listens for this so it can ease toward the target and
    re-send the tweened sequence as a ``PRIMARY`` ``set_knobs``.

    Carries the raw dict as posted, NOT the per-knob clamped value.
    """

    raw: dict


@dataclass(frozen=True)
class PromptApplied:
    """The named prompt (A) was re-encoded and is now live."""

    tags: str


@dataclass(frozen=True)
class PromptBlendEcho:
    """Echo of a prompt-blend slider target from a non-primary origin.
    Same shape as :class:`ParamsEcho`."""

    value: float


@dataclass(frozen=True)
class LoraCatalogUpdate:
    """The engine's LoRA catalog was refreshed (typically after enable /
    disable refit). ``catalog`` shape matches ``ready.lora_catalog``."""

    catalog: list


@dataclass(frozen=True)
class DepthApplied:
    """Pipeline depth retune landed. ``value`` is the clamped
    actually-applied depth."""

    value: int


@dataclass(frozen=True)
class ManualSlotCount:
    """Manual steering slot count after a manual_slot_add / manual_slot_pop
    (published on success AND refusal so the client's +/- UI resyncs
    either way). ``count`` is the controller's live slot count."""

    count: int


@dataclass(frozen=True)
class SwapReady:
    """Source swap completed. Carries enough state for the transport to
    crossfade into the new buffer and update its detected-metadata UI.

    ``initial_buffer`` is the new source's full playback buffer (float32
    [N, C]); transports serialize it as a follow-up binary frame.

    ``stems`` / ``stem_error`` carry the post-swap stem-extraction
    outcome bundled inside this event so a transport can ship the
    swap_ready JSON, the new-buffer binary, and either the stem_assets
    JSON + binary follow-ups OR the stem_failed JSON as one atomic
    sequence under its send lock. (The init handshake's stem payload
    does NOT flow through the bus; it ships inline before the
    subscriber is attached.)
    """

    duration: float
    sample_rate: int
    channels: int
    bpm: int
    key: str
    time_signature: str
    fixture_name: str | None
    initial_buffer: np.ndarray
    stems: dict | None = None              # {name: np.ndarray | torch.Tensor}
    stem_source_mode: str | None = None
    stem_error: str | None = None


@dataclass(frozen=True)
class SwapFailed:
    """Source swap aborted. ``build_command`` is populated only for the
    ``EngineNotBuiltError`` case so operators see exactly the build
    incantation that would unblock the swap."""

    error: str
    build_command: str | None = None


@dataclass(frozen=True)
class StemAssets:
    """Vocals / instruments stem buffers extracted from the active source.

    ``stems`` is keyed by name in display order; transports that ship
    the stems as binary follow-up frames must walk the dict in
    insertion order to match the wire's index alignment.
    """

    fixture_name: str | None
    source_mode: str               # "full" | "vocals" | "instruments"
    sample_rate: int
    channels: int
    frames: int
    stems: dict                    # {name: np.ndarray[float32, (C, N)]}


@dataclass(frozen=True)
class StemFailed:
    fixture_name: str | None
    error: str


@dataclass(frozen=True)
class TimbreSet:
    name: str
    duration: float                # post-cap clip seconds


@dataclass(frozen=True)
class TimbreCleared:
    pass


@dataclass(frozen=True)
class TimbreFailed:
    error: str


@dataclass(frozen=True)
class StructureSet:
    """Structure (semantic hint) ref applied.

    Wire payload (per ``api_layer_protocol_matrix.md`` Finding 3)
    carries only ``name`` and ``duration``. The log line additionally
    reports the post-pad/trim target length; transports that want it
    can read ``StreamingSession.snapshot()`` instead of widening this
    event.
    """

    name: str
    duration: float


@dataclass(frozen=True)
class StructureCleared:
    pass


@dataclass(frozen=True)
class StructureFailed:
    error: str


@dataclass(frozen=True)
class CommandFailed:
    """A capability-gated command was rejected because the session's
    backend doesn't declare the required capability (plan §3.4: loud
    failure, never a silent no-op).

    ``command`` is the wire command name (the ``requires``-tagged
    :class:`CommandSpec` in ``demos/realtime_motion_graph_web/
    protocol.py``); ``requires`` is the :class:`~acestep.streaming.
    generator_backend.Capabilities` field the command needs; ``error``
    is the human-readable reason (the mapped
    :class:`~acestep.streaming.generator_backend.UnsupportedOperation`
    message). With only the ACE backend registered this is wired but
    never fires for capability reasons other than ``lora`` on a
    LoRA-disabled session.
    """

    command: str
    requires: str
    error: str


@dataclass(frozen=True)
class SessionReady:
    """First event the session publishes once setup completes.

    Carries everything the transport needs to ship the wire's ``ready``
    JSON plus the binary initial-buffer follow-up. Adapter-supplied
    fields (``session_id``, ``checkpoint``, ``lora_dir``) are NOT here
    because the adapter already has them; this event carries only
    session-resolved values.
    """

    duration: float
    sample_rate: int
    channels: int
    bpm: int
    key: str
    time_signature: str
    pipeline_depth: int
    max_pipeline_depth: int
    checkpoint_scale: str | None
    lora_catalog: list
    lora_pending_enable: list
    initial_buffer: np.ndarray     # float32 [N, C]


@dataclass(frozen=True)
class SessionError:
    """Generic session-level error event. Init-time failures are raised
    as exceptions from ``StreamingSession.create``; this event is for
    runtime errors that don't stop the session."""

    code: str
    message: str
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SubscriberDropped:
    """Per-subscription terminal event. Delivered to a listener ONCE
    before its subscription is closed because its bounded queue stayed
    saturated past the policy's threshold. Never published to the bus."""

    reason: str


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------


class BackpressurePolicy(Enum):
    """How a :class:`Subscription` handles a publish into a full queue.

    On overflow the queue never evicts a ``NEVER_DROP`` event to make
    room for any other event (see :meth:`Subscription._post`); the
    eviction target is always the oldest *droppable* event.

    - ``NEVER_DROP``: never evicted, and never evicts. When a
      ``NEVER_DROP`` event arrives at a full queue with no droppable
      event to evict, the subscription is closed after delivering one
      :class:`SubscriberDropped`. Use for control-plane events where
      missing one corrupts the consumer.
    - ``DROP_OLDEST``: on full, evict the oldest *droppable* queued event
      to make room; if none exists (backlog is all control events), drop
      the incoming event instead. Use for high-rate recoverable events
      (slices).
    - ``COALESCE``: on full, walk the queue back-to-front and replace
      the most recent event of the SAME concrete type with the new one
      (so the consumer always sees the latest value). Falls back to
      ``DROP_OLDEST`` semantics if no same-type event is queued.
    """

    NEVER_DROP = "never_drop"
    DROP_OLDEST = "drop_oldest"
    COALESCE = "coalesce"


# Default per-event-type policies. Subscribers can override per
# subscription, but the bus uses these when no per-type policy was set.
_DEFAULT_POLICY: dict[type, BackpressurePolicy] = {
    AudioReady: BackpressurePolicy.DROP_OLDEST,
    ParamsUpdate: BackpressurePolicy.COALESCE,
    ParamsEcho: BackpressurePolicy.COALESCE,
    PromptBlendEcho: BackpressurePolicy.COALESCE,
}


def _policy_for(event: Any, override: BackpressurePolicy | None) -> BackpressurePolicy:
    if override is not None:
        return override
    return _DEFAULT_POLICY.get(type(event), BackpressurePolicy.NEVER_DROP)


class SubscriptionClosed(Exception):
    """Raised when a closed subscription is published to. Internal."""


class Subscription:
    """Bounded buffer + daemon drainer for one listener.

    Owns its own queue (under a lock — a plain :class:`queue.Queue` lacks
    the introspection needed for ``COALESCE`` and ``DROP_OLDEST``) and
    one drainer thread that invokes the listener callback in FIFO order.
    The listener runs on the drainer thread; if it blocks or raises,
    only this subscription is affected.
    """

    def __init__(
        self,
        listener: Callable[[Any], None],
        *,
        queue_size: int = 256,
        policy: BackpressurePolicy | None = None,
        name: str = "subscription",
    ) -> None:
        self._listener = listener
        self._max = max(1, int(queue_size))
        self._policy_override = policy
        self._name = name

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._buf: deque[Any] = deque()
        self._closed = False
        # When True, the drainer has already been told to deliver one
        # SubscriberDropped event and then exit. Guards against publishing
        # additional events after the close signal latches.
        self._closing = False

        self._thread = threading.Thread(
            target=self._drain, name=f"event-sub-{name}", daemon=True,
        )
        self._thread.start()

    # ---- bus-side -------------------------------------------------------

    def _post(self, event: Any) -> None:
        """Push an event into this subscription's queue. Non-blocking.

        Called only by :meth:`EventBus.publish` while holding no lock
        beyond the bus's subscriber-list lock. Honors the per-event
        policy on overflow.

        Overflow rule (correct-by-construction): a full queue never
        evicts a ``NEVER_DROP`` event to make room. Instead we drop the
        OLDEST *droppable* (non-``NEVER_DROP``) event, so a flood of
        high-rate slices can't knock a queued control frame
        (``SwapReady``, ``LoraCatalogUpdate``, …) out of the queue —
        which the old evict-absolute-oldest path could, silently losing
        an unrecoverable control event. The subscription is closed only
        when the backlog is *entirely* control events AND the incoming
        event is itself ``NEVER_DROP`` (a consumer so wedged it can't
        even drain the control plane). A lossy event arriving against an
        all-control backlog is simply dropped — losing a slice is
        recoverable; losing a control frame is not.
        """
        policy = _policy_for(event, self._policy_override)
        with self._cv:
            if self._closed or self._closing:
                return
            if len(self._buf) < self._max:
                self._buf.append(event)
                self._cv.notify()
                return
            # Queue full.
            #
            # COALESCE: replace the newest same-type event in place. This
            # doesn't grow the queue and never displaces a control event,
            # so try it first; the consumer always sees the latest value.
            if policy is BackpressurePolicy.COALESCE:
                etype = type(event)
                # Walk back-to-front so the newest same-type event is
                # the one we replace (consumers always see the latest).
                for i in range(len(self._buf) - 1, -1, -1):
                    if type(self._buf[i]) is etype:
                        self._buf[i] = event
                        return
            # Make room by dropping the OLDEST droppable event. Never
            # evict a NEVER_DROP (control-plane) event to do it.
            for i, queued in enumerate(self._buf):
                if _policy_for(
                    queued, self._policy_override,
                ) is not BackpressurePolicy.NEVER_DROP:
                    del self._buf[i]
                    self._buf.append(event)
                    self._cv.notify()
                    return
            # Backlog is entirely NEVER_DROP events (consumer wedged on
            # the control plane).
            if policy is not BackpressurePolicy.NEVER_DROP:
                # Incoming is lossy (e.g. an audio slice). Dropping it is
                # recoverable; keep the control backlog intact.
                return
            # Incoming is itself NEVER_DROP and there's no droppable event
            # to evict — close the subscription. Tell the drainer to
            # deliver one SubscriberDropped and exit. We do this by
            # clearing the buffer (which would otherwise stack
            # unservable events) and queueing the single notice.
            self._closing = True
            self._buf.clear()
            self._buf.append(SubscriberDropped(
                reason=f"queue overflow on {type(event).__name__}",
            ))
            self._cv.notify()

    # ---- listener-side --------------------------------------------------

    def close(self) -> None:
        """Stop draining; the drainer thread exits after delivering any
        already-queued events."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    @property
    def name(self) -> str:
        return self._name

    # ---- internal -------------------------------------------------------

    def _drain(self) -> None:
        while True:
            with self._cv:
                while not self._buf and not self._closed:
                    self._cv.wait()
                if not self._buf and self._closed:
                    return
                event = self._buf.popleft()
                closing_after = self._closing and not self._buf
            try:
                self._listener(event)
            except Exception:  # noqa: BLE001
                # Subscriber callbacks must not take down the drainer.
                # We deliberately swallow; the subscriber is expected to
                # have its own logging.
                pass
            if closing_after:
                with self._cv:
                    self._closed = True
                return


class EventBus:
    """Fan-out bus for typed events.

    The runner thread (publisher) never blocks on subscribers. Each
    subscription holds its own bounded queue and a daemon drainer
    thread that invokes the listener.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: list[Subscription] = []
        self._closed = False

    def subscribe(
        self,
        listener: Callable[[Any], None],
        *,
        queue_size: int = 256,
        policy: BackpressurePolicy | None = None,
        name: str = "subscription",
    ) -> Subscription:
        sub = Subscription(
            listener, queue_size=queue_size, policy=policy, name=name,
        )
        with self._lock:
            if self._closed:
                sub.close()
                return sub
            self._subs.append(sub)
        return sub

    def publish(self, event: Any) -> None:
        """Non-blocking fan-out to every subscriber.

        Caller MUST treat any numpy fields on ``event`` as immutable
        after the call.
        """
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            sub._post(event)

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            try:
                self._subs.remove(sub)
            except ValueError:
                pass
        sub.close()

    def close(self) -> None:
        """Stop accepting new subscriptions and close every existing one.
        Drainer threads exit after their current queue empties."""
        with self._lock:
            self._closed = True
            subs = list(self._subs)
            self._subs.clear()
        for sub in subs:
            sub.close()

    def subscribers(self) -> Iterable[Subscription]:
        with self._lock:
            return list(self._subs)
