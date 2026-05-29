"""Unit tests for the streaming :class:`EventBus` backpressure semantics.

Pure Python — no GPU, no model load. Exercises the overflow path of
:class:`acestep.streaming.events.Subscription`, which is exactly the
behavior that isn't reachable by the GPU-gated browser smoke test.

The contract under test: a full queue never evicts a ``NEVER_DROP``
(control-plane) event to make room. A flood of high-rate ``DROP_OLDEST``
slices must not knock a queued ``SwapReady`` (or any other control frame)
out of the queue; the subscription is closed only when the backlog is
entirely control events and an incoming control event can't be queued.

Determinism: the drainer runs on its own thread, so each test *parks* it
inside the listener (blocked on a gate) before filling the queue. With
the drainer parked, every subsequent ``publish`` lands in the bounded
buffer and the overflow logic runs synchronously on the publishing
thread, so the queue state is fully determined when we assert.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.streaming.events import (
    AudioReady,
    BackpressurePolicy,
    EventBus,
    ParamsEcho,
    SubscriberDropped,
    SwapReady,
)


# ---------------------------------------------------------------------------
# Event factories (minimal — the bus dispatches on type + policy only)
# ---------------------------------------------------------------------------


def _audio(tag: int = 0) -> AudioReady:
    """A DROP_OLDEST event. ``tag`` rides on ``num_gens`` so a test can
    tell which slices were dropped."""
    return AudioReady(
        audio=None, start_sample=tag, num_samples=0, channels=2,
        tick_ms=0.0, dec_ms=0.0, num_gens=tag, params={},
    )


def _swap(tag: int = 0) -> SwapReady:
    """A NEVER_DROP control event. ``tag`` rides on ``fixture_name``."""
    return SwapReady(
        duration=1.0, sample_rate=48000, channels=2, bpm=120, key="C",
        time_signature="4", fixture_name=str(tag), initial_buffer=None,
    )


def _echo(tag: int = 0) -> ParamsEcho:
    """A COALESCE event."""
    return ParamsEcho(raw={"n": tag})


# ---------------------------------------------------------------------------
# Parking harness
# ---------------------------------------------------------------------------


class _Harness:
    """Subscribes a listener that blocks on a gate so the drainer can be
    parked deterministically while a test fills the queue."""

    def __init__(self, bus: EventBus, queue_size: int,
                 policy: BackpressurePolicy | None = None):
        self.received: list = []
        self._entered = threading.Event()
        self._gate = threading.Event()
        self.sub = bus.subscribe(
            self._listener, queue_size=queue_size, policy=policy,
            name="test",
        )

    def _listener(self, ev) -> None:
        self.received.append(ev)
        self._entered.set()
        # Block every delivery until the test releases the gate. Once
        # released, queued events flush straight through.
        self._gate.wait(timeout=5)

    def park(self, bus: EventBus, parker) -> None:
        """Publish one event and block until the drainer is parked inside
        the listener holding it. Afterwards the buffer is empty and every
        further publish exercises the overflow path synchronously."""
        bus.publish(parker)
        assert self._entered.wait(timeout=5), "drainer never entered listener"

    def release_and_join(self) -> None:
        """Let the parked listener return, flush the queue, and join the
        drainer so ``received`` is final."""
        self._gate.set()
        self.sub.close()
        self.sub.join(timeout=5)


def _swap_tags(received) -> list:
    return [e.fixture_name for e in received if isinstance(e, SwapReady)]


def _audio_tags(received) -> list:
    return [e.num_gens for e in received if isinstance(e, AudioReady)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_under_capacity_delivers_in_order():
    """Sanity: below capacity, every event is delivered FIFO, untouched."""
    bus = EventBus()
    h = _Harness(bus, queue_size=8)
    h.park(bus, _audio(0))
    bus.publish(_swap(1))
    bus.publish(_audio(2))
    bus.publish(_echo(3))
    h.release_and_join()

    assert _swap_tags(h.received) == ["1"]
    assert _audio_tags(h.received) == [0, 2]
    assert any(isinstance(e, ParamsEcho) for e in h.received)


def test_drop_oldest_evicts_oldest_audio():
    """Pure-audio overflow drops the OLDEST slice (unchanged behavior for
    the all-droppable case)."""
    bus = EventBus()
    h = _Harness(bus, queue_size=2)
    h.park(bus, _audio(0))
    bus.publish(_audio(1))   # buf: [1]
    bus.publish(_audio(2))   # buf: [1, 2] full
    bus.publish(_audio(3))   # overflow -> evict oldest (1) -> [2, 3]
    h.release_and_join()

    delivered = _audio_tags(h.received)
    assert 1 not in delivered, "oldest slice should have been dropped"
    assert delivered == [0, 2, 3]


def test_control_event_survives_audio_flood():
    """THE regression test: a queued SwapReady must not be evicted by a
    flood of DROP_OLDEST slices. The pre-fix code evicted the absolute
    oldest, which could be the control event at the head."""
    bus = EventBus()
    h = _Harness(bus, queue_size=4)
    h.park(bus, _audio(-1))
    # Fill: control frame at the head, then audio behind it.
    bus.publish(_swap(99))   # buf: [swap99]
    bus.publish(_audio(0))   # buf: [swap99, a0]
    bus.publish(_audio(1))   # buf: [swap99, a0, a1]
    bus.publish(_audio(2))   # buf: [swap99, a0, a1, a2] full
    # Flood: each must evict an audio slice, never the swap.
    for t in range(3, 30):
        bus.publish(_audio(t))
    h.release_and_join()

    assert _swap_tags(h.received) == ["99"], (
        "control event was dropped by the audio flood"
    )
    assert not any(isinstance(e, SubscriberDropped) for e in h.received)


def test_incoming_control_evicts_audio_not_close():
    """An incoming NEVER_DROP event meeting a full queue that still holds
    droppable audio evicts the oldest audio — it does NOT tear the
    subscription down (the pre-fix code closed in this case)."""
    bus = EventBus()
    h = _Harness(bus, queue_size=3)
    h.park(bus, _audio(-1))
    bus.publish(_audio(0))
    bus.publish(_audio(1))
    bus.publish(_audio(2))   # full, all droppable
    bus.publish(_swap(7))    # incoming control -> evict oldest audio (0)
    h.release_and_join()

    assert _swap_tags(h.received) == ["7"]
    assert not any(isinstance(e, SubscriberDropped) for e in h.received)
    assert 0 not in _audio_tags(h.received)


def test_all_control_backlog_drops_incoming_lossy():
    """When the backlog is entirely control events, an incoming lossy
    slice is dropped — the control backlog stays intact and the
    subscription is NOT closed."""
    bus = EventBus()
    h = _Harness(bus, queue_size=2)
    h.park(bus, _swap(-1))
    bus.publish(_swap(1))    # buf: [swap1]
    bus.publish(_swap(2))    # buf: [swap1, swap2] full, all NEVER_DROP
    bus.publish(_audio(9))   # incoming lossy -> dropped
    h.release_and_join()

    assert _swap_tags(h.received) == ["-1", "1", "2"]
    assert _audio_tags(h.received) == [], "lossy event should be dropped"
    assert not any(isinstance(e, SubscriberDropped) for e in h.received)


def test_all_control_backlog_incoming_control_closes():
    """When the backlog is entirely control events AND another control
    event arrives, the subscription closes after delivering exactly one
    SubscriberDropped (the consumer is wedged on the control plane)."""
    bus = EventBus()
    h = _Harness(bus, queue_size=2)
    h.park(bus, _swap(-1))
    bus.publish(_swap(1))    # buf: [swap1]
    bus.publish(_swap(2))    # buf: [swap1, swap2] full, all NEVER_DROP
    bus.publish(_swap(3))    # incoming control, nothing droppable -> close
    h.release_and_join()

    assert isinstance(h.received[-1], SubscriberDropped)
    assert sum(isinstance(e, SubscriberDropped) for e in h.received) == 1
    # The unservable backlog (swap1, swap2) is cleared on close.
    assert _swap_tags(h.received) == ["-1"]


def test_coalesce_keeps_latest_same_type():
    """COALESCE replaces the newest same-type event in place on overflow,
    so the consumer always sees the latest value and the queue never
    grows past its bound."""
    bus = EventBus()
    h = _Harness(bus, queue_size=2)
    h.park(bus, _audio(-1))
    bus.publish(_echo(1))    # buf: [echo1]
    bus.publish(_echo(2))    # buf: [echo1, echo2] full
    bus.publish(_echo(3))    # COALESCE -> replace newest same-type (2)
    bus.publish(_echo(4))    # COALESCE -> replace newest same-type (3)
    h.release_and_join()

    echo_vals = [e.raw["n"] for e in h.received if isinstance(e, ParamsEcho)]
    assert echo_vals == [1, 4], f"expected latest coalesced value, got {echo_vals}"


def test_publish_after_close_is_ignored():
    """A closed subscription silently drops further publishes."""
    bus = EventBus()
    h = _Harness(bus, queue_size=4)
    h.park(bus, _audio(0))
    h.release_and_join()
    # Bus still references the (now closed) sub; publishing must not raise.
    bus.publish(_swap(1))
    bus.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
