"""Unit coverage for the slice-stream backpressure drop decision.

The failure mode under test: on a link that can't drain the slice stream
(~1-2.5 MB/s of heavily overlapping windows), the per-subscriber send
queue accumulates seconds of slices, each superseded by the next, and
every patch lands behind the playhead — heard as the raw source. The
serializer sheds windowed slices BEFORE encoding (so the delta mirror
only advances for slices actually sent) via two layers: an end-to-end
in-flight byte window (client acks ``slice_bytes_rx``) and a bus-queue
age backstop for the case TCP itself pushes back.

``_windowed_slice_drop_reason`` is the pure decision those layers run;
its end-to-end behavior is also exercised by
``scripts/flow_control_harness.py`` (manual, needs a live server).
"""

from demos.realtime_motion_graph_web.ws_adapter import (
    _windowed_slice_drop_reason,
)

_WINDOW = 256 * 1024
_MAX_AGE = 2.0


def _decide(acked, sent, age_s=0.0):
    return _windowed_slice_drop_reason(
        acked=acked,
        sent=sent,
        window_bytes=_WINDOW,
        age_s=age_s,
        max_age_s=_MAX_AGE,
    )


def test_healthy_link_sends():
    # A few slices in flight, fresh from the queue → no drop.
    assert _decide(acked=900_000, sent=900_000 + 50_000, age_s=0.01) is None


def test_window_exceeded_drops():
    # sent-minus-acked past the window → shed (the load-bearing layer).
    reason, detail = _decide(acked=0, sent=_WINDOW + 1)
    assert reason == "window"
    assert detail == float(_WINDOW + 1)


def test_at_window_boundary_sends():
    # Strictly greater-than: exactly at the window is still sent.
    assert _decide(acked=0, sent=_WINDOW) is None


def test_no_ack_disables_window_layer_only():
    # Old client (never acked): the window layer can't engage no matter
    # how much has been sent, but the age backstop still applies.
    assert _decide(acked=None, sent=10 * _WINDOW, age_s=0.01) is None
    reason, _ = _decide(acked=None, sent=10 * _WINDOW, age_s=_MAX_AGE + 0.1)
    assert reason == "age"


def test_age_backstop_drops():
    # Within the window but queued too long (send thread blocked) → shed.
    reason, detail = _decide(acked=100, sent=200, age_s=_MAX_AGE + 0.5)
    assert reason == "age"
    assert detail == _MAX_AGE + 0.5


def test_window_checked_before_age():
    # Both layers tripped → window wins (it sheds the unbounded backlog
    # before the age backstop is even consulted).
    reason, _ = _decide(acked=0, sent=_WINDOW + 1, age_s=_MAX_AGE + 5.0)
    assert reason == "window"
