"""Params-coalescing rule for the WS recv loop.

When a backlog of ~125 Hz ``params`` messages forms, the recv loop keeps only
the newest snapshot — except for ``slice_lead_s``, which the wire contract
defines as the WORST lead observed since the previous report. The feedback
controller widens playback lead on that worst value, so the minimum lead must
be folded forward across a coalesced run rather than overwritten by
newest-wins. ``_coalesced_slice_lead`` is the pure helper that decides it.
"""

from demos.realtime_motion_graph_web.ws_adapter import _coalesced_slice_lead


def test_min_of_two_leads():
    assert _coalesced_slice_lead(-1.25, 0.40) == -1.25
    assert _coalesced_slice_lead(0.40, -1.25) == -1.25


def test_carries_forward_when_new_is_omitted():
    # A superseding report with no slice_lead_s must not erase the worst lead
    # the previous report carried.
    assert _coalesced_slice_lead(-0.75, None) == -0.75


def test_takes_new_when_prev_is_omitted():
    assert _coalesced_slice_lead(None, -0.75) == -0.75


def test_none_when_neither_carries_a_lead():
    assert _coalesced_slice_lead(None, None) is None


def test_reviewer_repro_sequence():
    # Reviewer's queued sequence: leads -1.25, omitted, then 0.40. Coalescing
    # them newest-first must surface -1.25 (the worst), not 0.40.
    pending = {"slice_lead_s": -1.25}  # msg 1
    # msg 2: no slice_lead_s
    carried = _coalesced_slice_lead(pending.get("slice_lead_s"), None)
    pending = {"slice_lead_s": carried} if carried is not None else {}
    # msg 3: lead 0.40
    carried = _coalesced_slice_lead(pending.get("slice_lead_s"), 0.40)
    assert carried == -1.25
