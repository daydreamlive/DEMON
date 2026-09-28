"""Pure text-to-audio sessions: no source upload, synthesised null anchor.

``config.text_only`` tells the adapter that no binary PCM frame is
coming. Getting the gate wrong is not a cosmetic bug in either
direction: skip a frame the client DID send and every control message
after it is read as audio, and the session desynchronises; wait for a
frame the client did NOT send and ``ws.recv()`` blocks forever, because
the plugin has no handshake timeout to rescue it.

The gate is therefore two keys, not one. ``supports_text_only`` rides on
``init_ack``, which is emitted only when the config carries
``telemetry_version`` — so a client without it never saw the advert,
cannot know the frame is optional, and must still be sending one.

No GPU, no model load: the helpers under test are pure functions of the
config dict, same pattern as test_swap_resize_session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.streaming.config import SessionConfig
from acestep.streaming.families import TextOnlySpec, get_family
from acestep.streaming.source import SAMPLE_RATE
from demos.realtime_motion_graph_web.ws_adapter import (
    _effective_family,
    _silent_source_waveform,
    _text_only_requested,
    _text_only_spec,
)

SA3 = get_family("sa3").text_only
ACE = get_family("acestep").text_only


def test_both_keys_required_to_skip_the_upload():
    assert _text_only_requested({"text_only": True, "telemetry_version": 1})


def test_text_only_without_telemetry_still_reads_the_frame():
    # The client never saw init_ack, so it cannot know the frame is
    # optional -- it is sending one, and skipping it would desync.
    assert not _text_only_requested({"text_only": True})


def test_old_clients_are_untouched():
    assert not _text_only_requested({})
    assert not _text_only_requested({"telemetry_version": 1})
    assert not _text_only_requested({"text_only": False, "telemetry_version": 1})


def test_anchor_is_synthesised_at_the_requested_render_length():
    # Length matters: sa3_session derives source_duration_s from this, and
    # matching the requested render keeps the two in agreement.
    wf = _silent_source_waveform({"sa3_duration_s": 30.0}, SA3)
    assert wf.shape == (2, int(30.0 * SAMPLE_RATE))
    assert not wf.any(), "the null anchor must be exactly silent"


def test_anchor_falls_back_when_no_duration_declared():
    wf = _silent_source_waveform({}, SA3)
    assert wf.shape[1] == int(SA3.default_duration_s * SAMPLE_RATE)


def test_anchor_is_clamped_and_survives_junk():
    assert _silent_source_waveform({"sa3_duration_s": 99999.0}, SA3).shape[1] <= int(
        SA3.max_duration_s * SAMPLE_RATE)
    for junk in (None, "", "abc", -5.0, 0.0):
        wf = _silent_source_waveform({"sa3_duration_s": junk}, SA3)
        assert wf.shape[1] > 0


def test_ace_anchor_is_the_family_default_and_ignores_the_sa3_field():
    # ACE has no per-session duration field; the anchor is always its
    # default (the 60 s TRT profile every pod builds), as it was before.
    assert ACE.duration_field is None
    wf = _silent_source_waveform({"sa3_duration_s": 120.0}, ACE)
    assert wf.shape[1] == int(60.0 * SAMPLE_RATE)


def test_text_only_spec_validates_and_clamps():
    spec = TextOnlySpec(default_duration_s=10.0, max_duration_s=20.0, duration_field="len")
    assert spec.duration_s({}) == 10.0
    assert spec.duration_s({"len": 15}) == 15.0
    assert spec.duration_s({"len": 999}) == 20.0
    assert spec.duration_s({"len": 0.2}) == 1.0
    assert spec.duration_s({"len": "junk"}) == 10.0
    import pytest
    with pytest.raises(ValueError):
        TextOnlySpec(default_duration_s=30.0, max_duration_s=20.0)


def test_effective_family_and_advert_follow_the_spec():
    # The client's explicit backend wins; else the pod's family.
    assert _effective_family({"backend": "sa3"}, "acestep") == "sa3"
    assert _effective_family({}, "acestep") == "acestep"
    assert _text_only_spec({}, "sa3") is SA3
    assert _text_only_spec({"backend": "acestep"}, "sa3") is ACE
    # An unknown family advertises no text-only; create fails loudly later.
    assert _text_only_spec({"backend": "nope"}, "sa3") is None


def test_config_carries_text_only_and_defaults_off():
    assert SessionConfig.from_dict({}).text_only is False
    assert SessionConfig.from_dict({"text_only": True}).text_only is True
