"""Per-session stream geometry and plugin params for stacked-stream
layouts. Tier 1, no GPU.

* ``StreamLayout.tokens_per_stream`` may be a callable ``(duration_s) -> L``:
  the conditioning capture and the duration clamp follow each session's
  requested duration through it.
* A session's ``plugin_*`` config keys reach ``SessionConfig.plugin_params``
  (never the wire contract) and the layout's ``make_dit`` through the
  context view's ``session_params``.
"""

from __future__ import annotations

import pytest

from acestep.engine import sa3_trt
from acestep.plugins import StreamLayout
from acestep.streaming.config import SessionConfig
from tests.unit.test_sa3_song_length import DS, stub_vendor  # noqa: F401
from tests.unit.test_sa3_stream_layout import N_STREAMS, _LayoutRuntime, _bare_context


def _l_for(seconds: float) -> int:
    return int(round(seconds * 44100 / DS))


def test_tokens_per_stream_accepts_a_rule():
    layout = StreamLayout(n_streams=2, tokens_per_stream=_l_for)
    assert layout.stream_tokens(10.0) == _l_for(10.0)
    assert StreamLayout(n_streams=2, tokens_per_stream=7).stream_tokens(99.0) == 7
    assert StreamLayout(n_streams=2).stream_tokens(10.0) is None
    with pytest.raises(ValueError):
        StreamLayout(n_streams=2, tokens_per_stream=lambda d: 0).stream_tokens(1.0)
    with pytest.raises(ValueError):
        StreamLayout(n_streams=2, tokens_per_stream=0)


def test_context_follows_the_rule_per_duration(stub_vendor, monkeypatch):
    monkeypatch.setattr(sa3_trt, "max_dit_engine_latents", lambda _m: 1)
    c = _bare_context(_LayoutRuntime({"tokens_per_stream": _l_for}))
    for seconds in (5.0, 12.5, 30.0):
        d = c.clamp_duration_for_trt(seconds, backend="eager")
        assert d == pytest.approx(min(seconds, _l_for(seconds) * DS / 44100))
        cond = c.prepare_cond(prompt="p", duration=d, steps=8)
        assert cond.latent_frames == N_STREAMS * _l_for(seconds)
        assert cond.audio_sample_size == _l_for(seconds) * DS


def test_session_config_collects_plugin_params_off_the_wire():
    cfg = SessionConfig.from_dict({
        "prompt": "p", "plugin_x_engine": "on", "plugin_y": 3,
        "plugin_z": {"nested": 1}, "other": "ignored",
    })
    assert cfg.plugin_params == {"plugin_x_engine": "on", "plugin_y": 3}
    assert SessionConfig.from_dict({}).plugin_params == {}
    from demos.realtime_motion_graph_web.protocol import config_catalog

    assert "plugin_params" not in config_catalog()


def test_make_dit_sees_the_session_params():
    seen = []

    def _make_dit(context, latent_frames, seconds_total, backend):
        seen.append(dict(context.session_params))
        return None

    c = _bare_context(_LayoutRuntime({"make_dit": _make_dit}))
    assert c.session_params == {}
    c.make_dit(latent_frames=24, seconds_total=30.0, backend="eager")
    view = c.with_session_params({"plugin_x_engine": "off"})
    view.make_dit(latent_frames=24, seconds_total=30.0, backend="eager")
    # A label view keeps the params, and params keep the label.
    labelled = view.with_song_seconds(120.0)
    assert labelled.session_params == {"plugin_x_engine": "off"}
    assert labelled.song_seconds == 120.0
    assert labelled.with_session_params({"plugin_q": 1}).song_seconds == 120.0
    assert seen == [{}, {"plugin_x_engine": "off"}]
    assert c.session_params == {}            # the shared context is untouched
