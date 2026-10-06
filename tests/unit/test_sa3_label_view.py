"""Per-session song-length label (sa3_song_seconds) over a shared context."""

from acestep.engine.sa3_context import (
    DEFAULT_LOOP_WRAP_S,
    LEGACY_OUTRO_PAD_S,
    SONG_SECONDS_MAX,
    SA3Context,
    SA3LabelView,
)
from acestep.streaming.config import SessionConfig


def _bare_context(song_seconds=180.0, outro_pad_s=DEFAULT_LOOP_WRAP_S):
    ctx = object.__new__(SA3Context)
    ctx.song_seconds = song_seconds
    ctx.outro_pad_s = outro_pad_s
    ctx.model_id = "small-sfx"
    return ctx


def test_true_length_view_relabels_without_touching_the_shared_context():
    base = _bare_context()
    view = base.with_song_seconds(None)
    assert isinstance(view, SA3LabelView)
    assert view.cond_seconds_total(4.0) == 4.0
    assert view.outro_pad_s == LEGACY_OUTRO_PAD_S
    assert base.cond_seconds_total(4.0) == 180.0
    assert base.outro_pad_s == DEFAULT_LOOP_WRAP_S
    assert view.model_id == "small-sfx"


def test_numeric_label_is_clamped_and_writes_reach_the_shared_context():
    base = _bare_context()
    view = base.with_song_seconds(1000.0)
    assert view.song_seconds == SONG_SECONDS_MAX
    assert view.cond_seconds_total(20.0) == SONG_SECONDS_MAX
    view.some_cache = {"k": 1}
    assert base.some_cache == {"k": 1}
    assert base.song_seconds == 180.0


def test_session_config_parses_the_label_field():
    cfg = SessionConfig.from_dict({"prompt": "rain", "sa3_song_seconds": 0})
    assert cfg.family_config["sa3_song_seconds"] == 0.0
    cfg = SessionConfig.from_dict({"prompt": "rain"})
    assert cfg.family_config["sa3_song_seconds"] is None
