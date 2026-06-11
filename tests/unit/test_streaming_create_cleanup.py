from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from acestep.nodes.types import Audio
from acestep.streaming.config import SessionConfig
from acestep.streaming.session import StreamingSession


class _FakeEngine:
    _trt_engine = None
    lora_available = False


class _FakeSession:
    def __init__(self, tracker, **_kwargs):
        self.closed = False
        self.handler = SimpleNamespace(_diffusion_engine=_FakeEngine())
        tracker.sessions.append(self)

    def close(self):
        self.closed = True

    def encode_text(self, **_kwargs):
        return object()

    def stream(self, **_kwargs):
        stream = _FakeStream()
        return stream


class _FakeStream:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _source():
    return SimpleNamespace(
        latent=object(),
        context_latent=object(),
    )


def _audio():
    return Audio(
        waveform=torch.zeros((2, 9600), dtype=torch.float32),
        sample_rate=48_000,
    )


def _patch_lightweight_create(monkeypatch, tracker):
    import acestep.streaming.session as session_mod

    monkeypatch.setattr(session_mod, "max_profile_duration_s", lambda **_kwargs: 1.0)
    monkeypatch.setattr(
        session_mod,
        "Session",
        lambda **kwargs: _FakeSession(tracker, **kwargs),
    )
    monkeypatch.setattr(
        session_mod,
        "_resolve_bpm_key_source",
        lambda *_args, **_kwargs: (_source(), 120, "C major", "4"),
    )
    monkeypatch.setattr(
        session_mod,
        "extract_and_select_upload_stem",
        lambda waveform, **_kwargs: (None, None, _source(), waveform),
    )
    return session_mod


def test_create_closes_engine_session_when_conditioning_fails(monkeypatch):
    tracker = SimpleNamespace(sessions=[])
    session_mod = _patch_lightweight_create(monkeypatch, tracker)

    def fail_encode_cond_pair(*_args, **_kwargs):
        raise RuntimeError("conditioning failed")

    monkeypatch.setattr(session_mod, "encode_cond_pair", fail_encode_cond_pair)

    with pytest.raises(RuntimeError, match="conditioning failed"):
        StreamingSession.create(
            audio=_audio(),
            config=SessionConfig(),
            checkpoint="ckpt",
            decoder_backend="eager",
            vae_backend="eager",
            session_id="s1",
        )

    assert tracker.sessions
    assert tracker.sessions[0].closed


def test_create_closes_stream_and_engine_session_when_late_setup_fails(monkeypatch):
    tracker = SimpleNamespace(sessions=[], streams=[])
    session_mod = _patch_lightweight_create(monkeypatch, tracker)

    def fake_stream(self, **_kwargs):
        stream = _FakeStream()
        tracker.streams.append(stream)
        return stream

    monkeypatch.setattr(_FakeSession, "stream", fake_stream)
    monkeypatch.setattr(
        session_mod,
        "encode_cond_pair",
        lambda *_args, **_kwargs: (object(), object()),
    )

    def fail_audio_engine(*_args, **_kwargs):
        raise RuntimeError("audio engine failed")

    monkeypatch.setattr(session_mod, "AudioEngine", fail_audio_engine)

    with pytest.raises(RuntimeError, match="audio engine failed"):
        StreamingSession.create(
            audio=_audio(),
            config=SessionConfig(),
            checkpoint="ckpt",
            decoder_backend="eager",
            vae_backend="eager",
            session_id="s1",
        )

    assert tracker.streams
    assert tracker.streams[0].closed
    assert tracker.sessions
    assert tracker.sessions[0].closed
