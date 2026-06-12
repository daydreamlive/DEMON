from __future__ import annotations

import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import acestep.fixtures as fixtures_mod
from acestep.fixtures import fixture_stems, fixture_track_metadata
from acestep.nodes.types import Audio, Latent
from acestep.paths import user_uploads_dir
from acestep.streaming.source import _resolve_bpm_key_source
from acestep.track_assets import save_track_metadata, write_stem_wavs
from acestep.user_uploads import (
    enumerate_user_uploads,
    load_user_upload_stems,
    persist_user_upload_packet,
    unique_user_upload_name,
    user_upload_wipe_enabled,
)
from acestep.user_uploads import user_upload_audio, user_upload_sidecar


def _persist_stems(name, *, waveform, stems, sample_rate):
    """Write stem WAVs + track metadata, mirroring the production write
    order (assets first, metadata last) so the disk-derived manifest is
    honest. Replaces the removed save_user_upload_stems helper."""
    root = user_uploads_dir()
    write_stem_wavs(root, name, stems=stems, sample_rate=sample_rate)
    save_track_metadata(root, name, waveform=waveform, sample_rate=sample_rate)


def test_user_upload_auto_wipe_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DEMON_WIPE_USER_UPLOADS", raising=False)

    assert user_upload_wipe_enabled() is False


def test_user_upload_auto_wipe_requires_explicit_truthy_env(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("DEMON_WIPE_USER_UPLOADS", value)
        assert user_upload_wipe_enabled() is True

    for value in ("", "0", "false", "no", "off", "pod"):
        monkeypatch.setenv("DEMON_WIPE_USER_UPLOADS", value)
        assert user_upload_wipe_enabled() is False


def test_user_upload_stem_cache_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    waveform = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    stems = {
        "vocals": waveform + 1,
        "instruments": waveform + 2,
    }

    _persist_stems(
        "song.wav",
        waveform=waveform,
        stems=stems,
        sample_rate=48_000,
    )

    loaded = load_user_upload_stems("song.wav", waveform=waveform)

    assert loaded is not None
    assert torch.equal(loaded["vocals"], stems["vocals"])
    assert torch.equal(loaded["instruments"], stems["instruments"])
    assert not (tmp_path / "user_uploads" / "song.wav").is_file()
    assert (tmp_path / "user_uploads" / "song" / "track.json").is_file()
    assert (tmp_path / "user_uploads" / "song" / "stems" / "vocals.wav").is_file()
    assert (tmp_path / "user_uploads" / "song" / "stems" / "instruments.wav").is_file()
    assert not (tmp_path / "user_uploads" / "song" / "stems.safetensors").exists()


def test_user_upload_stem_cache_preserves_user_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    root = tmp_path / "user_uploads"
    root.mkdir()
    meta_path = root / "song" / "track.json"
    meta_path.parent.mkdir()
    meta_path.write_text(
        json.dumps({
            "display_name": "Edited title",
            "bpm": 123,
            "key": "D minor",
            "time_signature": "3",
        }),
        encoding="utf-8",
    )
    waveform = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    stems = {
        "vocals": waveform + 1,
        "instruments": waveform + 2,
    }

    _persist_stems(
        "song.wav",
        waveform=waveform,
        stems=stems,
        sample_rate=48_000,
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["display_name"] == "Edited title"
    assert meta["bpm"] == 123
    assert meta["key"] == "D minor"
    assert meta["time_signature"] == "3"
    assert meta["samples"] == 8


def test_user_upload_stem_cache_rejects_same_name_different_waveform(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    waveform = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    stems = {
        "vocals": waveform + 1,
        "instruments": waveform + 2,
    }
    _persist_stems(
        "song.wav",
        waveform=waveform,
        stems=stems,
        sample_rate=48_000,
    )

    changed = waveform.clone()
    changed[0, 0] += 1

    assert load_user_upload_stems("song.wav", waveform=changed) is None


def test_user_upload_stem_cache_rejects_missing_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    root = tmp_path / "user_uploads"
    root.mkdir()
    waveform = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    write_stem_wavs(
        root,
        "song.wav",
        stems={"vocals": waveform + 1, "instruments": waveform + 2},
        sample_rate=48_000,
    )

    assert load_user_upload_stems("song.wav", waveform=waveform) is None


def test_user_upload_packet_persists_complete_canonical_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    waveform = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    stems = {"vocals": waveform + 1, "instruments": waveform + 2}

    packet = persist_user_upload_packet(
        "song.wav",
        waveform=waveform,
        stems=stems,
        sources={
            "full": _prepared_source(1),
            "vocals": _prepared_source(2),
            "instruments": _prepared_source(3),
        },
        sample_rate=48_000,
        checkpoint="ckpt",
        bpm=120,
        key="C major",
        time_signature="4",
    )
    root = tmp_path / "user_uploads"

    assert packet.name == "song.wav"
    assert enumerate_user_uploads() == ["song.wav"]
    assert user_upload_audio("song.wav").is_file()
    assert user_upload_audio("song.wav") == root / "song" / "source.wav"
    for rel in (
        "source.wav",
        "track.json",
        "stems/vocals.wav",
        "stems/instruments.wav",
        "sidecars/full.json",
        "sidecars/full.safetensors",
        "sidecars/vocals.json",
        "sidecars/vocals.safetensors",
        "sidecars/instruments.json",
        "sidecars/instruments.safetensors",
    ):
        assert (root / "song" / rel).is_file()


def test_http_user_upload_route_serves_clean_source_wav(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    waveform = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    stems = {"vocals": waveform + 1, "instruments": waveform + 2}
    persist_user_upload_packet(
        "TomsDinerClip.wav",
        waveform=waveform,
        stems=stems,
        sources={
            "full": _prepared_source(1),
            "vocals": _prepared_source(2),
            "instruments": _prepared_source(3),
        },
        sample_rate=48_000,
        checkpoint="ckpt",
        bpm=120,
        key="C major",
        time_signature="4",
    )

    from demos.realtime_motion_graph_web.server import _process_request

    conn = type("Conn", (), {"remote_address": ("127.0.0.1", 1234)})()
    req = type(
        "Req",
        (),
        {"path": "/user_uploads/TomsDinerClip.wav", "headers": {}},
    )()

    res = _process_request(conn, req)

    assert res.status_code == 200
    assert bytes(res.body[:4]) == b"RIFF"
    assert user_upload_audio("TomsDinerClip.wav") == (
        tmp_path / "user_uploads" / "TomsDinerClip" / "source.wav"
    )


def test_unique_user_upload_name_uses_canonical_wav(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    root = tmp_path / "user_uploads"
    root.mkdir()
    (root / "song").mkdir()

    assert unique_user_upload_name("song.mp3") == "song (1).wav"


def test_source_mode_sidecar_lookup_uses_variant_name(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    root = tmp_path / "user_uploads"
    latent = torch.zeros(1, 2, 3)
    context = torch.ones(1, 4, 5)

    from acestep.sidecars import save_sidecar_pair

    sidecar_dir = root / "song" / "sidecars"
    save_sidecar_pair(
        sidecar_dir / "vocals.json",
        sidecar_dir / "vocals.safetensors",
        latent=latent,
        context_latent=context,
        checkpoint="ckpt",
        bpm=120,
        key="C major",
        time_signature="4",
        duration_s=1.0,
        samples=48_000,
        sample_rate=48_000,
        channels=2,
    )

    assert user_upload_sidecar("song.wav") is None
    sc = user_upload_sidecar("song.wav", "vocals")
    assert sc is not None
    assert sc.name == "song.wav.vocals"
    assert torch.equal(sc.latent, latent)


def test_fixture_stems_use_same_asset_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    root = tmp_path / "fixtures"
    name = "inside_confusion_loop_60s_gsm.wav"
    waveform = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    stems = {"vocals": waveform + 1, "instruments": waveform + 2}

    save_track_metadata(
        root,
        name,
        waveform=waveform,
        sample_rate=48_000,
        bpm=99,
        key="G# minor",
        time_signature="3",
    )
    write_stem_wavs(root, name, stems=stems, sample_rate=48_000)

    loaded = fixture_stems(name, waveform=waveform, sample_rate=48_000)

    assert loaded is not None
    assert torch.equal(loaded["vocals"], stems["vocals"])
    assert torch.equal(loaded["instruments"], stems["instruments"])
    assert fixture_track_metadata(name)["bpm"] == 99


def test_fixture_stems_fall_back_when_assets_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(fixtures_mod, "_resolve_fixture_asset", lambda *args, **kwargs: None)
    name = "inside_confusion_loop_60s_gsm.wav"
    waveform = torch.zeros(2, 8)

    assert fixture_stems(name, waveform=waveform, sample_rate=48_000) is None


class _FakeHandler:
    device = torch.device("cpu")
    dtype = torch.float32


class _FakeSession:
    handler = _FakeHandler()

    def __init__(self):
        self.prepare_calls = 0

    def prepare_source(self, audio):
        self.prepare_calls += 1
        return _prepared_source(fill=float(self.prepare_calls))


def _prepared_source(fill: float = 0.0):
    return type(
        "Prepared",
        (),
        {
            "latent": Latent(tensor=torch.full((1, 2, 3), fill)),
            "context_latent": Latent(tensor=torch.full((1, 4, 5), fill + 1)),
        },
    )()


def test_track_metadata_overrides_user_upload_sidecar_values(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    root = tmp_path / "user_uploads"
    name = "song.wav"
    waveform = torch.zeros(2, 48_000)
    root.mkdir()
    meta_path = root / "song" / "track.json"
    meta_path.parent.mkdir()
    meta_path.write_text(
        json.dumps({"bpm": 111, "key": "D minor", "time_signature": "3"}),
        encoding="utf-8",
    )
    from acestep.sidecars import save_sidecar_pair

    save_sidecar_pair(
        root / "song" / "sidecars" / "full.json",
        root / "song" / "sidecars" / "full.safetensors",
        latent=torch.zeros(1, 2, 3),
        context_latent=torch.ones(1, 4, 5),
        checkpoint="ckpt",
        bpm=120,
        key="C major",
        time_signature="4",
        duration_s=1.0,
        samples=48_000,
        sample_rate=48_000,
        channels=2,
    )

    source, bpm, key, time_signature = _resolve_bpm_key_source(
        _FakeSession(),
        audio_in=Audio(waveform=waveform, sample_rate=48_000),
        fixture_name=name,
        samples=48_000,
    )

    assert source.latent.tensor.shape == (1, 2, 3)
    assert (bpm, key, time_signature) == (111, "D minor", "3")


def test_live_user_upload_resolution_does_not_write_canonical_packet(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    name = "song.wav"
    root = tmp_path / "user_uploads"
    root.mkdir()
    waveform = torch.zeros(2, 48_000)
    meta_path = root / "song" / "track.json"
    meta_path.parent.mkdir()
    meta_path.write_text(
        json.dumps({"bpm": 112, "key": "E minor", "time_signature": "6"}),
        encoding="utf-8",
    )
    session = _FakeSession()

    _resolve_bpm_key_source(
        session,
        audio_in=Audio(waveform=waveform, sample_rate=48_000),
        fixture_name=name,
        samples=48_000,
    )

    assert session.prepare_calls == 1
    assert not (root / "song" / "source.wav").exists()
    assert not (root / "song" / "sidecars" / "full.json").exists()
    assert not (root / "song" / "sidecars" / "full.safetensors").exists()
    meta = json.loads((root / "song" / "track.json").read_text(encoding="utf-8"))
    assert (meta["bpm"], meta["key"], meta["time_signature"]) == (
        112,
        "E minor",
        "6",
    )
