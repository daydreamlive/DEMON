"""C2PA manifest round-trip through the track-asset WAV writers.

Pure CPU — no GPU, no model load. Exercises the real ``c2pa-python``
SDK (the ``provenance`` extra): :func:`acestep.track_assets.
write_track_wav` / :func:`write_stem_wavs` embed a signed manifest,
and we read it back with ``c2pa.Reader`` to assert the spec-02 §3
posture survives signing: the IPTC ``digitalSourceType``, the CAWG
training-and-data-mining assertion (``notAllowed`` everywhere), and
the custom ``com.daydream.session`` assertion.

The no-SDK degrade path is covered too (simulated by forcing the
import guard), because asset writing must behave identically when the
``provenance`` extra is absent.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

c2pa = pytest.importorskip("c2pa", reason="provenance extra not installed")

import acestep.provenance.manifest as manifest_mod
from acestep.provenance.manifest import (
    COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA,
    TRAINED_ALGORITHMIC_MEDIA,
    build_manifest_definition,
    embed_wav_manifest,
)
from acestep.track_assets import (
    save_track_metadata,
    source_audio_path,
    stem_audio_path,
    waveform_fingerprint,
    write_stem_wavs,
    write_track_wav,
)

SAMPLE_RATE = 48_000


@pytest.fixture(autouse=True)
def _isolated_provenance_dir(tmp_path, monkeypatch):
    """Every test signs with throwaway keys under its own tmp dir."""
    monkeypatch.setenv("ACESTEP_PROVENANCE_DIR", str(tmp_path / "provenance"))


def _waveform(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand((2, 4800), generator=g) * 0.5


def _read_active_manifest(path: Path) -> dict:
    with c2pa.Reader(str(path)) as reader:
        data = json.loads(reader.json())
    return data["manifests"][data["active_manifest"]]


def _assertion(active: dict, label: str) -> dict:
    matches = [a for a in active["assertions"] if a["label"].startswith(label)]
    assert matches, f"no {label!r} assertion in {active['assertions']}"
    assert len(matches) == 1
    return matches[0]["data"]


# ---------------------------------------------------------------------------
# Round trip through the writers
# ---------------------------------------------------------------------------


def test_write_track_wav_embeds_readable_manifest(tmp_path):
    root = tmp_path / "uploads"
    waveform = _waveform()

    write_track_wav(root, "song.wav", waveform=waveform, sample_rate=SAMPLE_RATE)

    p = source_audio_path(root, "song.wav")
    active = _read_active_manifest(p)

    # Claim generator identity: local self-signed, never Daydream.
    assert active["claim_generator_info"][0]["name"] == "DEMON (local, self-signed)"

    # c2pa.created with the composite digitalSourceType (the written
    # track IS seed audio, so it always carries the composite type).
    actions = _assertion(active, "c2pa.actions")["actions"]
    created = [a for a in actions if a["action"] == "c2pa.created"]
    assert len(created) == 1
    assert created[0]["digitalSourceType"] == COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA

    # CAWG do-not-train: every category notAllowed, on by default.
    entries = _assertion(active, "cawg.training-mining")["entries"]
    assert set(entries) == {
        "cawg.ai_generative_training",
        "cawg.ai_inference",
        "cawg.ai_training",
        "cawg.data_mining",
    }
    assert all(v == {"use": "notAllowed"} for v in entries.values())

    # com.daydream.session: no live session -> null identity fields,
    # but the seed fingerprint matches what track.json advertises.
    session = _assertion(active, "com.daydream.session")
    assert session["session_id"] is None
    assert session["model"] is None
    assert session["loras"] == []
    assert session["seed_waveform_sha256"] == waveform_fingerprint(waveform)

    # The signed WAV must still be a decodable WAV.
    data, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE
    assert data.shape == (4800, 2)


def test_write_stem_wavs_carries_seed_fingerprint_from_track_json(tmp_path):
    root = tmp_path / "uploads"
    waveform = _waveform()
    stems = {"vocals": waveform + 0.1, "instruments": waveform + 0.2}

    # Metadata first: the stem writer picks the seed fingerprint off disk.
    save_track_metadata(root, "song.wav", waveform=waveform, sample_rate=SAMPLE_RATE)
    write_stem_wavs(root, "song.wav", stems=stems, sample_rate=SAMPLE_RATE)

    for mode in ("vocals", "instruments"):
        active = _read_active_manifest(stem_audio_path(root, "song.wav", mode))
        actions = _assertion(active, "c2pa.actions")["actions"]
        assert actions[0]["digitalSourceType"] == COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA
        session = _assertion(active, "com.daydream.session")
        assert session["seed_waveform_sha256"] == waveform_fingerprint(waveform)


def test_write_stem_wavs_before_metadata_keeps_composite_type(tmp_path):
    """Production order (assets first, track.json last): no fingerprint
    on disk yet, but the source type stays composite — stems always
    derive from seed audio."""
    root = tmp_path / "uploads"
    waveform = _waveform()
    stems = {"vocals": waveform, "instruments": waveform}

    write_stem_wavs(root, "song.wav", stems=stems, sample_rate=SAMPLE_RATE)

    active = _read_active_manifest(stem_audio_path(root, "song.wav", "vocals"))
    actions = _assertion(active, "c2pa.actions")["actions"]
    assert actions[0]["digitalSourceType"] == COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA
    session = _assertion(active, "com.daydream.session")
    assert "seed_waveform_sha256" not in session


def test_embed_picks_up_live_session_summary(tmp_path):
    """With a session tap attached, the embedded com.daydream.session
    assertion carries that session's identity and counts."""
    from acestep.streaming.events import EventBus, PromptApplied
    from acestep.streaming import registry

    bus = EventBus()
    handle = registry.SessionHandle(
        id="prov-test-1",
        started_at=time.time(),
        inject=lambda data, audio: None,
        snapshot=lambda: {"lora_catalog": [{"id": "lead", "state": "enabled"}]},
        bus=bus,
        provenance_meta={"checkpoint": "ace_step_v1"},
    )
    registry.register(handle)
    try:
        bus.publish(PromptApplied(tags="dark techno"))
        root = tmp_path / "uploads"
        waveform = _waveform()
        write_track_wav(root, "live.wav", waveform=waveform, sample_rate=SAMPLE_RATE)
    finally:
        registry.unregister(handle.id)

    active = _read_active_manifest(source_audio_path(root, "live.wav"))
    session = _assertion(active, "com.daydream.session")
    assert session["session_id"] == "prov-test-1"
    assert session["model"] == "ace_step_v1"
    assert session["loras"] == ["lead"]
    assert session["session_log_sha256"]
    summary = session["timeline_summary"]
    assert summary["events"] >= 1
    assert isinstance(summary["duration_s"], float)


# ---------------------------------------------------------------------------
# Manifest definition builder (pure, no SDK involvement)
# ---------------------------------------------------------------------------


def test_build_manifest_definition_defaults_to_trained_media():
    manifest = build_manifest_definition(title="out.wav")
    actions = manifest["assertions"][0]["data"]["actions"]
    assert actions[0]["digitalSourceType"] == TRAINED_ALGORITHMIC_MEDIA


def test_build_manifest_definition_fingerprint_implies_composite():
    manifest = build_manifest_definition(
        title="out.wav", ingredient_fingerprint="ab" * 32,
    )
    actions = manifest["assertions"][0]["data"]["actions"]
    assert actions[0]["digitalSourceType"] == COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA
    session = manifest["assertions"][2]["data"]
    assert session["seed_waveform_sha256"] == "ab" * 32


def test_build_manifest_definition_explicit_source_type_wins():
    manifest = build_manifest_definition(
        title="out.wav",
        source_type=COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA,
    )
    actions = manifest["assertions"][0]["data"]["actions"]
    assert actions[0]["digitalSourceType"] == COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA


# ---------------------------------------------------------------------------
# Degrade path: writers must behave identically without the SDK
# ---------------------------------------------------------------------------


def test_writers_survive_missing_c2pa(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest_mod, "_import_c2pa", lambda: None)
    root = tmp_path / "uploads"
    waveform = _waveform()

    write_track_wav(root, "plain.wav", waveform=waveform, sample_rate=SAMPLE_RATE)

    p = source_audio_path(root, "plain.wav")
    data, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE and data.shape == (4800, 2)
    assert c2pa.Reader.try_create(str(p)) is None  # no manifest embedded
    assert not list(p.parent.glob("*.c2pa.tmp"))


def test_embed_failure_leaves_unsigned_wav_intact(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    waveform = _waveform()
    write_track_wav(root, "keep.wav", waveform=waveform, sample_rate=SAMPLE_RATE)
    p = source_audio_path(root, "keep.wav")
    before = p.read_bytes()

    def _boom(*args, **kwargs):
        raise RuntimeError("signer exploded")

    monkeypatch.setattr(manifest_mod, "build_manifest_definition", _boom)
    assert embed_wav_manifest(p) is False
    assert p.read_bytes() == before
    assert not list(p.parent.glob("*.c2pa.tmp"))
