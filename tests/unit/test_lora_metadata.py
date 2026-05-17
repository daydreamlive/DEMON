"""Tests for :mod:`acestep.lora_metadata`.

The loader's contract is "always returns a record" — these tests
exercise the three documented degradation paths (full sidecar, legacy
.trigger.txt only, neither), the malformed-sidecar warning fallback,
schema-version drift, primary/trigger-words mismatch, and the
``(path, mtime_ns)`` cache.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.lora_metadata import (
    CURRENT_SCHEMA_VERSION,
    LoraMetadata,
    clear_cache,
    load_lora_metadata,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()
    yield
    clear_cache()


def _weights(tmp_path: Path, stem: str = "industrial") -> Path:
    """Create a zero-byte stand-in for the LoRA weights file. The loader
    never reads the safetensors itself; it just uses the path to derive
    the sidecar location."""
    p = tmp_path / f"{stem}.safetensors"
    p.write_bytes(b"")
    return p


def _full_metadata(stem: str = "industrial") -> dict:
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "id": f"{stem}-v1",
        "name": "Industrial",
        "description": "Mechanical percussion, distorted synths.",
        "model": {
            "type": "lora",
            "base_model": "AceStep v1.5 Turbo",
            "base_model_scale": "2B",
            "architecture": "diffusion-transformer",
            "format": "safetensors",
        },
        "inference": {
            "trigger_words": ["roti-1ndstrl"],
            "primary_trigger_word": "roti-1ndstrl",
            "recommended_strength": 1.0,
            "recommended_steps": 8,
            "recommended_shift": 3.0,
            "recommended_guidance": 7.0,
        },
        "classification": {
            "primary_genre": "industrial",
            "secondary_genres": [],
            "tags": ["electronic", "percussive"],
            "moods": ["dark"],
        },
    }


def test_full_metadata_parses(tmp_path):
    weights = _weights(tmp_path)
    sidecar = tmp_path / "industrial.metadata.json"
    sidecar.write_text(json.dumps(_full_metadata()), encoding="utf-8")

    md = load_lora_metadata(weights)

    assert md.has_metadata is True
    assert md.id == "industrial"  # always the stem, not the sidecar id
    assert md.name == "Industrial"
    assert md.description == "Mechanical percussion, distorted synths."
    assert md.primary_trigger_word == "roti-1ndstrl"
    assert md.trigger_words == ["roti-1ndstrl"]
    assert md.recommended_strength == 1.0
    assert md.recommended_steps == 8
    assert md.recommended_shift == 3.0
    assert md.recommended_guidance == 7.0
    assert md.primary_genre == "industrial"
    assert md.tags == ["electronic", "percussive"]
    assert md.moods == ["dark"]


def test_legacy_trigger_txt_only(tmp_path):
    weights = _weights(tmp_path, stem="phonk")
    (tmp_path / "phonk.trigger.txt").write_text("roti-ph0nk\n", encoding="utf-8")

    md = load_lora_metadata(weights)

    assert md.has_metadata is False
    assert md.id == "phonk"
    assert md.name == "phonk"
    assert md.primary_trigger_word == "roti-ph0nk"
    assert md.trigger_words == ["roti-ph0nk"]
    assert md.recommended_strength is None
    assert md.description is None


def test_bare_safetensors(tmp_path):
    weights = _weights(tmp_path, stem="bare")

    md = load_lora_metadata(weights)

    assert md.has_metadata is False
    assert md.id == "bare"
    assert md.name == "bare"
    assert md.primary_trigger_word is None
    assert md.trigger_words == []


def test_malformed_json_falls_back_to_trigger_txt(tmp_path, caplog):
    weights = _weights(tmp_path, stem="broken")
    (tmp_path / "broken.metadata.json").write_text("{not valid json", encoding="utf-8")
    (tmp_path / "broken.trigger.txt").write_text("fallback-word", encoding="utf-8")

    with caplog.at_level("WARNING"):
        md = load_lora_metadata(weights)

    # The bad JSON gets warned about, but the loader returns the legacy
    # synthesized record instead of raising.
    assert md.has_metadata is False
    assert md.primary_trigger_word == "fallback-word"
    assert any("unreadable" in rec.message for rec in caplog.records)


def test_malformed_json_no_trigger_txt_returns_bare(tmp_path, caplog):
    weights = _weights(tmp_path, stem="broken2")
    (tmp_path / "broken2.metadata.json").write_text("garbage", encoding="utf-8")

    with caplog.at_level("WARNING"):
        md = load_lora_metadata(weights)

    assert md.has_metadata is False
    assert md.primary_trigger_word is None
    assert md.trigger_words == []


def test_unknown_schema_version_warns_but_parses(tmp_path, caplog):
    weights = _weights(tmp_path, stem="future")
    raw = _full_metadata("future")
    raw["schema_version"] = 99
    (tmp_path / "future.metadata.json").write_text(json.dumps(raw), encoding="utf-8")

    with caplog.at_level("WARNING"):
        md = load_lora_metadata(weights)

    assert md.has_metadata is True
    assert md.primary_trigger_word == "roti-1ndstrl"
    assert any("schema_version" in rec.message for rec in caplog.records)


def test_primary_not_in_trigger_words_warns_but_keeps_primary(tmp_path, caplog):
    weights = _weights(tmp_path, stem="mismatch")
    raw = _full_metadata("mismatch")
    raw["inference"]["trigger_words"] = ["abc"]
    raw["inference"]["primary_trigger_word"] = "xyz"
    (tmp_path / "mismatch.metadata.json").write_text(json.dumps(raw), encoding="utf-8")

    with caplog.at_level("WARNING"):
        md = load_lora_metadata(weights)

    assert md.primary_trigger_word == "xyz"
    assert md.trigger_words == ["abc"]
    assert any(
        "primary_trigger_word" in rec.message for rec in caplog.records
    )


def test_primary_missing_falls_back_to_first_trigger(tmp_path):
    weights = _weights(tmp_path, stem="nopri")
    raw = _full_metadata("nopri")
    raw["inference"]["trigger_words"] = ["first", "second"]
    del raw["inference"]["primary_trigger_word"]
    (tmp_path / "nopri.metadata.json").write_text(json.dumps(raw), encoding="utf-8")

    md = load_lora_metadata(weights)

    assert md.primary_trigger_word == "first"
    assert md.trigger_words == ["first", "second"]


def test_sidecar_id_mismatch_keeps_stem(tmp_path, caplog):
    weights = _weights(tmp_path, stem="actual-stem")
    raw = _full_metadata("actual-stem")
    raw["id"] = "different-id"
    (tmp_path / "actual-stem.metadata.json").write_text(
        json.dumps(raw), encoding="utf-8"
    )

    with caplog.at_level("WARNING"):
        md = load_lora_metadata(weights)

    assert md.id == "actual-stem"
    assert any("sidecar id" in rec.message for rec in caplog.records)


def test_mtime_cache_invalidates_on_change(tmp_path):
    weights = _weights(tmp_path, stem="industrial")
    sidecar = tmp_path / "industrial.metadata.json"
    sidecar.write_text(json.dumps(_full_metadata()), encoding="utf-8")

    md1 = load_lora_metadata(weights)
    assert md1.recommended_strength == 1.0

    # Touch + rewrite with a new strength. The cache key is (path,
    # mtime_ns) so the rewrite should be picked up; if we accidentally
    # key on something coarser this assertion catches it.
    raw2 = _full_metadata()
    raw2["inference"]["recommended_strength"] = 0.5
    # Bump mtime explicitly because on some filesystems back-to-back
    # writes can land in the same mtime_ns.
    import os
    import time
    time.sleep(0.01)
    sidecar.write_text(json.dumps(raw2), encoding="utf-8")
    new_t = time.time()
    os.utime(sidecar, (new_t, new_t))

    md2 = load_lora_metadata(weights)
    assert md2.recommended_strength == 0.5


def test_missing_optional_fields_default_to_none(tmp_path):
    weights = _weights(tmp_path, stem="sparse")
    raw = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "inference": {"primary_trigger_word": "go", "trigger_words": ["go"]},
    }
    (tmp_path / "sparse.metadata.json").write_text(json.dumps(raw), encoding="utf-8")

    md = load_lora_metadata(weights)

    assert md.has_metadata is True
    assert md.id == "sparse"
    assert md.name == "sparse"
    assert md.description is None
    assert md.primary_trigger_word == "go"
    assert md.recommended_strength is None
    assert md.primary_genre is None
    assert md.tags == []


def test_base_model_scale_round_trips(tmp_path):
    weights = _weights(tmp_path, stem="xl_one")
    raw = _full_metadata("xl_one")
    raw["model"]["base_model_scale"] = "5B"
    raw["model"]["base_model"] = "AceStep v1.5 XL Turbo"
    (tmp_path / "xl_one.metadata.json").write_text(json.dumps(raw), encoding="utf-8")

    md = load_lora_metadata(weights)

    assert md.base_model_scale == "5B"
    assert md.base_model == "AceStep v1.5 XL Turbo"


def test_model_block_missing_yields_null_scale(tmp_path):
    weights = _weights(tmp_path, stem="nomodel")
    raw = _full_metadata("nomodel")
    del raw["model"]
    (tmp_path / "nomodel.metadata.json").write_text(json.dumps(raw), encoding="utf-8")

    md = load_lora_metadata(weights)

    # Sparse model block must NOT crash; callers treat None as
    # "compatible with everything" so they don't hide the LoRA.
    assert md.base_model is None
    assert md.base_model_scale is None


def test_empty_strings_filtered_from_trigger_lists(tmp_path):
    weights = _weights(tmp_path, stem="empties")
    raw = _full_metadata("empties")
    raw["inference"]["trigger_words"] = ["good", "", "  ".strip()]
    raw["classification"]["tags"] = ["a", "", None]
    (tmp_path / "empties.metadata.json").write_text(json.dumps(raw), encoding="utf-8")

    md = load_lora_metadata(weights)

    assert md.trigger_words == ["good"]
    assert md.tags == ["a"]
