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
    canonical_sa3_lineage,
    clear_cache,
    load_lora_metadata,
    sniff_lora_file,
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


def test_dotted_stem_resolves_sidecar(tmp_path):
    """A filename with a dot in the stem (e.g. a version like
    ``acestep1.5``) must still resolve its sibling sidecar. The old
    ``with_suffix("").with_suffix(...)`` trick truncated at the internal
    dot and silently dropped the metadata, so the UI showed the raw stem
    instead of the display name."""
    stem = "alt_pop50-acestep1.5-dora-v2"
    weights = _weights(tmp_path, stem=stem)
    (tmp_path / f"{stem}.metadata.json").write_text(
        json.dumps(_full_metadata(stem)), encoding="utf-8"
    )
    (tmp_path / f"{stem}.trigger.txt").write_text("ignored", encoding="utf-8")

    md = load_lora_metadata(weights)

    assert md.has_metadata is True
    assert md.id == stem
    assert md.name == "Industrial"  # came from the sidecar, not the stem
    assert md.primary_trigger_word == "roti-1ndstrl"


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


# ---------------------------------------------------------------------------
# Header sniffing: lora_family + SA3 lora_config synthesis
# ---------------------------------------------------------------------------
#
# These build REAL (tiny) safetensors files because the sniff reads the
# actual on-disk header; the zero-byte _weights stand-in above stays
# valid for the sidecar-only tests (an unreadable header degrades to
# family=None, never raises).

import os
import time

import torch
from safetensors.torch import save_file


def _bump_mtime(p: Path) -> None:
    """Force a distinct mtime_ns so back-to-back writes can't land in
    the same filesystem timestamp granule."""
    t = time.time() + 1.0
    os.utime(p, (t, t))


def _write_sa3_lora(
    tmp_path: Path,
    stem: str = "sa3lora",
    config: dict | None = None,
    omit_config: bool = False,
) -> Path:
    """Tiny SA3-format LoRA: parametrize-style keys at index 0, fp16,
    lora_config JSON embedded in the safetensors header — exactly the
    layout the vendored stable-audio-3 trainer saves."""
    prefix = "model.transformer.layers.0.self_attn.to_qkv.parametrizations.weight.0"
    tensors = {
        f"{prefix}.lora_A": torch.zeros(4, 16, dtype=torch.float16),
        f"{prefix}.lora_B": torch.zeros(16, 4, dtype=torch.float16),
    }
    if config is None:
        config = {
            "rank": 4,
            "alpha": 4,
            "adapter_type": "lora",
            "base_model": "medium-base",
        }
    metadata = None if omit_config else {"lora_config": json.dumps(config)}
    p = tmp_path / f"{stem}.safetensors"
    save_file(tensors, str(p), metadata=metadata)
    return p


def _write_ace_lora(tmp_path: Path, stem: str = "acelora") -> Path:
    """Tiny ACE/PEFT-format LoRA: lora_A.weight / lora_B.weight pairs."""
    tensors = {
        "base_model.model.q.lora_A.weight": torch.zeros(4, 16),
        "base_model.model.q.lora_B.weight": torch.zeros(8, 4),
    }
    p = tmp_path / f"{stem}.safetensors"
    save_file(tensors, str(p))
    return p


def test_sniff_sa3_family_and_config(tmp_path):
    p = _write_sa3_lora(tmp_path)
    family, cfg = sniff_lora_file(p)
    assert family == "sa3"
    assert cfg["rank"] == 4
    assert cfg["adapter_type"] == "lora"
    assert cfg["base_model"] == "medium-base"

    md = load_lora_metadata(p)
    assert md.lora_family == "sa3"
    assert md.adapter_type == "lora"
    assert md.rank == 4
    assert md.base_model == "medium-base"  # synthesized: no sidecar
    assert md.has_metadata is False


def test_sniff_sa3_rank_inferred_without_config(tmp_path):
    """No embedded lora_config: family still detected from key shapes
    and rank inferred from the first adapter tensor (upstream's
    infer_global_rank rule: lora_A shape[0])."""
    p = _write_sa3_lora(tmp_path, omit_config=True)
    family, cfg = sniff_lora_file(p)
    assert family == "sa3"
    assert cfg == {"rank": 4}

    md = load_lora_metadata(p)
    assert md.lora_family == "sa3"
    assert md.rank == 4
    assert md.adapter_type is None
    assert md.base_model is None


def test_sniff_ace_family(tmp_path):
    p = _write_ace_lora(tmp_path)
    family, cfg = sniff_lora_file(p)
    assert family == "ace"
    assert cfg == {}

    md = load_lora_metadata(p)
    assert md.lora_family == "ace"
    assert md.adapter_type is None
    assert md.rank is None


def test_sniff_non_lora_safetensors_is_family_none(tmp_path):
    p = tmp_path / "weights.safetensors"
    save_file({"encoder.weight": torch.zeros(2, 2)}, str(p))
    family, cfg = sniff_lora_file(p)
    assert family is None
    assert cfg == {}
    assert load_lora_metadata(p).lora_family is None


def test_sniff_unreadable_header_degrades_to_none(tmp_path):
    """Zero-byte and garbage files must never raise — the sniff runs
    on every catalog refresh."""
    empty = tmp_path / "empty.safetensors"
    empty.write_bytes(b"")
    garbage = tmp_path / "garbage.safetensors"
    garbage.write_bytes(b"\xff" * 64)

    assert sniff_lora_file(empty) == (None, {})
    assert sniff_lora_file(garbage) == (None, {})
    assert load_lora_metadata(empty).lora_family is None
    assert load_lora_metadata(garbage).lora_family is None


def test_sidecar_wins_display_fields_family_stays_header_truth(tmp_path):
    """A sidecar owns display/trigger/base_model fields, but the
    weight-format family always comes from the header — a sidecar
    cannot claim a different format than the file actually has."""
    p = _write_sa3_lora(tmp_path, stem="styled")
    raw = _full_metadata("styled")
    raw["model"]["base_model"] = "sidecar-says-this"
    (tmp_path / "styled.metadata.json").write_text(
        json.dumps(raw), encoding="utf-8"
    )

    md = load_lora_metadata(p)

    assert md.has_metadata is True
    assert md.name == "Industrial"  # sidecar display name wins
    assert md.base_model == "sidecar-says-this"  # sidecar wins over header
    assert md.lora_family == "sa3"  # header truth
    assert md.adapter_type == "lora"
    assert md.rank == 4


def test_cache_invalidates_on_weights_replacement(tmp_path):
    """Replacing the weights file (same path, new content + mtime) must
    re-sniff: header-derived fields would otherwise go stale. This is
    exactly what the old sidecar-only cache key got wrong."""
    p = _write_ace_lora(tmp_path, stem="swapme")
    assert load_lora_metadata(p).lora_family == "ace"

    _write_sa3_lora(tmp_path, stem="swapme")
    _bump_mtime(p)

    assert load_lora_metadata(p).lora_family == "sa3"


def test_cache_invalidates_when_sidecar_appears(tmp_path):
    """Bare records are now cached (keyed on the weights mtime), so a
    sidecar appearing later must change the key and be picked up."""
    p = _write_ace_lora(tmp_path, stem="latecomer")
    md1 = load_lora_metadata(p)
    assert md1.has_metadata is False

    sidecar = tmp_path / "latecomer.metadata.json"
    sidecar.write_text(json.dumps(_full_metadata("latecomer")), encoding="utf-8")
    _bump_mtime(sidecar)

    md2 = load_lora_metadata(p)
    assert md2.has_metadata is True
    assert md2.name == "Industrial"
    assert md2.lora_family == "ace"


def test_cache_invalidates_when_trigger_txt_appears(tmp_path):
    """Same for the legacy .trigger.txt sidecar: the old code never
    cached bare records so a trigger appearing was picked up instantly;
    the composite key must preserve that behavior."""
    p = _write_ace_lora(tmp_path, stem="trig")
    assert load_lora_metadata(p).primary_trigger_word is None

    trigger = tmp_path / "trig.trigger.txt"
    trigger.write_text("late-word", encoding="utf-8")
    _bump_mtime(trigger)

    assert load_lora_metadata(p).primary_trigger_word == "late-word"


def test_to_wire_carries_family_fields(tmp_path):
    wire = load_lora_metadata(_write_sa3_lora(tmp_path)).to_wire()
    assert wire["lora_family"] == "sa3"
    assert wire["adapter_type"] == "lora"
    assert wire["rank"] == 4


# ---------------------------------------------------------------------------
# canonical_sa3_lineage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling,expected",
    [
        # runtime ids pass through
        ("medium", "medium"),
        ("small-music", "small-music"),
        ("small-sfx", "small-sfx"),
        # trainer "-base" spellings
        ("medium-base", "medium"),
        ("small-music-base", "small-music"),
        ("small-sfx-base", "small-sfx"),
        # full HF repo ids, with and without the org prefix
        ("stabilityai/stable-audio-3-medium", "medium"),
        ("stabilityai/stable-audio-3-medium-base", "medium"),
        ("stable-audio-3-small-music-base", "small-music"),
        # underfit's short trainer spelling (seen in the wild)
        ("sa3-medium", "medium"),
        ("sa3-small-music-base", "small-music"),
        # case-insensitive, whitespace-tolerant
        ("Medium-Base", "medium"),
        ("  small-music  ", "small-music"),
        # unrecognized -> None (permissive downstream)
        ("acestep-v15-turbo", None),
        ("large", None),
        ("", None),
        (None, None),
    ],
)
def test_canonical_sa3_lineage(spelling, expected):
    assert canonical_sa3_lineage(spelling) == expected
