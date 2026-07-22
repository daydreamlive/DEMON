"""Server-side LoRA reference resolution + catalog compatibility.

Covers the two pieces that moved LoRA name/scale resolution from
clients into the server:

* ``lora_scale_compatible`` — the scale axis of the backend-owned
  ``lora_compatible`` predicate (permissive on unknowns).
* ``resolve_lora_reference`` — exact catalog id first, then a
  case-insensitive stem/display-name alias lookup restricted to the
  compatible subset (mirrors the demo webapp's ``buildLoraAliasMap``).
* The session plumbing that uses them: ``lora_catalog_payload``
  annotating each entry with the backend verdict, and
  ``_resolve_lora_id`` mapping aliases to canonical ids. Exercised
  against stub engine/backend objects — no GPU, no model load.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.lora_metadata import clear_cache, lora_scale_compatible
from acestep.streaming.session import StreamingSession, resolve_lora_reference


# ---------------------------------------------------------------------------
# lora_scale_compatible
# ---------------------------------------------------------------------------


class TestLoraScaleCompatible:
    def test_matching_scales(self):
        assert lora_scale_compatible("2B", "2B")
        assert lora_scale_compatible("5B", "5B")

    def test_mismatched_scales(self):
        assert not lora_scale_compatible("2B", "5B")
        assert not lora_scale_compatible("5B", "2B")

    def test_unknown_on_either_side_is_compatible(self):
        assert lora_scale_compatible(None, "2B")
        assert lora_scale_compatible("2B", None)
        assert lora_scale_compatible(None, None)
        assert lora_scale_compatible("", "2B")


# ---------------------------------------------------------------------------
# resolve_lora_reference
# ---------------------------------------------------------------------------

_ENTRIES = [
    ("ambient-v1", "Ambient", True),
    ("ambient-xl-v1", "Ambient", False),
    ("deep_house-v1", "Deep House", True),
    ("metalcore-xl-v1", "Metalcore", False),
    ("bare-lora", None, True),
]


class TestResolveLoraReference:
    def test_exact_id_wins(self):
        assert resolve_lora_reference("ambient-v1", _ENTRIES) == "ambient-v1"

    def test_exact_id_wins_even_when_incompatible(self):
        # Exact-id clients keep today's behavior, including today's
        # engine-side refusal for an entry the engine can't load.
        assert (
            resolve_lora_reference("ambient-xl-v1", _ENTRIES)
            == "ambient-xl-v1"
        )

    def test_stem_alias_is_case_insensitive(self):
        assert resolve_lora_reference("AMBIENT-V1", _ENTRIES) == "ambient-v1"

    def test_display_name_resolves_to_compatible_variant(self):
        # "Ambient" names both scale variants; only the compatible one
        # enters the alias map, so the reference lands on it.
        assert resolve_lora_reference("Ambient", _ENTRIES) == "ambient-v1"
        assert resolve_lora_reference("ambient", _ENTRIES) == "ambient-v1"

    def test_alias_of_incompatible_only_entry_misses(self):
        # "Metalcore" exists only as an incompatible variant: alias
        # lookup is restricted to the compatible subset, so no match.
        assert resolve_lora_reference("Metalcore", _ENTRIES) is None
        assert resolve_lora_reference("metalcore", _ENTRIES) is None

    def test_unknown_reference_misses(self):
        assert resolve_lora_reference("synthwave", _ENTRIES) is None

    def test_none_display_name_is_safe(self):
        assert resolve_lora_reference("bare-lora", _ENTRIES) == "bare-lora"
        assert resolve_lora_reference("BARE-LORA", _ENTRIES) == "bare-lora"


# ---------------------------------------------------------------------------
# Session plumbing (stubbed engine/backend, unbound methods)
# ---------------------------------------------------------------------------


class _Desc:
    def __init__(self, path: Path):
        self.id = path.stem
        self.name = path.stem
        self.path = str(path)
        self.state = "registered"
        self.strength = 0.0
        self.materialized_bytes = 0


class _StubEngine:
    def __init__(self, descs):
        self._descs = descs

    def list_loras(self):
        return self._descs


class _ScaleBackend:
    """Backend stub whose predicate is the ACE scale axis for a 2B
    checkpoint."""

    def lora_compatible(self, metadata: dict) -> bool:
        return lora_scale_compatible(metadata.get("base_model_scale"), "2B")


class _StubSession:
    lora_available = True

    def __init__(self, descs):
        self.engine_obj = _StubEngine(descs)
        self.backend = _ScaleBackend()

    lora_catalog_payload = StreamingSession.lora_catalog_payload
    _resolve_lora_id = StreamingSession._resolve_lora_id


def _write_lora(tmp_path: Path, stem: str, name: str, scale: str) -> Path:
    p = tmp_path / f"{stem}.safetensors"
    p.write_bytes(b"")
    sidecar = tmp_path / f"{stem}.metadata.json"
    sidecar.write_text(
        json.dumps({
            "schema_version": 1,
            "id": stem,
            "name": name,
            "model": {"base_model": "acestep", "base_model_scale": scale},
        }),
        encoding="utf-8",
    )
    return p


def _stub_session(tmp_path: Path) -> _StubSession:
    clear_cache()
    return _StubSession([
        _Desc(_write_lora(tmp_path, "ambient-v1", "Ambient", "2B")),
        _Desc(_write_lora(tmp_path, "ambient-xl-v1", "Ambient", "5B")),
        _Desc(_write_lora(tmp_path, "metalcore-xl-v1", "Metalcore", "5B")),
    ])


class TestSessionPlumbing:
    def test_catalog_entries_carry_backend_verdict(self, tmp_path):
        catalog = _stub_session(tmp_path).lora_catalog_payload()
        verdicts = {e["id"]: e["compatible"] for e in catalog}
        assert verdicts == {
            "ambient-v1": True,
            "ambient-xl-v1": False,
            "metalcore-xl-v1": False,
        }
        # Annotation only — the full catalog still ships (the demo's
        # show_incompatible_loras toggle needs the incompatible rows).
        assert len(catalog) == 3

    def test_resolve_lora_id_prefers_exact_then_alias(self, tmp_path):
        ss = _stub_session(tmp_path)
        assert ss._resolve_lora_id("ambient-xl-v1") == "ambient-xl-v1"
        assert ss._resolve_lora_id("Ambient") == "ambient-v1"
        assert ss._resolve_lora_id("ambient") == "ambient-v1"

    def test_resolve_lora_id_falls_through_on_miss(self, tmp_path):
        ss = _stub_session(tmp_path)
        # Alias restricted to the compatible subset: "Metalcore" only
        # exists as a 5B variant on this 2B stub, so the reference
        # passes through unchanged (and fails engine-side, as today).
        assert ss._resolve_lora_id("Metalcore") == "Metalcore"
        assert ss._resolve_lora_id("nope") == "nope"

    def test_resolve_lora_id_without_engine_is_identity(self, tmp_path):
        ss = _stub_session(tmp_path)
        ss.engine_obj = None
        assert ss._resolve_lora_id("Ambient") == "Ambient"


# ---------------------------------------------------------------------------
# Family axis: ACE backend predicate + mixed-family catalog annotation
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
from safetensors.torch import save_file

from acestep.engine.lora import EagerLoRAManager
from acestep.streaming.ace_backend import ACEStepBackend


def _write_sa3_file(tmp_path: Path, stem: str = "sa3style") -> Path:
    """Tiny SA3-format LoRA file (parametrize-style keys), real bytes so
    the metadata sniff classifies it."""
    prefix = (
        "model.transformer.layers.0.self_attn.to_qkv"
        ".parametrizations.weight.0"
    )
    p = tmp_path / f"{stem}.safetensors"
    save_file(
        {
            f"{prefix}.lora_A": torch.zeros(4, 16, dtype=torch.float16),
            f"{prefix}.lora_B": torch.zeros(16, 4, dtype=torch.float16),
        },
        str(p),
        metadata={
            "lora_config": json.dumps(
                {"rank": 4, "alpha": 4, "adapter_type": "lora",
                 "base_model": "medium-base"}
            )
        },
    )
    return p


class _AceStub:
    """Bare-attribute stand-in so ACEStepBackend.lora_compatible can run
    unbound without constructing the full backend."""

    def __init__(self, scale="2B"):
        self.checkpoint_scale = scale


class TestAceFamilyCompatible:
    def test_sa3_family_is_hard_no(self):
        assert not ACEStepBackend.lora_compatible(
            _AceStub(), {"lora_family": "sa3"},
        )

    def test_sa3_family_is_hard_no_even_with_matching_scale(self):
        # The family axis is checked before scale: an SA3 file whose
        # sidecar happens to claim a matching scale still can't load.
        assert not ACEStepBackend.lora_compatible(
            _AceStub("2B"),
            {"lora_family": "sa3", "base_model_scale": "2B"},
        )

    def test_ace_family_falls_through_to_scale_axis(self):
        assert ACEStepBackend.lora_compatible(
            _AceStub("2B"), {"lora_family": "ace", "base_model_scale": "2B"},
        )
        assert not ACEStepBackend.lora_compatible(
            _AceStub("2B"), {"lora_family": "ace", "base_model_scale": "5B"},
        )

    def test_unknown_family_stays_permissive(self):
        assert ACEStepBackend.lora_compatible(
            _AceStub("2B"), {"lora_family": None, "base_model_scale": None},
        )
        assert ACEStepBackend.lora_compatible(_AceStub("2B"), {})


class _FamilyBackend:
    """Stub backend running the REAL ACE predicate (family + scale)."""

    checkpoint_scale = "2B"
    lora_compatible = ACEStepBackend.lora_compatible


class TestMixedFamilyCatalog:
    def test_catalog_annotates_sa3_entry_incompatible(self, tmp_path):
        clear_cache()
        ss = _StubSession([
            _Desc(_write_lora(tmp_path, "ambient-v1", "Ambient", "2B")),
            _Desc(_write_sa3_file(tmp_path, "sa3style")),
        ])
        ss.backend = _FamilyBackend()
        verdicts = {
            e["id"]: e["compatible"] for e in ss.lora_catalog_payload()
        }
        assert verdicts == {"ambient-v1": True, "sa3style": False}

    def test_alias_of_sa3_only_name_misses(self, tmp_path):
        """Alias resolution is restricted to the compatible subset, so a
        display-name reference to an SA3-only entry misses; the exact id
        still passes through (and fails loudly at the enable boundary)."""
        clear_cache()
        ss = _StubSession([_Desc(_write_sa3_file(tmp_path, "sa3style"))])
        ss.backend = _FamilyBackend()
        assert ss._resolve_lora_id("sa3style") == "sa3style"  # exact id
        assert ss._resolve_lora_id("SA3STYLE") == "SA3STYLE"  # alias: miss


# ---------------------------------------------------------------------------
# Hard format validation at the enable boundary (ACE manager)
# ---------------------------------------------------------------------------


def _tiny_eager_manager() -> EagerLoRAManager:
    decoder = nn.Module()
    decoder.q = nn.Linear(16, 8, bias=False)
    return EagerLoRAManager(decoder=decoder, device=torch.device("cpu"))


class TestAceEnableRejectsWrongFamily:
    def test_sa3_file_raises_loudly(self, tmp_path):
        """The exact-id bypass means an SA3 file CAN reach the ACE
        enable path; it must raise, not silently no-op (the pre-fix
        behavior: empty pair map -> empty deltas -> 'enabled' with
        nothing applied)."""
        p = _write_sa3_file(tmp_path)
        mgr = _tiny_eager_manager()
        mgr.register_lora(str(p))
        import pytest
        with pytest.raises(RuntimeError, match="not an ACE-Step LoRA"):
            mgr.enable_lora("sa3style", strength=1.0)

    def test_sa3_file_error_names_the_family(self, tmp_path):
        p = _write_sa3_file(tmp_path)
        mgr = _tiny_eager_manager()
        mgr.register_lora(str(p))
        import pytest
        with pytest.raises(RuntimeError, match="SA3"):
            mgr.enable_lora("sa3style", strength=1.0)

    def test_non_lora_file_raises_without_sa3_hint(self, tmp_path):
        p = tmp_path / "notalora.safetensors"
        save_file({"encoder.weight": torch.zeros(2, 2)}, str(p))
        mgr = _tiny_eager_manager()
        mgr.register_lora(str(p))
        import pytest
        with pytest.raises(RuntimeError, match="not an ACE-Step LoRA") as ei:
            mgr.enable_lora("notalora", strength=1.0)
        assert "SA3" not in str(ei.value)

    def test_real_ace_file_still_loads(self, tmp_path):
        """Regression guard for the new raise: a well-formed PEFT file
        through the REAL _compute_deltas still enables."""
        p = tmp_path / "real.safetensors"
        save_file(
            {
                "base_model.model.q.lora_A.weight": torch.ones(4, 16) * 0.5,
                "base_model.model.q.lora_B.weight": torch.ones(8, 4) * 0.5,
            },
            str(p),
        )
        mgr = _tiny_eager_manager()
        mgr.register_lora(str(p))
        mgr.enable_lora("real", strength=1.0)
        assert mgr.get_lora("real").state == "enabled"


# ---------------------------------------------------------------------------
# Duplicate-stem collision handling in discovery
# ---------------------------------------------------------------------------

from acestep.paths import assign_lora_ids


class TestAssignLoraIds:
    def test_unique_stems_keep_bare_ids(self, tmp_path):
        paths = [tmp_path / "a.safetensors", tmp_path / "b.safetensors"]
        assert assign_lora_ids(paths) == [
            (paths[0], "a"), (paths[1], "b"),
        ]

    def test_collision_first_keeps_stem_later_gets_parent_suffix(self, tmp_path):
        p1 = tmp_path / "lib" / "vibes.safetensors"
        p2 = tmp_path / "training_out" / "vibes.safetensors"
        assert assign_lora_ids([p1, p2]) == [
            (p1, "vibes"), (p2, "vibes--training_out"),
        ]

    def test_triple_collision_same_parent_name_gets_numeric_suffix(self, tmp_path):
        p1 = tmp_path / "r1" / "out" / "x.safetensors"
        p2 = tmp_path / "r2" / "out" / "x.safetensors"
        p3 = tmp_path / "r3" / "out" / "x.safetensors"
        ids = [lid for _, lid in assign_lora_ids([p1, p2, p3])]
        assert ids[0] == "x"
        assert ids[1] == "x--out"
        assert ids[2] == "x--out-2"
        assert len(set(ids)) == 3

    def test_register_library_registers_all_collided_files(self, tmp_path, monkeypatch):
        """End-to-end through the eager manager: a same-stem pair in two
        subdirectories must yield TWO catalog entries (the old
        first-wins registrar silently dropped the second)."""
        (tmp_path / "lib").mkdir()
        (tmp_path / "extra").mkdir()
        (tmp_path / "lib" / "dupe.safetensors").write_bytes(b"")
        (tmp_path / "extra" / "dupe.safetensors").write_bytes(b"")

        mgr = _tiny_eager_manager()
        ids = mgr.register_library(tmp_path)

        assert sorted(ids) == ["dupe", "dupe--extra"] or sorted(ids) == [
            "dupe", "dupe--lib",
        ]
        assert len(mgr.list_loras()) == 2
