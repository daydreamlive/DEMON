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
