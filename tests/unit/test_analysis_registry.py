"""AnalyzerSpec validation, option coercion and the built-in ``notes`` spec."""

from __future__ import annotations

import pytest

from acestep.analysis import registry
from acestep.analysis.registry import AnalyzerOption, AnalyzerSpec


def _spec(**kw) -> AnalyzerSpec:
    base = dict(
        name="fake",
        display_name="Fake",
        item_kind="note",
        make_engine=lambda size, device: None,
        probe=lambda: None,
        sizes=("a", "b"),
        default_size="a",
    )
    base.update(kw)
    return AnalyzerSpec(**base)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"name": "Bad-Name"}, "must match"),
        ({"display_name": ""}, "display_name"),
        ({"item_kind": "chord"}, "item_kind"),
        ({"placement": "tpu"}, "placement"),
        ({"default_size": "z"}, "default_size"),
        ({"window_s": 0}, "window_s"),
        ({"options": (AnalyzerOption("x", "int"), AnalyzerOption("x", "int"))}, "duplicate"),
    ],
)
def test_spec_rejects_bad_declarations(kw, match):
    with pytest.raises(ValueError, match=match):
        _spec(**kw)


def test_engine_id_defaults_to_name():
    assert _spec().engine_id == "fake"
    assert _spec(engine_name="model_x").engine_id == "model_x"


def test_resolve_size_order(monkeypatch):
    spec = _spec(size_env="FAKE_ANALYZER_SIZE")
    monkeypatch.delenv("FAKE_ANALYZER_SIZE", raising=False)
    assert spec.resolve_size() == "a"
    monkeypatch.setenv("FAKE_ANALYZER_SIZE", " B ")
    assert spec.resolve_size() == "b"
    assert spec.resolve_size("a") == "a"
    with pytest.raises(ValueError, match="not available"):
        spec.resolve_size("zz")


def test_validate_options_defaults_types_bounds():
    seen = []

    def upper(v):
        seen.append(v)
        return [s.upper() for s in v]

    spec = _spec(options=(
        AnalyzerOption("f", "float", default=1.5, bounds=(1.0, 2.0)),
        AnalyzerOption("i", "int", default=3),
        AnalyzerOption("s", "str", default="x", bounds=("x", "y")),
        AnalyzerOption("l", "str_list", default=None, validate=upper),
    ))
    assert spec.validate_options(None) == {"f": 1.5, "i": 3, "s": "x", "l": None}
    out = spec.validate_options({"f": 2, "i": 4.9, "s": "y", "l": ("a", "b")})
    assert out == {"f": 2.0, "i": 4, "s": "y", "l": ["A", "B"]}
    assert isinstance(out["f"], float)
    assert seen == [["a", "b"]]
    # None means "use the default".
    assert spec.validate_options({"f": None})["f"] == 1.5
    for bad, match in [
        ({"nope": 1}, "unknown option"),
        ({"f": 2.5}, "within"),
        ({"f": True}, "number"),
        ({"f": "1"}, "number"),
        ({"s": "z"}, "one of"),
        ({"s": 3}, "string"),
        ({"l": "abc"}, "list of strings"),
        ({"l": ["a", 1]}, "list of strings"),
    ]:
        with pytest.raises(ValueError, match=match):
            spec.validate_options(bad)


def test_register_is_idempotent_per_object():
    spec = _spec(name="fake_register_once")
    try:
        assert registry.register(spec) is spec
        assert registry.register(spec) is spec
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_spec(name="fake_register_once"))
        assert registry.get_analyzer("fake_register_once") is spec
    finally:
        registry._ANALYZERS.pop("fake_register_once", None)


def test_get_analyzer_unknown_lists_known():
    with pytest.raises(KeyError, match="notes"):
        registry.get_analyzer("does_not_exist")


def test_available_analyzers_probe_failure_is_reason():
    def boom():
        raise RuntimeError("probe broke")

    spec = _spec(name="fake_probe_raises", probe=boom)
    registry.register(spec)
    try:
        avail = registry.available_analyzers()
        assert avail["fake_probe_raises"] == "RuntimeError: probe broke"
    finally:
        registry._ANALYZERS.pop("fake_probe_raises", None)


def test_notes_spec_declaration():
    spec = registry.get_analyzer("notes")
    assert spec.display_name == "Notes (MuScriptor)"
    assert spec.engine_id == "muscriptor"
    assert spec.item_kind == "note"
    assert tuple(spec.sizes) == ("small", "medium", "large")
    assert spec.default_size == "small"
    assert spec.size_env == "DEMON_ANALYSIS_NOTES_MODEL"
    assert (spec.window_s, spec.context_s, spec.min_sr) == (4.0, 0.5, 16000)
    assert spec.placement == "gpu"
    assert dict(spec.resident_gb) == {"small": 0.6, "medium": 1.2, "large": 3.5}
    assert "CC BY-NC 4.0" in spec.licence
    assert [o.name for o in spec.options] == ["size", "instruments", "horizon_s"]


def test_notes_options():
    spec = registry.get_analyzer("notes")
    assert spec.validate_options({}) == {"size": None, "instruments": None, "horizon_s": 8.0}
    assert spec.validate_options({"size": "large", "horizon_s": 30})["horizon_s"] == 30.0
    assert spec.validate_options({"instruments": []})["instruments"] is None
    for bad, match in [
        ({"size": "huge"}, "one of"),
        ({"horizon_s": 3.9}, "within"),
        ({"horizon_s": 31}, "within"),
        ({"stride": 1}, "unknown option"),
    ]:
        with pytest.raises(ValueError, match=match):
            spec.validate_options(bad)


def test_notes_instruments_validated_by_muscriptor():
    pytest.importorskip("muscriptor")
    spec = registry.get_analyzer("notes")
    out = spec.validate_options({"instruments": ["acoustic_piano", "drums"]})
    assert out["instruments"] == ["acoustic_piano", "drums"]
    with pytest.raises(ValueError, match="unknown instrument"):
        spec.validate_options({"instruments": ["kazoo_of_doom"]})


def test_notes_probe(monkeypatch):
    spec = registry.get_analyzer("notes")
    monkeypatch.setenv("DEMON_MIDI_DISABLE", "1")
    assert "disabled" in spec.probe()
    monkeypatch.delenv("DEMON_MIDI_DISABLE")
    monkeypatch.setenv("DEMON_ANALYSIS_NOTES_MODEL", "gigantic")
    assert "not available" in spec.probe()
    monkeypatch.delenv("DEMON_ANALYSIS_NOTES_MODEL")
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a: None if name == "muscriptor" else real(name, *a),
    )
    assert "not installed" in spec.probe()


def test_broken_builtin_module_raises(monkeypatch):
    """A built-in analyzer module that fails to import is a bug, not
    "no analyzers": the error surfaces at lookup time."""
    monkeypatch.setattr(
        registry, "_BUILTIN_MODULES",
        ("acestep.analysis.notes_engine", "acestep.analysis._no_such_analyzer"),
    )
    with pytest.raises(ModuleNotFoundError, match="_no_such_analyzer"):
        registry.list_analyzers()
