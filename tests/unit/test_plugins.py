"""Plugin foundation + model-extension contract (Tier 1, no GPU).

The highest-risk property here is not "does a plugin load" but "does a
plugin's control actually reach the validation machinery". Knob coercion
passes UNKNOWN names through verbatim and unclamped by design (curve
specs, telemetry and lora_blend rely on it), so an extension knob that
fails to reach the session's spec map does not fail loudly — it silently
accepts anything a client sends. Several tests below exist specifically
to pin that down.
"""

from __future__ import annotations

import json

import pytest

from acestep.plugins import (
    PLUGIN_API_VERSION,
    BackendVerdict,
    ConfigField,
    ExtensionConfigError,
    ModelExtension,
    ModelExtensionRuntime,
    ModelExtensionSpec,
    PluginManifest,
    PluginRegistry,
    PluginRegistrationError,
    Rejection,
    validate_extension_config,
)
from acestep.plugins.model_extensions import ModelDescriptor
from acestep.plugins.selection import (
    SelectedExtension,
    load_extension_config,
    select_model_extension,
    split_qualified_id,
)
from acestep.streaming.knobs import KnobSpec, coerce_knob_values

PLUGIN_ID = "demo_ext"


def _manifest(plugin_id=PLUGIN_ID, version="1.2.3"):
    return PluginManifest(
        id=plugin_id, version=version, plugin_api=PLUGIN_API_VERSION,
    )


def _registry(plugin_id=PLUGIN_ID, seen=None):
    return PluginRegistry(_manifest(plugin_id), seen=seen)


def _knob(name, **kw):
    kw.setdefault("default", 1.0)
    kw.setdefault("min_val", 0.0)
    kw.setdefault("max_val", 3.0)
    return KnobSpec(name, **kw)


class _NoopExtension(ModelExtension):
    """Neutral extension used to prove the contract without private code."""

    def __init__(self, context):
        self.context = context
        self.installed_with = None
        self.runtime = None

    def validate_base_model(self, descriptor):
        if descriptor.model_id == "unsupported":
            return Rejection("test extension does not support this model")
        return None

    def config_fingerprint(self):
        return f"scale={self.context.config.get('scale')}"

    def install(self, model, model_config):
        self.installed_with = (model, model_config)
        self.runtime = _NoopRuntime()
        return self.runtime


class _NoopRuntime(ModelExtensionRuntime):
    def __init__(self):
        self.controls = None
        self.closes = 0
        self.decorated = 0

    def supports_dit_backend(self, backend):
        if backend == "tensorrt":
            return BackendVerdict(False, "no accelerated plan for this graph")
        return BackendVerdict(True)

    def decorate_conditioning(self, bundle, source_latent_bct=None):
        self.decorated += 1
        return {**bundle, "extension_cond": source_latent_bct}

    def apply_controls(self, values):
        self.controls = dict(values)

    def status(self):
        return {"ready": True}

    def close(self):
        self.closes += 1


def _spec(**kw):
    kw.setdefault("name", "demo")
    kw.setdefault("family", "sa3")
    kw.setdefault("create", _NoopExtension)
    kw.setdefault("knob_specs", (_knob(f"plugin_{PLUGIN_ID}_strength"),))
    return ModelExtensionSpec(**kw)


# ---- registration validation ------------------------------------------------


def test_valid_extension_registers_with_a_qualified_id():
    registry = _registry()
    qualified = registry.add_model_extension(_spec())

    assert qualified == f"{PLUGIN_ID}.demo"
    assert "demo" in registry.model_extensions


def test_registering_one_capability_name_twice_is_rejected():
    # The reachable duplicate: one plugin registering the same name twice.
    # Without this check the second silently overwrites the first in
    # _model_extensions, and the operator gets whichever won.
    registry = _registry()
    registry.add_model_extension(_spec())

    with pytest.raises(PluginRegistrationError, match="duplicate capability"):
        registry.add_model_extension(_spec())


def test_duplicate_capability_id_is_rejected_across_registries():
    seen: set = set()
    _registry(seen=seen).add_model_extension(_spec())

    with pytest.raises(PluginRegistrationError, match="duplicate capability"):
        _registry(seen=seen).add_model_extension(_spec())


def test_unknown_family_is_rejected():
    with pytest.raises(PluginRegistrationError, match="unknown backend family"):
        _registry().add_model_extension(_spec(family="not_a_family"))


def test_non_callable_create_is_rejected():
    with pytest.raises(PluginRegistrationError, match="create must be callable"):
        _registry().add_model_extension(_spec(create="nope"))


def test_unnamespaced_knob_is_rejected():
    with pytest.raises(PluginRegistrationError, match="must be namespaced"):
        _registry().add_model_extension(
            _spec(knob_specs=(_knob("strength"),)),
        )


def test_knob_namespaced_to_another_plugin_is_rejected():
    with pytest.raises(PluginRegistrationError, match="must be namespaced"):
        _registry().add_model_extension(
            _spec(knob_specs=(_knob("plugin_other_plugin_strength"),)),
        )


def test_no_core_knob_can_reach_the_plugin_namespace():
    # The namespace regex IS the shadowing defense, so the property it
    # depends on is worth pinning: nothing in the core registry or any
    # family universe is named plugin_*. If a core knob ever were, it
    # could be shadowed by a plugin control that still passes validation.
    from acestep.streaming.families import FAMILY_KNOB_UNIVERSES
    from acestep.streaming.knobs import knob_specs

    names = {spec.name for spec in knob_specs(True, loras=["<lora_id>"])}
    for universe in FAMILY_KNOB_UNIVERSES.values():
        names.update(spec.name for spec in universe())

    assert "seed" in names          # the set is non-empty and real
    assert "sa3_denoise" in names
    assert not [n for n in names if n.startswith("plugin_")]


def test_non_numeric_knob_type_is_rejected():
    spec = KnobSpec(
        f"plugin_{PLUGIN_ID}_mode", default="a", type="enum", options=("a", "b"),
    )
    with pytest.raises(PluginRegistrationError, match="must be one of"):
        _registry().add_model_extension(_spec(knob_specs=(spec,)))


def test_default_outside_its_own_bounds_is_rejected():
    # Coercion clamps, so this default would silently become a different
    # number than the one the plugin declared.
    bad = _knob(f"plugin_{PLUGIN_ID}_strength", default=9.0, max_val=3.0)
    with pytest.raises(PluginRegistrationError, match="outside"):
        _registry().add_model_extension(_spec(knob_specs=(bad,)))


def test_inverted_bounds_are_rejected():
    bad = _knob(f"plugin_{PLUGIN_ID}_strength", min_val=2.0, max_val=1.0)
    with pytest.raises(PluginRegistrationError, match="below"):
        _registry().add_model_extension(_spec(knob_specs=(bad,)))


def test_duplicate_knob_names_within_one_spec_are_rejected():
    name = f"plugin_{PLUGIN_ID}_strength"
    with pytest.raises(PluginRegistrationError, match="duplicate knob"):
        _registry().add_model_extension(
            _spec(knob_specs=(_knob(name), _knob(name))),
        )


def test_invalid_capability_name_is_rejected():
    with pytest.raises(PluginRegistrationError, match="invalid capability name"):
        _registry().add_model_extension(_spec(name="Bad Name"))


# ---- the silent-passthrough hole -------------------------------------------


def test_extension_knob_in_the_spec_map_is_clamped():
    # The load-bearing property: once an extension knob reaches a session's
    # spec map, an out-of-range client value is CLAMPED like any core knob.
    name = f"plugin_{PLUGIN_ID}_strength"
    specs = {name: _knob(name, min_val=0.0, max_val=3.0)}

    clean, errors = coerce_knob_values({name: 99.0}, specs)

    assert clean[name] == 3.0
    assert errors


def test_extension_knob_missing_from_the_spec_map_is_not_validated():
    # The failure mode this whole namespacing/composition design exists to
    # prevent: an unwired knob is passed through verbatim and unclamped,
    # with no error. If composition regresses, values stop being coerced
    # and nothing complains.
    name = f"plugin_{PLUGIN_ID}_strength"

    clean, errors = coerce_knob_values({name: 99.0}, {})

    assert clean[name] == 99.0
    assert not errors


# ---- configuration ----------------------------------------------------------


def _schema():
    return {
        "scale": ConfigField(type="float", required=True),
        "label": ConfigField(type="str", required=False, default="x"),
    }


def test_config_validation_applies_defaults_and_coerces():
    out = validate_extension_config("p.e", _schema(), {"scale": 2})
    assert out == {"scale": 2.0, "label": "x"}


def test_config_rejects_unknown_keys():
    with pytest.raises(ExtensionConfigError, match="unknown configuration key"):
        validate_extension_config("p.e", _schema(), {"scale": 1.0, "typo": 1})


def test_config_rejects_missing_required_key():
    with pytest.raises(ExtensionConfigError, match="missing required"):
        validate_extension_config("p.e", _schema(), {})


def test_config_rejects_wrong_type():
    with pytest.raises(ExtensionConfigError, match="wrong type"):
        validate_extension_config("p.e", _schema(), {"scale": "loud"})


def test_config_rejects_bool_for_number():
    # bool is an int subclass in Python; accepting True as 1.0 would be a
    # silent misreading of an operator's config file.
    with pytest.raises(ExtensionConfigError, match="wrong type"):
        validate_extension_config("p.e", _schema(), {"scale": True})


def test_config_rejects_non_integer_float_for_int():
    schema = {"steps": ConfigField(type="int")}
    with pytest.raises(ExtensionConfigError, match="wrong type"):
        validate_extension_config("p.e", schema, {"steps": 1.5})


def test_config_file_is_keyed_by_qualified_id(tmp_path):
    path = tmp_path / "ext.json"
    path.write_text(json.dumps({"demo_ext.demo": {"scale": 2.0}}), encoding="utf-8")

    assert load_extension_config(path, "demo_ext.demo") == {"scale": 2.0}


def test_config_file_without_the_selected_entry_fails(tmp_path):
    path = tmp_path / "ext.json"
    path.write_text(json.dumps({"other.thing": {}}), encoding="utf-8")

    with pytest.raises(ExtensionConfigError, match="no entry for"):
        load_extension_config(path, "demo_ext.demo")


def test_missing_config_file_fails_with_the_path(tmp_path):
    with pytest.raises(ExtensionConfigError, match="not found"):
        load_extension_config(tmp_path / "absent.json", "demo_ext.demo")


def test_malformed_config_file_fails(tmp_path):
    path = tmp_path / "ext.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ExtensionConfigError, match="not valid JSON"):
        load_extension_config(path, "demo_ext.demo")


def test_no_config_path_means_empty_config():
    assert load_extension_config(None, "demo_ext.demo") == {}


# ---- selection --------------------------------------------------------------


def _loaded_plugins(spec=None):
    from acestep.plugins.api import LoadedPlugin

    spec = spec or _spec()
    return {
        PLUGIN_ID: LoadedPlugin(
            manifest=_manifest(), model_extensions={spec.name: spec},
        )
    }


def test_no_selection_returns_none():
    assert select_model_extension(None, family="sa3") is None
    assert select_model_extension("", family="sa3") is None


def test_selection_builds_a_configured_extension(tmp_path):
    path = tmp_path / "ext.json"
    path.write_text(
        json.dumps({f"{PLUGIN_ID}.demo": {"scale": 2.0}}), encoding="utf-8",
    )
    spec = _spec(config_schema={"scale": ConfigField(type="float")})

    selected = select_model_extension(
        f"{PLUGIN_ID}.demo",
        family="sa3",
        config_path=path,
        plugins=_loaded_plugins(spec),
    )

    assert isinstance(selected, SelectedExtension)
    assert selected.config == {"scale": 2.0}
    assert selected.knob_specs[0].name == f"plugin_{PLUGIN_ID}_strength"


def test_selection_of_a_missing_plugin_fails():
    from acestep.plugins.errors import PluginNotFoundError

    with pytest.raises(PluginNotFoundError, match="not installed"):
        select_model_extension("absent.demo", family="sa3", plugins={})


def test_selection_of_a_missing_capability_fails():
    from acestep.plugins.errors import PluginNotFoundError

    with pytest.raises(PluginNotFoundError, match="no model extension"):
        select_model_extension(
            f"{PLUGIN_ID}.absent", family="sa3", plugins=_loaded_plugins(),
        )


def test_selection_across_families_fails():
    with pytest.raises(ExtensionConfigError, match="backend family"):
        select_model_extension(
            f"{PLUGIN_ID}.demo", family="acestep", plugins=_loaded_plugins(),
        )


def test_malformed_qualified_id_fails():
    from acestep.plugins.errors import PluginNotFoundError

    for bad in ("noplugin", ".demo", "plugin."):
        with pytest.raises(PluginNotFoundError, match="invalid model extension"):
            split_qualified_id(bad)


def test_public_status_omits_configuration(tmp_path):
    path = tmp_path / "ext.json"
    path.write_text(
        json.dumps({f"{PLUGIN_ID}.demo": {"scale": 7.5}}), encoding="utf-8",
    )
    selected = select_model_extension(
        f"{PLUGIN_ID}.demo",
        family="sa3",
        config_path=path,
        plugins=_loaded_plugins(_spec(config_schema={"scale": ConfigField("float")})),
    )

    status = selected.public_status()

    assert status["id"] == f"{PLUGIN_ID}.demo"
    assert "7.5" not in json.dumps(status)
    assert "scale" not in json.dumps(status)


# ---- lifecycle --------------------------------------------------------------


def _selected():
    return select_model_extension(
        f"{PLUGIN_ID}.demo", family="sa3", plugins=_loaded_plugins(),
    )


def test_full_lifecycle_install_decorate_apply_status_close():
    selected = _selected()
    descriptor = ModelDescriptor(
        family="sa3", model_id="medium", checkpoint_dir="/models/x",
    )
    selected.validate_base_model(descriptor)

    runtime = selected.install(model="MODEL", model_config={"k": 1})
    assert selected.extension.installed_with == ("MODEL", {"k": 1})

    bundle = {"cross_attn_cond": 1}
    decorated = runtime.decorate_conditioning(bundle, "LATENT")
    assert decorated["extension_cond"] == "LATENT"
    # Fresh mapping: the input may already be referenced by in-flight work.
    assert decorated is not bundle
    assert "extension_cond" not in bundle

    runtime.apply_controls({f"plugin_{PLUGIN_ID}_strength": 2.0})
    assert runtime.controls == {f"plugin_{PLUGIN_ID}_strength": 2.0}

    assert runtime.status() == {"ready": True}

    runtime.close()
    runtime.close()
    assert runtime.closes == 2  # idempotent from the core's perspective


def test_base_model_veto_aborts():
    selected = _selected()
    descriptor = ModelDescriptor(
        family="sa3", model_id="unsupported", checkpoint_dir="/models/x",
    )
    with pytest.raises(ExtensionConfigError, match="refused the selected base"):
        selected.validate_base_model(descriptor)


def test_cache_key_tracks_configuration():
    spec = _spec(config_schema={"scale": ConfigField(type="float")})

    def _select(scale, tmp):
        path = tmp / "c.json"
        path.write_text(
            json.dumps({f"{PLUGIN_ID}.demo": {"scale": scale}}), encoding="utf-8",
        )
        return select_model_extension(
            f"{PLUGIN_ID}.demo",
            family="sa3",
            config_path=path,
            plugins=_loaded_plugins(spec),
        )

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        assert _select(1.0, tmp).cache_key() != _select(2.0, tmp).cache_key()
        assert _select(1.0, tmp).cache_key() == _select(1.0, tmp).cache_key()


def test_partial_install_failure_unwinds():
    """An install that mutates the model then raises must still unwind."""
    closed = []

    class _Exploding(ModelExtension):
        def __init__(self, context):
            self.context = context

        def close(self):
            closed.append("extension")

        def install(self, model, model_config):
            raise RuntimeError("boom after attaching")

    selected = select_model_extension(
        f"{PLUGIN_ID}.demo",
        family="sa3",
        plugins=_loaded_plugins(_spec(create=_Exploding)),
    )

    with pytest.raises(RuntimeError, match="boom after attaching"):
        selected.install(model="MODEL", model_config={})

    assert closed == ["extension"]


def test_install_returning_a_non_runtime_is_rejected():
    # Caught at boot rather than as an AttributeError at a tick boundary:
    # the base class gives every hook a neutral default, so a runtime that
    # skips it is missing hooks the streaming runner will call.
    class _Structural:
        def apply_controls(self, values):
            pass

    class _Ext(ModelExtension):
        def __init__(self, context):
            pass

        def install(self, model, model_config):
            return _Structural()

    selected = select_model_extension(
        f"{PLUGIN_ID}.demo",
        family="sa3",
        plugins=_loaded_plugins(_spec(create=_Ext)),
    )
    with pytest.raises(ExtensionConfigError, match="does not subclass"):
        selected.install(model="MODEL", model_config={})


def test_partial_install_unwinds_without_a_close_override():
    # ModelExtension.close() defaults to a no-op, so an extension whose
    # install cannot fail dirty does not have to define one — the core's
    # unwind path must not blow up on its absence.
    class _Exploding(ModelExtension):
        def __init__(self, context):
            pass

        def install(self, model, model_config):
            raise RuntimeError("boom")

    selected = select_model_extension(
        f"{PLUGIN_ID}.demo",
        family="sa3",
        plugins=_loaded_plugins(_spec(create=_Exploding)),
    )
    with pytest.raises(RuntimeError, match="boom"):
        selected.install(model="MODEL", model_config={})


def test_install_returning_none_is_rejected():
    class _Bad(ModelExtension):
        def __init__(self, context):
            pass

        def install(self, model, model_config):
            return None

    selected = select_model_extension(
        f"{PLUGIN_ID}.demo",
        family="sa3",
        plugins=_loaded_plugins(_spec(create=_Bad)),
    )
    with pytest.raises(ExtensionConfigError, match="must return a runtime"):
        selected.install(model="MODEL", model_config={})


def test_backend_verdicts_are_reported():
    runtime = _selected().install(model="M", model_config={})
    assert runtime.supports_dit_backend("eager").ok
    verdict = runtime.supports_dit_backend("tensorrt")
    assert not verdict.ok
    assert verdict.reason


def test_default_runtime_hooks_are_neutral():
    # A minimal extension implementing nothing must behave exactly like no
    # extension at all.
    runtime = ModelExtensionRuntime()
    bundle = {"a": 1}

    assert runtime.supports_dit_backend("tensorrt").ok
    assert runtime.supports_codec_backend("tensorrt").ok
    assert runtime.decorate_conditioning(bundle) is bundle
    assert runtime.status() == {}
    assert runtime.metrics() == {}
    runtime.apply_controls({})
    runtime.close()
