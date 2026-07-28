"""Entry-point plugin discovery (Tier 1, no GPU).

The asymmetry under test: a plugin the operator EXPLICITLY selected fails
the process on any error, while an unselected broken plugin is isolated.
Silently dropping a selected model extension would leave the server
generating with a different model than the one that was asked for, and
the output would look entirely plausible.
"""

from __future__ import annotations

import sys
import types

import pytest

from acestep.plugins import (
    PLUGIN_API_VERSION,
    ConfigField,
    ModelExtension,
    ModelExtensionSpec,
    PluginIncompatibleError,
    PluginLoadError,
    PluginNotFoundError,
)
from acestep.plugins import discovery
from acestep.streaming.knobs import KnobSpec


class _FakeDist:
    def __init__(self, version):
        self.version = version


class _FakeEntryPoint:
    """Mirrors importlib.metadata.EntryPoint semantics.

    Notably ``load()`` resolves to the target ATTRIBUTE, not the module,
    and discovery imports ``module`` by name — so the fake registers a
    real module in ``sys.modules`` and exercises the real import path. An
    earlier version of this fake returned the module from ``load()``,
    which hid a genuine discovery bug.
    """

    def __init__(self, name, module, *, version="1.0.0", attr="register",
                 import_error=None):
        self.name = name
        self.module = f"{name}._fake_plugin_module"
        self.attr = attr
        self.value = f"{self.module}:{attr}"
        self.dist = _FakeDist(version)
        if import_error is None:
            sys.modules[self.module] = module
        else:
            sys.modules.pop(self.module, None)
        self._import_error = import_error

    def load(self):
        return getattr(sys.modules[self.module], self.attr)


def _plugin_module(
    plugin_id, *, api=PLUGIN_API_VERSION, register=None, requires=None,
):
    module = types.ModuleType(f"{plugin_id}.plugin")
    module.PLUGIN_API = api
    if requires is not None:
        module.REQUIRES_DEMON = requires
    if register is not None:
        module.register = register
    else:
        def _register(registry):
            registry.add_model_extension(
                ModelExtensionSpec(
                    name="demo",
                    family="sa3",
                    create=lambda ctx: ModelExtension(),
                    config_schema={"scale": ConfigField(type="float", required=False)},
                    knob_specs=(
                        KnobSpec(
                            f"plugin_{plugin_id}_strength",
                            default=1.0, min_val=0.0, max_val=3.0,
                        ),
                    ),
                )
            )
        module.register = _register
    return module


def _patch_entry_points(monkeypatch, entry_points):
    monkeypatch.setattr(
        discovery, "_iter_entry_points",
        lambda group: sorted(entry_points, key=lambda ep: ep.name),
    )


def test_discovers_and_registers_a_valid_plugin(monkeypatch):
    _patch_entry_points(
        monkeypatch, [_FakeEntryPoint("good_ext", _plugin_module("good_ext"))],
    )

    plugins = discovery.discover_plugins()

    assert set(plugins) == {"good_ext"}
    plugin = plugins["good_ext"]
    assert plugin.manifest.version == "1.0.0"
    assert plugin.capability_ids == ["demo"]


def test_discovery_is_deterministic_by_entry_point_name(monkeypatch):
    _patch_entry_points(monkeypatch, [
        _FakeEntryPoint("zeta_ext", _plugin_module("zeta_ext")),
        _FakeEntryPoint("alpha_ext", _plugin_module("alpha_ext")),
    ])

    assert list(discovery.discover_plugins()) == ["alpha_ext", "zeta_ext"]


def test_wrong_plugin_api_is_rejected(monkeypatch):
    _patch_entry_points(monkeypatch, [
        _FakeEntryPoint(
            "old_ext", _plugin_module("old_ext", api=PLUGIN_API_VERSION + 1),
        ),
    ])

    with pytest.raises(PluginIncompatibleError, match="plugin API"):
        discovery.discover_plugins(required=["old_ext"])


def test_missing_plugin_api_declaration_is_rejected(monkeypatch):
    module = _plugin_module("bare_ext")
    del module.PLUGIN_API
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("bare_ext", module)])

    with pytest.raises(PluginLoadError, match="PLUGIN_API"):
        discovery.discover_plugins(required=["bare_ext"])


def test_missing_register_callable_is_rejected(monkeypatch):
    module = _plugin_module("noreg_ext")
    del module.register
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("noreg_ext", module)])

    with pytest.raises(PluginLoadError, match="register"):
        discovery.discover_plugins(required=["noreg_ext"])


def test_import_failure_is_reported_as_a_load_error(monkeypatch):
    _patch_entry_points(monkeypatch, [
        _FakeEntryPoint("broken_ext", None, import_error=True),
    ])

    with pytest.raises(PluginLoadError, match="failed to import"):
        discovery.discover_plugins(required=["broken_ext"])


def test_register_failure_is_reported(monkeypatch):
    def _bad_register(registry):
        raise RuntimeError("registration exploded")

    _patch_entry_points(monkeypatch, [
        _FakeEntryPoint(
            "bad_ext", _plugin_module("bad_ext", register=_bad_register),
        ),
    ])

    with pytest.raises(PluginLoadError, match="register\\(\\) failed"):
        discovery.discover_plugins(required=["bad_ext"])


def test_registration_error_surfaces_unchanged(monkeypatch):
    from acestep.plugins import PluginRegistrationError

    def _register(registry):
        registry.add_model_extension(
            ModelExtensionSpec(
                name="demo", family="sa3", create=lambda ctx: ModelExtension(),
                # Unnamespaced: must fail registration, not become a
                # generic load error that hides the real cause.
                knob_specs=(KnobSpec("strength", default=1.0, max_val=1.0),),
            )
        )

    _patch_entry_points(monkeypatch, [
        _FakeEntryPoint("ns_ext", _plugin_module("ns_ext", register=_register)),
    ])

    with pytest.raises(PluginRegistrationError, match="namespaced"):
        discovery.discover_plugins(required=["ns_ext"])


def test_unselected_broken_plugin_is_isolated(monkeypatch):
    _patch_entry_points(monkeypatch, [
        _FakeEntryPoint("broken_ext", None, import_error=True),
        _FakeEntryPoint("good_ext", _plugin_module("good_ext")),
    ])

    # Nobody selected the broken one, so one bad optional plugin must not
    # take the server down.
    plugins = discovery.discover_plugins()

    assert set(plugins) == {"good_ext"}


def test_selected_broken_plugin_is_fatal(monkeypatch):
    _patch_entry_points(monkeypatch, [
        _FakeEntryPoint("broken_ext", None, import_error=True),
        _FakeEntryPoint("good_ext", _plugin_module("good_ext")),
    ])

    with pytest.raises(PluginLoadError):
        discovery.discover_plugins(required=["broken_ext"])


def test_selected_but_uninstalled_plugin_is_fatal(monkeypatch):
    _patch_entry_points(monkeypatch, [
        _FakeEntryPoint("good_ext", _plugin_module("good_ext")),
    ])

    with pytest.raises(PluginNotFoundError, match="not installed"):
        discovery.discover_plugins(required=["absent_ext"])


def test_capability_ids_are_unique_across_plugins(monkeypatch):
    # Two plugins each registering "demo" is fine: the qualified ids
    # differ, so neither is silently dropped.
    _patch_entry_points(monkeypatch, [
        _FakeEntryPoint("one_ext", _plugin_module("one_ext")),
        _FakeEntryPoint("two_ext", _plugin_module("two_ext")),
    ])

    plugins = discovery.discover_plugins()

    assert plugins["one_ext"].capability_ids == ["demo"]
    assert plugins["two_ext"].capability_ids == ["demo"]


def test_requires_demon_is_recorded_but_not_enforced(monkeypatch):
    # Deliberately advisory: DEMON's distribution version tracks the model
    # generation, not an API contract. A plugin demanding an impossible
    # version must still load rather than fail on a meaningless comparison.
    _patch_entry_points(monkeypatch, [
        _FakeEntryPoint(
            "req_ext", _plugin_module("req_ext", requires=">=99.0"),
        ),
    ])

    plugins = discovery.discover_plugins(required=["req_ext"])

    assert plugins["req_ext"].manifest.requires_demon == ">=99.0"


def test_no_plugins_installed_is_the_stock_path(monkeypatch):
    _patch_entry_points(monkeypatch, [])
    assert discovery.discover_plugins() == {}
