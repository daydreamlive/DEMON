"""Family conformance: every registered ``FamilySpec`` satisfies the
platform contract, on CPU, without constructing a backend.

This is the test that makes "another engineer can add a family" a
checkable claim (docs/FAMILIES.md). It runs over
``acestep.streaming.families.FAMILY_SPECS`` and over the views derived
from it, so a spec that drifts from what the core reads fails here
rather than at a pod's first session.
"""

import re

import pytest

from acestep.streaming import families
from acestep.streaming.families import (
    CHECKPOINT_ALIASES,
    SA3_TEXT_ONLY_MAX_DURATION_S,
    TextOnlySpec,
    DEFAULT_FAMILY,
    FAMILIES,
    FAMILY_KNOB_UNIVERSES,
    FAMILY_SPECS,
    PROMPT_POLICIES,
    SESSION_CREATORS,
    WARMUP_POLICIES,
    WARMUP_POLICY_NAMES,
    FamilySpec,
    get_family,
    make_backend,
    resolve_checkpoint,
)
from acestep.streaming.knobs import KnobSpec

SPECS = sorted(FAMILY_SPECS.values(), key=lambda s: s.name)
IDS = [s.name for s in SPECS]


# ---- the spec itself --------------------------------------------------------

@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_spec_identity(spec: FamilySpec):
    assert re.match(r"^[a-z][a-z0-9_]{0,31}$", spec.name)
    assert spec.display_name.strip()
    assert FAMILY_SPECS[spec.name] is spec


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_spec_callables(spec: FamilySpec):
    assert callable(spec.make_backend)
    assert callable(spec.knob_universe)
    assert spec.create_session is None or callable(spec.create_session)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_spec_warmup_policy_is_known(spec: FamilySpec):
    assert spec.warmup_policy in WARMUP_POLICY_NAMES


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_spec_declares_boot_policy(spec: FamilySpec):
    # Every registered family boots through its own check; a pod that
    # cannot serve what it was asked for must stop at boot, not at the
    # first session.
    assert callable(spec.preflight), f"{spec.name} has no preflight"
    assert callable(spec.create_session), f"{spec.name} has no create_session"
    assert spec.prompt_policy in PROMPT_POLICIES
    assert spec.text_only is None or isinstance(spec.text_only, TextOnlySpec)
    assert spec.shutdown is None or callable(spec.shutdown)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_knob_universe_is_a_list_of_knobspecs(spec: FamilySpec):
    universe = spec.knob_universe()
    assert isinstance(universe, list) and universe, "empty knob universe"
    assert all(isinstance(k, KnobSpec) for k in universe)
    # A name may repeat across a family's own variants (ACE's SDE and
    # non-SDE manifests) only with identical semantics; the homonym guard
    # (test_knob_homonyms.py) checks the semantics, this checks the type.
    assert all(isinstance(k.name, str) and k.name for k in universe)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_checkpoint_aliases_resolve_to_this_family(spec: FamilySpec):
    for alias, model_id in spec.checkpoint_aliases.items():
        assert resolve_checkpoint(alias) == (spec.name, model_id)
        # An alias must not be mistakable for a family name or a bare
        # checkpoint directory of the default family.
        assert alias not in FAMILY_SPECS


# ---- the registry -------------------------------------------------------------

def test_default_family_is_registered_and_owns_unaliased_names():
    assert DEFAULT_FAMILY in FAMILY_SPECS
    assert resolve_checkpoint("some-checkpoint-dir") == (DEFAULT_FAMILY, "some-checkpoint-dir")


def test_aliases_are_unique_across_families():
    seen: dict = {}
    for spec in SPECS:
        for alias in spec.checkpoint_aliases:
            assert alias not in seen, f"alias {alias!r} in {seen[alias]} and {spec.name}"
            seen[alias] = spec.name


def test_get_family_unknown_fails_loudly():
    with pytest.raises(ValueError, match="unknown backend family 'nope'"):
        get_family("nope")


def test_make_backend_unknown_fails_loudly():
    with pytest.raises(ValueError, match="unknown backend family 'nope'"):
        make_backend("nope", None)


def test_spec_validation_rejects_bad_declarations():
    good = dict(
        name="demo", display_name="Demo",
        make_backend=lambda ss: None, knob_universe=lambda: [],
    )
    FamilySpec(**good)
    with pytest.raises(ValueError, match="must match"):
        FamilySpec(**{**good, "name": "Demo-Family"})
    with pytest.raises(ValueError, match="display_name"):
        FamilySpec(**{**good, "display_name": ""})
    with pytest.raises(ValueError, match="warmup_policy"):
        FamilySpec(**{**good, "warmup_policy": "sometimes"})
    with pytest.raises(ValueError, match="prompt_policy"):
        FamilySpec(**{**good, "prompt_policy": "poetry"})
    with pytest.raises(ValueError, match="non-empty strings"):
        FamilySpec(**{**good, "checkpoint_aliases": {"": "x"}})


def test_register_rejects_duplicate_alias_and_missing_default():
    a = FamilySpec(name="acestep", display_name="A", make_backend=lambda ss: None,
                   knob_universe=lambda: [], checkpoint_aliases={"xl": "a"})
    b = FamilySpec(name="other", display_name="B", make_backend=lambda ss: None,
                   knob_universe=lambda: [], checkpoint_aliases={"xl": "b"})
    with pytest.raises(ValueError, match="claimed by both"):
        families._register(a, b)
    with pytest.raises(ValueError, match="DEFAULT_FAMILY"):
        families._register(b)


# ---- the derived views never drift from the specs -------------------------

def test_derived_views_match_the_specs():
    assert set(FAMILIES) == set(FAMILY_SPECS)
    assert set(WARMUP_POLICIES) == set(FAMILY_SPECS)
    assert set(FAMILY_KNOB_UNIVERSES) == set(FAMILY_SPECS)
    for name, spec in FAMILY_SPECS.items():
        assert FAMILIES[name] is spec.make_backend
        assert WARMUP_POLICIES[name] == spec.warmup_policy
        assert FAMILY_KNOB_UNIVERSES[name] is spec.knob_universe
        for alias, model_id in spec.checkpoint_aliases.items():
            assert CHECKPOINT_ALIASES[alias] == (name, model_id)
    assert set(SESSION_CREATORS) == {
        n for n, s in FAMILY_SPECS.items() if s.create_session is not None
    }
    assert set(CHECKPOINT_ALIASES) == {
        a for s in FAMILY_SPECS.values() for a in s.checkpoint_aliases
    }


# ---- what the in-tree families promise today ----------------------------

def test_in_tree_families_declare_what_the_pods_rely_on():
    # The controlnet pool selects --model-extension on sa3; an ACE pod
    # has no extension hooks and must refuse a selection.
    assert get_family("sa3").supports_extensions is True
    assert get_family("acestep").supports_extensions is False
    # The ACE server warmup path keys on this exact policy name.
    assert get_family("acestep").warmup_policy == "ace_trt"
    # Every family owns its create path; StreamingSession.create only
    # dispatches (ace_session.py / sa3_session.py).
    assert get_family("sa3").create_session is not None
    assert get_family("acestep").create_session is not None
    # /api/enhance infers its policy from the family.
    assert get_family("sa3").prompt_policy == "sa3"
    assert get_family("acestep").prompt_policy == "acestep"
    # Only SA3 serves an operator-supplied checkpoint directory today.
    assert get_family("sa3").accepts_checkpoint_dir is True
    assert get_family("acestep").accepts_checkpoint_dir is False
    # Both families run text-only sessions (verified on a 5090 with the
    # headless probe, 2026-09-24); SA3 lets the client size the render.
    assert get_family("acestep").text_only.duration_field is None
    assert get_family("sa3").text_only.duration_field == "sa3_duration_s"
    # SA3 process-caches its model and may host an extension attached to
    # it; the server's shutdown path detaches through this hook.
    assert get_family("sa3").shutdown is not None
    assert get_family("acestep").shutdown is None


def test_sa3_text_only_cap_matches_the_backend():
    # The registry mirrors the backend's constant so it never imports the
    # backend module; this pins the two together.
    from acestep.streaming.sa3_backend import SA3_MAX_DURATION_S

    assert SA3_TEXT_ONLY_MAX_DURATION_S == SA3_MAX_DURATION_S
    assert get_family("sa3").text_only.max_duration_s == SA3_MAX_DURATION_S


def test_extension_selection_refuses_a_family_without_hooks(tmp_path):
    """A plugin for a family that cannot install it fails at selection,
    not silently at the first session."""
    from acestep.plugins import ModelExtension, ModelExtensionSpec
    from acestep.plugins.api import LoadedPlugin, PluginManifest, PLUGIN_API_VERSION
    from acestep.plugins.errors import ExtensionConfigError
    from acestep.plugins.selection import select_model_extension

    class _Ext(ModelExtension):
        def __init__(self, context):
            pass

    spec = ModelExtensionSpec(name="demo", family="acestep", create=_Ext)
    plugins = {
        "p": LoadedPlugin(
            manifest=PluginManifest(id="p", version="0", plugin_api=PLUGIN_API_VERSION),
            model_extensions={"demo": spec},
        )
    }
    with pytest.raises(ExtensionConfigError, match="does not host model extensions"):
        select_model_extension("p.demo", family="acestep", plugins=plugins)
