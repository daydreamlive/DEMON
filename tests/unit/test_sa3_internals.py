"""Structural canary for the Tier-2 SA3 internals contract (no GPU).

:mod:`acestep.engine.sa3_internals` encapsulates the vendored
``stable_audio_3`` object-graph shapes that a model extension has to
traverse. Those shapes belong to a third-party package, so they can drift
under us on a vendor bump — and the failure mode for an out-of-tree
extension is not an ImportError, it is a model that runs and produces
subtly wrong output.

This test is the tripwire. It fails HERE, in public CI, the moment the
accessors stop matching reality. Most of it runs against a synthetic
stand-in so it needs neither checkpoints nor a GPU; the parts that need
the vendored source skip cleanly when it is absent.
"""

from __future__ import annotations

import pytest

from acestep.engine import sa3_internals as internals
from acestep.engine.sa3_helpers import sa3_vendor_present


class _Blocks(list):
    pass


class _Transformer:
    def __init__(self, depth):
        self.layers = _Blocks(object() for _ in range(depth))


class _Dit:
    def __init__(self, depth):
        self.transformer = _Transformer(depth)


class _Wrapper:
    def __init__(self, depth):
        self.model = _Dit(depth)
        self.diffusion_objective = "rectified_flow"


class _Model:
    def __init__(self, depth):
        self.model = _Wrapper(depth)


class _Sam:
    """Mirrors the attribute chain the accessors document."""

    def __init__(self, depth=4):
        self.model = _Model(depth)


def test_accessors_traverse_the_documented_paths():
    sam = _Sam(depth=4)

    assert internals.trunk_wrapper(sam) is sam.model.model
    assert internals.trunk_module(sam) is sam.model.model.model
    assert internals.trunk_blocks(sam) is sam.model.model.model.transformer.layers
    assert len(internals.trunk_blocks(sam)) == 4


def test_check_layout_accepts_the_expected_shape():
    internals.check_layout(_Sam(depth=2))


def test_check_layout_rejects_a_trunk_with_no_blocks():
    with pytest.raises(internals.LayoutError, match="no transformer blocks"):
        internals.check_layout(_Sam(depth=0))


def test_missing_attribute_names_the_failing_hop_and_the_vendor_pin():
    sam = _Sam()
    del sam.model.model.model.transformer.layers

    with pytest.raises(internals.LayoutError) as excinfo:
        internals.trunk_blocks(sam)

    message = str(excinfo.value)
    # The whole point is an actionable message rather than a bare
    # AttributeError from four levels down someone else's package.
    assert "trunk_blocks" in message
    assert "layers" in message
    assert internals.VENDOR_SHA in message
    assert "sa3_internals" in message


def test_replace_trunk_returns_the_previous_wrapper():
    sam = _Sam()
    original = internals.trunk_wrapper(sam)
    sentinel = object()

    previous = internals.replace_trunk(sam, sentinel)

    assert previous is original
    assert internals.trunk_wrapper(sam) is sentinel
    # Restoration is the caller's job and must round-trip exactly.
    internals.replace_trunk(sam, previous)
    assert internals.trunk_wrapper(sam) is original


def test_dit_architecture_config_extracts_objective_and_kwargs():
    config = {
        "model": {
            "diffusion": {
                "diffusion_objective": "rectified_flow",
                "config": {"depth": 24, "embed_dim": 1536},
            }
        }
    }

    objective, kwargs = internals.dit_architecture_config(config)

    assert objective == "rectified_flow"
    assert kwargs == {"depth": 24, "embed_dim": 1536}
    # Must be a copy: extensions mutate it to build a shallower branch.
    kwargs["depth"] = 12
    assert config["model"]["diffusion"]["config"]["depth"] == 24


def test_dit_architecture_config_defaults_the_objective():
    objective, _ = internals.dit_architecture_config(
        {"model": {"diffusion": {"config": {}}}},
    )
    assert objective == "v"


def test_dit_architecture_config_rejects_a_foreign_config():
    with pytest.raises(internals.LayoutError, match="model.diffusion.config"):
        internals.dit_architecture_config({"model": {}})


def test_api_version_and_vendor_pin_are_declared():
    # Extensions assert these at install; they must exist and be usable
    # as equality/identity checks.
    assert isinstance(internals.API_VERSION, int)
    assert internals.API_VERSION >= 1
    assert isinstance(internals.VENDOR_SHA, str) and len(internals.VENDOR_SHA) == 40


@pytest.mark.skipif(
    not sa3_vendor_present(), reason="vendored stable_audio_3 not on disk",
)
def test_vendored_dit_class_matches_the_documented_construction_surface():
    # The real canary: the vendored transformer must still be importable
    # where we look for it, and still accept the constructor arguments an
    # extension builds a derived branch from.
    import inspect

    cls = internals.dit_class()
    params = inspect.signature(cls.__init__).parameters

    for expected in ("depth", "diffusion_objective"):
        assert expected in params, (
            f"vendored DiffusionTransformer no longer accepts {expected!r}; "
            f"extensions building a derived branch will break "
            f"(pin {internals.VENDOR_SHA})"
        )
