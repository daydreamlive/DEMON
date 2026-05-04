"""Tests for the unified ``acestep.LoRA`` node.

The node sits in front of the manager and is responsible for the
swap-on-selection-change and the per-instance "previously enabled id"
bookkeeping. We mock the engine so the assertions are about node
behavior, not manager internals.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.nodes.lora_nodes import LoRA, _NONE
from acestep.nodes.types import ModelHandle


def _make_engine(catalog_ids=("a", "b")):
    """Mock engine with list_loras / enable_lora / disable_lora / set_lora_strength."""
    engine = MagicMock()
    engine.lora_available = True
    engine.list_loras.return_value = [
        MagicMock(id=lid) for lid in catalog_ids
    ]
    return engine


def _make_handle(engine):
    handler = MagicMock()
    handler._diffusion_engine = engine
    return ModelHandle(handler=handler)


def test_none_selection_is_passthrough():
    """Default state: no LoRA selected, manager untouched."""
    engine = _make_engine()
    node = LoRA()
    out = node.execute(model=_make_handle(engine), lora=_NONE, strength=1.0)
    assert "model" in out
    engine.enable_lora.assert_not_called()
    engine.disable_lora.assert_not_called()


def test_select_enables_at_strength():
    engine = _make_engine()
    node = LoRA()
    node.execute(model=_make_handle(engine), lora="a", strength=0.5)

    engine.enable_lora.assert_called_once_with("a", strength=0.5)
    assert node._enabled_id == "a"


def test_strength_change_triggers_set_strength_not_re_enable():
    engine = _make_engine()
    node = LoRA()
    node.execute(model=_make_handle(engine), lora="a", strength=0.5)
    engine.enable_lora.reset_mock()

    node.execute(model=_make_handle(engine), lora="a", strength=0.7)

    engine.enable_lora.assert_not_called()
    engine.set_lora_strength.assert_called_once_with("a", 0.7)


def test_swap_disables_old_enables_new():
    """Changing the dropdown selection must drop the previous LoRA cleanly."""
    engine = _make_engine()
    node = LoRA()
    node.execute(model=_make_handle(engine), lora="a", strength=0.5)
    engine.enable_lora.reset_mock()

    node.execute(model=_make_handle(engine), lora="b", strength=0.4)

    engine.disable_lora.assert_called_once_with("a")
    engine.enable_lora.assert_called_once_with("b", strength=0.4)
    assert node._enabled_id == "b"


def test_select_then_none_disables():
    engine = _make_engine()
    node = LoRA()
    node.execute(model=_make_handle(engine), lora="a", strength=0.5)

    node.execute(model=_make_handle(engine), lora=_NONE, strength=0.5)

    engine.disable_lora.assert_called_once_with("a")
    assert node._enabled_id is None


def test_unavailable_backend_warns_but_passes_through():
    """When the engine has no LoRA manager, the node must not crash."""
    engine = MagicMock()
    engine.lora_available = False
    node = LoRA()
    out = node.execute(model=_make_handle(engine), lora="a", strength=0.5)
    assert "model" in out
    engine.enable_lora.assert_not_called()


def test_definition_options_include_none_sentinel():
    defn = LoRA.get_definition()
    lora_param = next(p for p in defn.params if p.name == "lora")
    assert lora_param.type == "select"
    assert _NONE in (lora_param.options or ())
