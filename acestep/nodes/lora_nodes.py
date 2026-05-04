"""Single LoRA node: select a LoRA from the library, set its strength.

The node is the user-facing surface for the unified LoRA manager owned
by ``DiffusionEngine``. The same code path runs against TRT (refit) or
PyTorch (in-place ``param.data`` writeback) decoders — the node never
sees the difference.

Users drop ``*.safetensors`` into ``acestep.paths.loras_dir()`` and pick
from the dropdown. Each node instance owns one library entry: changing
the selection disables the previous one and enables the new one;
moving the strength slider issues a refit at the new strength.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Optional

from loguru import logger

from .base import BaseNode, NodeDefinition, NodeParam, NodePort, NodeRegistry
from .types import ModelHandle


# Sentinel option meaning "this slot is empty / disable any previously
# enabled LoRA on this node". Sits at the top of the dropdown so a fresh
# node defaults to "no LoRA applied".
_NONE = "(none)"


def _scan_library_options() -> tuple[str, ...]:
    """Return the dropdown options: ``(none)`` plus every LoRA stem.

    Called from ``get_definition()`` at node-class import time so the
    bridge picks up whatever LoRAs exist when the plugin loads.
    """
    from acestep.paths import discover_loras

    try:
        files = discover_loras()
    except Exception as e:
        logger.warning("LoRA library scan failed: %s", e)
        return (_NONE,)
    return (_NONE,) + tuple(p.stem for p in files)


@NodeRegistry.register
class LoRA(BaseNode):
    """Apply a LoRA from the library at a given strength.

    Stack multiple instances of this node in series to combine multiple
    LoRAs; each node owns its own library entry independently. Strength
    is hot-swappable: dragging the slider issues a single refit per
    delta and the next forward pass sees the new weights.

    Inputs:
        model: the upstream MODEL handle.

    Outputs:
        model: the same handle, returned unchanged (the engine state
            it points at is what mutated).

    Params:
        lora: dropdown selection from the LoRA library (sentinel
            ``(none)`` disables any LoRA previously enabled by this
            node).
        strength: slider in [0, 2]. 0 keeps the LoRA enabled but
            contributes nothing (placeholder pattern for ramping up
            mid-stream).
    """

    node_type_id: ClassVar[str] = "acestep.LoRA"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="LoRA",
            category="model",
            description="Apply a LoRA from the library to the model.",
            inputs=(
                NodePort(name="model", type="MODEL"),
            ),
            outputs=(
                NodePort(name="model", type="MODEL"),
            ),
            params=(
                NodeParam(
                    name="lora", type="select", default=_NONE,
                    description="LoRA from the library",
                    options=_scan_library_options(),
                ),
                NodeParam(
                    name="strength", type="number", default=1.0,
                    description="LoRA strength",
                    min=0.0, max=2.0, step=0.05,
                ),
            ),
        )

    def __init__(self) -> None:
        # Per-instance state: the lora id this node currently has
        # enabled in the engine. Used to detect dropdown swaps so the
        # previously-enabled entry can be disabled before enabling the
        # new one.
        self._enabled_id: Optional[str] = None

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        model_handle: ModelHandle = kwargs["model"]
        selection = kwargs.get("lora", _NONE)
        strength = float(kwargs.get("strength", 1.0))

        engine = self._resolve_engine(model_handle)
        if engine is None or not engine.lora_available:
            if selection != _NONE:
                logger.warning(
                    "LoRA node: backend has no LoRA manager available; "
                    "selection %r ignored",
                    selection,
                )
            return {"model": model_handle}

        target_id = self._resolve_target_id(engine, selection)

        # Swap: previously enabled id no longer matches.
        if self._enabled_id is not None and self._enabled_id != target_id:
            try:
                engine.disable_lora(self._enabled_id)
            except Exception as e:
                logger.warning(
                    "LoRA node: disable_lora(%s) failed: %s",
                    self._enabled_id, e,
                )
            self._enabled_id = None

        if target_id is None:
            return {"model": model_handle}

        if self._enabled_id is None:
            engine.enable_lora(target_id, strength=strength)
            self._enabled_id = target_id
        else:
            # Already enabled; idempotent strength update.
            engine.set_lora_strength(target_id, strength)

        return {"model": model_handle}

    @staticmethod
    def _resolve_engine(model_handle: ModelHandle):
        handler = getattr(model_handle, "handler", None)
        return getattr(handler, "_diffusion_engine", None)

    @staticmethod
    def _resolve_target_id(engine, selection: str) -> Optional[str]:
        """Translate a dropdown value into a manager-known LoRA id.

        ``(none)`` and empty strings are the disabled state. Anything
        else is treated first as an existing id; if the manager doesn't
        know it but the same stem exists in the library directory,
        register-by-path and use the resulting id (covers the case
        where a LoRA was added to the directory after the manager
        scanned the library).
        """
        if not selection or selection == _NONE:
            return None

        existing = {d.id for d in engine.list_loras()}
        if selection in existing:
            return selection

        from acestep.paths import discover_loras
        for p in discover_loras():
            if p.stem == selection:
                return engine.register_lora(str(p))

        logger.warning(
            "LoRA node: selection %r not in library and not registered",
            selection,
        )
        return None
