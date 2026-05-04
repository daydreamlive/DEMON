"""LoRA loading and application nodes.

Both nodes route through the unified ``DiffusionEngine`` LoRA API,
which dispatches internally to the eager (PyTorch in-place writeback)
or TRT (IRefitter) backend depending on what's loaded. Callers don't
need to branch.
"""

from __future__ import annotations

from typing import Any, ClassVar

from loguru import logger

from .base import BaseNode, NodeDefinition, NodeParam, NodePort, NodeRegistry
from .types import LoRA, ModelHandle


@NodeRegistry.register
class LoadLoRA(BaseNode):
    """Load a LoRA adapter from a safetensors file.

    Returns a wire payload carrying the path + scale; the actual
    materialization happens in :class:`ApplyLoRA`.
    """

    node_type_id: ClassVar[str] = "acestep.LoadLoRA"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Load LoRA",
            category="model",
            description="Load a LoRA adapter from a safetensors file.",
            inputs=(),
            outputs=(
                NodePort(name="lora", type="LORA"),
            ),
            params=(
                NodeParam(
                    name="path", type="string", default="",
                    description="Path to .safetensors LoRA file",
                ),
                NodeParam(
                    name="scale", type="number", default=1.0,
                    description="LoRA strength",
                    min=0.0, max=2.0, step=0.05,
                ),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        path = kwargs["path"]
        scale = kwargs.get("scale", 1.0)
        return {"lora": LoRA(path=str(path), scale=float(scale))}


@NodeRegistry.register
class ApplyLoRA(BaseNode):
    """Apply a LoRA adapter to the model and return the modified handle."""

    node_type_id: ClassVar[str] = "acestep.ApplyLoRA"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Apply LoRA",
            category="model",
            description="Apply a LoRA adapter to the model for generation.",
            inputs=(
                NodePort(name="model", type="MODEL"),
                NodePort(name="lora", type="LORA"),
            ),
            outputs=(
                NodePort(name="model", type="MODEL"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        model_handle: ModelHandle = kwargs["model"]
        lora: LoRA = kwargs["lora"]
        handler = model_handle.handler

        engine = getattr(handler, "_diffusion_engine", None)
        if engine is None or not engine.lora_available:
            logger.warning(
                "ApplyLoRA: no LoRA backend available on this model; "
                "skipping %s", lora.path,
            )
            return {"model": model_handle}

        lora_id = engine.apply_lora(lora.path, lora.scale)
        # Stack of ids so RemoveLoRA can pop the most recent.  Same
        # contract the old per-backend stacks honored, just unified.
        if not hasattr(handler, "_active_lora_ids"):
            handler._active_lora_ids = []
        handler._active_lora_ids.append(lora_id)
        return {"model": model_handle}


@NodeRegistry.register
class RemoveLoRA(BaseNode):
    """Remove the most recently applied LoRA from the model."""

    node_type_id: ClassVar[str] = "acestep.RemoveLoRA"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Remove LoRA",
            category="model",
            description="Remove the most recently applied LoRA adapter.",
            inputs=(
                NodePort(name="model", type="MODEL"),
            ),
            outputs=(
                NodePort(name="model", type="MODEL"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        model_handle: ModelHandle = kwargs["model"]
        handler = model_handle.handler
        engine = getattr(handler, "_diffusion_engine", None)
        if engine is None:
            return {"model": model_handle}

        ids = getattr(handler, "_active_lora_ids", [])
        if ids:
            engine.remove_lora(ids.pop())
        return {"model": model_handle}
