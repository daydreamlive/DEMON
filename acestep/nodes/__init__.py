"""ACE-Step node system: typed, composable operations for audio generation."""

from .types import (
    Audio,
    ChannelGuidanceEntry,
    CLIPHandle,
    Conditioning,
    ConditioningEntry,
    Config,
    Curve,
    Latent,
    Mask,
    ModelHandle,
    VAEHandle,
    all_type_names,
    get_type_class,
    types_compatible,
)
from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry

__all__ = [
    # Types
    "Audio",
    "ChannelGuidanceEntry",
    "CLIPHandle",
    "Conditioning",
    "ConditioningEntry",
    "Config",
    "Curve",
    "Latent",
    "Mask",
    "ModelHandle",
    "VAEHandle",
    "all_type_names",
    "get_type_class",
    "types_compatible",
    # Framework
    "BaseNode",
    "NodeDefinition",
    "NodePort",
    "NodeRegistry",
]
