"""Frozen dataclasses + exceptions for the activation-steering surface.

Pure data; no torch / disk I/O so torch-free consumers (MCP catalog,
UI metadata) can import without paying for the controller's torch
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoAxis:
    """One verified auto-path steering axis.

    ``sign`` is folded into alpha at config-build time so the on-disk
    cache stays raw (the manual path reads the same cache and is
    sign-agnostic by design).
    """

    name: str           # knob name, e.g. "steer_bright"
    axis: str           # disk-side axis tag, e.g. "brightness"
    probe_layer: int
    probe_step: int
    sign: float         # -1.0 flips the raw vector direction; 1.0 leaves it
    layer_offset: int   # added to probe_layer at inject time
    blurb: str          # operator-facing effect description


@dataclass(frozen=True)
class CatalogEntry:
    """One pre-built (axis, build_layer, build_step) cell from disk."""

    index: int
    axis: str
    build_layer: int
    build_step: int
    filename: str


@dataclass(frozen=True)
class KnobNames:
    """Wire knob names for one manual steering slot."""

    src: str    # man_src_<N>
    layer: str  # man_layer_<N>
    step: str   # man_step_<N>
    alpha: str  # man_alpha_<N>


class SteeringError(Exception):
    """Base class for slot-registry errors raised by SteeringController."""


class CapacityError(SteeringError):
    """``add_slot`` called when the registry is already at ``slot_cap``."""


class EmptyError(SteeringError):
    """``pop_slot`` called when the registry has no allocated slots."""
