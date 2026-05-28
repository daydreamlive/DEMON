"""Activation-steering engine surface.

Wire knob names:
  - ``steer_bright`` / ``steer_warm`` / ``steer_rough`` / ``steer_density``:
    the four verified auto-path axes.
  - ``man_src_<N>`` / ``man_layer_<N>`` / ``man_step_<N>`` / ``man_alpha_<N>``:
    LIFO-numbered manual slot quadruples.

Auto-axis alpha is sign-corrected at config-build time; the on-disk
cache stays raw so the manual path (direction-agnostic by design) sees
the literal probe direction.
"""

from .catalog import enumerate_catalog, load_auto_vectors, load_vector
from .controller import SteeringController
from .hub import ensure_steering_vectors, upload_steering_vectors
from .policy import (
    AUTO_AXES,
    MANUAL_CATALOG_AXES,
    MANUAL_MAX_LAYER,
    MANUAL_MAX_STEP,
    MANUAL_SLOT_CAP,
    MANUAL_SLOT_DEFAULT_COUNT,
    PROBE_N,
    fractional_inject_step,
)
from .types import (
    AutoAxis,
    CapacityError,
    CatalogEntry,
    EmptyError,
    KnobNames,
    SteeringError,
)

__all__ = [
    "AUTO_AXES",
    "AutoAxis",
    "CapacityError",
    "CatalogEntry",
    "EmptyError",
    "KnobNames",
    "MANUAL_CATALOG_AXES",
    "MANUAL_MAX_LAYER",
    "MANUAL_MAX_STEP",
    "MANUAL_SLOT_CAP",
    "MANUAL_SLOT_DEFAULT_COUNT",
    "PROBE_N",
    "SteeringController",
    "SteeringError",
    "ensure_steering_vectors",
    "enumerate_catalog",
    "fractional_inject_step",
    "load_auto_vectors",
    "load_vector",
    "upload_steering_vectors",
]
