"""Optional product mapping for the Stable Audio 3 denoise control.

The corrected upstream schedule makes ``sigma_max`` mathematically monotonic,
but measured audio change is still concentrated near the top of its range. In
a 41-point sweep over nine clips, less than one fifth of the available change
occurred below sigma 0.70. The knot table below inverts that measured curve so
equal control movements target roughly equal movements in the referee score.

The referee combines harmonic and rhythmic change and reproduced two separate
listening-order judgments that a loudness-based measure did not. It validates
the ordering and the location of the dead region, not a claim that every step
is equally perceptible. This is also one global mapping: sparse acoustic input
can move faster than dense electronic material.

Set ``DEMON_SA3_DENOISE_MAPPING=identity`` to bypass the product mapping while
retaining the upstream monotonic-schedule bugfix.
"""

from __future__ import annotations

import os
import typing as tp

__all__ = [
    "denoise_mapping_mode",
    "dial_to_entry_sigma",
    "map_denoise_to_entry_sigma",
]


# Dial position -> entry sigma, obtained by inverting the measured change curve.
# Both endpoints remain exact: zero preserves the source and one starts from
# pure noise. Values between measured knots interpolate linearly.
_CALIBRATION: tp.Sequence[tuple[float, float]] = (
    (0.00, 0.0000),
    (0.10, 0.3786),
    (0.20, 0.5762),
    (0.30, 0.6853),
    (0.40, 0.7321),
    (0.50, 0.7683),
    (0.65, 0.8376),
    (0.80, 0.8843),
    (1.00, 1.0000),
)


def denoise_mapping_mode() -> str:
    """Return ``calibrated`` (default) or the rollback mode ``identity``."""
    mode = os.environ.get("DEMON_SA3_DENOISE_MAPPING", "calibrated").strip().lower()
    if mode not in ("calibrated", "identity"):
        raise ValueError(
            "DEMON_SA3_DENOISE_MAPPING must be calibrated|identity, "
            f"got {mode!r}"
        )
    return mode


def dial_to_entry_sigma(dial: float) -> float:
    """Interpolate the measured monotonic mapping, clamped to ``[0, 1]``."""
    position = min(max(float(dial), 0.0), 1.0)
    if position <= _CALIBRATION[0][0]:
        return _CALIBRATION[0][1]

    for (p0, s0), (p1, s1) in zip(_CALIBRATION, _CALIBRATION[1:]):
        if position <= p1:
            fraction = (position - p0) / (p1 - p0)
            return s0 + (s1 - s0) * fraction

    return _CALIBRATION[-1][1]


def map_denoise_to_entry_sigma(dial: float) -> float:
    """Apply the selected product mapping without changing endpoint semantics."""
    clamped = min(max(float(dial), 0.0), 1.0)
    if denoise_mapping_mode() == "identity":
        return clamped
    return dial_to_entry_sigma(clamped)
