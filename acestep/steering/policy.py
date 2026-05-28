"""Activation-steering policy: which axes are exposed, where they
inject, and how schedules translate.

Pure tables + one pure function. No torch, no disk I/O. The
SteeringController consumes this module; consumers that only need to
*describe* the surface (MCP, UI catalog builders) can import here
without torch.
"""

from __future__ import annotations

import re

from .types import AutoAxis


# Default probe schedule. Used when the bundle subpath doesn't encode
# one (older bundles) or for callers that don't have a subpath handy.
PROBE_N: int = 8


_PROBE_N_RE = re.compile(r"_n(\d+)(?:_|$)")


def parse_probe_n(subpath: str) -> int:
    """Pull the probe schedule N out of a bundle subpath.

    Subpaths look like ``v15-turbo/shift3.5_n8_seed1528``; the ``_n<int>_``
    token names the schedule. Falls back to ``PROBE_N`` when absent.
    """
    m = _PROBE_N_RE.search(subpath)
    return int(m.group(1)) if m else PROBE_N


# v1.5 turbo decoder has 24 DiT blocks → legal layers 0..23.
MANUAL_MAX_LAYER: int = 23

# Matches the demo's ``steps_override`` MIDI cap of 16 (0..15).
# Values past the live ``steps_override - 1`` silently no-op.
MANUAL_MAX_STEP: int = 15

MANUAL_SLOT_DEFAULT_COUNT: int = 1

# Soft cap on registered slots; sized so the UI doesn't collapse if
# someone stress-tests the surface (engine cost per slot is negligible).
MANUAL_SLOT_CAP: int = 16


# Manual catalog index order: axis-major per this list, then
# build_layer asc, then build_step asc. Pins each (axis, layer, step)
# cell to a stable index across sessions.
MANUAL_CATALOG_AXES: tuple[str, ...] = (
    "brightness",
    "warmth",
    "roughness",
    "density",
    "attack",
    "tonality",
    "punch",
    "bass_emphasis",
)


# Auto-path axes — only the four whose prompt-to-metric premise
# verified in PROMPT_BASELINE.md. The broken-premise four (attack,
# tonality, punch, bass_emphasis) stay reachable via the manual
# catalog only.
#
# warmth: sign=-1 flips the raw vector so positive alpha tilts warmer
# (raw probe direction is reversed for this axis).
# density: layer_offset=-3 from PHASE3_ANALYSIS.md (28/30 transfer
# pairs preferred 3 layers shallower than the probe).
AUTO_AXES: tuple[AutoAxis, ...] = (
    AutoAxis(
        name="steer_bright",
        axis="brightness",
        probe_layer=9,
        probe_step=3,
        sign=1.0,
        layer_offset=0,
        blurb=(
            "positive alpha shifts spectral centroid up "
            "(brighter, more highs)"
        ),
    ),
    AutoAxis(
        name="steer_warm",
        axis="warmth",
        probe_layer=15,
        probe_step=0,
        sign=-1.0,
        layer_offset=0,
        blurb=(
            "positive alpha tilts the spectrum toward bass (warmer); "
            "vector is sign-corrected from the raw probe direction"
        ),
    ),
    AutoAxis(
        name="steer_rough",
        axis="roughness",
        probe_layer=9,
        probe_step=3,
        sign=1.0,
        layer_offset=0,
        blurb=(
            "positive alpha increases spectral flatness "
            "(grittier, noisier); magnitude is small at this probe cell"
        ),
    ),
    AutoAxis(
        name="steer_density",
        axis="density",
        probe_layer=18,
        probe_step=3,
        sign=1.0,
        layer_offset=-3,
        blurb=(
            "positive alpha thins the texture toward sparse/minimal"
        ),
    ),
)


def fractional_inject_step(
    probe_step: int,
    inject_n: int,
    *,
    probe_n: int = PROBE_N,
) -> int:
    """Translate a probe-schedule step into the inject schedule.

    Returns ``round(probe_step / probe_n * inject_n)`` clamped to
    ``[0, inject_n - 1]``. Identity at same schedule
    (``inject_n == probe_n`` returns ``probe_step`` exactly).
    """
    if inject_n < 1:
        inject_n = 1
    if probe_n < 1:
        probe_n = 1
    s = int(round(probe_step / probe_n * inject_n))
    if s < 0:
        return 0
    if s >= inject_n:
        return inject_n - 1
    return s
