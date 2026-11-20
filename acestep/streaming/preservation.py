"""Transport-neutral validation for a circular source-preservation envelope."""

import math

MAX_PRESERVATION_POINTS = 256


def parse_preservation_curve(value):
    """None/empty clears. Otherwise accept 2..256 finite numbers in [0, 1].

    Samples are equally spaced over playable source time, endpoint excluded;
    interpolation wraps from the last sample back to the first. Reject whole
    malformed updates so a partial/bad gesture cannot erase the previous one.
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("preservation curve must be a list or null")
    if not value:
        return None
    if not 2 <= len(value) <= MAX_PRESERVATION_POINTS:
        raise ValueError("preservation curve needs 2..256 samples")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not 0 <= v <= 1 or not math.isfinite(v) for v in value):
        raise ValueError("preservation samples must be finite numbers in [0, 1]")
    return tuple(float(v) for v in value)
