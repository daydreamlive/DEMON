"""Command origin for the streaming-session API.

Two ``StreamingSession`` verbs branch on origin: ``set_knobs`` and
``set_prompt_blend``.

- ``PRIMARY`` is the browser, or any direct API caller that owns the UI
  layer's smoothing tweens. The session applies the command and emits
  the corresponding *applied* / *update* event.
- ``EXTERNAL`` is the MCP control bus, or any secondary transport that
  pushes a target value without owning the UI tween. For the two
  origin-dependent verbs the session does NOT mutate state; it emits an
  *echo* event so the primary transport's UI can smooth toward the
  target and re-send the tweened sequence as ``PRIMARY``.

All other operations are origin-agnostic. They accept an ``origin``
kwarg for log traceability only; behavior is identical.

The ops that already pass through ``SessionState.pending_*`` queues
(``enable_lora``, ``disable_lora``, ``set_depth``, ``swap_source``)
use those queues as the typed dispatch surface (each entry's tuple /
dict shape is fixed by the consumer in ``apply_pending``), so no
wrapper Command dataclasses are introduced here. The public surface
is the typed method on ``StreamingSession``; the internal carrier is
the existing ``SessionState`` field.
"""

from __future__ import annotations

from enum import Enum


class CommandOrigin(Enum):
    """Where a command came from.

    Only two operations branch on this. See module docstring.
    """

    PRIMARY = "primary"
    EXTERNAL = "external"
