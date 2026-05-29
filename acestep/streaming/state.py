"""Mutable session state for a streaming generative session.

Single source of truth for one streaming session's mutable cells.
The dispatcher and the runner share this object instead of a
constellation of list-wrapped ``*_ref`` cells.

Per-subscriber transport state (``client_mirror``, zstd context,
slice epoch counters, ``send_lock``) is intentionally **not** here.
That state lives in the transport adapter because each subscriber may
have attached at a different point in the stream and serializes its
own delta basis.

The lock is taken explicitly by callers that perform non-atomic
mutations (``pending_*`` list mutation, ``swap_pending`` dict
mutation). Plain field reads/writes rely on CPython's GIL atomicity.
``RLock`` (rather than ``Lock``) so a closure can re-enter under an
already-held lock without deadlocking.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


def _default_swap_pending() -> dict:
    """Empty swap_pending slot. Mirrors the shape callers expect.

    ``waveform`` is a decoded ``torch.Tensor`` ([≤2, N], float32): the
    transport-edge byte decode lives in the WS adapter so the
    runner-side drain consumes a value object, not wire bytes."""
    return {
        "waveform": None,
        "tags": None,
        "key": None,
        "time_signature": None,
        "fixture_name": None,
        "stem_source_mode": None,
    }


def _default_params() -> dict:
    """Runtime telemetry dict populated by the runner each tick and
    read by the transport adapter for the ``params_update`` wire event."""
    return {"num_gens": 0, "tick_ms": 0.0, "dec_ms": 0.0}


@dataclass
class SessionState:
    """Single source of truth for one streaming session's mutable state.

    Required fields (passed at construction) cover everything resolved
    during the per-connect setup: source latent + audio context, the
    A/B conditioning pairs, the initial prompts, and the active
    pipeline depth. Defaulted fields cover slider values, optional
    overrides, pending cross-thread queues, telemetry, and lifecycle.
    """

    # === Source / model context (resolved before SessionState is built) ===
    source: Any                          # PreparedSource (avoid heavy import)
    bpm: int
    key: str
    time_signature: str
    duration: float
    n_channels: int
    playback_samples: int

    # === Conditioning cache + prompts ===
    cond_pair: tuple                     # (cond_silence, cond_full) for prompt A
    cond_pair_b: tuple                   # (cond_silence_b, cond_full_b) for prompt B
    prompt_text: str                     # prompt A (read by runner each tick)
    prompt_text_b: str                   # prompt B (dispatcher only)
    current_depth: int                   # active pipeline_depth

    # === Slider values driving the engine ===
    prompt_blend: float = 0.0
    timbre_strength: float = 1.0

    # === Timbre override (uploaded ref vs. self) ===
    timbre_latent: Any = None            # Latent | None
    timbre_name: str | None = None

    # === Structure (semantic-hint) override ===
    struct_audio: Any = None             # torch.Tensor [C, N] | None
    struct_context: Any = None           # Latent | None
    struct_name: str | None = None

    # === Pending cross-thread queues (drained inside before_tick) ===
    # ``pending_enable`` carries (id, strength_or_None) tuples. Applying
    # at-strength avoids a first-window-without-LoRA artifact when the
    # runner's per-tick set_strength catch-up only kicks in after tick 1.
    pending_enable: list = field(default_factory=list)
    pending_disable: list = field(default_factory=list)
    pending_depth: int | None = None
    swap_pending: dict = field(default_factory=_default_swap_pending)

    # === Activity gating for the idle pause ===
    # The runner reads ``last_activity_ts`` each tick. Dispatchers bump
    # it only on meaningful messages (see ``last_params_raw`` diff in
    # backend's _dispatch_message) so the 125 Hz params heartbeat
    # doesn't defeat the pause.
    last_activity_ts: float = field(default_factory=time.monotonic)
    last_params_raw: Any = None          # dict | None

    # === Runtime telemetry dict mutated by the runner each tick ===
    params: dict = field(default_factory=_default_params)

    # === Session lifecycle (runner sets to False on tick error) ===
    running: bool = True

    # === Runner display / motion (read each tick) ===
    sde_curve_display: Any = None
    motion_val: float = 0.0

    # === Single re-entrant lock for cross-thread mutations ===
    # Held by callers that mutate ``pending_*`` lists, ``swap_pending``
    # dict, or that read/write multiple fields atomically. Plain
    # single-field reads/writes don't take the lock (GIL atomicity).
    _lock: threading.RLock = field(default_factory=threading.RLock)
