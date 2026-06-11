"""GeneratorBackend: the universal seam between the streaming runner and
whatever generates the audio.

The runner (:mod:`acestep.streaming.pipeline_runner`) owns pacing, the
adaptive playback lead, idle pause staging, loop-band decode targeting,
crossfading, and emission. Everything model-shaped — knob supply and
translation, generation, decode/render, stall signaling, per-generation
params echo — lives behind this protocol. The ACE-Step diffusion stack
implements it in :mod:`acestep.streaming.ace_backend`; future families
(SA3 via a ``ModelAdapter`` inside the diffusion backend, token/AR
models like Magenta RT2 directly) implement it without forking the
runner.

Design reference: ``round_3_BACKEND_PLAN_FINAL.md`` (repo root of the
authoring worktree). Deviations from that document's sketch, forced by
the real runner loop and recorded here so the doc can be updated from
working code:

* ``tick(knobs) -> AudioChunk`` is split into :meth:`~GeneratorBackend.produce`
  (one generation step) and :meth:`~GeneratorBackend.render_window`
  (audio at a position from current state). The runner performs up to
  two renders per tick (main window + loop-band wrap), gap-fills from
  the backend's cached state when production stalls, and keeps
  rendering through DiT-pause — one fused call cannot express that.
* :meth:`~GeneratorBackend.produce` takes a mode because the historical
  loop runs the full knob-translation/prepare path on EVERY active tick
  (shared-curve writes land even on ticks that skip the engine), and
  the idle DiT-pause path re-adopts cached state *as a fresh
  generation* (it bumps ``num_gens``, restamps the params echo, and
  feeds the feedback history). Byte-identical extraction requires
  modeling those modes explicitly.
* :meth:`~GeneratorBackend.sync_source` exists because source-swap /
  walk-chunk cache resets happen at a fixed point in the historical
  loop (after idle staging, before the knob read), and the runner's
  stall-deferral choreography observes backend cache state both before
  and after that point.
* Stall signaling is two boolean hooks at the two points the loop
  pre-covers today: refit work that happens inside ``before_tick``
  (:meth:`~GeneratorBackend.has_pending_refit`, checked before the hook
  runs) and rebuilds triggered by knob changes observed this tick
  (:meth:`~GeneratorBackend.rebuild_imminent`, checked after the knob
  read). The stall-magnitude estimate (prewarm) stays in the runner: it
  is learned from observed inter-write gaps, which are runner state.
* GPU timing (``tick_ms`` / ``dec_ms``) is measured inside the backend
  (it owns the device synchronization points); the runner reads
  :attr:`~GeneratorBackend.last_tick_ms` / :attr:`~GeneratorBackend.last_dec_ms`
  only for its latency trace line.
* :meth:`~GeneratorBackend.knob_specs` takes the session's enabled-LoRA
  id list instead of the plan's no-arg sketch: the enabled set is
  session-tracked state (pending initial enables at ``ready`` time,
  live toggles afterwards) that the backend cannot observe on its own,
  while WHICH specs that set expands to stays family knowledge behind
  the seam.

Transport-agnostic: numpy in, numpy out. No demo imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, runtime_checkable

import numpy as np

# Produce modes (see GeneratorBackend.produce):
#   "generate": run the engine for one step (the normal path).
#   "reuse":    DiT-pause — re-adopt cached state as this tick's fresh
#               result without touching the engine.
#   "skip":     stall deferral — run the prepare path but produce
#               nothing; the runner gap-fills from cached state.
ProduceMode = Literal["generate", "reuse", "skip"]


class UnsupportedOperation(Exception):
    """A capability-gated operation was invoked on a backend whose
    :class:`Capabilities` mask doesn't include it.

    Raised by the session's capability gate (and, for future backends,
    by backend control methods themselves) and mapped by the session to
    the typed ``command_failed`` wire event — never a silent no-op
    (plan §3.4). ``capability`` is the :class:`Capabilities` field name
    the operation requires.
    """

    def __init__(self, capability: str, message: str = ""):
        super().__init__(message or f"requires capability {capability!r}")
        self.capability = capability


@dataclass(frozen=True)
class TickContext:
    """Runner-side facts a backend may need for one tick.

    ``playhead_s`` is the monotonic estimate of the client's audible
    playhead; ``buffer_duration_s`` is the playable audio buffer length.
    Walk-window chunk selection is the current consumer; append-only
    backends ignore both.
    """

    playhead_s: float
    buffer_duration_s: float


@dataclass(frozen=True)
class AudioChunk:
    """One rendered window of audio, placed in song time.

    ``pcm`` is interleaved ``[frames, channels]`` float32 at the
    backend's declared :class:`AudioGeometry` sample rate.
    ``start_sample`` is the absolute song position of ``pcm[0]``.
    Backends with ``Capabilities.refines_audio`` may re-render
    previously emitted regions; append-only backends always return new
    regions.
    """

    pcm: np.ndarray
    start_sample: int


@dataclass(frozen=True)
class AudioGeometry:
    """Declared audio shape of a backend's output.

    ``duration_s = None`` is reserved for endless streams (the v2
    ``append`` song shape); fixed-duration backends always declare a
    real duration. ``chunk_rate_hz`` is the generation cadence (latent
    frame rate for diffusion, frame rate for AR models) — the lead
    tuning's prior, not a wire constant.
    """

    sample_rate: int
    channels: int
    chunk_rate_hz: float
    duration_s: Optional[float]


@dataclass(frozen=True)
class Capabilities:
    """What this backend can honor. Consumed by the session's typed ops
    (loud rejection of unsupported commands), the wire ``ready`` payload
    (Phase 2), and client panel gating. The union list from the plan;
    every field defaults False so a new backend starts minimal and opts
    in explicitly."""

    refines_audio: bool = False
    swap: bool = False
    timbre: bool = False
    structure: bool = False
    lora: bool = False
    stems: bool = False
    loop_band: bool = False
    depth: bool = False
    curves: bool = False
    notes_conditioning: bool = False
    # Activation steering (per-layer residual shifts driven by the
    # steer_* / man_*_<N> knobs and the manual_slot_add/pop commands).
    # True only when the backend has a steering controller with a
    # reachable vector bundle for its checkpoint.
    steering: bool = False


@dataclass(frozen=True)
class LeadProfile:
    """Default adaptive-lead bounds for this producer class.

    For a diffusion backend these mirror the historical runner
    defaults. For an append-only backend the buffered lead IS the
    knob-to-ear latency (committed audio cannot be revised), so its
    profile should sit near the underrun floor. ``SessionConfig``'s
    ``lead_*`` fields override per session; ``None`` here means "no
    backend opinion, use the runner's default".
    """

    floor_s: Optional[float] = None
    ceiling_s: Optional[float] = None
    release_tau_s: Optional[float] = None


@runtime_checkable
class GeneratorBackend(Protocol):
    """The universal generation seam. See module docstring.

    Per-tick call order from the runner (one iteration of its loop):

    1. :meth:`has_pending_refit` (+ :meth:`has_renderable_state`) —
       stall-deferral choreography around ``before_tick``.
    2. Runner idle staging (may end the iteration).
    3. :meth:`sync_source` — source-swap / walk-chunk cache resets.
    4. :meth:`read_knobs` — the coerced knob view for this tick.
    5. :meth:`rebuild_imminent` — second stall-deferral point.
    6. :meth:`produce` with the runner-chosen mode.
    7. :meth:`render_window` (windowed; possibly twice for a loop-band
       wrap) or :meth:`render_full` (legacy full-buffer mode).
    8. :meth:`on_fresh_generation` when produce reported fresh.

    Threading contract: every method is called on the runner thread
    only. Cross-thread mutations reach the backend the same way they
    reached the old loop body — via the session's ``before_tick``
    rendezvous — so implementations need no internal locking beyond
    what the wrapped engine already does.
    """

    name: str

    # GPU timing of the most recent produce / render, for the runner's
    # latency trace. Backend-measured because the backend owns the
    # device synchronization points.
    last_tick_ms: float
    last_dec_ms: float

    def capabilities(self) -> Capabilities: ...

    def geometry(self) -> AudioGeometry: ...

    def lead_profile(self) -> LeadProfile: ...

    def knob_specs(self, lora_ids=()) -> list:
        """The backend-owned knob manifest: the list of ``KnobSpec``
        (:mod:`acestep.streaming.knobs`) this backend exposes for the
        current session. Family knowledge lives here; the shared
        registry module stays pure schema/coercion machinery.

        ``lora_ids`` is the session's enabled-LoRA id set (the initial
        enable set at session start, the live set after runtime
        toggles) — session-tracked state the backend folds into its
        per-id strength knobs. Backends without the ``lora``
        capability ignore it.
        """
        ...

    # ---- hot loop --------------------------------------------------------

    def sync_source(self, ctx: TickContext) -> None:
        """Reconcile per-tick source state before the knob read:
        source-identity / length change detection (swap path) and
        walk-window chunk selection, including any cache invalidation
        those imply. No-op for backends without a positional source."""
        ...

    def read_knobs(self) -> dict:
        """Return the coerced knob view for this tick (the backend owns
        its knob state; the registry's coercion has already been
        applied upstream of it)."""
        ...

    def produce(self, knobs: dict, ctx: TickContext, mode: ProduceMode) -> bool:
        """Run one tick of the prepare+generate path.

        The prepare half (knob translation, conditioning/curve updates)
        runs in EVERY mode — live control changes must keep landing on
        in-flight work even when the engine step itself is skipped. The
        generate half depends on ``mode`` (see :data:`ProduceMode`).

        Returns True when this tick ended with a fresh generation to
        render and book-keep ("reuse" counts as fresh — that is the
        historical DiT-pause behavior); False when there is nothing new
        (mid-flight engine, or "skip") and the runner should gap-fill.
        """
        ...

    def render_window(self, t_start_s: float) -> Optional[AudioChunk]:
        """Render audio at song position ``t_start_s`` from the current
        generation state (fresh or cached). Position-honoring backends
        (``refines_audio``) decode exactly there; append-only backends
        ignore the hint and return their next frontier chunk. Returns
        None when no renderable state exists yet."""
        ...

    def render_full(self) -> Optional[AudioChunk]:
        """Render the entire playable buffer (legacy full-buffer path,
        ``vae_window <= 0``). May return None to skip emission this
        tick (e.g. the latent-MSE skip when the result barely moved).
        """
        ...

    def has_renderable_state(self) -> bool:
        """True once at least one successful :meth:`produce` has left
        state behind that :meth:`render_window` can re-render (gates
        gap-fill, DiT-pause reuse, and stall pre-coverage)."""
        ...

    def playable_duration_s(self) -> Optional[float]:
        """Current playable song duration in seconds, or None to let
        the runner fall back to the audio buffer length (walk-window
        mode does this). Tracks crop and source swaps."""
        ...

    # ---- stall signaling --------------------------------------------------

    def has_pending_refit(self) -> bool:
        """True when the NEXT ``before_tick`` will perform blocking
        refit work (e.g. a queued LoRA enable/disable). Checked by the
        runner BEFORE invoking ``before_tick`` so it can pre-cover the
        buffer and defer the hook by one iteration."""
        ...

    def rebuild_imminent(self, knobs: dict) -> bool:
        """True when producing with these knob values will trigger a
        blocking pipeline rebuild (e.g. a step-count change). Called
        exactly once per tick, after the knob read; implementations may
        use it to advance their change-detection state."""
        ...

    # ---- bookkeeping ------------------------------------------------------

    def on_fresh_generation(self, knobs: dict) -> None:
        """Mirror per-generation state (params echo, counters, sampled
        trace) after a fresh produce+render completed. A gap-fill tick
        never calls this."""
        ...
