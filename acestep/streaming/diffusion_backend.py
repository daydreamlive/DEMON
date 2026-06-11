"""DiffusionBackend: the shared Tier-1 skeleton for diffusion families.

``round_3_BACKEND_PLAN_FINAL.md`` §2: the in-process diffusion stack
behind the :class:`~acestep.streaming.generator_backend.GeneratorBackend`
seam, parameterized by ``(ModelAdapter, codec)`` — the Tier-2 model
seam (:mod:`acestep.engine.model_adapter`) and the family's
latent→audio decoder. This base owns what every diffusion family
shares:

* the ``produce()`` mode skeleton (``generate`` / ``reuse`` / ``skip``)
  with its renderable-state caching, GPU timing capture, and the
  prepare-runs-every-mode rule (live control changes must keep landing
  on in-flight work even when the engine step is skipped);
* the renderable-state predicate the runner's gap-fill / DiT-pause /
  stall pre-coverage choreography gates on;
* neutral defaults for the contract hooks a family may not need
  (``sync_source``, stall signaling, ``lead_profile``).

Families subclass and implement ``_prepare_tick`` / ``_generate`` /
``_after_produce`` plus the contract surface (``capabilities`` /
``geometry`` / ``knob_specs`` / render methods):

* :class:`~acestep.streaming.ace_backend.ACEStepBackend` — ACE-Step
  v1.5. Its Tier-2 adapter is pipeline-owned (the default
  ``ACEAdapter`` built inside its ``StreamHandle``'s pipeline) and its
  codec is the engine ``Session`` (windowed VAE decode), so it passes
  ``codec=session`` and leaves ``adapter`` None here.
* the SA3 backend — owns both: an ``SA3Adapter`` over the shared
  ``StreamPipeline`` and a SAME-decode codec with the 44.1→48 kHz
  resample at the decode boundary.
"""

from __future__ import annotations

import time

import torch

from acestep.streaming.generator_backend import (
    LeadProfile,
    ProduceMode,
    TickContext,
)


class DiffusionBackend:
    """Shared diffusion-family Tier-1 mechanics. See module docstring."""

    name = "diffusion"

    def __init__(self, *, adapter=None, codec=None):
        # Tier-2 model adapter + family codec. Either may be None when
        # the concrete family owns the object elsewhere (ACE's adapter
        # lives on its StreamHandle's pipeline).
        self.adapter = adapter
        self.codec = codec

        # Most recent successful generation; feeds gap-fill, DiT-pause
        # reuse, and stall pre-coverage (has_renderable_state).
        self._last_result_latent = None
        # Result of THIS tick's produce (None on skip / mid-flight).
        self._current_result = None

        # GPU timing of the most recent produce / render, read by the
        # runner for its latency trace.
        self.last_tick_ms = 0.0
        self.last_dec_ms = 0.0

    # ---- contract defaults --------------------------------------------------

    def lead_profile(self) -> LeadProfile:
        # No opinion beyond the runner's historical defaults; the
        # SessionConfig lead_* fields keep overriding per session.
        return LeadProfile()

    def sync_source(self, ctx: TickContext) -> None:
        # No positional source by default.
        pass

    def has_pending_refit(self) -> bool:
        return False

    def rebuild_imminent(self, knobs: dict) -> bool:
        return False

    def has_renderable_state(self) -> bool:
        return self._last_result_latent is not None

    # ---- produce-mode skeleton ----------------------------------------------

    def produce(self, knobs: dict, ctx: TickContext, mode: ProduceMode) -> bool:
        """The historical loop's produce shape, family-independent.

        The prepare half (:meth:`_prepare_tick`) runs in EVERY mode;
        the generate half (:meth:`_generate`) only in ``"generate"``.
        ``"reuse"`` re-adopts the cached latent as a fresh result
        (DiT-pause), ``"skip"`` produces nothing and the runner
        gap-fills. Timing brackets the engine step, as before.
        """
        prep = self._prepare_tick(knobs, ctx)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if mode == "reuse":
            result_latent = self._last_result_latent
        elif mode == "skip":
            result_latent = None
        else:
            result_latent = self._generate(prep)

        # Cache the most recent successful latent so the DiT-pause and
        # gap-fill paths have something to feed the renderer.
        if result_latent is not None:
            self._last_result_latent = result_latent

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_tick_ms = (time.perf_counter() - t0) * 1000
        self.last_dec_ms = 0.0

        self._current_result = result_latent
        is_fresh = result_latent is not None
        self._after_produce(prep, result_latent, is_fresh)
        return is_fresh

    # ---- family hooks --------------------------------------------------------

    def _prepare_tick(self, knobs: dict, ctx: TickContext) -> dict:
        """Translate knobs, write shared curves / conditioning onto
        in-flight work, and assemble everything :meth:`_generate`
        needs. Runs on every active tick regardless of mode."""
        raise NotImplementedError

    def _generate(self, prep: dict):
        """Run one engine step from the prepared state; return the
        family's renderable result (or None mid-flight)."""
        raise NotImplementedError

    def _after_produce(self, prep: dict, result_latent, is_fresh: bool) -> None:
        """Per-produce bookkeeping (history rings, params-echo stash).
        Default: nothing."""
