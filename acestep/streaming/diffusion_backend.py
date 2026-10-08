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

import numpy as np
import torch

from acestep.engine.stream import next_submit_seq
from acestep.streaming.generator_backend import (
    AudioChunk,
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
        # Submit stamp (acestep.engine.stream.next_submit_seq) of the
        # request _last_result_latent was generated from: the freshness
        # signal behind bounce_ready.
        self._last_result_seq = 0
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

    def lora_compatible(self, metadata: dict) -> bool:
        # Permissive default per the seam contract: a family that can't
        # classify a LoRA against its engine treats it as loadable.
        # Families with a real axis override (ACE: base-model scale).
        return True

    # ---- LoRA facade (D2): engine_obj-delegating defaults -------------------
    #
    # ACE's LoRA managers live inside its DiffusionEngine (``engine_obj``
    # on the concrete backend); these defaults route the session's LoRA
    # plumbing there so ACE behavior is unchanged by the facade. SA3
    # overrides the whole block with its own manager. A family with
    # neither (engine_obj None / absent) reports unavailable and fails
    # loudly on mutation.

    def _lora_engine(self):
        return getattr(self, "engine_obj", None)

    def _require_lora_engine(self):
        eng = self._lora_engine()
        if eng is None:
            raise RuntimeError(
                f"backend {self.name!r} has no LoRA engine; the session's "
                "capability gate should have rejected this command"
            )
        return eng

    def lora_available(self) -> bool:
        eng = self._lora_engine()
        return bool(eng is not None and getattr(eng, "lora_available", False))

    def register_lora(self, path: str) -> str:
        return self._require_lora_engine().register_lora(path)

    def prewarm_lora(self, lora_id: str):
        return self._require_lora_engine().prewarm_lora(lora_id)

    def enable_lora(self, lora_id: str, strength=None) -> None:
        self._require_lora_engine().enable_lora(lora_id, strength=strength)

    def disable_lora(self, lora_id: str) -> None:
        self._require_lora_engine().disable_lora(lora_id)

    def set_lora_strength(self, lora_id: str, strength: float) -> None:
        self._require_lora_engine().set_lora_strength(lora_id, strength)

    def list_loras(self) -> list:
        eng = self._lora_engine()
        return eng.list_loras() if eng is not None else []

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
            pipe = self._stream_pipeline()
            ticks_before = getattr(pipe, "ticks", None)
            result_latent = self._generate(prep)
            if result_latent is not None:
                self._last_result_seq = self._finished_seq(pipe, ticks_before)

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

    # ---- bounce (Capabilities.bounce) ------------------------------------------

    def _stream_pipeline(self):
        """The StreamPipeline this backend submits to (None if none)."""
        return getattr(self, "pipeline", None)

    def _finished_seq(self, pipe_before, ticks_before) -> int:
        """Submit stamp of the latent ``_generate`` just returned.

        A pipeline that did not tick produced it synchronously from the
        current state (e.g. a zero-denoise passthrough), so it is as
        fresh as a new mark."""
        pipe = self._stream_pipeline()
        if pipe is None or (
            pipe is pipe_before and getattr(pipe, "ticks", None) == ticks_before
        ):
            return next_submit_seq()
        req = getattr(pipe, "last_finished_request", None)
        return int(getattr(req, "submit_seq", 0) or 0)

    def bounce_mark(self) -> int:
        """Freshness mark for a bounce: taken on the runner thread once
        every earlier control change has been applied."""
        return next_submit_seq()

    def bounce_ready(self, mark: int) -> bool:
        """True once the latest generation began denoising after
        ``mark`` — i.e. it reflects every change applied before it."""
        return self._last_result_latent is not None and self._last_result_seq > mark

    def render_bounce(self):
        """The whole playable window of the latest generation as one
        :class:`AudioChunk` (``start_sample=0``), or None when the
        backend has no fixed window (walk mode) or no state yet.

        Default: tile :meth:`render_window` across the window with the
        runner's 25 ms crossfade at the seams, so the result is exactly
        what the stream would converge to. Runner thread only."""
        dur = self.playable_duration_s()
        if dur is None or self.vae_window <= 0 or not self.has_renderable_state():
            return None
        sr = self.geometry().sample_rate
        total = int(round(dur * sr))
        xfade = min(1200, int(round(self.vae_window * sr)) // 4)
        timing = (self.last_tick_ms, self.last_dec_ms)
        out = None
        pos = written = 0
        try:
            while written < total:
                chunk = self.render_window(pos / sr)
                if chunk is None or len(chunk.pcm) == 0:
                    return None
                pcm = np.array(chunk.pcm, dtype=np.float32)
                if out is None:
                    out = np.zeros((total, pcm.shape[1]), dtype=np.float32)
                s = max(0, int(chunk.start_sample))
                e = min(s + len(pcm), total)
                if e <= written:
                    return None  # render made no progress; don't spin
                n = min(xfade, max(0, written - s), e - s)
                if n > 0:
                    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
                    pcm[:n] = out[s:s + n] * (1 - ramp) + pcm[:n] * ramp
                out[s:e] = pcm[:e - s]
                written = e
                pos = max(pos + 1, e - xfade)
        finally:
            # A bounce is not a tick: keep the latency trace honest.
            self.last_tick_ms, self.last_dec_ms = timing
        return AudioChunk(pcm=out, start_sample=0)

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
