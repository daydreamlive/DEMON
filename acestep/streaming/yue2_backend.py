"""YuE2Backend: YuE2's acoustic NAR in the ring, behind the GeneratorBackend seam.

The third :class:`~acestep.streaming.diffusion_backend.DiffusionBackend`
family, parameterized by (:class:`~acestep.engine.yue2_adapter.
YuE2Adapter`, the :class:`~acestep.engine.yue2_context.YuE2Context`
codec). Shaped like the SA3 backend:

* **What runs in the ring**: the NAR flow-matching solve of the WHOLE
  song, one midpoint step per tick, conditioned on a per-song bundle
  (AR-prefix KV) that rides ``SlotRequest.aux_cond`` by reference.
* **What is conditioning**: score plan, semantic tokens and the KV
  prefill. They run at session create (synchronously, before ``ready``)
  and, for a prompt change, outside the hot loop
  (:meth:`YuE2Backend.handle_set_prompt`): the new bundle is published
  under ``_control_lock`` and in-flight slots finish on the old one.
* **The source canvas** is the song's own clean latent (the anchor, one
  full solve at create). ``yue2_denoise`` < 1 re-noises the anchor with
  the seed's noise and integrates the truncated grid from there
  (audio-to-audio toward the live conditioning); ``x0_target`` locks
  toward it; ``feedback`` blends the latest outputs into it.

Control surface: ``yue2_denoise`` (prefixed; ACE's ``denoise`` means
something else) plus the shared ``x0_target`` / ``feedback`` /
``feedback_depth`` / ``seed``. The step count stays at the released
32-step grid: the shared ``steps_override`` knob (default 8, at most 16)
cannot express it, and ``yue2_denoise`` already shortens the work. ``prompt`` is a fast
restyle (same score and semantic tokens, new ``[Tags]`` prefix);
``set_prompt_blend`` is a hard A/B switch at 0.5 (two prefixes of
different lengths have no KV to interpolate). Lyrics and song length
are fixed for the session. No LoRA, no CFG, no per-frame curves.

Settled short-circuit: with fixed conditioning, seed and knobs and no
feedback, the ring re-renders the same latent forever. Once a latent
generated under the current signature has emerged, :meth:`_generate`
stops ticking and the renderer keeps playing that latent, so the GPU is
idle until something moves.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional

import torch

from acestep.engine.obs import logger
from acestep.nodes.interpolation import INTERPOLATIONS
from acestep.streaming.diffusion_backend import DiffusionBackend
from acestep.streaming.generator_backend import (
    AudioChunk,
    AudioGeometry,
    Capabilities,
    TickContext,
)
from acestep.streaming.knobs import KnobSpec, knob_specs as registry_knob_specs

SAMPLE_RATE = 48000
LATENT_RATE_HZ = 25.0
SAMPLES_PER_FRAME = 1920

#: Longest song: the flexible NAR TRT profile's 2500 frames at 25 Hz.
YUE2_MAX_SONG_S = 100.0

#: The full decode is 64 samples shorter than T frames of 1920.
DECODE_TAIL_SAMPLES = 64

MAX_FEEDBACK_DEPTH = int(
    next(s for s in registry_knob_specs(False) if s.name == "feedback_depth").max_val
)


def playable_seconds(frames: int) -> float:
    """Length of the full decode of ``frames`` latent frames, seconds."""
    return (SAMPLES_PER_FRAME * int(frames) - DECODE_TAIL_SAMPLES) / SAMPLE_RATE


def yue2_knob_specs() -> list:
    """The YuE2 knob manifest (also the homonym-guard universe).

    ``seed``, ``x0_target``, ``feedback`` and ``feedback_depth`` come
    FROM the shared registry by name, so their semantics cannot fork
    from ACE's. ``yue2_denoise`` is prefixed: it truncates the released
    step grid, not ACE's k1 strength. No ``steps_override``: the shared
    spec (default 8, max 16) cannot express the released 32 steps."""
    shared = {s.name: s for s in registry_knob_specs(False)}
    return [
        KnobSpec(
            "yue2_denoise", default=1.0, max_val=1.0, group="yue2",
            description=(
                "How much of the song's acoustic solve is redone per "
                "generation: 1.0 renders from pure noise (32 midpoint steps), "
                "lower values re-noise the song anchor and run only the last "
                "ceil(32*d) steps (faster updates, closer to the anchor). "
                "Distinct from ACE's 'denoise', hence the prefix."
            ),
        ),
        shared["x0_target"],
        shared["feedback"],
        shared["feedback_depth"],
        shared["seed"],
    ]


class YuE2Backend(DiffusionBackend):
    """See module docstring.

    Decoupled from the YuE2Context for testability: ``codec`` is anything
    with ``decode_window(latent_btc, start_frame, n_frames) -> [2, N]``
    and ``decode_full(latent_btc) -> [2, N]`` (optionally
    ``release_bundle(bundle)``); ``restyler(tags, epoch) -> bundle``
    builds a bundle for new style tags on the session's composition.
    """

    name = "yue2"

    def __init__(
        self,
        *,
        adapter,
        codec,
        bundle,
        anchor_latent: torch.Tensor,
        knob_state,
        state=None,
        restyler: Optional[Callable] = None,
        bundle_b=None,
        prompt_tags: Optional[str] = None,
        prompt_tags_b: Optional[str] = None,
        steps: int = 32,
        depth: int = 1,
        default_seed: int = 0,
        vae_window_s: float = 0.4,
    ):
        super().__init__(adapter=adapter, codec=codec)
        self.knob_state = knob_state
        self.state = state
        self._restyler = restyler
        self._steps = int(steps)
        self._depth = int(depth)
        self._default_seed = int(default_seed)
        self.vae_window = float(vae_window_s)
        self._frames = int(bundle.frames)
        if anchor_latent.shape[1] != self._frames:
            raise ValueError(
                f"yue2 anchor has {anchor_latent.shape[1]} frames, bundle {self._frames}"
            )
        # The song anchor: source canvas and x0 lock target, [1, T, 64].
        self._anchor = anchor_latent

        # Conditioning control state, swapped by the command thread and
        # snapshotted by the runner under _control_lock (never held
        # across pipeline work).
        self._control_lock = threading.Lock()
        self._bundle_a = bundle
        self._bundle_b = bundle_b if bundle_b is not None else bundle
        self._blend = 0.0
        self._active_bundle = bundle
        self._tags_a = prompt_tags
        self._tags_b = prompt_tags_b if prompt_tags_b not in (None, "", prompt_tags) else None
        self._cond_epoch = 0
        # (bundle, epoch, tags) for emerged-latent attribution, bounded.
        self._cond_history: list = [(bundle, 0, prompt_tags)]
        if self._bundle_b is not bundle:
            self._cond_history.append((self._bundle_b, 0, self._tags_b))

        self._latent_history: deque = deque(maxlen=MAX_FEEDBACK_DEPTH)
        # Settled short-circuit bookkeeping: the signature each submitted
        # request was built under, and the signature of the last latent
        # that emerged. A request can finish up to ``steps + depth`` ticks
        # after it was submitted (one per tick at queue_cap 1), so the
        # bookkeeping must outlive a full solve: 4x the step count covers
        # every depth the ring allows with room to spare.
        self._submitted: deque = deque(maxlen=4 * self._steps)
        self._emerged_signature = None
        self._emerged_request = None
        self._emerged_marker = None
        self._rendered_for = None
        self._rendered_pcm = None
        self._last_prep = None

        self.pipeline = self._build_pipeline(self._steps)

    # ---- assembly -----------------------------------------------------------

    @classmethod
    def from_context(cls, context, *, composition, bundle, anchor_latent, steps: int = 32,
                     **kwargs) -> "YuE2Backend":
        """Production assembly over a :class:`~acestep.engine.yue2_context.
        YuE2Context`: the adapter on the context's velocity backend (TRT
        when built), the context as codec, and a restyler that prefills a
        new bundle for the session's composition."""
        from acestep.engine.yue2_adapter import YuE2Adapter

        adapter = YuE2Adapter(
            context.velocity, steps=steps, device=context.device, dtype=torch.bfloat16,
        )

        def restyler(tags: str, epoch: int):
            return context.bundle(composition, style=tags, epoch=epoch)

        return cls(adapter=adapter, codec=context, bundle=bundle, anchor_latent=anchor_latent,
                   restyler=restyler, steps=steps, **kwargs)

    def _build_pipeline(self, steps: int):
        from acestep.engine.diffusion import DiffusionConfig
        from acestep.engine.stream import StreamPipeline

        self.adapter.steps = int(steps)
        config = DiffusionConfig(
            infer_steps=int(steps), infer_method="ode", noise_on_cpu=True,
            dcw_enabled=False,  # ACE wavelet corrector; off for YuE2
        )
        return StreamPipeline(
            None, config, pipeline_depth=self._depth, adapter=self.adapter, queue_cap=1,
        )

    # ---- contract ------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        # loop_band / render_anchor_queue: windowed rendering through the
        # shared runner, exactly as SA3. Everything else is off in v1.
        return Capabilities(refines_audio=True, loop_band=True, render_anchor_queue=True)

    def geometry(self) -> AudioGeometry:
        return AudioGeometry(
            sample_rate=SAMPLE_RATE, channels=2, chunk_rate_hz=LATENT_RATE_HZ,
            duration_s=self.playable_duration_s(),
        )

    def knob_specs(self, lora_ids=()) -> list:
        return yue2_knob_specs()

    def max_duration_s(self):
        return YUE2_MAX_SONG_S

    def playable_duration_s(self):
        return playable_seconds(self._frames)

    def read_knobs(self) -> dict:
        return self.knob_state.get_all_values()

    # ---- control: prompt ------------------------------------------------------

    def handle_set_prompt(self, tags: str, *, tags_b: Optional[str] = None) -> None:
        """Fast restyle: rebuild the conditioning for new style tags on
        the session's fixed score and semantic tokens (a KV prefill, not
        a re-compose), then publish atomically. In-flight slots finish
        on the bundle they were submitted with; an absent / empty /
        identical ``tags_b`` resets B to A."""
        if self._restyler is None:
            raise RuntimeError("YuE2Backend was constructed without a restyler")
        t0 = time.perf_counter()
        epoch = self._cond_epoch + 1
        bundle = self._restyler(tags, epoch)
        self._check_geometry(bundle)
        if tags_b and tags_b != tags:
            bundle_b = self._restyler(tags_b, epoch)
            self._check_geometry(bundle_b)
        else:
            bundle_b = bundle
        with self._control_lock:
            self._bundle_a, self._bundle_b = bundle, bundle_b
            self._active_bundle = self._select(self._blend)
            self._cond_epoch = epoch
            self._tags_a = tags
            self._tags_b = tags_b if (tags_b and tags_b != tags) else None
            self._cond_history.append((bundle, epoch, tags))
            if bundle_b is not bundle:
                self._cond_history.append((bundle_b, epoch, tags_b))
            del self._cond_history[:-6]
        logger.info(
            "yue2_prompt_applied tags={!r} tags_b={!r} cond_epoch={} restyle_ms={:.1f}",
            tags, tags_b, epoch, (time.perf_counter() - t0) * 1000,
        )

    def _check_geometry(self, bundle) -> None:
        if int(bundle.frames) != self._frames:
            raise ValueError(
                f"yue2 restyle changed the song length ({self._frames} -> "
                f"{bundle.frames} frames); a restyle keeps the semantic tokens"
            )

    def handle_set_prompt_blend(self, value: float) -> None:
        """Hard A/B switch at 0.5 (no KV interpolation between prefixes
        of different lengths)."""
        v = max(0.0, min(1.0, float(value)))
        with self._control_lock:
            self._blend = v
            self._active_bundle = self._select(v)

    def _select(self, v: float):
        return self._bundle_b if v >= 0.5 else self._bundle_a

    # ---- produce hooks ---------------------------------------------------------

    def _prepare_tick(self, knobs: dict, ctx: TickContext) -> dict:
        x0_str = float(knobs.get("x0_target", 0.0))
        # Shared override: a strength change lands on in-flight slots
        # within one tick (the ACE / SA3 convention).
        self.pipeline.set_shared_curve("x0_target_strength", x0_str)
        try:
            fb_depth_raw = float(knobs.get("feedback_depth", 1.0))
        except (TypeError, ValueError):
            fb_depth_raw = 1.0
        return {
            "denoise": float(knobs.get("yue2_denoise", 1.0)),
            "seed": int(knobs.get("seed", self._default_seed)),
            "x0_target": x0_str,
            "feedback": float(knobs.get("feedback", 0.0)),
            "feedback_depth": max(1, min(MAX_FEEDBACK_DEPTH, int(round(fb_depth_raw)))),
        }

    def _signature(self, prep: dict, bundle) -> Optional[tuple]:
        """What a generation depends on; None when it depends on history
        (feedback) and so never settles."""
        if prep["feedback"] > 0.0:
            return None
        return (id(bundle), id(self._anchor), round(prep["denoise"], 4), prep["seed"],
                round(prep["x0_target"], 4))

    def is_settled(self, prep: dict) -> bool:
        """True when the latest emerged latent is exactly what the ring
        would render now (nothing changed since)."""
        with self._control_lock:
            bundle = self._active_bundle
        sig = self._signature(prep, bundle)
        return sig is not None and sig == self._emerged_signature

    def _source_for(self, prep: dict) -> torch.Tensor:
        """The anchor, feedback-blended with a past output when asked
        (the ACE / SA3 delay-tap, verbatim)."""
        source = self._anchor
        if prep["feedback"] > 0.0 and self._latent_history:
            tap = min(prep["feedback_depth"] - 1, len(self._latent_history) - 1)
            method = getattr(self.state, "interp_feedback", "slerp") if self.state is not None else "slerp"
            source = INTERPOLATIONS[method](source, self._latent_history[tap], prep["feedback"])
        return source

    def _generate(self, prep: dict):
        from acestep.engine.stream import SlotRequest

        if self.is_settled(prep):
            return None

        with self._control_lock:
            bundle = self._active_bundle
        request = SlotRequest(
            seed=prep["seed"],
            denoise=prep["denoise"],
            source_latents=self._source_for(prep),
            # The lock target stays the clean anchor (feedback is upstream
            # of it), and identity against the live anchor tells a stale
            # latent apart after a re-anchor.
            x0_target=self._anchor,
            x0_target_strength=prep["x0_target"],
            aux_cond=bundle,
            latent_frames=self._frames,
        )
        self._submitted.append((request, self._signature(prep, bundle)))
        self.pipeline.submit(request)
        latent = self.pipeline.tick()
        if latent is None:
            return None
        req = getattr(self.pipeline, "last_finished_request", None)
        if req is not None and req.x0_target is not self._anchor:
            logger.info("yue2_gen_discarded reason=anchor_replaced")
            return None
        self._emerged_request = req
        self._emerged_signature = next(
            (sig for r, sig in self._submitted if r is req), None,
        )
        return latent

    def _cond_meta_for(self, bundle) -> tuple:
        with self._control_lock:
            history = tuple(self._cond_history)
        for b, epoch, tags in history:
            if b is bundle:
                return epoch, tags
        return None, None

    def _after_produce(self, prep: dict, result_latent, is_fresh: bool) -> None:
        self._last_prep = prep
        if not is_fresh:
            return
        self._latent_history.appendleft(result_latent.detach().clone())
        req = self._emerged_request
        if self.state is None or req is None:
            return
        epoch, tags = self._cond_meta_for(req.aux_cond)
        p = self.state.params
        p["gen_yue2_denoise"] = round(float(req.denoise), 4)
        p["gen_seed"] = int(req.seed)
        p["gen_cond_epoch"] = epoch
        p["gen_prompt"] = tags
        marker = (p["gen_yue2_denoise"], epoch, p["gen_seed"])
        if marker != self._emerged_marker:
            self._emerged_marker = marker
            logger.info(
                "yue2_gen_emerged denoise={} seed={} cond_epoch={} tags={!r}",
                p["gen_yue2_denoise"], p["gen_seed"], epoch, tags,
            )

    # ---- rendering -------------------------------------------------------------

    def window_frames(self) -> int:
        """Latent frames per window render: ``vae_window`` rounded up to
        whole 5-frame decoder cores."""
        from acestep.engine.yue2_trt import frames_for_seconds

        return frames_for_seconds(self.vae_window, LATENT_RATE_HZ)

    def render_window(self, t_start_s: float):
        decode_src = (
            self._current_result if self._current_result is not None
            else self._last_result_latent
        )
        if decode_src is None:
            return None
        n = self.window_frames()
        start = int(round(float(t_start_s) * LATENT_RATE_HZ))
        start = max(0, min(start, self._frames - n))
        t0 = time.perf_counter()
        audio = self.codec.decode_window(decode_src, start, n)
        pcm = audio.clamp(-1, 1).cpu().numpy().T
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_dec_ms += (time.perf_counter() - t0) * 1000
        start_sample = start * SAMPLES_PER_FRAME
        playable = SAMPLES_PER_FRAME * self._frames - DECODE_TAIL_SAMPLES
        return AudioChunk(pcm=pcm[: max(0, playable - start_sample)], start_sample=start_sample)

    def render_full(self):
        if self._current_result is None:
            return None
        latent = self._current_result
        if self._rendered_for is not latent:
            self._rendered_pcm = self.codec.decode_full(latent).clamp(-1, 1).cpu().numpy().T
            self._rendered_for = latent
        return AudioChunk(pcm=self._rendered_pcm, start_sample=0)

    # ---- teardown / bookkeeping -------------------------------------------------

    def close(self) -> None:
        """Free this session's conditioning bundles (GPU KV). The model
        is process-cached and outlives the session."""
        release = getattr(self.codec, "release_bundle", None)
        with self._control_lock:
            bundles = {id(b): b for b, _, _ in self._cond_history}
            self._cond_history = []
        if release is not None:
            for bundle in bundles.values():
                release(bundle)

    def on_fresh_generation(self, knobs: dict) -> None:
        if self.state is None:
            return
        p = self.state.params
        p["num_gens"] = p.get("num_gens", 0) + 1
        p["tick_ms"] = self.last_tick_ms
        p["dec_ms"] = self.last_dec_ms
        prep = self._last_prep
        if prep:
            p["yue2_denoise"] = round(prep["denoise"], 2)
            p["seed"] = prep["seed"]
            p["x0_target"] = round(prep["x0_target"], 2)
            p["feedback"] = round(prep["feedback"], 2)
            p["feedback_depth"] = prep["feedback_depth"]
        p["_prompt"] = getattr(self.state, "prompt_text", "")
