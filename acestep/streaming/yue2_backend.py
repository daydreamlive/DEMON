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
  and, for a prompt change, on the context worker
  (:mod:`acestep.streaming.yue2_recompose`): the new song is published
  under ``_control_lock``, the ring drops the old song's in-flight
  slots and renders the new song's anchor with one full solve.
* **The source canvas** is the song's own clean latent (the anchor, one
  full solve at create or after a re-compose). ``yue2_denoise`` < 1
  re-noises the anchor with the seed's noise and integrates the
  truncated grid from there (audio-to-audio toward the live
  conditioning); ``x0_target`` locks toward it; ``feedback`` blends the latest outputs into it.

Control surface: ``yue2_denoise`` (prefixed; ACE's ``denoise`` means
something else) plus the shared ``x0_target`` / ``feedback`` /
``feedback_depth`` / ``seed``. The step count stays at the released
32-step grid: the shared ``steps_override`` knob (default 8, at most 16)
cannot express it, and ``yue2_denoise`` already shortens the work.
``prompt`` re-composes the session's lyrics under the new tags at the
session's length (a ``[Tags]``-only restyle with frozen semantics is
nearly inaudible, experiment E3); ``set_prompt_blend`` is a hard switch
at 0.5 between the A and B songs. Lyrics and song length are fixed for
the session. No LoRA, no CFG, no per-frame curves.

Settled short-circuit: with fixed conditioning, seed and knobs and no
feedback, the ring re-renders the same latent forever. Once a latent
generated under the current signature has emerged, :meth:`_generate`
stops ticking and the renderer keeps playing that latent, so the GPU is
idle until something moves. Slots still in flight when the ring settles
(depth >= 2) hold redundant or stale work and are dropped, so the next
change starts clean instead of first finishing a frozen slot.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Callable, Optional

import torch

from acestep.engine.obs import logger
from acestep.engine.yue2_adapter import YuE2Adapter
from acestep.engine.yue2_trt import FLEX_FRAMES, SAMPLES_PER_FRAME
from acestep.nodes.interpolation import INTERPOLATIONS
from acestep.streaming.diffusion_backend import DiffusionBackend
from acestep.streaming.generator_backend import (
    AudioChunk,
    AudioGeometry,
    Capabilities,
    TickContext,
)
from acestep.streaming.knobs import KnobSpec, knob_specs as registry_knob_specs
from acestep.streaming.yue2_recompose import Recomposer, Song, run_inline

SAMPLE_RATE = YuE2Adapter.sample_rate
LATENT_RATE_HZ = YuE2Adapter.latent_rate_hz

#: Longest song: the flexible NAR TRT profile's 2500 frames at 25 Hz.
YUE2_MAX_SONG_S = FLEX_FRAMES[1] / LATENT_RATE_HZ

#: The full decode is 64 samples shorter than T frames of 1920.
DECODE_TAIL_SAMPLES = 64

#: Decoded windows kept per latent (0.4 s stereo float32 each, ~150 KB).
WINDOW_CACHE_MAX = 512

#: Runner pacing while settled (also the worst-case extra knob latency).
SETTLED_NAP_S = 0.02

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
    ``release_bundle(bundle)``); ``recompose(tags, epoch) -> Song``
    composes the session's lyrics under new style tags at the session's
    length, and ``submit(fn, *args)`` runs it off the command thread
    (the context's worker; inline in tests). ``on_error(code, message)``
    reports a failed re-compose to the client (the session's
    ``SessionError`` event).
    """

    name = "yue2"

    def __init__(
        self,
        *,
        adapter,
        codec,
        song: Song,
        knob_state,
        state=None,
        song_b: Optional[Song] = None,
        recompose: Optional[Callable[[str, int], Song]] = None,
        submit: Callable = run_inline,
        steps: int = 32,
        depth: int = 1,
        vae_window_s: float = 0.4,
        gpu_gate=None,
        settled_nap_s: float = SETTLED_NAP_S,
        on_error: Optional[Callable[[str, str], None]] = None,
    ):
        super().__init__(adapter=adapter, codec=codec)
        # Excludes a conditioning-worker CUDA graph capture while the ring
        # does GPU work (YuE2Context.gpu_gate).
        self._gpu_gate = gpu_gate if gpu_gate is not None else threading.Lock()
        self.knob_state = knob_state
        self.state = state
        self._steps = int(steps)
        self._depth = int(depth)
        self._default_seed = song.seed
        self.vae_window = float(vae_window_s)
        self._frames = song.frames
        if song.anchor is None or song.anchor.shape[1] != self._frames:
            raise ValueError("yue2 session song needs an anchor of the song's length")

        # Song control state, swapped by the command / worker threads and
        # snapshotted by the runner under _control_lock (never held
        # across pipeline work).
        self._control_lock = threading.Lock()
        self._song_a = song
        self._song_b = song_b if song_b is not None else song
        self._b_follows_a = song_b is None
        self._blend = 0.0
        self._active = song
        self._cond_epoch = 0
        self._closed = False
        self._on_error = on_error
        # Every song this session holds a bundle for: the runner frees
        # the replaced ones (_release_retired), close frees the rest.
        self._songs: list = [song] + ([song_b] if song_b is not None else [])
        self._recomposer = None
        if recompose is not None:
            self._recomposer = Recomposer(
                build=recompose, submit=submit, publish=self._publish_song,
                release=self._release_song, on_failure=self._recompose_failed,
            )

        self._latent_history: deque = deque(maxlen=MAX_FEEDBACK_DEPTH)
        # Settled short-circuit bookkeeping: (request, signature, song)
        # for each submitted request, and the signature of the last
        # latent that emerged. A request can finish up to ``steps +
        # depth`` ticks after it was submitted (one per tick at
        # queue_cap 1), so the bookkeeping must outlive a full solve.
        self._submitted: deque = deque(maxlen=4 * self._steps)
        self._emerged_signature = None
        self._emerged_request = None
        self._emerged_song = None
        self._emerged_marker = None
        self._submitted_song = None
        self._rendered_for = None
        self._rendered_pcm = None
        self._settled_last = False
        self._settled_nap_s = float(settled_nap_s)
        self._window_cache: OrderedDict = OrderedDict()
        self._window_cache_src = None
        self._last_prep = None

        self.pipeline = self._build_pipeline(self._steps)

    @classmethod
    def from_context(cls, context, *, composition, song: Song, steps: int = 32,
                     **kwargs) -> "YuE2Backend":
        """Production assembly over a :class:`~acestep.engine.yue2_context.
        YuE2Context`: the adapter on the context's velocity backend (TRT
        when built), the context as codec, and a re-compose of the
        session's lyrics at the session's length on the context worker."""
        from acestep.engine.yue2_adapter import YuE2Adapter

        adapter = YuE2Adapter(
            context.velocity, steps=steps, device=context.device, dtype=torch.bfloat16,
        )
        frames = song.frames

        def recompose(tags: str, epoch: int) -> Song:
            new = context.compose(style=tags, lyrics=composition.lyrics, seed=composition.seed,
                                  max_frames=frames, exact_frames=frames)
            return Song(bundle=context.bundle(new, epoch=epoch), anchor=None, tags=tags, epoch=epoch)

        return cls(adapter=adapter, codec=context, song=song, recompose=recompose,
                   submit=context.submit, gpu_gate=context.gpu_gate, steps=steps, **kwargs)

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
        """Re-compose for new style tags in the background (see
        :mod:`acestep.streaming.yue2_recompose`); returns at once and the
        ring keeps playing the current song until the new one is
        published. A slot whose tags did not change is left alone, and so
        is a slot already re-composing for exactly these tags (a repeated
        click keeps the job in flight); an absent / empty / identical
        ``tags_b`` makes B follow A."""
        if self._recomposer is None:
            raise RuntimeError("YuE2Backend was constructed without a recompose function")
        follows = not (tags_b and tags_b != tags)
        with self._control_lock:
            self._cond_epoch += 1
            epoch = self._cond_epoch
            # A job still building these tags may publish (it re-checks
            # under this lock), so it counts as the slot's state.
            keep_a = tags == self._recomposer.inflight("a")
            keep_b = not follows and tags_b == self._recomposer.inflight("b")
            redo_a = not keep_a and tags != self._song_a.tags
            redo_b = not follows and not keep_b and tags_b != self._song_b.tags
            self._b_follows_a = follows
            # Going back to the playing prompt abandons the job in flight
            # for that slot (latest wins). Under this lock, so a publish
            # racing it re-checks and releases instead.
            if not (redo_a or keep_a):
                self._recomposer.cancel("a")
            if not (redo_b or keep_b):
                self._recomposer.cancel("b")
            if follows and not (redo_a or keep_a):
                self._song_b = self._song_a
                self._active = self._select(self._blend)
        logger.info("yue2_recompose_requested tags={!r} tags_b={!r} cond_epoch={} a={} b={}",
                    tags, tags_b, epoch, redo_a, redo_b)
        if redo_a:
            self._recomposer.request("a", tags, epoch)
        if redo_b:
            self._recomposer.request("b", tags_b, epoch)

    def _publish_song(self, slot: str, song: Song, build_ms: float,
                      is_current: Callable[[], bool] = lambda: True) -> None:
        """Worker-side: swap a finished song in atomically. The ring
        renders it with one full solve before it is heard. A song whose
        job was cancelled, a B song after B went back to following A, and
        anything finishing after :meth:`close` are released instead."""
        if song.frames != self._frames:
            logger.error("yue2_recompose_rejected frames={} session_frames={} (forced length "
                         "not honoured)", song.frames, self._frames)
            self._release_song(song)
            return
        with self._control_lock:
            reason = ("closed" if self._closed
                      else "superseded" if not is_current()
                      else "b_follows_a" if slot == "b" and self._b_follows_a
                      else None)
            if reason is None:
                self._install_song(slot, song)
        if reason is not None:
            logger.info("yue2_recompose_dropped slot={} tags={!r} reason={}",
                        slot, song.tags, reason)
            self._release_song(song)
            return
        if self.state is not None:
            # Wake the runner's idle pause: the new song must be rendered
            # even when no knob moves.
            self.state.last_activity_ts = time.monotonic()
        logger.info("yue2_recompose_published slot={} tags={!r} cond_epoch={} build_ms={:.0f}",
                    slot, song.tags, song.epoch, build_ms)

    def _install_song(self, slot: str, song: Song) -> None:
        """Under ``_control_lock``."""
        if slot == "a":
            self._song_a = song
            if self._b_follows_a:
                self._song_b = song
        else:
            self._song_b = song
        self._active = self._select(self._blend)
        self._songs.append(song)

    def _release_retired(self) -> None:
        """Runner-side, under the GPU gate: free the bundle of every song
        that no slot plays and no in-flight solve uses. In-flight slots
        only ever hold ``_submitted_song`` (a song change restarts the
        ring), so a replaced song is freed on the tick after the ring
        moves off it, on the only thread that runs the velocity."""
        with self._control_lock:
            keep = (self._song_a, self._song_b, self._active, self._submitted_song)
            retired = [s for s in self._songs if all(s is not k for k in keep)]
            if not retired:
                return
            self._songs = [s for s in self._songs if all(s is not r for r in retired)]
        for old in retired:
            logger.info("yue2_song_released tags={!r} cond_epoch={}", old.tags, old.epoch)
            self._release_song(old)

    def _recompose_failed(self, slot: str, tags: str, exc: BaseException) -> None:
        """The ring keeps playing the current song; tell the client."""
        if self._on_error is not None and not self._closed:
            self._on_error("yue2_recompose_failed",
                           f"Re-compose for {tags!r} failed: {type(exc).__name__}: {exc}")

    def _release_song(self, song: Song) -> None:
        release = getattr(self.codec, "release_bundle", None)
        if release is not None:
            release(song.bundle)

    def handle_set_prompt_blend(self, value: float) -> None:
        """Hard A/B switch at 0.5 between the two songs."""
        v = max(0.0, min(1.0, float(value)))
        with self._control_lock:
            self._blend = v
            self._active = self._select(v)

    def _select(self, v: float) -> Song:
        return self._song_b if v >= 0.5 else self._song_a

    # ---- produce hooks ---------------------------------------------------------

    def produce(self, knobs: dict, ctx: TickContext, mode) -> bool:
        with self._gpu_gate:
            fresh = super().produce(knobs, ctx, mode)
        if not fresh and self._settled_last and self._settled_nap_s > 0:
            # Nothing to generate: pace the runner, which would otherwise
            # spin re-rendering cached windows and starve the conditioning
            # worker's Python AR loop of the GIL.
            time.sleep(self._settled_nap_s)
        return fresh

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

    def _signature(self, prep: dict, song: Song) -> Optional[tuple]:
        """What a generation depends on; None when it depends on history
        (feedback) and so never settles. A song without an anchor renders
        its anchor whatever the knobs say."""
        if song.anchor is None:
            return ("anchor", id(song))
        if prep["feedback"] > 0.0:
            return None
        return (id(song), id(song.anchor), round(prep["denoise"], 4), prep["seed"],
                round(prep["x0_target"], 4))

    def is_settled(self, prep: dict) -> bool:
        """True when the latest emerged latent is exactly what the ring
        would render now (nothing changed since)."""
        with self._control_lock:
            song = self._active
        sig = self._signature(prep, song)
        return sig is not None and sig == self._emerged_signature

    def _source_for(self, prep: dict, song: Song) -> torch.Tensor:
        """The anchor, feedback-blended with a past output when asked
        (the ACE / SA3 delay-tap, verbatim)."""
        source = song.anchor
        if prep["feedback"] > 0.0 and self._latent_history:
            tap = min(prep["feedback_depth"] - 1, len(self._latent_history) - 1)
            method = getattr(self.state, "interp_feedback", "slerp") if self.state is not None else "slerp"
            source = INTERPOLATIONS[method](source, self._latent_history[tap], prep["feedback"])
        return source

    def _request_for(self, prep: dict, song: Song):
        from acestep.engine.stream import SlotRequest

        if song.anchor is None:
            # The song's anchor: one full solve from its own seed, no
            # source, no lock target.
            return SlotRequest(seed=song.seed, denoise=1.0, aux_cond=song.bundle,
                               latent_frames=self._frames)
        return SlotRequest(
            seed=prep["seed"],
            denoise=prep["denoise"],
            source_latents=self._source_for(prep, song),
            x0_target=song.anchor,
            x0_target_strength=prep["x0_target"],
            aux_cond=song.bundle,
            latent_frames=self._frames,
        )

    def _restart_ring(self, prep: dict) -> None:
        """Drop every in-flight slot and queued request: a fresh pipeline
        over the same adapter (no weights, no engine rebuild)."""
        self.pipeline = self._build_pipeline(self._steps)
        self.pipeline.set_shared_curve("x0_target_strength", prep["x0_target"])
        self._submitted.clear()

    def _generate(self, prep: dict):
        self._release_retired()
        settled = self.is_settled(prep)
        if settled and not self._settled_last and self.pipeline.active_slots:
            # depth >= 2: the other slots hold the same request (redundant)
            # or an older one (stale). Left alone they freeze mid-solve and
            # the next change first emits one of them.
            self._restart_ring(prep)
        self._settled_last = settled
        if settled:
            return None
        with self._control_lock:
            song = self._active
        if song is not self._submitted_song:
            if self._submitted_song is not None and self.pipeline.active_slots:
                # The old song's slots would be discarded when they emerge;
                # start the new song now instead of after them.
                self._restart_ring(prep)
            self._submitted_song = song
        request = self._request_for(prep, song)
        self._submitted.append((request, self._signature(prep, song), song))
        self.pipeline.submit(request)
        latent = self.pipeline.tick()
        if latent is None:
            return None
        req = getattr(self.pipeline, "last_finished_request", None)
        entry = next((e for e in self._submitted if e[0] is req), None)
        if entry is None:
            return latent
        _, signature, emerged_song = entry
        with self._control_lock:
            active = self._active
        if emerged_song is not active:
            logger.info("yue2_gen_discarded reason=song_replaced")
            return None
        if signature is not None and signature[0] == "anchor":
            signature = self._adopt_anchor(emerged_song, latent, prep)
        self._emerged_request = req
        self._emerged_song = emerged_song
        self._emerged_signature = signature
        return latent

    def _adopt_anchor(self, song: Song, latent: torch.Tensor, prep: dict) -> Optional[tuple]:
        """A song's first full solve becomes its anchor. Returns the
        signature the emerged latent now satisfies: the knobs' own when
        they ask for exactly the anchor (song seed, full solve, no
        feedback), else one that never matches, so the ring goes on to
        render the knobs over the new anchor."""
        if song.anchor is not None:  # a stale anchor request finishing late
            return ("anchored", id(song))
        song.anchor = latent.detach().clone()
        self._latent_history.clear()  # feedback never reaches across songs
        logger.info("yue2_song_anchored tags={!r} cond_epoch={}", song.tags, song.epoch)
        if prep["seed"] == song.seed and prep["denoise"] >= 1.0 and prep["feedback"] == 0.0:
            return self._signature(prep, song)
        return ("anchored", id(song))

    def _after_produce(self, prep: dict, result_latent, is_fresh: bool) -> None:
        self._last_prep = prep
        if not is_fresh:
            return
        self._latent_history.appendleft(result_latent.detach().clone())
        req, song = self._emerged_request, self._emerged_song
        if self.state is None or req is None or song is None:
            return
        p = self.state.params
        p["gen_yue2_denoise"] = round(float(req.denoise), 4)
        p["gen_seed"] = int(req.seed)
        p["gen_cond_epoch"] = song.epoch
        p["gen_prompt"] = song.tags
        self._stamp_nar_path(song)
        marker = (p["gen_yue2_denoise"], song.epoch, p["gen_seed"], id(song))
        if marker != self._emerged_marker:
            self._emerged_marker = marker
            logger.info(
                "yue2_gen_emerged denoise={} seed={} cond_epoch={} tags={!r}",
                p["gen_yue2_denoise"], p["gen_seed"], song.epoch, song.tags,
            )

    def _stamp_nar_path(self, song: Song) -> None:
        """``yue2_nar``: the NAR path of the song that just emerged. A
        re-composed song can leave the TRT profile (conditioning past its
        token bound) and run eager at several times the tick."""
        from acestep.streaming.yue2_session import nar_backend_for

        nar = nar_backend_for(song.bundle, self.adapter.velocity)
        previous = self.state.params.get("yue2_nar")
        self.state.params["yue2_nar"] = nar
        if previous is not None and previous != nar:
            logger.warning("yue2_nar_path_changed from={} to={} tags={!r} frames={} cond_tokens={}",
                           previous, nar, song.tags, song.frames,
                           getattr(song.bundle, "cond_tokens", None))

    # ---- rendering -------------------------------------------------------------

    def window_frames(self) -> int:
        """Latent frames per window render: ``vae_window`` rounded up to
        whole 5-frame decoder cores."""
        from acestep.engine.yue2_trt import frames_for_seconds

        return frames_for_seconds(self.vae_window, LATENT_RATE_HZ)

    def render_window(self, t_start_s: float):
        with self._gpu_gate:
            return self._render_window(t_start_s)

    def _render_window(self, t_start_s: float):
        decode_src = (
            self._current_result if self._current_result is not None
            else self._last_result_latent
        )
        if decode_src is None:
            return None
        n = self.window_frames()
        start = int(round(float(t_start_s) * LATENT_RATE_HZ))
        start = max(0, min(start, self._frames - n))
        start_sample = start * SAMPLES_PER_FRAME
        # The runner re-renders windows of a settled latent many times a
        # second; the decode is deterministic per (latent, start), so a
        # repeat costs a CPU copy instead of GPU time (which a background
        # re-compose needs). The runner crossfades in place: hand out copies.
        if self._window_cache_src is not decode_src:
            self._window_cache.clear()
            self._window_cache_src = decode_src
        pcm = self._window_cache.get(start)
        if pcm is None:
            t0 = time.perf_counter()
            audio = self.codec.decode_window(decode_src, start, n)
            pcm = audio.clamp(-1, 1).cpu().numpy().T
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.last_dec_ms += (time.perf_counter() - t0) * 1000
            playable = SAMPLES_PER_FRAME * self._frames - DECODE_TAIL_SAMPLES
            pcm = pcm[: max(0, playable - start_sample)]
            self._window_cache[start] = pcm
            while len(self._window_cache) > WINDOW_CACHE_MAX:
                self._window_cache.popitem(last=False)
        else:
            self._window_cache.move_to_end(start)
        return AudioChunk(pcm=pcm.copy(), start_sample=start_sample)

    def render_full(self):
        with self._gpu_gate:
            return self._render_full()

    def _render_full(self):
        if self._current_result is None:
            return None
        latent = self._current_result
        if self._rendered_for is not latent:
            self._rendered_pcm = self.codec.decode_full(latent).clamp(-1, 1).cpu().numpy().T
            self._rendered_for = latent
        return AudioChunk(pcm=self._rendered_pcm, start_sample=0)

    # ---- teardown / bookkeeping -------------------------------------------------

    def close(self) -> None:
        """Free this session's conditioning bundles (GPU KV) and drop any
        re-compose still queued or running (a running one releases its
        bundle when it finishes, without waiting here). The model is process-cached and
        outlives the session."""
        with self._control_lock:
            # A re-compose finishing from here on releases its song
            # (_publish_song checks this flag under the same lock).
            self._closed = True
        if self._recomposer is not None:
            self._recomposer.cancel_all()
        with self._control_lock:
            songs = {id(s): s for s in [*self._songs, self._song_a, self._song_b, self._active]}
            self._songs = []
        for song in songs.values():
            self._release_song(song)

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
