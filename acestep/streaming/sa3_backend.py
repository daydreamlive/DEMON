"""SA3Backend: Stable Audio 3 behind the GeneratorBackend seam.

The second :class:`~acestep.streaming.diffusion_backend.DiffusionBackend`
family, parameterized by (:class:`~acestep.engine.sa3_adapter.SA3Adapter`,
:class:`~acestep.engine.sa3_context.SA3SAMECodec`). It owns BOTH halves
of the parameterization (unlike ACE, whose adapter is pipeline-default
and whose codec is the engine Session): a shared
:class:`~acestep.engine.stream.StreamPipeline` is built here with
``engine=None`` and the SA3 adapter, and rendering decodes through the
SAME codec with the 44.1 → 48 kHz delivery resample applied at the
decode boundary (round_3 decision 2: AudioEngine / worklet / client
stay 48 kHz-untouched in v1; ``geometry().sample_rate`` declares the
DELIVERED rate, 48000 — native-44.1k delivery later is a geometry +
client change only).

v1 surface (everything else off, capability-gated): ``prompt`` (one
conditioning bundle at a time, swapped per-prompt via
:meth:`SA3Backend.set_prompt`), fixed duration, ``seed``, ``steps_override``,
and ``sa3_denoise`` — SA3's ``init_noise_level``, the audio-to-audio
blend against the source anchor. The name is load-bearing: ACE's
``denoise`` is a different control, and the homonym rule
(``tests/unit/test_knob_homonyms.py``) forbids reusing the name with
different semantics. Continuity comes the same way the spike demo
proved (``demos/test_stream_sa3_graph.py``): every emit is a
partial-denoise cover of the SAME source latent at the same seed, so
advancing playback windows reconstruct one evolving song.

Capabilities: ``refines_audio`` only. No swap/timbre/structure/LoRA/
stems/loop-band/depth/curves until validated (canonical plan Phase 5).
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch

from acestep.engine.obs import logger
from acestep.streaming.diffusion_backend import DiffusionBackend
from acestep.streaming.generator_backend import (
    AudioChunk,
    AudioGeometry,
    Capabilities,
    TickContext,
)
from acestep.streaming.knobs import KnobSpec, knob_specs as registry_knob_specs

# Delivery rate (v1): SA3's native 44.1 kHz is resampled at the decode
# boundary so everything downstream of the backend stays at the engine
# rate. See module docstring / round_3 decision 2.
DELIVERY_SAMPLE_RATE = 48000
SA3_SAMPLE_RATE = 44100
SA3_LATENT_RATE_HZ = 44100.0 / 4096.0


def sa3_knob_specs() -> list:
    """The SA3 family knob manifest (backend-owned, plan §3.3).

    ``seed`` and ``steps_override`` are genuinely neutral controls and
    are taken FROM the shared registry by name, so their semantics can
    never fork from ACE's (the homonym test would catch it; this makes
    the fork impossible instead). ``sa3_denoise`` is family-prefixed
    because ACE's ``denoise`` means something else.
    """
    shared = {s.name: s for s in registry_knob_specs(False)}
    return [
        KnobSpec(
            "sa3_denoise", default=1.0, max_val=1.0, group="sa3",
            description=(
                "SA3 init_noise_level: fresh-noise vs source-anchor mix "
                "at slot init (1.0 = generate from pure noise, lower = "
                "closer cover of the source). Distinct from ACE's "
                "'denoise' (k1 strength), hence the prefix."
            ),
        ),
        shared["seed"],
        shared["steps_override"],
    ]


class SA3Backend(DiffusionBackend):
    """Stable Audio 3 streaming generation. See module docstring.

    Decoupled from :class:`~acestep.engine.sa3_context.SA3Context` for
    testability: takes the adapter, codec, conditioning, and a
    schedule-builder factory (``steps -> (denoise -> schedule)``)
    directly, so unit tests drive it with a mock DiT and codec. The
    production assembly (context → adapter/codec/cond → backend) is
    :meth:`from_context`.
    """

    name = "sa3"

    def __init__(
        self,
        *,
        adapter,
        codec,
        cond,
        schedule_builder_factory: Callable[[int], Callable],
        knob_state,
        state=None,
        source_latent_bct: Optional[torch.Tensor] = None,
        steps: int = 8,
        depth: int = 4,
        default_seed: int = 1528,
        vae_window_s: float = 3.0,
        # SA3 checkpoints are ``diffusion_objective: rf_denoiser`` —
        # upstream samples them with pingpong ONLY (euler isn't even
        # offered in their UI for this objective, and 8-step euler is
        # audibly degraded). Determinism — and therefore window-splice
        # continuity — is preserved via the seeded per-slot renoise
        # stream (SlotRequest.sde_noise_seeded), the spike pipeline's
        # per-slot generator semantics.
        sampler: str = "pingpong",
        prompt_rebuilder: Optional[Callable] = None,
    ):
        super().__init__(adapter=adapter, codec=codec)
        self._cond = cond
        self._schedule_builder_factory = schedule_builder_factory
        # ``(tags, steps) -> (cond, steps -> (denoise -> schedule))`` —
        # the per-prompt re-conditioning hook behind :meth:`set_prompt`.
        # Supplied by :meth:`from_context` (a closure over the
        # SA3Context); None on directly-constructed test backends, where
        # set_prompt fails loudly instead.
        self._prompt_rebuilder = prompt_rebuilder
        self.knob_state = knob_state
        self.state = state
        self._steps = int(steps)
        self._depth = int(depth)
        self._default_seed = int(default_seed)
        self.vae_window = float(vae_window_s)
        # "pingpong"/"sde" (rf_denoiser-native, deterministic via seeded
        # renoise) | "ode" (euler; off-objective for SA3, debug only)
        self._sampler = sampler

        # Source anchor for audio-to-audio: engine layout [1, T, 256].
        self._source_latent_btc = (
            source_latent_bct.movedim(1, 2).contiguous()
            if source_latent_bct is not None else None
        )

        # Rendered-audio cache: one full decode+resample per fresh
        # latent (SAME-S decodes the whole window in ~11 ms); window
        # renders slice it, so gap-fill re-renders are bit-stable.
        # Windowed codecs (SAME-L / medium: full decode ~80 ms) bypass
        # the cache and decode per render instead — see render_window.
        self._rendered_for = None     # latent tensor identity
        self._rendered_48k = None     # np.ndarray [N, C] float32
        self._windowed_codec = hasattr(codec, "decode_window")

        self.pipeline = self._build_pipeline(self._steps)

    # ---- assembly -----------------------------------------------------------

    @classmethod
    def from_context(
        cls,
        context,
        *,
        prompt: str,
        duration_s: float,
        knob_state,
        state=None,
        source_audio=None,
        cond=None,
        source_latent_bct=None,
        **kwargs,
    ) -> "SA3Backend":
        """Production assembly over a loaded
        :class:`~acestep.engine.sa3_context.SA3Context`.

        ``cond`` / ``source_latent_bct`` accept precomputed values so the
        serving-layer create path (:mod:`acestep.streaming.sa3_session`),
        which runs ``prepare_cond`` + source encode itself before the
        session exists, doesn't pay them twice; absent, they're computed
        here (the in-process assembly the GPU smoke validated).

        Component selection is the context's call (``make_dit`` /
        ``make_codec``): small runs the torch DiT + SAME-S full-decode
        codec; medium gets the TRT DiT engine (when built) and the
        SAME-L windowed codec."""
        from acestep.engine.sa3_adapter import SA3Adapter

        steps = int(kwargs.get("steps", 8))
        if cond is None:
            cond = context.prepare_cond(
                prompt=prompt, duration=duration_s, steps=steps,
            )
        source_latent = (
            source_latent_bct if source_latent_bct is not None
            else context.encode_source(source_audio, cond.audio_sample_size)
            if source_audio is not None else None
        )
        adapter = SA3Adapter(
            context.make_dit(
                latent_frames=cond.latent_frames,
                seconds_total=duration_s,
            ),
            schedule_builder=context.make_schedule_builder(cond, steps),
            device=context.device,
            dtype=context.dtype,
        )

        def _prompt_rebuilder(tags: str, steps_now: int):
            # Per-prompt re-conditioning (set_prompt): same fixed
            # duration, fresh T5Gemma capture + a schedule-builder
            # factory closed over the NEW cond's sched_args.
            new_cond = context.prepare_cond(
                prompt=tags, duration=duration_s, steps=steps_now,
            )
            return new_cond, (
                lambda s, _c=new_cond: context.make_schedule_builder(_c, s)
            )

        return cls(
            adapter=adapter,
            codec=context.make_codec(),
            cond=cond,
            schedule_builder_factory=(
                lambda s: context.make_schedule_builder(cond, s)
            ),
            knob_state=knob_state,
            state=state,
            source_latent_bct=source_latent,
            prompt_rebuilder=_prompt_rebuilder,
            **kwargs,
        )

    def _build_pipeline(self, steps: int):
        from acestep.engine.diffusion import DiffusionConfig
        from acestep.engine.stream import StreamPipeline

        self.adapter.schedule_builder = self._schedule_builder_factory(steps)
        config = DiffusionConfig(
            infer_steps=int(steps),
            infer_method="sde" if self._sampler in ("sde", "pingpong") else "ode",
            noise_on_cpu=True,
            dcw_enabled=False,  # ACE wavelet corrector semantics; off for SA3
        )
        return StreamPipeline(
            None, config, pipeline_depth=self._depth, adapter=self.adapter,
        )

    # ---- contract ------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(refines_audio=True)

    def geometry(self) -> AudioGeometry:
        return AudioGeometry(
            sample_rate=DELIVERY_SAMPLE_RATE,
            channels=2,
            chunk_rate_hz=SA3_LATENT_RATE_HZ,
            duration_s=self.playable_duration_s(),
        )

    def knob_specs(self, lora_ids=()) -> list:
        return sa3_knob_specs()

    def playable_duration_s(self):
        return self._cond.audio_sample_size / SA3_SAMPLE_RATE

    def read_knobs(self) -> dict:
        return self.knob_state.get_all_values()

    def rebuild_imminent(self, knobs: dict) -> bool:
        return int(knobs.get("steps_override", self._steps)) != self._steps

    # ---- control (universal): per-prompt re-conditioning ------------------------

    def set_prompt(self, tags: str, tags_b: Optional[str] = None) -> None:
        """Re-run ``prepare_cond`` for ``tags`` and swap the conditioning
        bundle (the session dispatches its ``set_prompt`` here — plan §2:
        prompt is the universal control). Per-prompt, OUTSIDE the hot
        loop: one T5Gemma capture on the dispatcher thread, then two
        GIL-atomic reference swaps; in-flight slots finish on the old
        bundle, the next ``submit`` carries the new one.

        SA3 v1 has no A/B conditioning cache, so a distinct ``tags_b``
        is not honored — logged loudly rather than silently blended.
        """
        if self._prompt_rebuilder is None:
            raise RuntimeError(
                "SA3Backend was constructed without a prompt_rebuilder; "
                "set_prompt requires the from_context assembly"
            )
        if tags_b and tags_b != tags:
            logger.warning(
                "sa3_prompt_b_ignored tags_b={!r} reason=no_ab_blend_v1",
                tags_b,
            )
        cond, sched_factory = self._prompt_rebuilder(tags, self._steps)
        if int(cond.latent_frames) != int(self._cond.latent_frames):
            # Duration is fixed for the session lifetime, so the latent
            # geometry must hold: a mismatch would desync the ring
            # buffer, the source anchor, and the cond bundle.
            raise ValueError(
                f"sa3 prompt swap changed latent_frames "
                f"({self._cond.latent_frames} -> {cond.latent_frames}); "
                f"duration is fixed per session"
            )
        self._schedule_builder_factory = sched_factory
        self.adapter.schedule_builder = sched_factory(self._steps)
        self._cond = cond
        logger.info("sa3_prompt_applied tags={!r}", tags)

    # ---- produce hooks ---------------------------------------------------------

    def _prepare_tick(self, knobs: dict, ctx: TickContext) -> dict:
        return {
            "denoise": float(knobs.get("sa3_denoise", 1.0)),
            "seed": int(knobs.get("seed", self._default_seed)),
            "steps": int(knobs.get("steps_override", self._steps)),
        }

    def _generate(self, prep: dict):
        from acestep.engine.stream import SlotRequest

        if prep["steps"] != self._steps:
            # Step-count change: schedules are (steps+1,)-shaped, so
            # the ring buffer rebuilds — the SA3 analog of ACE's
            # rebuild-signature stall, pre-covered via rebuild_imminent.
            self._steps = prep["steps"]
            self.pipeline = self._build_pipeline(self._steps)

        self.pipeline.submit(SlotRequest(
            seed=prep["seed"],
            denoise=prep["denoise"],
            source_latents=self._source_latent_btc,
            aux_cond=self._cond.cond_bundle,
            latent_frames=self._cond.latent_frames,
            # Deterministic pingpong: identical requests must replay the
            # same trajectory or advancing windows splice different
            # realizations (incoherent audio). See SlotRequest.
            sde_noise_seeded=True,
        ))
        return self.pipeline.tick()  # engine-layout [1, T, 256] | None

    def _after_produce(self, prep: dict, result_latent, is_fresh: bool) -> None:
        self.last_denoise = prep["denoise"]
        self._last_prep = prep

    # ---- rendering -------------------------------------------------------------

    def _rendered_audio(self, latent_btc: torch.Tensor):
        """Full decode + delivery resample, cached per latent identity."""
        if self._rendered_for is latent_btc and self._rendered_48k is not None:
            return self._rendered_48k
        import torchaudio

        t0 = time.perf_counter()
        audio_ct = self.codec.decode_full(latent_btc.movedim(1, 2))
        # The decode boundary (round_3 decision 2): one whole-window
        # resample per generation, so window slices share one filter
        # pass and seams can't come from per-slice resampling.
        audio_48 = torchaudio.functional.resample(
            audio_ct.float(), SA3_SAMPLE_RATE, DELIVERY_SAMPLE_RATE,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_dec_ms += (time.perf_counter() - t0) * 1000
        self._rendered_48k = audio_48.clamp(-1, 1).cpu().numpy().T  # [N, C]
        self._rendered_for = latent_btc
        return self._rendered_48k

    def render_window(self, t_start_s: float):
        decode_src = (
            self._current_result if self._current_result is not None
            else self._last_result_latent
        )
        if decode_src is None:
            return None
        if self._windowed_codec:
            return self._render_window_via_codec(decode_src, t_start_s)
        audio = self._rendered_audio(decode_src)
        n = int(round(self.vae_window * DELIVERY_SAMPLE_RATE))
        start = int(round(t_start_s * DELIVERY_SAMPLE_RATE))
        start = max(0, min(start, max(0, audio.shape[0] - n)))
        return AudioChunk(pcm=audio[start:start + n], start_sample=start)

    def _render_window_via_codec(self, latent_btc: torch.Tensor, t_start_s: float):
        """Windowed-codec render (SAME-L / medium): decode ONLY a small
        latent window around the target, then resample that window.

        44.1k↔48k bookkeeping uses the exact 147:160 ratio. The decode
        request carries a 588-sample (= 640 at 48 k) guard margin on
        each side so the resampler's filter edges land outside the kept
        slice; the runner's 25 ms crossfade against the live buffer
        covers the (deterministic) window seams, exactly as it does for
        ACE's windowed VAE decode.
        """
        import torchaudio

        n48 = int(round(self.vae_window * DELIVERY_SAMPLE_RATE))
        dur48 = int(round((self.playable_duration_s() or 0.0) * DELIVERY_SAMPLE_RATE))
        start48 = int(round(t_start_s * DELIVERY_SAMPLE_RATE))
        start48 = max(0, min(start48, max(0, dur48 - n48)))

        m44 = 588                                # guard margin; 588*160/147 == 640
        start44 = (start48 * SA3_SAMPLE_RATE) // DELIVERY_SAMPLE_RATE
        n44 = -(-n48 * 147 // 160)               # ceil to cover n48 after resample
        lo44 = max(0, start44 - m44)
        lead44 = start44 - lo44
        total44 = lead44 + n44 + m44

        t0 = time.perf_counter()
        audio_ct = self.codec.decode_window(
            latent_btc.movedim(1, 2), lo44, total44,
        )
        audio48 = torchaudio.functional.resample(
            audio_ct.float(), SA3_SAMPLE_RATE, DELIVERY_SAMPLE_RATE,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_dec_ms += (time.perf_counter() - t0) * 1000

        lead48 = (lead44 * DELIVERY_SAMPLE_RATE) // SA3_SAMPLE_RATE
        pcm48 = audio48[:, lead48:lead48 + n48]
        if pcm48.shape[-1] < n48:
            pcm48 = torch.nn.functional.pad(pcm48, (0, n48 - pcm48.shape[-1]))
        pcm = pcm48.clamp(-1, 1).cpu().numpy().T  # [N, C]
        return AudioChunk(pcm=pcm, start_sample=start48)

    def render_full(self):
        if self._current_result is None:
            return None
        return AudioChunk(
            pcm=self._rendered_audio(self._current_result), start_sample=0,
        )

    # ---- bookkeeping -------------------------------------------------------------

    def on_fresh_generation(self, knobs: dict) -> None:
        if self.state is None:
            return
        p = self.state.params
        p["num_gens"] = p.get("num_gens", 0) + 1
        p["tick_ms"] = self.last_tick_ms
        p["dec_ms"] = self.last_dec_ms
        prep = getattr(self, "_last_prep", None)
        if prep:
            p["sa3_denoise"] = round(prep["denoise"], 2)
            p["seed"] = prep["seed"]
            p["steps_override"] = prep["steps"]
        p["_prompt"] = getattr(self.state, "prompt_text", "")
