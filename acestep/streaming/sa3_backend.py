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

v1 surface (everything else off, capability-gated): ``prompt`` (fixed
per conditioning bundle), fixed duration, ``seed``, ``steps_override``,
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
        vae_window_s: float = 0.36,
        sampler: str = "ode",
    ):
        super().__init__(adapter=adapter, codec=codec)
        self._cond = cond
        self._schedule_builder_factory = schedule_builder_factory
        self.knob_state = knob_state
        self.state = state
        self._steps = int(steps)
        self._depth = int(depth)
        self._default_seed = int(default_seed)
        self.vae_window = float(vae_window_s)
        self._sampler = sampler  # "ode" (continuity) | "sde" (pingpong)

        # Source anchor for audio-to-audio: engine layout [1, T, 256].
        self._source_latent_btc = (
            source_latent_bct.movedim(1, 2).contiguous()
            if source_latent_bct is not None else None
        )

        # Rendered-audio cache: one full decode+resample per fresh
        # latent (SAME-S decodes the whole window in ~11 ms); window
        # renders slice it, so gap-fill re-renders are bit-stable.
        self._rendered_for = None     # latent tensor identity
        self._rendered_48k = None     # np.ndarray [N, C] float32

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
        **kwargs,
    ) -> "SA3Backend":
        """Production assembly over a loaded
        :class:`~acestep.engine.sa3_context.SA3Context`."""
        from acestep.engine.sa3_adapter import SA3Adapter
        from acestep.engine.sa3_context import SA3SAMECodec

        steps = int(kwargs.get("steps", 8))
        cond = context.prepare_cond(
            prompt=prompt, duration=duration_s, steps=steps,
        )
        source_latent = (
            context.encode_source(source_audio, cond.audio_sample_size)
            if source_audio is not None else None
        )
        adapter = SA3Adapter(
            context.dit,
            schedule_builder=context.make_schedule_builder(cond, steps),
            device=context.device,
            dtype=context.dtype,
        )
        return cls(
            adapter=adapter,
            codec=SA3SAMECodec(context),
            cond=cond,
            schedule_builder_factory=(
                lambda s: context.make_schedule_builder(cond, s)
            ),
            knob_state=knob_state,
            state=state,
            source_latent_bct=source_latent,
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
        audio = self._rendered_audio(decode_src)
        n = int(round(self.vae_window * DELIVERY_SAMPLE_RATE))
        start = int(round(t_start_s * DELIVERY_SAMPLE_RATE))
        start = max(0, min(start, max(0, audio.shape[0] - n)))
        return AudioChunk(pcm=audio[start:start + n], start_sample=start)

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
