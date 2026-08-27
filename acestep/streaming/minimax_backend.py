"""MiniMaxBackend: MiniMax-Music3 behind the Tier-1 GeneratorBackend seam.

The streaming shape here is the SA3 one — every emit is a
partial-denoise cover of the SAME source latent at the SAME seed, so
advancing playback windows reconstruct one evolving song — but the
conditioning story is different enough to be worth stating plainly.

MiniMax's DiT has no cross-attention and no text input. Its only
conditioning is ``encoder_hidden_states`` ``[B, T, 2048]``, produced by
running an 8.58B autoregressive LM over the prompt and fusing its
per-frame hidden states. That stage costs seconds and cannot be put in
a tick. So this backend treats conditioning as a *captured composition*:
:class:`~acestep.engine.minimax_context.MiniMaxContext` runs the AR
stage once at session create, and the stream then covers that fixed
musical idea indefinitely. A prompt change re-runs the AR stage on the
dispatcher thread and swaps the capture in, exactly as SA3 swaps a
re-encoded T5Gemma bundle.

What that costs: ``set_prompt`` is seconds, not one pipeline flush.
What survives: everything DEMON steers solver-side lands in one tick —
denoise, seed, the source lock, the feedback delay tap, and the shared
curves. Two extra family knobs fall out of the architecture rather than
being invented for it: ``minimax_cond_strength`` interpolates the
capture toward zeros (which is literally the model's own unconditional
CFG branch, so 0.0 is a defined operating point, not an extrapolation),
and prompt blending slerps between two captures per frame.

Delivery is 48 kHz because ``pipeline_runner`` hardcodes it and never
calls ``geometry()``. MiniMax is native 44.1 kHz. Unlike SA3 this
backend can resample without a guard margin: its decoder is
deterministic and cheap enough to decode the whole song at once, so the
full 44.1 kHz render is resampled once and window rendering is pure
indexing into the cached 48 kHz buffer. No window seams exist to fix.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import torch

from acestep.engine.minimax_adapter import (
    MINIMAX_COND_DIM,
    MINIMAX_LATENT_CHANNELS,
    MINIMAX_SAMPLE_RATE,
    MINIMAX_UPSAMPLE,
)
from acestep.engine.obs import logger
from acestep.engine.stream import SlotRequest, StreamPipeline
from acestep.streaming.diffusion_backend import DiffusionBackend
from acestep.streaming.generator_backend import (
    AudioChunk,
    AudioGeometry,
    Capabilities,
    TickContext,
)
from acestep.streaming.knobs import (
    KnobSpec,
    lora_strength_spec,
)
from acestep.streaming.knobs import knob_specs as registry_knob_specs

# The runner's world is 48 kHz (pipeline_runner.SAMPLE_RATE); it never
# reads geometry(). Resample at the decode boundary or nothing works.
DELIVERY_SAMPLE_RATE = 48000

MINIMAX_LATENT_RATE_HZ = float(MINIMAX_SAMPLE_RATE) / float(MINIMAX_UPSAMPLE)

# The AR stage emits 25 frames/s; the DiT latent runs at 86.133 Hz, so
# each AR frame covers 3.4453 latent frames. Upstream renders in
# 200-AR-frame windows (8.0 s = 689 latent frames) in all three of its
# implementations, and the DiT is trained at that span — so it is the
# natural song length for a session even though shorter spans are
# mechanically legal.
MINIMAX_AR_FRAME_RATE_HZ = 25.0
MINIMAX_CHUNK_AR_FRAMES = 200


def minimax_latent_frames(duration_s: float) -> int:
    """Latent frame count for ``duration_s`` of audio."""
    return int(duration_s * MINIMAX_LATENT_RATE_HZ)


def minimax_knob_specs(loras=()) -> list:
    """The MiniMax knob manifest.

    Shared knobs are taken BY OBJECT out of the registry rather than
    re-declared, so a semantic fork is impossible rather than merely
    detected by the homonym guard.
    """
    shared = {s.name: s for s in registry_knob_specs(False)}
    specs = [
        KnobSpec(
            name="minimax_denoise",
            default=1.0,
            min_val=0.0,
            max_val=1.0,
            group="minimax",
            description=(
                "Cover strength. 1.0 regenerates from noise; lower values "
                "start each generation as a partially noised copy of the "
                "source anchor, which is what keeps consecutive emissions "
                "coherent."
            ),
        ),
        KnobSpec(
            name="minimax_shift",
            default=1.0,
            min_val=0.25,
            max_val=4.0,
            group="minimax",
            description=(
                "Schedule warp. >1 spends more steps near noise "
                "(structure), <1 near the data (refinement)."
            ),
        ),
        KnobSpec(
            name="minimax_cond_strength",
            default=1.0,
            min_val=0.0,
            max_val=1.5,
            group="minimax",
            description=(
                "How strongly the captured composition asserts itself. "
                "Interpolates the AR conditioning toward zeros, which is "
                "the model's own unconditional branch."
            ),
        ),
    ]
    specs += [
        shared["seed"],
        shared["steps_override"],
        shared["x0_target"],
        shared["feedback"],
        shared["feedback_depth"],
    ]
    specs += [lora_strength_spec(lid) for lid in (loras or ())]
    return specs


class MiniMaxBackend(DiffusionBackend):
    """See module docstring."""

    name = "minimax"

    def __init__(
        self,
        *,
        adapter,
        codec,
        cond,
        cond_b=None,
        schedule_builder_factory,
        knob_state,
        state=None,
        context=None,
        source_latent_bct: Optional[torch.Tensor] = None,
        duration_s: float = 8.0,
        steps: int = 8,
        depth: int = 4,
        vae_window_s: float = 0.36,
        seed: int = 1528,
    ):
        super().__init__(adapter=adapter, codec=codec)

        # Held for handle_set_prompt, which has to re-run the AR stage.
        self._context = context

        # Read unguarded by session.py when it builds the runner.
        self.vae_window = float(vae_window_s)

        self.knob_state = knob_state
        self.state = state
        self._duration_s = float(duration_s)
        self._latent_frames = minimax_latent_frames(duration_s)
        self._steps = int(steps)
        self._depth = int(depth)
        self._seed = int(seed)

        # Conditioning captures. Guarded because handle_set_prompt runs
        # on the dispatcher thread while produce() runs on the runner
        # thread, and a multi-field publish is not GIL-atomic.
        self._control_lock = threading.Lock()
        self._cond = cond
        self._cond_b = cond_b
        self._blend = 0.0
        self._active_cond = self._compose_cond(cond, cond_b, 0.0)

        self._schedule_builder_factory = schedule_builder_factory

        # The song this stream is covering. None until the first
        # generation lands, at which point we adopt it — the model has
        # no shipped audio encoder, so "continue from your own
        # generation" is the anchor path that actually works.
        self._source_latent = source_latent_bct

        # Decoded-audio cache, keyed by the identity of the latent it
        # came from. render_window is called up to twice per tick and a
        # full decode is ~10^2 ms, so this is load-bearing, not an
        # optimization.
        # Keyed by the latent OBJECT, held by strong reference. An
        # id()-keyed cache can stale-hit once the old tensor is freed and
        # a new one is allocated at the same address, which would serve
        # the previous generation's audio forever.
        self._decode_src: Optional[torch.Tensor] = None
        self._decode_pcm: Optional[np.ndarray] = None

        # Feedback delay tap: a ring of recent finished latents, blended
        # back into the anchor before the next submit. Depth is bounded
        # by the shared registry spec so the ring size and the knob's
        # range can never drift apart.
        self._max_feedback_depth = int(
            next(
                s.max_val for s in registry_knob_specs(False)
                if s.name == "feedback_depth"
            )
        )
        self._latent_history: list = []

        self._pending_steps: Optional[int] = None
        self._last_request = None

        self.pipeline = self._build_pipeline(self._steps)

    # ---- construction --------------------------------------------------------

    @classmethod
    def from_context(
        cls,
        context,
        *,
        cond,
        cond_b=None,
        knob_state,
        state=None,
        source_latent_bct=None,
        duration_s: float = 8.0,
        steps: int = 8,
        depth: int = 4,
        vae_window_s: float = 0.36,
        dit_backend: str = "eager",
        codec_backend: str = "eager",
    ):
        from acestep.engine.minimax_adapter import MiniMaxAdapter

        latent_frames = minimax_latent_frames(duration_s)
        dit = context.make_dit(
            latent_frames=latent_frames, backend=dit_backend,
        )
        codec = context.make_codec(backend=codec_backend)

        def _factory(active_cond, step_count):
            return context.make_schedule_builder(active_cond, step_count)

        adapter = MiniMaxAdapter(
            dit,
            schedule_builder=_factory(cond, steps),
            device=context.device,
            dtype=context.dtype,
        )
        return cls(
            adapter=adapter,
            codec=codec,
            cond=cond,
            cond_b=cond_b,
            schedule_builder_factory=_factory,
            knob_state=knob_state,
            state=state,
            context=context,
            source_latent_bct=source_latent_bct,
            duration_s=duration_s,
            steps=steps,
            depth=depth,
            vae_window_s=vae_window_s,
        )

    def _build_pipeline(self, steps: int) -> StreamPipeline:
        from acestep.engine.diffusion import DiffusionConfig

        config = DiffusionConfig(
            infer_steps=int(steps),
            # MiniMax's reference sampler is plain forward Euler on a
            # uniform schedule; the ODE path is the faithful one.
            infer_method="ode",
            noise_on_cpu=True,
            # ACE wavelet-corrector semantics; not MiniMax's.
            dcw_enabled=False,
        )
        return StreamPipeline(
            None, config, pipeline_depth=self._depth, adapter=self.adapter,
        )

    # ---- conditioning --------------------------------------------------------

    @staticmethod
    def _compose_cond(cond, cond_b, blend: float) -> dict:
        """Active conditioning bundle for the A/B blend position.

        Endpoints return the verbatim bundle object so an accelerated
        wrapper's identity-keyed staging cache stays warm. Interior
        points slerp per token: a linear midpoint collapses the norm of
        the conditioning and sounds washed out, the same failure SA3
        and ACE both hit.
        """
        if cond_b is None or blend <= 1e-6:
            return cond
        if blend >= 1.0 - 1e-6:
            return cond_b

        a = cond["encoder_hidden_states"].float()
        b = cond_b["encoder_hidden_states"].float()
        a_n = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        b_n = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        omega = (a_n * b_n).sum(-1, keepdim=True).clamp(-1.0, 1.0).acos()
        sin_omega = omega.sin().clamp_min(1e-6)
        w_a = ((1.0 - blend) * omega).sin() / sin_omega
        w_b = (blend * omega).sin() / sin_omega
        merged = dict(cond)
        merged["encoder_hidden_states"] = (w_a * a + w_b * b).to(a.dtype)
        return merged

    def _cond_for_tick(self, strength: float) -> dict:
        """Apply ``minimax_cond_strength`` to the active capture.

        Scaling toward zero walks toward the model's own unconditional
        branch (the reference pipeline's negative CFG input is literally
        ``torch.zeros_like(condition)``), so the whole 0..1 range is a
        defined operating point.
        """
        with self._control_lock:
            active = self._active_cond
        if abs(strength - 1.0) < 1e-6:
            return active
        out = dict(active)
        out["encoder_hidden_states"] = active["encoder_hidden_states"] * strength
        return out

    def handle_set_prompt(self, tags, *, tags_b=None) -> None:
        """Re-run the AR stage and swap the capture in.

        Runs on the dispatcher thread. Seconds, not milliseconds — the
        cost is an 8.58B LM pass, which is the price of MiniMax having
        no text path into its DiT.
        """
        context = getattr(self, "_context", None)
        if context is None:
            raise RuntimeError("minimax backend has no context to recompose with")

        cond = context.prepare_cond(prompt=tags, duration_s=self._duration_s)
        cond_b = (
            context.prepare_cond(prompt=tags_b, duration_s=self._duration_s)
            if tags_b else None
        )
        got = cond["encoder_hidden_states"].shape[1]
        if got != self._latent_frames:
            raise ValueError(
                "minimax prompt swap changed latent geometry "
                f"({self._latent_frames} -> {got}); duration is fixed for "
                "the session lifetime"
            )

        with self._control_lock:
            self._cond = cond
            self._cond_b = cond_b
            self._active_cond = self._compose_cond(cond, cond_b, self._blend)
            self.adapter.schedule_builder = self._schedule_builder_factory(
                self._active_cond, self._steps
            )
        # The schedule cache is keyed by denoise alone, so a builder
        # swap is invisible to it without this.
        self.pipeline.invalidate_schedule_cache()
        logger.info("minimax_prompt_swapped frames={}", got)

    def handle_set_prompt_blend(self, value: float) -> None:
        blend = max(0.0, min(1.0, float(value)))
        with self._control_lock:
            self._blend = blend
            self._active_cond = self._compose_cond(
                self._cond, self._cond_b, blend,
            )

    # ---- Tier-1 contract -----------------------------------------------------

    def capabilities(self) -> Capabilities:
        # Deliberately minimal, the way SA3 started. swap/write_audio
        # need an audio encoder this checkpoint does not ship converted;
        # LoRA needs a refit story. Each earns its flag on evidence.
        return Capabilities(
            refines_audio=True,
            loop_band=True,
            render_anchor_queue=True,
            depth=True,
            curves=True,
        )

    def geometry(self) -> AudioGeometry:
        return AudioGeometry(
            sample_rate=DELIVERY_SAMPLE_RATE,
            channels=2,
            chunk_rate_hz=MINIMAX_LATENT_RATE_HZ,
            duration_s=self.playable_duration_s(),
        )

    def knob_specs(self, lora_ids=()) -> list:
        return minimax_knob_specs(loras=list(lora_ids or ()))

    def read_knobs(self) -> dict:
        return self.knob_state.get_all_values()

    def playable_duration_s(self) -> Optional[float]:
        return self._duration_s

    def rebuild_imminent(self, knobs: dict) -> bool:
        want = int(knobs.get("steps_override", self._steps) or self._steps)
        want = max(1, want)
        if want != self._steps:
            self._pending_steps = want
            return True
        self._pending_steps = None
        return False

    # ---- produce -------------------------------------------------------------

    def _prepare_tick(self, knobs: dict, ctx: TickContext) -> dict:
        shift = float(knobs.get("minimax_shift", 1.0) or 1.0)
        if abs(shift - self.adapter.shift_alpha) > 1e-6:
            self.adapter.shift_alpha = shift
            self.pipeline.invalidate_schedule_cache()

        x0_strength = float(knobs.get("x0_target", 0.0) or 0.0)
        # Shared curves land on IN-FLIGHT slots on the very next tick,
        # bypassing the ring drain — this is why a knob move is felt
        # immediately even at depth 4.
        self.pipeline.set_shared_curve("x0_target_strength", x0_strength)

        return {
            "denoise": float(knobs.get("minimax_denoise", 1.0) or 1.0),
            "cond_strength": float(knobs.get("minimax_cond_strength", 1.0) or 1.0),
            "x0_target": x0_strength,
            "seed": int(knobs.get("seed", self._seed) or self._seed),
            "feedback": float(knobs.get("feedback", 0.0) or 0.0),
            "feedback_depth": int(knobs.get("feedback_depth", 1) or 1),
        }

    def _tapped_source(self, prep: dict) -> Optional[torch.Tensor]:
        """Anchor with the feedback delay tap applied.

        Blends a past output latent back into the source so the stream
        drifts rather than orbiting one fixed idea. Deliberately
        UPSTREAM of x0_target, which stays locked to the clean anchor —
        feedback moves the song, the source lock pulls it home.
        """
        anchor = self._source_latent
        if anchor is None:
            return None
        amount = prep["feedback"]
        if amount <= 1e-6 or not self._latent_history:
            return anchor
        depth = max(1, min(prep["feedback_depth"], len(self._latent_history)))
        past = self._latent_history[-depth]
        if past.shape != anchor.shape:
            return anchor
        return (1.0 - amount) * anchor + amount * past

    def _generate(self, prep: dict):
        if self._pending_steps is not None:
            self._steps = self._pending_steps
            self._pending_steps = None
            # The schedule builder is parameterized by the step count, so
            # it has to be rebuilt alongside the pipeline or the adapter
            # keeps handing back a schedule of the previous length.
            with self._control_lock:
                self.adapter.schedule_builder = self._schedule_builder_factory(
                    self._active_cond, self._steps
                )
            self.pipeline = self._build_pipeline(self._steps)

        denoise = prep["denoise"]
        # Without an anchor the first generation must run from pure
        # noise; a partial denoise of nothing is not defined.
        if self._source_latent is None:
            denoise = 1.0

        request = SlotRequest(
            seed=prep["seed"],
            denoise=denoise,
            source_latents=self._tapped_source(prep),
            aux_cond=self._cond_for_tick(prep["cond_strength"]),
            latent_frames=self._latent_frames,
            # The morph target stays the CLEAN anchor, not the
            # feedback-blended source: x0_target is a source lock,
            # feedback is deliberately upstream of it. Attached whenever
            # an anchor exists so a strength bump via the shared curve
            # engages on in-flight slots too.
            x0_target=self._source_latent,
            x0_target_strength=prep["x0_target"],
            # Deterministic per-slot renoise. Without it, consecutive
            # covers are different realizations of the same latent and
            # advancing playback windows splice incoherently.
            sde_noise_seeded=True,
        )
        self.pipeline.submit(request)
        self._last_request = request
        return self.pipeline.tick()

    def _after_produce(self, prep: dict, result_latent, is_fresh: bool) -> None:
        if not is_fresh or result_latent is None:
            return
        # Adopt the first completed generation as the song this stream
        # covers from here on. MiniMax ships no converted audio encoder,
        # so "continue from your own generation" is the anchor path that
        # actually exists.
        if self._source_latent is None:
            self._source_latent = result_latent.detach().clone()
            logger.info(
                "minimax_anchor_adopted frames={}", result_latent.shape[1],
            )
        self._latent_history.append(result_latent.detach())
        if len(self._latent_history) > self._max_feedback_depth:
            self._latent_history.pop(0)

    def on_fresh_generation(self, knobs: dict) -> None:
        # Per-generation params echo. Nothing family-specific to mirror
        # yet; the runner calls this only on a fresh produce+render.
        pass

    # ---- render --------------------------------------------------------------

    def _decoded(self) -> Optional[np.ndarray]:
        """Full song at 48 kHz, ``[frames, channels]``, cached.

        Decoding the whole 8 s at once and resampling it in one piece
        means window rendering never crosses a decode or resampler
        boundary, so there are no seams to crossfade and no guard
        margins to get wrong.
        """
        latent = self._last_result_latent
        if latent is None:
            return None
        if latent is self._decode_src and self._decode_pcm is not None:
            return self._decode_pcm

        with torch.no_grad():
            # Engine layout [1, T, C] -> MiniMax-native [1, C, T].
            audio = self.codec.decode_full(latent.movedim(1, 2))
        pcm = self._to_delivery(audio)
        self._decode_src = latent
        self._decode_pcm = pcm
        return pcm

    @staticmethod
    def _to_delivery(audio: torch.Tensor) -> np.ndarray:
        """MiniMax-native 44.1 kHz ``[C, N]`` -> 48 kHz ``[N, C]`` float32."""
        import torchaudio

        if audio.ndim == 3:
            audio = audio[0]
        audio = audio.detach().float().cpu()
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        resampled = torchaudio.functional.resample(
            audio, MINIMAX_SAMPLE_RATE, DELIVERY_SAMPLE_RATE,
        )
        return resampled.transpose(0, 1).contiguous().numpy().astype(np.float32)

    def render_window(self, t_start_s: float) -> Optional[AudioChunk]:
        pcm = self._decoded()
        if pcm is None:
            return None
        total = pcm.shape[0]
        start = int(round(float(t_start_s) * DELIVERY_SAMPLE_RATE))
        start = max(0, min(start, max(0, total - 1)))
        length = int(round(self.vae_window * DELIVERY_SAMPLE_RATE))
        end = min(total, start + max(1, length))
        # .copy() is mandatory: the runner crossfades INTO the array we
        # return, in place, and this one is our decode cache.
        return AudioChunk(pcm=pcm[start:end].copy(), start_sample=start)

    def render_full(self) -> Optional[AudioChunk]:
        pcm = self._decoded()
        if pcm is None:
            return None
        return AudioChunk(pcm=pcm.copy(), start_sample=0)

    # ---- teardown ------------------------------------------------------------

    def close(self) -> None:
        self._decode_pcm = None
        self._decode_src = None
        self._source_latent = None
        self._last_result_latent = None
        self._current_result = None
