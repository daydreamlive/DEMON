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
calls ``geometry()``. MiniMax is native 44.1 kHz, so every render
crosses a resampler as well as a decoder, and both want a guard.

Rendering is WINDOWED: each call decodes a fixed 56-frame span around
the requested slice and keeps the middle. An earlier version decoded the
whole song on every fresh latent and indexed into the result, which is
O(song length) per generation where this is O(window) -- 44.5 ms against
~5 ms at 8 s, and 346 ms against the same ~5 ms at 60 s. It also arrived
as a spike on one tick in four, which the runner's lead controller reads
as a longer inter-write interval and answers by inflating playback lead,
coupling knob-to-ear latency to song duration. The decoder is
deterministic (no inference-time noise to seed) and purely
convolutional, so a guarded window is exact rather than approximate.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
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
# each AR frame covers 3.4453 latent frames (441/128 exactly).
#
# Upstream renders in 200-AR-frame windows on a 100-frame hop, carrying
# 172 latent frames of overlap between windows. That is an INFERENCE
# CONTRACT, not a training span: the transformer config carries no
# max_position_embeddings, its RoPE is computed for whatever length
# arrives, and nothing upstream states a trained window. Treating 200 as
# a model limit was an error this file used to make. It is the default
# request only.
MINIMAX_AR_FRAME_RATE_HZ = 25.0
MINIMAX_CHUNK_AR_FRAMES = 200

# --- windowed decode -------------------------------------------------------
#
# 44100 and 48000 reduce to 147:160. The consequence is load-bearing: a
# latent frame is 512 native samples and gcd(512, 147) == 1, so a frame
# boundary is NOT an integer delivery sample. Resampling a block that
# starts on a frame boundary therefore lands on a different sample phase
# than resampling the whole song does, and the window disagrees with the
# full decode by up to half a sample -- which on broadband material is a
# ~17% relative error, not a rounding detail. The fix is to trim the
# decoded block forward to a multiple of 147 native samples so the
# resampled block starts on an exact delivery sample. SA3 solves the same
# problem the same way with its 588-sample (4 x 147) guard.
_RESAMPLE_NUM = 160   # 48000 / gcd(44100, 48000)
_RESAMPLE_DEN = 147   # 44100 / gcd(44100, 48000)

# Measured, not guessed: scripts/minimax/minimax_decode_profile.py decodes
# a slice, compares it against the same span of a full decode, and
# profiles the error inward from the edge. Peak error reaches the fp32
# floor (4.4e-3) at 9 latent frames and is flat thereafter; an analytic
# walk of the conv stack puts the one-sided field at 10. 12 buys margin
# over both, plus the sub-frame slack the 147-alignment trim needs, for
# a few frames of decode nobody will notice.
MINIMAX_VAE_GUARD_FRAMES = 12

# Fixed decode span, the ACE pattern. A constant shape means a live
# vae_window change can never shrink the guard below its converged floor,
# and it is the shape a TensorRT decoder engine would be built at.
#
# 58 = 34 keep + 24 guard. The keep number is not the window's frame
# count but its worst case: the span is converted with floor on one end
# and ceil on the other, so a 0.36 s window (31.008 frames of native
# audio) covers 33 frames when it starts just inside a frame boundary.
# Sizing this to the average instead of the worst case fails on roughly
# one start position in three, which is exactly the sort of thing that
# survives a spot check and dies in a stream.
MINIMAX_VAE_DECODE_FRAMES = 58

# Sampler defaults, measured rather than inherited. The reference
# pipeline runs 30 unwarped steps at guidance 1.7; that is 60 forwards
# per generation, more than a real-time ring wants to spend. The grid in
# ``scripts/minimax/minimax_quality_ablation.py`` says where the cost can
# come off and where it cannot:
#
#   * Guidance is not optional. Unguided sampling plateaus at ~0.11
#     log-mel from the reference and stays there through 40 steps; eight
#     guided steps beat forty unguided ones. Dropping CFG is the single
#     largest quality loss available on this model.
#   * Step count trades against schedule warp almost one for one. The
#     unwarped schedule needs 30 steps; warping toward the noise end
#     buys back most of it, and (16 steps, shift 2.0) lands at log-mel
#     0.032 / latent cosine 0.9993 against the reference for 32
#     forwards -- close enough that the residual is bf16 rounding.
#
# Hence: 16 steps, shift 2.0, guidance 1.7. Raising steps toward 30 is
# only worth it alongside lowering shift back toward 1.0; the two are a
# matched pair, not independent quality dials.
MINIMAX_DEFAULT_STEPS = 16
MINIMAX_DEFAULT_SHIFT = 2.0
MINIMAX_DEFAULT_GUIDANCE = 1.7


def minimax_latent_frames(duration_s: float) -> int:
    """Latent frame count for ``duration_s`` of audio."""
    return int(duration_s * MINIMAX_LATENT_RATE_HZ)


def minimax_delivery_samples(latent_frames: int) -> int:
    """Delivery-rate length of ``latent_frames``, the resampler's answer.

    The single source of this number. It was previously derived twice --
    once as ``round(duration_s * 48000)`` for the ring buffer and once
    implicitly by the resampler -- which left the ring 34 samples longer
    than any decode produced, so the song's last ~0.7 ms was never
    written. ``torchaudio.functional.resample`` returns
    ``ceil(n * new / orig)``; match it exactly rather than approximately.
    """
    native = int(latent_frames) * MINIMAX_UPSAMPLE
    return -(-native * _RESAMPLE_NUM // _RESAMPLE_DEN)


# One render window plus a full guard on each side. Below this a song
# cannot fill a window without reading its own guard twice, and the
# cyclic wrap stops being a margin and becomes the signal.
MINIMAX_MIN_LATENT_FRAMES = MINIMAX_VAE_DECODE_FRAMES


def minimax_max_vae_window_s(
    decode_frames: int = MINIMAX_VAE_DECODE_FRAMES,
    guard: int = MINIMAX_VAE_GUARD_FRAMES,
) -> float:
    """Widest wire slice the fixed decode span can serve with full guard.

    Two frames of headroom rather than none: the kept span is converted
    to frames with floor on one end and ceil on the other, so a window
    whose native length is 31.008 frames can still touch 33 of them
    depending on where it starts.
    """
    return (decode_frames - 2 * guard - 2) / MINIMAX_LATENT_RATE_HZ


@dataclass(frozen=True)
class DecodePlan:
    """Where a windowed decode reads from and what it keeps.

    Pure arithmetic, deliberately separated from the decode so it can be
    tested exhaustively without weights -- the off-by-one that matters
    here is a sample-phase error, which is invisible in a listening test
    and expensive in a GPU one.

    ``frame_start`` may be negative or run past the end of the song: the
    guard wraps cyclically, because the ring buffer loops and the song's
    tail genuinely is what plays into its head. Only the guard wraps --
    the kept span is clamped to the song, and the runner asks for the
    wrapped remainder in a separate call.
    """
    frame_start: int      # first latent frame to decode; wraps
    frames: int           # always the fixed decode span
    trim_native: int      # native samples dropped for 147-alignment
    offset: int           # index into the resampled block where the keep starts
    length: int           # delivery samples to keep


def plan_decode_window(
    start_48k: int,
    length_48k: int,
    total_frames: int,
    *,
    guard: int = MINIMAX_VAE_GUARD_FRAMES,
    decode_frames: int = MINIMAX_VAE_DECODE_FRAMES,
) -> DecodePlan:
    """Plan a decode that serves ``[start_48k, start_48k + length_48k)``.

    The kept span is converted to latent frames, padded to the fixed
    decode span, trimmed forward to a 147-sample boundary so the
    resample lands on the same phase grid a whole-song resample would,
    and the offset back into the block is then exact integer arithmetic.
    """
    span = min(decode_frames, total_frames)
    p0 = start_48k * _RESAMPLE_DEN / _RESAMPLE_NUM
    p1 = (start_48k + length_48k) * _RESAMPLE_DEN / _RESAMPLE_NUM
    keep0 = int(math.floor(p0 / MINIMAX_UPSAMPLE))
    keep1 = int(math.ceil(p1 / MINIMAX_UPSAMPLE))
    keep = keep1 - keep0
    if keep + 2 * guard > span and total_frames > span:
        raise ValueError(
            f"window of {length_48k} delivery samples needs {keep} latent "
            f"frames, which leaves less than {guard} frames of guard inside "
            f"a {span}-frame decode; lower vae_window (max "
            f"{minimax_max_vae_window_s(span, guard):.4f}s) or widen "
            "MINIMAX_VAE_DECODE_FRAMES"
        )

    # The whole song fits inside one decode: no windowing to do, and
    # anchoring at 0 keeps the offset arithmetic trivially exact.
    f0 = 0 if total_frames <= span else keep0 - (span - keep) // 2

    n0 = f0 * MINIMAX_UPSAMPLE
    trim = (-n0) % _RESAMPLE_DEN
    n0 += trim
    # n0 is now a multiple of 147, so this is exact rather than rounded.
    offset = start_48k - (n0 // _RESAMPLE_DEN) * _RESAMPLE_NUM
    return DecodePlan(
        frame_start=f0, frames=span, trim_native=trim,
        offset=offset, length=length_48k,
    )


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
            default=MINIMAX_DEFAULT_SHIFT,
            min_val=0.25,
            max_val=4.0,
            group="minimax",
            description=(
                "Schedule warp. >1 spends more steps near noise "
                "(structure), <1 near the data (refinement). Matched to "
                "the step count: the default 16 steps want 2.0, and 30 "
                "steps want 1.0."
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
        KnobSpec(
            name="minimax_guidance",
            default=MINIMAX_DEFAULT_GUIDANCE,
            min_val=1.0,
            max_val=3.0,
            group="minimax",
            description=(
                "Classifier-free guidance scale, the reference "
                "pipeline's own parameter. 1.0 turns guidance off and "
                "halves the compute, but this model needs guidance more "
                "than it needs steps -- unguided output plateaus well "
                "short of the reference no matter how many steps it gets."
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
        steps: int = MINIMAX_DEFAULT_STEPS,
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
        self._uncond_cache: Optional[torch.Tensor] = None

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
        self._last_prep: Optional[dict] = None

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
        steps: int = MINIMAX_DEFAULT_STEPS,
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

    def _uncond_bundle(self) -> dict:
        """The model's unconditional branch: an all-zeros capture.

        Cached on the instance because it is 689x2048 and identical
        every tick, and because an accelerated wrapper's staging cache
        is keyed by tensor identity -- a fresh ``zeros_like`` each tick
        would miss it every time.

        Note that ``minimax_cond_strength`` at 0.0 makes the positive
        bundle equal to this one, at which point guidance is a no-op
        rather than an error: the guidance direction around a point is
        zero. That is a coherent operating point, not a trap.
        """
        zeros = self._uncond_cache
        with self._control_lock:
            cond = self._active_cond["encoder_hidden_states"]
        if (
            zeros is None
            or zeros.shape != cond.shape
            or zeros.dtype != cond.dtype
            or zeros.device != cond.device
        ):
            zeros = torch.zeros_like(cond)
            self._uncond_cache = zeros
        return {"encoder_hidden_states": zeros}

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
        shift = float(
            knobs.get("minimax_shift", MINIMAX_DEFAULT_SHIFT)
            or MINIMAX_DEFAULT_SHIFT
        )
        if abs(shift - self.adapter.shift_alpha) > 1e-6:
            self.adapter.shift_alpha = shift
            self.pipeline.invalidate_schedule_cache()

        x0_strength = float(knobs.get("x0_target", 0.0) or 0.0)
        # Shared curves land on IN-FLIGHT slots on the very next tick,
        # bypassing the ring drain — this is why a knob move is felt
        # immediately even at depth 4.
        self.pipeline.set_shared_curve("x0_target_strength", x0_strength)

        prep = {
            "denoise": float(knobs.get("minimax_denoise", 1.0) or 1.0),
            "cond_strength": float(knobs.get("minimax_cond_strength", 1.0) or 1.0),
            "guidance": float(
                knobs.get("minimax_guidance", MINIMAX_DEFAULT_GUIDANCE)
                or MINIMAX_DEFAULT_GUIDANCE
            ),
            "x0_target": x0_strength,
            "seed": int(knobs.get("seed", self._seed) or self._seed),
            "feedback": float(knobs.get("feedback", 0.0) or 0.0),
            "feedback_depth": int(knobs.get("feedback_depth", 1) or 1),
        }
        self._last_prep = prep
        return prep

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

        guidance = prep["guidance"]
        request = SlotRequest(
            seed=prep["seed"],
            denoise=denoise,
            source_latents=self._tapped_source(prep),
            aux_cond=self._cond_for_tick(prep["cond_strength"]),
            # Guidance, the reference pipeline's way. Its negative branch
            # is literally zeros, so the uncond bundle is free to build
            # and exactly the one the model was trained against.
            #
            # apg_eta/apg_norm_threshold/apg_momentum reduce DEMON's APG
            # to textbook CFG. That is not a stylistic preference: stock
            # APG measures ~4x worse against the reference here, because
            # its norm cap is calibrated for ACE's latent scale and
            # throttles a 689-frame guidance delta almost to nothing.
            neg_aux_cond=self._uncond_bundle() if guidance != 1.0 else None,
            guidance_curve=guidance if guidance != 1.0 else None,
            apg_momentum=0.0,
            apg_eta=1.0,
            apg_norm_threshold=0.0,
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
        """Mirror per-generation telemetry into session params.

        ``num_gens`` and ``tick_ms`` ride the binary slice header, and
        ``num_gens`` divided by wall time IS the throughput metric this
        project reports. Not writing them left the family invisible to
        every existing instrument -- and ``dec_ms`` in particular read a
        flat 0.0, which is exactly how a whole-song decode stayed hidden
        behind a 0.0 ms median render.
        """
        if self.state is None:
            return
        p = self.state.params
        p["num_gens"] = p.get("num_gens", 0) + 1
        p["tick_ms"] = self.last_tick_ms
        p["dec_ms"] = self.last_dec_ms
        prep = self._last_prep
        if prep:
            p["minimax_denoise"] = round(prep["denoise"], 2)
            p["minimax_guidance"] = round(prep["guidance"], 2)
            p["minimax_cond_strength"] = round(prep["cond_strength"], 2)
            p["seed"] = prep["seed"]
            p["steps_override"] = self._steps

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
    def _to_delivery(audio: torch.Tensor, *, trim_native: int = 0) -> np.ndarray:
        """MiniMax-native 44.1 kHz ``[C, N]`` -> 48 kHz ``[N, C]`` float32.

        ``trim_native`` drops samples from the head BEFORE resampling, so
        the block starts on a 147-sample boundary and the resampler lands
        on the same phase grid as a whole-song resample. See the note on
        ``_RESAMPLE_DEN``.
        """
        import torchaudio

        if audio.ndim == 3:
            audio = audio[0]
        audio = audio.detach().float().cpu()
        if trim_native:
            audio = audio[:, trim_native:]
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        resampled = torchaudio.functional.resample(
            audio, MINIMAX_SAMPLE_RATE, DELIVERY_SAMPLE_RATE,
        )
        return resampled.transpose(0, 1).contiguous().numpy().astype(np.float32)

    def delivery_samples(self) -> int:
        """Length of the song in delivery samples."""
        return minimax_delivery_samples(self._latent_frames)

    def render_window(self, t_start_s: float) -> Optional[AudioChunk]:
        """Decode ONLY this window, with a cyclic guard on each side.

        The previous implementation decoded the whole song on every fresh
        latent and indexed into the result. That is O(song length) per
        generation where this is O(window): measured 44.5 ms against
        ~5 ms at 8 s, and 346 ms against the same ~5 ms at 60 s. The
        whole-song version also arrived as a spike on one tick in four,
        which the runner's lead controller reads as a longer inter-write
        interval and answers by inflating playback lead -- coupling
        knob-to-ear latency to song duration, which is precisely what
        the windowed contract exists to prevent.
        """
        latent = self._last_result_latent
        if latent is None:
            return None
        total_frames = int(latent.shape[1])
        total = self.delivery_samples()
        start = int(round(float(t_start_s) * DELIVERY_SAMPLE_RATE))
        start = max(0, min(start, max(0, total - 1)))
        length = max(1, int(round(self.vae_window * DELIVERY_SAMPLE_RATE)))
        # Only the guard wraps. The kept span stops at the song's end and
        # the runner asks for the wrapped remainder in its own call.
        length = min(length, total - start)

        t0 = time.perf_counter()
        plan = plan_decode_window(start, length, total_frames)
        with torch.no_grad():
            idx = torch.arange(
                plan.frame_start, plan.frame_start + plan.frames,
                device=latent.device,
            ) % total_frames
            # Engine layout [1, T, C] -> MiniMax-native [1, C, T].
            sl = latent.index_select(1, idx).movedim(1, 2)
            audio = self.codec.decode_full(sl)
        pcm = self._to_delivery(audio, trim_native=plan.trim_native)
        self.last_dec_ms = (time.perf_counter() - t0) * 1000.0

        lo = max(0, min(plan.offset, pcm.shape[0]))
        out = pcm[lo:lo + plan.length]
        if out.shape[0] < plan.length:
            # Only reachable at the tail of a song shorter than one decode
            # span. Pad rather than short-return: the runner sizes its
            # underrun arithmetic against vae_window.
            out = np.pad(out, ((0, plan.length - out.shape[0]), (0, 0)))
        # Owned array, always: the runner crossfades INTO what we return.
        return AudioChunk(pcm=np.ascontiguousarray(out), start_sample=start)

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
