"""MiniMaxBackend: MiniMax-Music3 behind the Tier-1 GeneratorBackend seam.

MiniMax-Music3 is autoregressive. Its 8.58B Global LM writes one 25 Hz
acoustic frame at a time over a KV cache and stops when the piece is
done; the flow-matching DiT renders those frames into audio in
overlapping windows, carrying latent context between them. It is a
*producer*, not a one-shot sampler, and this backend drives it as one:
the AR stage and the chunked renderer run on a generation worker, and
the frontier they advance is handed to the runner as append-only audio.

That makes this the second append-only family behind the seam, after
MRT2, and it inherits that family's shape:

* ``render_window`` IGNORES the runner's position hint and returns the
  next frontier chunk. Committed audio is never re-rendered --
  ``Capabilities.refines_audio`` is False, and it is false all the way
  down: the AR stage cannot revise a frame it has emitted.
* Song shape is a rolling window. The frontier writes advance modulo
  ``window_s`` and the player loops it, so the "song" is a tape being
  overwritten just behind the playhead.
* Each emission re-emits the previous one's last ``XFADE`` samples at
  its head, so the runner's unconditional leading-edge crossfade blends
  identical samples instead of smearing new audio against last lap's.

What this backend deliberately does NOT do is run the ring buffer and
the batch-axis staircase. Those exist to make a one-shot, whole-song
diffusion model behave like a stream by keeping several partial
generations of the SAME song in flight on the batch axis. None of it
applies here, and one fact settles it: chunk k's carry is chunk k-1's
committed output, so consecutive renders are strictly dependent and
there is nothing to pipeline. An earlier version of this file froze the
AR output into a static conditioning tensor and ran the ring over it,
which is a streaming model converted into a one-shot model so that
streaming machinery could be applied to it.

## What it costs, measured on a 5090

Two rates, and they are NOT the same number -- conflating them is what
produced this integration's first round of throughput claims:

* **25 Hz** is the AR acoustic frame rate (40 ms per frame).
* **86.133 Hz** (44100/512) is the DiT latent frame rate (11.6 ms).

Against those, each stage measured alone
(``scripts/minimax/minimax_ar_bench.py``, ``minimax_stream_bench.py``):

    AR emission          53.6 ms/frame  = 0.75x realtime, flat in context
    chunk render, TRT    513 ms per 4.0 s commit = 7.8x realtime
    chunk render, eager  835 ms per 4.0 s commit = 4.8x realtime
    guarded decode       ~6 ms/window, negligible

And the two together, session means over 70-100 s runs, which is the
number that matters and is NOT the one the parts predict:

    hop=100    AR 57-61 ms/frame   render 855-1030 ms   end to end 0.54x
    hop=25     AR 54.2 ms/frame    render 518 ms        end to end 0.48x

**Co-residency lands on the render, and it scales with how long the AR
runs between renders.** With the AR stage's 17.4 GB pinned alongside the
renderer, a hop of 100 frames means ~6 s of language model between DiT
turns, and the chunk render roughly doubles (513 -> ~1030 ms) because
the DiT's weights are gone from cache by the time it runs again. At
hop=25 the renders are frequent enough to stay warm and the chunk render
comes in at 518 ms -- its isolated speed. Benchmarking either stage
alone, or benchmarking with a hop that does not match production,
overstates the pipeline.

The AR stage is the bottleneck and it is **under realtime**, so this
backend cannot sustain a live stream on this hardware: the playhead laps
the frontier and the listener hears the rolling window repeat. That is
reported, not hidden -- ``frontier_lead_s`` and ``ar_realtime`` ride the
params echo on every generation, and both are session means rather than
last-sample values (a single sample moves the figure by 15%).

Worth being precise about whose limitation that is, because it is
measurable and was measured (``minimax_ar_bench.py --profile``). Of a
52.6 ms frame, only 22.3 ms is GPU kernel time: the GPU **idles ~60% of
every frame** waiting on Python to launch the next of ~3900 kernels,
while the GEMMs that do run already reach **86% of the card's memory
bandwidth**. The stage is dispatch-bound, not bandwidth-bound, and
upstream serves this checkpoint through SGLang rather than a plain torch
loop. 0.75x is a property of this dependency-free reimplementation, not
a measurement of the model's ceiling.

The consequence for hardware: a faster card scales only the busy 22 ms,
so an H100 SXM projects to ~0.9x realtime for this stage alone and still
short end to end, while removing the dispatch gap with CUDA graphs would
reach 1.79x on the 5090. Deliberately not taken -- this family is a
backend-generality demonstration, not a speed target.

## What DEMON's value proposition buys here

Knob-to-ear on this family is **seconds, not milliseconds**, and the two
stages have different floors for different reasons. Measured
knob-to-frontier (add the playback lead for knob-to-ear):

    knob                        hop=100    hop=25
    minimax_guidance (render)   6.7-7.1 s   1.65 s
    minimax_temperature (AR)    8.6-9.8 s   10.8 s
    end-to-end throughput          0.54x     0.48x

A **renderer** knob waits for the next chunk render to start, which
means waiting for the AR stage to fill the next hop. ``minimax_hop`` is
the lever for it and trades directly against throughput, because a
smaller hop re-renders more of the same audio per second committed.

An **AR** knob does not benefit, and the reason is geometric rather than
budgetary. A frame written now sits inside a 200-frame conditioning
window whose commit region ends 150 frames before the window does, so
that frame cannot be committed until the LM has written up to 150 more
of them -- 6 s of audio, ~8 s of wall clock, no matter what the hop is.
Shrinking it means shrinking the chunk or the lookahead, which is a
quality question (the committed region would sit at the edge of the
model's window) rather than a scheduling one, and it is unmeasured.

So the *live steering* half of DEMON's proposition applies, at a
timescale of seconds rather than the ~60-230 ms the diffusion families
reach. What genuinely lands:

* AR sampling (``minimax_temperature`` / ``_top_k`` / ``_ar_guidance``)
  steers the composition **as it is being written**, at 40 ms
  granularity on the emission frontier. That is a stronger control than
  the cover architecture had, which could only re-render a frozen
  composition.
* ``set_prompt`` re-prefills the LM against the existing audio history
  instead of restarting the piece -- see
  :meth:`~acestep.engine.minimax_ar.MiniMaxARStream.reprompt`. A live
  prompt change on an autoregressive model, for the cost of a prefill.
* Renderer controls (steps / shift / guidance / cond strength) apply at
  the next chunk boundary.

## Delivery

MiniMax is native 44.1 kHz; ``pipeline_runner`` is 48 kHz and never
calls ``geometry()``. The conversion is append-only and has to stay
phase-exact across block boundaries, which is what
:class:`_DeliveryResampler` is for.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import torch

from acestep.engine.minimax_ar import ARControls
from acestep.engine.minimax_render import (
    AR_FRAME_RATE_HZ,
    CARRY_LATENT_FRAMES,
    CHUNK_AR_FRAMES,
    DECODE_GUARD_FRAMES,
    DEFAULT_GUIDANCE,
    DEFAULT_SHIFT,
    DEFAULT_STEPS,
    HOP_AR_FRAMES,
    MINIMAX_SAMPLE_RATE,
    MiniMaxChunkRenderer,
    MiniMaxLatentDecoder,
    MiniMaxLatentStream,
    RenderControls,
    latent_origin,
)
from acestep.engine.obs import logger
from acestep.streaming.generator_backend import (
    AudioChunk,
    AudioGeometry,
    Capabilities,
    LeadProfile,
    ProduceMode,
    TickContext,
    UnsupportedOperation,
)
from acestep.streaming.knobs import KnobSpec
from acestep.streaming.knobs import knob_specs as registry_knob_specs

# The runner's world is 48 kHz (pipeline_runner.SAMPLE_RATE); it never
# reads geometry(). Resample at the emission boundary or nothing works.
DELIVERY_SAMPLE_RATE = 48000

# 44100 and 48000 reduce to 147:160.
_RESAMPLE_DEN = 147   # native samples per ratio unit
_RESAMPLE_NUM = 160   # delivery samples per ratio unit

# Rolling-window song shape: the synthetic duration the session declares
# and the player loops.
DEFAULT_WINDOW_S = 60.0

# Overlap re-emitted at each chunk head so the runner's leading-edge
# crossfade blends against identical samples. Matches the runner's own
# fade length (min(1200, len // 4) at 48 kHz = 25 ms).
XFADE = 1200

# Cap on one emission. Keeps a post-stall burst from writing a
# multi-second slab in one tick.
MAX_EMIT_S = 1.5

# AR frames per worker step. Small enough that a stop request is honored
# promptly; large enough that per-call overhead is noise against
# 53 ms/frame. Control changes do not wait on it -- ARControls is read
# per frame inside the session.
AR_BATCH_FRAMES = 25

# The AR stage's resident footprint (17.9 GB of weights plus its KV
# cache). Stated so a deployment that cannot afford it picks a different
# checkpoint rather than discovering this at session create.
AR_RESIDENT_VRAM_GB = 21.0


def minimax_knob_specs(loras=()) -> list:
    """The MiniMax knob manifest.

    Shared knobs are taken BY OBJECT out of the registry rather than
    re-declared, so a semantic fork is impossible rather than merely
    detected by the homonym guard. The family-prefixed ones split by
    which stage they steer, because the two stages have very different
    latencies and an operator should be able to tell which is which:
    ``minimax_temperature`` / ``_top_k`` / ``_ar_guidance`` change what
    the LM writes next; the rest change how the DiT renders it.
    """
    shared = {s.name: s for s in registry_knob_specs(False)}
    specs = [
        KnobSpec(
            name="minimax_temperature",
            default=1.0,
            min_val=0.1,
            max_val=2.0,
            group="minimax",
            description=(
                "AR sampling temperature, applied after the top-k cut. "
                "Steers the composition as the LM writes it: higher "
                "wanders, lower commits to the caption. Lands on the "
                "next 40 ms frame the LM emits, which reaches the ear "
                "one chunk hop later."
            ),
        ),
        KnobSpec(
            name="minimax_top_k",
            default=50,
            min_val=1,
            max_val=200,
            type="int",
            group="minimax",
            description=(
                "AR top-k. 50 is the reference recipe. Narrowing it "
                "tightens the piece toward the model's confident "
                "continuations."
            ),
        ),
        KnobSpec(
            name="minimax_ar_guidance",
            default=1.5,
            min_val=1.0,
            max_val=3.0,
            group="minimax",
            description=(
                "Classifier-free guidance inside the AR stage, against "
                "a twin whose caption tokens are masked out. 1.5 is "
                "upstream's fixed value; it decides how literally the "
                "caption is obeyed."
            ),
        ),
        KnobSpec(
            name="minimax_guidance",
            default=DEFAULT_GUIDANCE,
            min_val=1.0,
            max_val=3.0,
            group="minimax",
            description=(
                "Renderer classifier-free guidance, the reference "
                "pipeline's own parameter. 1.0 turns it off and halves "
                "the render cost, but this model needs guidance more "
                "than it needs steps -- unguided output plateaus well "
                "short of the reference no matter how many steps it "
                "gets."
            ),
        ),
        KnobSpec(
            name="minimax_shift",
            default=DEFAULT_SHIFT,
            min_val=0.25,
            max_val=4.0,
            group="minimax",
            description=(
                "Schedule warp. >1 spends more steps near noise "
                "(structure), <1 near the data (refinement). Matched to "
                "the step count: 16 steps want 2.0, 30 steps want 1.0."
            ),
        ),
        KnobSpec(
            name="minimax_cond_strength",
            default=1.0,
            min_val=0.0,
            max_val=1.5,
            group="minimax",
            description=(
                "How strongly the AR stage's conditioning asserts "
                "itself in the render. Interpolates toward zeros, which "
                "is the model's own unconditional branch, so 0.0 is a "
                "defined operating point rather than an extrapolation."
            ),
        ),
        KnobSpec(
            name="minimax_hop",
            default=HOP_AR_FRAMES,
            min_val=10,
            max_val=150,
            type="int",
            group="minimax",
            description=(
                "AR frames committed per render, at 25 Hz. The "
                "latency lever for the RENDERER knobs only: measured "
                "guidance-to-frontier is 7.1 s at the default 100 "
                "(4.0 s of audio) and 1.6 s at 25, for a drop from "
                "0.54x to 0.46x realtime. It does NOT help the AR "
                "knobs -- their floor is the conditioning window's "
                "150-frame lookahead, which the hop does not move."
            ),
        ),
        KnobSpec(
            name="minimax_steps",
            default=DEFAULT_STEPS,
            min_val=8,
            max_val=40,
            type="int",
            group="minimax",
            description=(
                "Sampler steps. NOT the shared steps_override, which "
                "means ACE's turbo step count: its default of 8 is an "
                "audibly broken render here (log-mel 0.24 from the "
                "reference against 0.03) and its ceiling of 16 excludes "
                "the reference's own 30. Matched to minimax_shift -- "
                "30/1.0, 20/1.5, 16/2.0, 12/3.0 are the measured pairs, "
                "and lowering steps without raising shift gives up most "
                "of what the steps were buying."
            ),
        ),
        KnobSpec(
            name="minimax_lead",
            default=1.0,
            min_val=0.3,
            max_val=8.0,
            group="minimax",
            description=(
                "Target generation lead over the playhead in seconds. "
                "Append-only audio cannot be revised, so this IS the "
                "knob-to-ear floor once generation is fast enough to "
                "choose. On hardware where it is not, it never binds."
            ),
        ),
    ]
    # ``seed`` is taken from the shared registry by name so a semantic
    # fork is impossible; ``steps_override`` deliberately is NOT (see
    # minimax_steps above).
    specs += [shared["seed"]]
    return specs


# ---------------------------------------------------------------------------
# 44.1 kHz -> 48 kHz, append-only
# ---------------------------------------------------------------------------


class _DeliveryResampler:
    """Seamless append-only 44100 -> 48000.

    A whole-song resample and a block-wise one agree only if every block
    starts on the same sample phase. 44100/48000 reduces to 147/160 and
    ``gcd(512, 147) == 1``, so a latent-frame boundary is NOT a delivery
    sample: resampling from one lands up to half a sample off the grid a
    whole-song resample would use, which on broadband material is a ~17%
    relative error rather than a rounding detail. SA3 solves the same
    problem with the same 147 alignment.

    So block boundaries are pinned to multiples of 147 native samples,
    where the two grids coincide exactly, and each block is resampled
    with ``PAD`` native samples of filter context on both sides that are
    then discarded. torchaudio's kernel spans 6 input samples at this
    ratio; 588 is four ratio units of margin over it.
    """

    PAD = 4 * _RESAMPLE_DEN  # 588 native samples

    def __init__(self, channels: int = 2):
        self.channels = int(channels)
        self._buf: Optional[np.ndarray] = None   # [C, N] native
        self._base = 0        # absolute native index of _buf[:, 0]
        self._emitted = 0     # native samples converted; multiple of 147

    @property
    def native_available(self) -> int:
        if self._buf is None:
            return self._base
        return self._base + int(self._buf.shape[1])

    @property
    def emitted_native(self) -> int:
        return self._emitted

    def push(self, start_native: int, pcm: np.ndarray) -> None:
        """Append decoded native audio ``[C, N]`` starting at
        ``start_native`` (absolute). Must abut what is already held."""
        block = np.ascontiguousarray(pcm, dtype=np.float32)
        if self._buf is None:
            self._base = int(start_native)
            self._buf = block
            return
        if start_native != self.native_available:
            raise ValueError(
                f"native audio starts at {start_native}, frontier is at "
                f"{self.native_available}"
            )
        self._buf = np.concatenate((self._buf, block), axis=1)

    def pop(self) -> Optional[np.ndarray]:
        """Next delivery-rate block ``[N, C]`` float32, or None."""
        import torchaudio

        usable = self.native_available - self.PAD
        end = (usable // _RESAMPLE_DEN) * _RESAMPLE_DEN
        if end <= self._emitted:
            return None

        lo = max(self._base, self._emitted - self.PAD)
        hi = min(self.native_available, end + self.PAD)
        block = torch.from_numpy(
            self._buf[:, lo - self._base:hi - self._base]
        )
        resampled = torchaudio.functional.resample(
            block, MINIMAX_SAMPLE_RATE, DELIVERY_SAMPLE_RATE,
        )
        head = (self._emitted - lo) // _RESAMPLE_DEN * _RESAMPLE_NUM
        length = (end - self._emitted) // _RESAMPLE_DEN * _RESAMPLE_NUM
        out = resampled[:, head:head + length]

        self._emitted = end
        # Nothing behind this is ever re-read; the left pad is all the
        # history the next block needs.
        self._trim(self._emitted - self.PAD)
        return np.ascontiguousarray(
            out.transpose(0, 1).numpy(), dtype=np.float32,
        )

    def _trim(self, keep_from: int) -> None:
        if self._buf is None:
            return
        drop = int(keep_from) - self._base
        if drop <= 0:
            return
        self._buf = np.ascontiguousarray(self._buf[:, drop:])
        self._base = int(keep_from)

    def close(self) -> None:
        self._buf = None


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class MiniMaxBackend:
    """See module docstring. Append-only, in-process, rolling window."""

    name = "minimax"

    def __init__(
        self,
        *,
        ar_stream,
        renderer: MiniMaxChunkRenderer,
        codec,
        knob_state,
        state=None,
        context=None,
        window_s: float = DEFAULT_WINDOW_S,
        steps: int = DEFAULT_STEPS,
        seed: int = 1528,
        start_worker: bool = True,
    ):
        self._context = context
        self.knob_state = knob_state
        self.state = state
        # The create path resolves the step count (family floor vs an
        # explicit SessionConfig.steps); publish it into the bank so the
        # first produce() reads back what was resolved instead of the
        # spec default silently undoing it. This is exactly how the
        # shared steps_override used to reset every session to ACE's 8.
        knob_state.update({"minimax_steps": int(steps)})
        self.window_s = float(window_s)
        self.window_samples = int(round(self.window_s * DELIVERY_SAMPLE_RATE))
        self._steps = int(steps)
        self._seed = int(seed)

        # Runner slice-width bookkeeping. PipelineRunner reads vae_window
        # for its stall/shortfall math and its windowed-mode switch; the
        # emission itself is frontier-driven and variable-length.
        self.vae_window = 1.0
        self.decode_span_s = 0.0
        self.last_tick_ms = 0.0
        self.last_dec_ms = 0.0

        self.ar = ar_stream
        self.renderer = renderer
        self.latents = MiniMaxLatentStream(renderer, hop_ar_frames=HOP_AR_FRAMES)
        self.decoder = MiniMaxLatentDecoder(codec, guard=DECODE_GUARD_FRAMES)
        self.resampler = _DeliveryResampler(channels=2)

        # ---- frontier state (runner thread only) ----
        self._abs_written = 0
        self._pending: deque = deque()
        self._pending_samples = 0
        self._tail: Optional[np.ndarray] = None
        self._playhead_wrapped_prev = 0.0
        self._playhead_laps = 0
        self._echo: dict = {}

        # ---- shared with the worker ----
        self._lock = threading.Lock()
        self._out: deque = deque()          # delivery-rate blocks [N, 2]
        self._out_samples = 0
        self._render_controls = RenderControls(
            steps=self._steps, shift=DEFAULT_SHIFT,
            guidance=DEFAULT_GUIDANCE, cond_strength=1.0, seed=self._seed,
        )
        self._hop = HOP_AR_FRAMES
        self._lead_target_s = 1.0
        self._playhead_abs_s = 0.0
        self._reprompt_request: Optional[tuple] = None
        self._stop = threading.Event()

        # ---- telemetry (worker writes, runner reads) ----
        self.ar_frames = 0
        self.chunks = 0
        self.ar_ms_per_frame = 0.0
        self.chunk_render_ms = 0.0
        # Cumulative, because the per-sample values above are the LAST
        # frame batch and the LAST chunk. Both are noisy enough that a
        # single sample moves an end-to-end realtime figure by 15%, which
        # is how a throughput claim ends up depending on which tick it
        # was read at.
        self.ar_wall_s = 0.0
        self.render_wall_s = 0.0
        self.ar_finished = False
        self._exhausted_logged = False
        self.last_reprompt_s = 0.0
        # Controls the most recently COMMITTED chunk was rendered under.
        # A knob moved while a render is in flight lands on the chunk
        # after it, so "the frontier advanced" is not the same event as
        # "the frontier advanced under the new setting"; anything
        # measuring renderer knob-to-ear has to read this, not the
        # frontier alone.
        self.last_commit_controls = self._render_controls

        self._worker: Optional[threading.Thread] = None
        if start_worker:
            self._worker = threading.Thread(
                target=self._run, name="minimax-generate", daemon=True,
            )
            self._worker.start()

    # ---- generation worker --------------------------------------------------

    def _snapshot(self):
        with self._lock:
            return self._render_controls, self._hop, self._lead_target_s

    @property
    def mean_ar_ms_per_frame(self) -> float:
        """Session-mean AR cost. The value a throughput claim should
        quote; ar_ms_per_frame is the last batch only."""
        if not self.ar_frames:
            return 0.0
        return self.ar_wall_s * 1000.0 / self.ar_frames

    @property
    def mean_chunk_render_ms(self) -> float:
        """Session-mean chunk render cost."""
        if not self.chunks:
            return 0.0
        return self.render_wall_s * 1000.0 / self.chunks

    def frontier_s(self) -> float:
        """Seconds of audio committed to the delivery frontier so far."""
        return self.resampler.emitted_native / MINIMAX_SAMPLE_RATE

    def _run(self) -> None:
        try:
            self._generate_loop()
        except Exception as exc:  # pragma: no cover - worker guard
            logger.error("minimax_worker_died error={}", exc)
            raise

    def _generate_loop(self) -> None:
        while not self._stop.is_set():
            controls, hop, lead_s = self._snapshot()
            if hop != self.latents.hop_ar_frames:
                # Chunk geometry is fixed; only the commit advance moves,
                # and it may move between chunks but never inside one.
                try:
                    self.latents.hop_ar_frames = self._validated_hop(hop)
                except ValueError as exc:
                    logger.warning("minimax_hop_rejected {}", exc)

            with self._lock:
                pending, self._reprompt_request = self._reprompt_request, None
            if pending is not None:
                self._apply_reprompt(*pending)

            # Credit pacing. Only binds on hardware fast enough to
            # choose; at 0.68x realtime the frontier never gets ahead.
            if self.frontier_s() - self._playhead_abs_s > lead_s:
                self._stop.wait(0.02)
                continue

            if self._advance_ar():
                continue
            if self._render_chunk(controls):
                continue

            # Nothing left to write. The AR stage has finished and the
            # frames it wrote since the last chunk are fewer than a
            # conditioning window, so they can never be rendered: up to
            # chunk-minus-hop frames of composition are dropped at the
            # end of a piece. Said once, loudly, because the alternative
            # is a stream that quietly stops extending.
            if not self._exhausted_logged:
                self._exhausted_logged = True
                unrendered = (
                    self.latents.frames_available
                    - self.latents._next_ar
                )
                logger.info(
                    "minimax_ar_exhausted frames={} chunks={} committed_s={:.1f} "
                    "unrendered_frames={}",
                    self.ar_frames, self.chunks, self.frontier_s(),
                    max(0, unrendered),
                )
            self._stop.wait(0.1 if self.ar_finished else 0.01)

    def _validated_hop(self, hop: int) -> int:
        renderer = self.renderer
        max_commit = renderer.latent_frames - renderer.carry_latent_frames
        hop = max(1, min(int(hop), renderer.chunk_ar_frames))
        if latent_origin(hop) + 1 > max_commit:
            raise ValueError(
                f"hop {hop} commits past the chunk after a "
                f"{renderer.carry_latent_frames}-frame carry"
            )
        return hop

    def _advance_ar(self) -> bool:
        """Emit AR frames if the next chunk's window is short. Returns
        True when work was done."""
        if self.ar.finished:
            self.ar_finished = True
            return False
        need = (
            self.latents._next_ar
            + self.renderer.chunk_ar_frames
            - self.latents.frames_available
        )
        if need <= 0:
            return False

        self.ar.set_controls(self._ar_controls())
        started = time.perf_counter()
        emitted = self.ar.advance(min(need, AR_BATCH_FRAMES))
        if emitted is None:
            self.ar_finished = self.ar.finished
            return False
        self.latents.push_frames(emitted)
        n = int(emitted.shape[1])
        self.ar_frames += n
        spent = time.perf_counter() - started
        self.ar_wall_s += spent
        self.ar_ms_per_frame = spent * 1000.0 / n
        return True

    def _ar_controls(self) -> ARControls:
        values = self.knob_state.get_all_values()
        return ARControls(
            temperature=float(values.get("minimax_temperature", 1.0) or 1.0),
            top_k=int(values.get("minimax_top_k", 50) or 50),
            guidance=float(values.get("minimax_ar_guidance", 1.5) or 1.5),
        )

    def _render_chunk(self, controls: RenderControls) -> bool:
        if not self.latents.can_render():
            return False
        out = self.latents.render_next(controls)
        if out is None:
            return False
        self.chunks += 1
        self.chunk_render_ms = self.renderer.last_render_ms
        self.render_wall_s += self.renderer.last_render_ms / 1000.0
        self.last_commit_controls = controls

        # Decode everything the new commit made exactly decodable, then
        # convert. Both are cheap next to the render, and capping them
        # would only defer the same work to the next iteration while the
        # audio ring waits for it.
        while True:
            decoded = self.decoder.decode_next(self.latents)
            if decoded is None:
                break
            start_native, audio = decoded
            self.resampler.push(start_native, audio.float().cpu().numpy())
            self.last_dec_ms = self.decoder.last_decode_ms

        while True:
            block = self.resampler.pop()
            if block is None:
                break
            with self._lock:
                self._out.append(block)
                self._out_samples += int(block.shape[0])

        # Release latent that no future carry and no future decode guard
        # can reach.
        keep = min(
            latent_origin(self.latents._next_ar),
            self.decoder.decoded_frames,
        ) - DECODE_GUARD_FRAMES
        self.latents.trim_latent(max(0, keep))
        return True

    def _apply_reprompt(self, tags: str, tags_b) -> None:
        try:
            self.last_reprompt_s = self.ar.reprompt(tags)
            logger.info(
                "minimax_prompt_swapped seconds={:.2f} frames={}",
                self.last_reprompt_s, self.ar.frames_emitted,
            )
        except Exception as exc:
            logger.error("minimax_reprompt_failed error={}", exc)

    # ---- session control hooks ----------------------------------------------

    def handle_set_prompt(self, tags: str, *, tags_b=None) -> None:
        """Queue a live prompt change onto the generation worker.

        Not applied inline: the KV cache belongs to the worker thread and
        rebuilding it under the runner would block the tick. The worker
        picks it up at its next loop iteration, at most one AR batch
        (~1 s of wall clock) away, and the change costs a prefill rather
        than a regeneration -- the music already written is kept.
        """
        if tags_b:
            logger.warning(
                "minimax_prompt_b_ignored reason=no_ab_blend_on_ar_prefix",
            )
        with self._lock:
            self._reprompt_request = (tags, tags_b)

    def handle_set_prompt_blend(self, value: float) -> None:
        """Not available on this family, and loudly so.

        The other families blend two conditioning TENSORS. MiniMax's
        conditioning is a KV prefix inside an 8.58B LM: there is no
        second one to interpolate toward without running a second LM,
        and interpolating hidden states across two prefixes is not a
        defined operation on this checkpoint. A silent no-op would read
        as "the blend knob does nothing on this model", which is the
        same symptom as a bug.
        """
        raise UnsupportedOperation(
            "prompt_blend",
            "minimax conditioning is an autoregressive KV prefix, not a "
            "tensor pair; use set_prompt, which re-prefills against the "
            "audio already written",
        )

    def close(self) -> None:
        self._stop.set()
        worker = getattr(self, "_worker", None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=5.0)
        self.ar.close()
        self.latents.close()
        self.resampler.close()
        self._pending.clear()
        self._out.clear()

    # ---- Tier-1 contract: declarations ---------------------------------------

    def capabilities(self) -> Capabilities:
        # Everything defaults False. No refinement (append-only, and the
        # AR stage cannot revise an emitted frame), no positional source
        # so no swap/timbre/structure/stems/write_audio, no LoRA refit
        # story on this checkpoint, no ring so no depth/curves/loop_band.
        return Capabilities()

    def geometry(self) -> AudioGeometry:
        # chunk_rate_hz is "the generation cadence". For this family that
        # is the AR ACOUSTIC frame rate, 25 Hz -- NOT the 86.133 Hz DiT
        # latent rate, and not comparable to another family's frame rate
        # as a throughput figure. The two were conflated once already.
        return AudioGeometry(
            sample_rate=DELIVERY_SAMPLE_RATE,
            channels=2,
            chunk_rate_hz=AR_FRAME_RATE_HZ,
            duration_s=self.window_s,
        )

    def lead_profile(self) -> LeadProfile:
        # No opinion: emission is frontier-driven, so the runner's
        # adaptive lead positions nothing. The lead that matters is the
        # minimax_lead knob, which paces the worker.
        return LeadProfile()

    def knob_specs(self, lora_ids=()) -> list:
        return minimax_knob_specs()

    def lora_available(self) -> bool:
        return False

    def lora_compatible(self, metadata: dict) -> bool:
        return False

    def list_loras(self) -> list:
        return []

    # ---- Tier-1 contract: hot loop -------------------------------------------

    def sync_source(self, ctx: TickContext) -> None:
        # No positional source. Unwrap the playhead here, once per tick
        # and before produce, so the worker's credit pacing sees a
        # monotonic clock across window laps.
        pos = ctx.playhead_s
        if pos < self._playhead_wrapped_prev - self.window_s * 0.5:
            self._playhead_laps += 1
        self._playhead_wrapped_prev = pos
        with self._lock:
            self._playhead_abs_s = self._playhead_laps * self.window_s + pos

    def read_knobs(self) -> dict:
        return self.knob_state.get_all_values()

    def has_pending_refit(self) -> bool:
        return False

    def rebuild_imminent(self, knobs: dict) -> bool:
        # Nothing here blocks on a rebuild: there is no pipeline to
        # reconstruct, and a step-count change is picked up by the next
        # chunk's control snapshot.
        return False

    def has_renderable_state(self) -> bool:
        return self._abs_written > 0 or self._pending_samples > 0

    def playable_duration_s(self) -> Optional[float]:
        return self.window_s

    def produce(self, knobs: dict, ctx: TickContext, mode: ProduceMode) -> bool:
        """Publish controls to the worker and drain what it produced.

        The mode distinction is a no-op here, as it is for every
        append-only family: there is no expensive local generate step to
        skip, and music must keep flowing through DiT-pause idle.
        """
        started = time.perf_counter()

        controls = RenderControls(
            steps=max(
                1, int(knobs.get("minimax_steps", self._steps) or self._steps)
            ),
            shift=float(knobs.get("minimax_shift", DEFAULT_SHIFT) or DEFAULT_SHIFT),
            guidance=float(
                knobs.get("minimax_guidance", DEFAULT_GUIDANCE) or DEFAULT_GUIDANCE
            ),
            cond_strength=float(knobs.get("minimax_cond_strength", 1.0) or 1.0),
            seed=int(knobs.get("seed", self._seed) or self._seed),
        )
        hop = int(knobs.get("minimax_hop", HOP_AR_FRAMES) or HOP_AR_FRAMES)
        lead = float(knobs.get("minimax_lead", 1.0) or 1.0)

        with self._lock:
            self._render_controls = controls
            self._hop = hop
            self._lead_target_s = lead
            blocks = list(self._out)
            self._out.clear()
            self._out_samples = 0

        for block in blocks:
            self._pending.append(block)
            self._pending_samples += int(block.shape[0])

        self._echo = {
            "minimax_shift": controls.shift,
            "minimax_guidance": controls.guidance,
            "minimax_cond_strength": controls.cond_strength,
            "minimax_hop": hop,
            "minimax_lead": lead,
            "minimax_temperature": knobs.get("minimax_temperature"),
            "minimax_top_k": knobs.get("minimax_top_k"),
            "minimax_ar_guidance": knobs.get("minimax_ar_guidance"),
            "seed": controls.seed,
            "minimax_steps": controls.steps,
        }
        self.last_tick_ms = (time.perf_counter() - started) * 1000.0

        if self._pending_samples == 0:
            # The ACE backend's GPU step paces the runner loop; here the
            # generation worker does, so nap instead of spinning the
            # runner at CPU speed.
            time.sleep(0.01)
            return False
        return True

    def render_window(self, t_start_s: float) -> Optional[AudioChunk]:
        """Emit the next frontier chunk. The position hint is ignored:
        append-only means there is exactly one place new audio can go.

        Returns None when nothing new is pending, which is correct rather
        than a stall -- committed audio is already in the ring and never
        changes.
        """
        if self._pending_samples == 0:
            return None

        wrapped_start = self._abs_written % self.window_samples
        room = self.window_samples - wrapped_start
        budget = min(int(MAX_EMIT_S * DELIVERY_SAMPLE_RATE), room)

        parts = []
        taken = 0
        while self._pending and taken < budget:
            arr = self._pending.popleft()
            if taken + arr.shape[0] > budget:
                cut = budget - taken
                parts.append(arr[:cut])
                self._pending.appendleft(arr[cut:])
                taken = budget
            else:
                parts.append(arr)
                taken += int(arr.shape[0])
        self._pending_samples -= taken
        new_pcm = parts[0] if len(parts) == 1 else np.concatenate(parts)

        # Overlap head: re-emit the tail of the previous emission so the
        # runner's leading-edge crossfade blends identical samples.
        # Skipped across the wrap seam (once per lap) and on the first
        # chunk.
        head = None
        if self._tail is not None and 0 < XFADE <= wrapped_start:
            head = self._tail[-XFADE:]

        if head is not None:
            pcm = np.concatenate([head, new_pcm])
            start_sample = wrapped_start - head.shape[0]
        else:
            pcm = np.array(new_pcm, copy=True)  # runner mutates in place
            start_sample = wrapped_start

        # Update the tail from pristine data BEFORE handing the chunk out
        # (the runner crossfades the array it is given, in place).
        if self._tail is None:
            self._tail = np.array(new_pcm[-XFADE:], copy=True)
        else:
            self._tail = np.concatenate([self._tail, new_pcm])[-XFADE:].copy()

        self._abs_written += taken
        return AudioChunk(pcm=pcm, start_sample=int(start_sample))

    def render_full(self) -> Optional[AudioChunk]:
        # Legacy full-buffer mode (vae_window <= 0) never applies: this
        # backend always declares a positive window.
        return None

    # ---- bookkeeping ---------------------------------------------------------

    def on_fresh_generation(self, knobs: dict) -> None:
        """Mirror per-generation telemetry into session params.

        The two MiniMax-specific numbers here are the ones a listener can
        actually hear: ``frontier_lead_s`` is how far generation is ahead
        of the playhead (negative means the playhead has lapped the
        frontier and the rolling window is repeating), and
        ``ar_realtime`` is the AR stage's rate against the 40 ms of audio
        each frame represents.
        """
        if self.state is None:
            return
        p = self.state.params
        p["num_gens"] = p.get("num_gens", 0) + 1
        p["tick_ms"] = self.last_tick_ms
        p["dec_ms"] = self.last_dec_ms
        for name, val in self._echo.items():
            if val is None:
                continue
            p[name] = round(float(val), 3)
        p["_prompt"] = getattr(self.state, "prompt_text", "")
        p["ar_frames"] = self.ar_frames
        p["ar_ms_per_frame"] = round(self.mean_ar_ms_per_frame, 1)
        p["ar_realtime"] = round(
            (1000.0 / AR_FRAME_RATE_HZ) / max(self.mean_ar_ms_per_frame, 1e-6), 3,
        )
        p["chunks"] = self.chunks
        p["chunk_render_ms"] = round(self.mean_chunk_render_ms, 1)
        p["frontier_lead_s"] = round(self.frontier_s() - self._playhead_abs_s, 2)
        p["ar_finished"] = self.ar_finished

    # ---- construction --------------------------------------------------------

    @classmethod
    def from_context(
        cls,
        context,
        *,
        prompt: str,
        lyrics: str = "",
        knob_state,
        state=None,
        window_s: float = DEFAULT_WINDOW_S,
        steps: int = DEFAULT_STEPS,
        seed: int = 1528,
        max_ar_frames: Optional[int] = None,
        dit_backend: str = "eager",
        codec_backend: str = "eager",
    ) -> "MiniMaxBackend":
        """Open the AR session and the renderer against a loaded stack.

        The AR stage must be RESIDENT for this backend, not paged: it
        runs continuously rather than once per composition, and moving
        18 GB across PCIe between chunks would cost more than the chunks
        do. That is the real deployment constraint this architecture
        adds -- the renderer alone fits comfortably on a 24 GB card and
        the pair does not.
        """
        renderer = MiniMaxChunkRenderer(
            context.make_dit(
                latent_frames=context.chunk_latent_frames, backend=dit_backend,
            ),
            context.condition_encoder,
            device=context.device,
            dtype=context.dtype,
            chunk_ar_frames=CHUNK_AR_FRAMES,
            carry_latent_frames=CARRY_LATENT_FRAMES,
            latent_channels=context.latent_channels,
        )
        ar_stream = context.open_ar_stream(
            prompt=prompt, lyrics=lyrics, seed=seed, max_frames=max_ar_frames,
        )
        return cls(
            ar_stream=ar_stream,
            renderer=renderer,
            codec=context.make_codec(backend=codec_backend),
            knob_state=knob_state,
            state=state,
            context=context,
            window_s=window_s,
            steps=steps,
            seed=seed,
        )
