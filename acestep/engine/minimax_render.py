"""MiniMax-Music3's own chunked render loop, driven incrementally.

The AR stage writes 25 Hz frames one at a time
(:class:`~acestep.engine.minimax_ar.MiniMaxARStream`); this module turns
that arriving stream into a continuous 86.133 Hz latent, one overlapping
chunk at a time, and then into a continuous 44.1 kHz PCM frontier.

Two rates, and conflating them has already cost this integration one
round of wrong numbers, so state them once:

* **25 Hz** is the AR *acoustic frame* rate. One frame is 40 ms. It is
  what the LM emits and what the ceiling of 9000 frames (= 360 s)
  counts.
* **86.133 Hz** (44100 / 512) is the *DiT latent* rate. One latent frame
  is 512 native samples, 11.6 ms.

They differ by exactly 441/128. A count in one is never a count in the
other, and neither is comparable to another model family's frame rate.

The chunk geometry is upstream's: a 200-AR-frame conditioning window
(689 latent frames), advanced 100 AR frames at a time, carrying 172
latent frames of already-committed latent as left context. Read as a
streaming loop, that arrangement says:

    |<-- carry 172 -->|<---- commit 344 ---->|<-- lookahead 173 -->|
    |<------------------- chunk 689 ------------------------------>|

The carry is locked to committed audio at every sampler step, so the
chunk continues the piece instead of restarting it. The commit region is
the only part kept. The lookahead is rendered and thrown away: it exists
so the committed region is never generated at the edge of the model's
window, where it has no future context to be consistent with.

That discarded lookahead is not waste, it is *latency*, and it is the
dominant term in this family's knob-to-ear time: nothing an operator
does can be heard until the AR stage has written 100 frames past the
point it would affect. ``hop_ar_frames`` is the lever, and it trades
exactly against throughput --- see
``scripts/minimax/minimax_stream_bench.py``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import torch

from acestep.engine.obs import logger

# ---------------------------------------------------------------------------
# Geometry. Two rates; see the module docstring.
# ---------------------------------------------------------------------------

AR_FRAME_RATE_HZ = 25.0
MINIMAX_SAMPLE_RATE = 44100
MINIMAX_UPSAMPLE = 512
LATENT_RATE_HZ = MINIMAX_SAMPLE_RATE / MINIMAX_UPSAMPLE  # 86.1328125

# Latent frames per AR frame, as the exact integer ratio the checkpoint's
# ConditionEncoder uses (output_sampling_rate * input_hop_length over
# input_sampling_rate * output_hop_length = 44100*960 / 24000*512).
LATENT_PER_AR_NUM = 441
LATENT_PER_AR_DEN = 128

# Upstream's inference chunking. These are an inference contract, not a
# trained span: transformer/config.json carries no max_position_embeddings
# and its RoPE is built for whatever length arrives.
CHUNK_AR_FRAMES = 200
HOP_AR_FRAMES = 100
CARRY_LATENT_FRAMES = 172

# Decoder guard. The DAV decoder's one-sided receptive field converges at
# 9 latent frames (measured: scripts/minimax/minimax_decode_profile.py);
# 12 buys margin over both that and the analytic walk of the conv stack.
DECODE_GUARD_FRAMES = 12

# Sampler operating point, measured rather than inherited. The reference
# pipeline runs 30 unwarped steps at guidance 1.7. The grid in
# scripts/minimax/minimax_quality_ablation.py says 16 steps at shift 2.0
# lands at log-mel 0.032 / latent cosine 0.9993 against it, and that
# guidance is not optional -- eight guided steps beat forty unguided.
DEFAULT_STEPS = 16
DEFAULT_SHIFT = 2.0
DEFAULT_GUIDANCE = 1.7


def latent_origin(ar_frame_index: int) -> int:
    """Absolute latent frame at which AR frame ``ar_frame_index`` starts.

    Floor of the exact ratio, which is what keeps a long stream from
    drifting: a hop of 100 AR frames is 344.53 latent frames, so placing
    every chunk at a constant 344 would slip half a latent frame per hop
    -- 0.6 s of conditioning-to-latent skew over a six-minute piece.
    Deriving each origin from its absolute AR index instead makes the
    hop alternate 344/345 and the skew stay bounded by one frame.
    """
    return int(ar_frame_index) * LATENT_PER_AR_NUM // LATENT_PER_AR_DEN


def chunk_latent_frames(ar_frames: int = CHUNK_AR_FRAMES) -> int:
    """Latent length the ConditionEncoder produces for ``ar_frames``."""
    return int(ar_frames) * LATENT_PER_AR_NUM // LATENT_PER_AR_DEN


def build_schedule(steps: int, shift: float) -> torch.Tensor:
    """``(steps + 1,)`` MiniMax-time schedule, 0 (noise) to 1 (data).

    MiniMax's own sampler walks a uniform grid; ``shift`` is the
    Flux/SD3 warp DEMON applies on top, expressed here in the noise
    coordinate ``s = 1 - t`` so that ``shift > 1`` means "spend more
    steps near noise" the same way it does for every other family in
    this tree. Applying it directly to ``t`` would invert that sense,
    which is the sort of sign error that sounds like a taste difference.
    """
    steps = max(1, int(steps))
    alpha = float(shift)
    if alpha <= 0.0:
        raise ValueError(f"shift must be > 0, got {shift}")
    s = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    if abs(alpha - 1.0) > 1e-6:
        s = alpha * s / (1.0 + (alpha - 1.0) * s)
    return (1.0 - s).to(torch.float32)


@dataclass(frozen=True)
class RenderControls:
    """Renderer-side controls, snapshotted once per chunk.

    Per chunk rather than per step because a chunk is the atomic unit of
    committed audio: a change applied mid-sampler would denoise the first
    half of a window under one setting and the second half under another,
    which is not a crossfade, it is an inconsistency inside one window.
    """

    steps: int = DEFAULT_STEPS
    shift: float = DEFAULT_SHIFT
    guidance: float = DEFAULT_GUIDANCE
    cond_strength: float = 1.0
    seed: int = 0


class MiniMaxChunkRenderer:
    """One chunk of flow-matching render, with committed left context.

    Stateless with respect to the stream: the caller owns the committed
    latent and hands in the carry slice. That keeps the sampler testable
    against a stub DiT and keeps the frontier bookkeeping in one place
    (:class:`MiniMaxLatentStream`).
    """

    def __init__(
        self,
        dit,
        cond_encoder,
        *,
        device,
        dtype,
        chunk_ar_frames: int = CHUNK_AR_FRAMES,
        carry_latent_frames: int = CARRY_LATENT_FRAMES,
        latent_channels: int = 128,
    ):
        self.dit = dit
        self.cond_encoder = cond_encoder
        self.device = torch.device(device)
        self.dtype = dtype
        self.latent_channels = int(latent_channels)
        self.chunk_ar_frames = int(chunk_ar_frames)
        self.carry_latent_frames = int(carry_latent_frames)
        self.latent_frames = chunk_latent_frames(self.chunk_ar_frames)
        if self.carry_latent_frames >= self.latent_frames:
            raise ValueError(
                f"carry {self.carry_latent_frames} must be shorter than the "
                f"{self.latent_frames}-frame chunk"
            )
        # A batch-1 TensorRT engine cannot take the conditional and
        # unconditional branches in one forward; eager can, and halving
        # the launch count matters more there than anywhere else.
        self.trt_batch1 = bool(getattr(dit, "trt_batch1", False))
        self.last_forwards = 0
        self.last_render_ms = 0.0

    # ---- conditioning ------------------------------------------------------

    @torch.no_grad()
    def encode_cond(self, frame_hiddens: torch.Tensor) -> torch.Tensor:
        """``[1, ar_frames, 32768]`` -> ``[1, latent_frames, 2048]``."""
        if frame_hiddens.shape[1] != self.chunk_ar_frames:
            raise ValueError(
                f"cond window is {frame_hiddens.shape[1]} AR frames, "
                f"expected {self.chunk_ar_frames}"
            )
        cond = self.cond_encoder(
            frame_hiddens.to(device=self.device, dtype=self.dtype)
        )
        if cond.shape[1] != self.latent_frames:
            raise RuntimeError(
                f"ConditionEncoder returned {cond.shape[1]} latent frames, "
                f"expected {self.latent_frames}"
            )
        return cond

    def _velocity(self, x_1ct, t: float, cond, uncond, guidance: float):
        """One guided velocity evaluation, MiniMax convention throughout
        (``t`` runs 0 noise -> 1 data; ``v`` points from noise to data)."""
        if guidance == 1.0 or uncond is None:
            self.last_forwards += 1
            if self.trt_batch1:
                return self.dit.step_bundle(
                    x_1ct, t, {"encoder_hidden_states": cond},
                ).to(dtype=x_1ct.dtype, copy=True)
            t_b = torch.full((1,), t, device=x_1ct.device, dtype=x_1ct.dtype)
            return self.dit(x_1ct, t_b, cond)

        if self.trt_batch1:
            self.last_forwards += 2
            v_pos = self.dit.step_bundle(
                x_1ct, t, {"encoder_hidden_states": cond},
            ).to(dtype=x_1ct.dtype, copy=True)
            v_neg = self.dit.step_bundle(
                x_1ct, t, {"encoder_hidden_states": uncond},
            ).to(dtype=x_1ct.dtype, copy=True)
        else:
            self.last_forwards += 1  # one batch-2 forward
            x2 = torch.cat((x_1ct, x_1ct), dim=0)
            t_b = torch.full((2,), t, device=x_1ct.device, dtype=x_1ct.dtype)
            both = self.dit(x2, t_b, torch.cat((cond, uncond), dim=0))
            v_pos, v_neg = both[0:1], both[1:2]
        # Textbook CFG, which is what this checkpoint was trained with.
        # DEMON's APG operator measures ~4x worse here: its norm cap is
        # calibrated for ACE's latent scale and throttles a 689-frame
        # guidance delta nearly to nothing.
        return v_neg + (v_pos - v_neg) * guidance

    # ---- render ------------------------------------------------------------

    @torch.no_grad()
    def render(
        self,
        frame_hiddens: torch.Tensor,
        *,
        carry: Optional[torch.Tensor],
        controls: RenderControls,
        chunk_index: int,
    ) -> torch.Tensor:
        """Render one chunk from AR frames. ``[1, 128, L]``.

        ``carry`` is ``[1, 128, carry_latent_frames]`` of already-committed
        latent that this chunk must continue from, or None for the first
        chunk of a stream. The caller slices the commit region out of the
        result; the whole chunk is returned so a caller can inspect the
        lookahead it is discarding.
        """
        return self.render_cond(
            self.encode_cond(frame_hiddens),
            carry=carry, controls=controls, chunk_index=chunk_index,
        )

    @torch.no_grad()
    def render_cond(
        self,
        cond: torch.Tensor,
        *,
        carry: Optional[torch.Tensor],
        controls: RenderControls,
        chunk_index: int,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The sampler itself, over already-encoded conditioning.

        Split out from :meth:`render` so the shipping trajectory can be
        driven from a stored conditioning tensor and compared against a
        hand-written Euler loop on identical noise -- the equivalence
        gate in ``scripts/minimax/minimax_quality_ablation.py``. Without
        that split the only thing measurable is a sampler nobody ships.

        ``noise`` overrides the seeded draw, for exactly that comparison.
        """
        started = time.perf_counter()
        self.last_forwards = 0

        strength = float(controls.cond_strength)
        if abs(strength - 1.0) > 1e-6:
            # Scaling toward zero walks toward the model's own
            # unconditional branch -- the reference pipeline's negative
            # input is literally zeros_like(condition) -- so the whole
            # 0..1 range is a defined operating point rather than an
            # extrapolation.
            cond = cond * strength
        guidance = float(controls.guidance)
        uncond = torch.zeros_like(cond) if guidance != 1.0 else None

        # MiniMax-native layout is [B, C, T] throughout this module; the
        # engine-layout transpose the ring backend needed does not exist
        # here because there is no shared batched pipeline to feed. The
        # state stays fp32 and is cast per forward: sixteen Euler
        # accumulations in bf16 lose more than the cast costs.
        if noise is not None:
            x = noise.to(device=self.device, dtype=torch.float32).clone()
        else:
            generator = torch.Generator(device=self.device).manual_seed(
                (int(controls.seed) * 1_000_003 + int(chunk_index)) % (2**31 - 1)
            )
            x = torch.randn(
                (1, self.latent_channels, cond.shape[1]),
                generator=generator, device=self.device, dtype=torch.float32,
            )

        carry_noise = None
        n_carry = 0
        if carry is not None:
            n_carry = int(carry.shape[-1])
            if n_carry:
                carry = carry.to(device=self.device, dtype=torch.float32)
                # A fixed draw for the whole chunk: the locked region has
                # to sit on ONE forward trajectory, not a fresh one per
                # step, or the model is asked to denoise a moving target.
                carry_noise = x[..., :n_carry].clone()

        schedule = build_schedule(controls.steps, controls.shift)
        for i in range(controls.steps):
            t = float(schedule[i])
            if carry_noise is not None:
                # MiniMax's forward interpolant: x_t = (1-t)*noise + t*data.
                x[..., :n_carry] = (1.0 - t) * carry_noise + t * carry
            v = self._velocity(
                x.to(self.dtype), t, cond, uncond, guidance,
            )
            dt = float(schedule[i + 1]) - t
            x = x + dt * v.float()

        if carry_noise is not None:
            # t = 1 exactly: the locked region IS the committed latent.
            # Restoring it rather than trusting the last Euler step keeps
            # the seam exact instead of one step's error away from it.
            x[..., :n_carry] = carry

        self.last_render_ms = (time.perf_counter() - started) * 1000.0
        return x.to(self.dtype)


class MiniMaxLatentStream:
    """The frontier: AR frames in, committed latent out, one chunk at a time.

    Owns the committed latent buffer and the chunk bookkeeping. It does
    not own the AR stage or the decoder -- the backend drives all three,
    so the pieces stay independently testable and the GPU work stays on
    one thread.
    """

    def __init__(
        self,
        renderer: MiniMaxChunkRenderer,
        *,
        hop_ar_frames: int = HOP_AR_FRAMES,
        latent_channels: int = 128,
    ):
        self.renderer = renderer
        self.hop_ar_frames = max(1, int(hop_ar_frames))
        self.latent_channels = int(latent_channels)

        chunk = renderer.chunk_ar_frames
        carry = renderer.carry_latent_frames
        max_commit = renderer.latent_frames - carry
        # The commit region has to fit after the carry. hop is in AR
        # frames and the commit in latent frames, so convert before
        # comparing -- a hop that looks legal at 25 Hz can overrun at
        # 86 Hz.
        if latent_origin(self.hop_ar_frames) + 1 > max_commit:
            raise ValueError(
                f"hop of {self.hop_ar_frames} AR frames commits up to "
                f"{latent_origin(self.hop_ar_frames) + 1} latent frames, "
                f"which does not fit after a {carry}-frame carry in a "
                f"{renderer.latent_frames}-frame chunk (max hop "
                f"{max_commit * LATENT_PER_AR_DEN // LATENT_PER_AR_NUM})"
            )
        if self.hop_ar_frames > chunk:
            raise ValueError(
                f"hop {self.hop_ar_frames} exceeds the {chunk}-frame chunk"
            )

        # AR frames received but not yet consumed by a chunk, kept as one
        # growing tensor on the render device.
        self._frames: Optional[torch.Tensor] = None
        self._frames_base = 0     # absolute AR index of _frames[:, 0]
        self._chunk_index = 0
        self._next_ar = 0         # absolute AR index of the next chunk's window

        # Committed latent, absolute-indexed from song start.
        self._latent: Optional[torch.Tensor] = None
        self.committed_frames = 0
        self._latent_base = 0     # absolute latent index of _latent[..., 0]

        self.chunks_rendered = 0
        self.last_commit_frames = 0

    # ---- input -------------------------------------------------------------

    def push_frames(self, frame_hiddens: torch.Tensor) -> None:
        """Append AR frames ``[1, k, 32768]`` to the pending window."""
        block = frame_hiddens.to(device=self.renderer.device)
        if self._frames is None:
            self._frames = block
        else:
            self._frames = torch.cat((self._frames, block), dim=1)

    @property
    def frames_available(self) -> int:
        """Absolute AR index one past the last frame received."""
        if self._frames is None:
            return self._frames_base
        return self._frames_base + int(self._frames.shape[1])

    def can_render(self) -> bool:
        """True when a full conditioning window has arrived."""
        return (
            self.frames_available
            >= self._next_ar + self.renderer.chunk_ar_frames
        )

    # ---- output ------------------------------------------------------------

    def _carry_slice(self, origin: int) -> Optional[torch.Tensor]:
        carry = self.renderer.carry_latent_frames
        if self._latent is None or self.committed_frames == 0 or carry == 0:
            return None
        lo = origin - self._latent_base
        if lo < 0 or origin + carry > self.committed_frames:
            # Only reachable if the caller trimmed committed latent that
            # a later chunk still needs. Loud, because the alternative is
            # a silent discontinuity every hop.
            raise RuntimeError(
                f"carry [{origin}, {origin + carry}) is outside the retained "
                f"latent [{self._latent_base}, {self.committed_frames})"
            )
        return self._latent[..., lo:lo + carry]

    def render_next(self, controls: RenderControls) -> Optional[tuple]:
        """Render the next chunk if its window has arrived.

        Returns ``(start_latent_frame, committed_latent)`` for the region
        this chunk newly commits, or None when there is not enough AR
        input yet.
        """
        if not self.can_render():
            return None

        renderer = self.renderer
        origin = latent_origin(self._next_ar)
        window_lo = self._next_ar - self._frames_base
        window = self._frames[
            :, window_lo:window_lo + renderer.chunk_ar_frames
        ].contiguous()

        carry = self._carry_slice(origin) if self.committed_frames else None
        n_carry = 0 if carry is None else int(carry.shape[-1])

        chunk = renderer.render(
            window, carry=carry, controls=controls,
            chunk_index=self._chunk_index,
        )

        # Commit from the end of the carry up to where the NEXT chunk's
        # carry begins, so consecutive commits abut exactly and nothing
        # is ever committed twice.
        next_origin = latent_origin(self._next_ar + self.hop_ar_frames)
        commit_end_abs = next_origin + renderer.carry_latent_frames
        commit_start_abs = origin + n_carry
        lo = n_carry
        hi = commit_end_abs - origin
        if hi > chunk.shape[-1]:
            raise RuntimeError(
                f"commit region ends at chunk offset {hi}, past the "
                f"{chunk.shape[-1]}-frame chunk"
            )
        committed = chunk[..., lo:hi].contiguous()

        if self._latent is None:
            self._latent = committed
            self._latent_base = commit_start_abs
        else:
            if commit_start_abs != self.committed_frames:
                raise RuntimeError(
                    f"commit starts at {commit_start_abs} but the frontier is "
                    f"at {self.committed_frames}"
                )
            self._latent = torch.cat((self._latent, committed), dim=-1)
        self.committed_frames = commit_start_abs + int(committed.shape[-1])
        self.last_commit_frames = int(committed.shape[-1])

        self._chunk_index += 1
        self._next_ar += self.hop_ar_frames
        self.chunks_rendered += 1

        # Release AR frames no future window can reach.
        keep_from = self._next_ar - self._frames_base
        if keep_from > 0:
            self._frames = self._frames[:, keep_from:].contiguous()
            self._frames_base = self._next_ar

        return commit_start_abs, committed

    def latent_slice(self, start: int, end: int) -> torch.Tensor:
        """Committed latent ``[start, end)`` in absolute frames."""
        if start < self._latent_base or end > self.committed_frames:
            raise ValueError(
                f"[{start}, {end}) is outside the retained latent "
                f"[{self._latent_base}, {self.committed_frames})"
            )
        lo = start - self._latent_base
        return self._latent[..., lo:lo + (end - start)]

    def trim_latent(self, keep_from: int) -> None:
        """Drop committed latent before ``keep_from``. The caller is
        responsible for keeping enough for the next carry and the next
        decode guard; :meth:`_carry_slice` raises rather than splicing a
        discontinuity if it does not."""
        if self._latent is None:
            return
        keep_from = max(self._latent_base, min(int(keep_from), self.committed_frames))
        drop = keep_from - self._latent_base
        if drop <= 0:
            return
        self._latent = self._latent[..., drop:].contiguous()
        self._latent_base = keep_from

    def close(self) -> None:
        self._frames = None
        self._latent = None


class MiniMaxLatentDecoder:
    """Committed latent frontier -> 44.1 kHz PCM frontier, guarded.

    Decodes only regions whose guard is committed on BOTH sides, so every
    sample it emits is what a whole-song decode would have produced. That
    costs ``DECODE_GUARD_FRAMES`` of latency (12 frames, 139 ms) and is
    the honest version of a problem the cover backend used to solve by
    decoding the entire song on every generation.
    """

    def __init__(self, codec, *, guard: int = DECODE_GUARD_FRAMES):
        self.codec = codec
        self.guard = int(guard)
        self.decoded_frames = 0   # absolute latent frames turned into PCM
        self.last_decode_ms = 0.0

    def decodable_upto(self, committed_frames: int) -> int:
        """Highest latent frame that can be decoded exactly right now."""
        return max(0, committed_frames - self.guard)

    @torch.no_grad()
    def decode_next(self, stream: MiniMaxLatentStream, *, max_frames: int = 0):
        """Decode the next exact span. Returns ``(start_native, pcm[2, N])``
        or None when nothing new is decodable."""
        end = self.decodable_upto(stream.committed_frames)
        if max_frames > 0:
            end = min(end, self.decoded_frames + int(max_frames))
        if end <= self.decoded_frames:
            return None
        start = self.decoded_frames

        lo = max(0, start - self.guard)
        hi = min(stream.committed_frames, end + self.guard)
        started = time.perf_counter()
        audio = self.codec.decode_full(stream.latent_slice(lo, hi))
        self.last_decode_ms = (time.perf_counter() - started) * 1000.0

        head = (start - lo) * MINIMAX_UPSAMPLE
        length = (end - start) * MINIMAX_UPSAMPLE
        self.decoded_frames = end
        return start * MINIMAX_UPSAMPLE, audio[..., head:head + length]
