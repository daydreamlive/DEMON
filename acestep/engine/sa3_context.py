"""SA3Context: Stable Audio 3 model state for one process.

Peer of :class:`~acestep.engine.model_context.ModelContext` in role,
not in surface: it owns the loaded SA3 model (DiT + SAME pretransform +
conditioner), per-prompt conditioning capture, source encoding, and
decode — everything family-private. All of it delegates to the
validated spike-branch helpers ported into ``scripts/sa3/``
(``load_local_model`` with the bundled-t5gemma patch,
``prepare_sa3_conditioning``, ``encode_sa3_source``,
``decode_sa3_latent``), reached through
:mod:`acestep.engine.sa3_helpers`.

Conditioning is per-prompt and OUTSIDE the hot loop (one
``prepare_cond`` per prompt/duration change, exactly like ACE's
one-time ``encode_cond_pair``); the hot loop only ever sees the
captured ``cond_bundle`` dict riding ``SlotRequest.aux_cond``. The
T5Gemma conditioner stays private to this context — nothing above the
backend seam touches it.

Requires the managed ``stable_audio_3`` source. ``demon-setup`` fetches
the pinned checkout, and construction retries that same vendoring path
when it is absent.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Mapping

import numpy as np
import torch

from acestep.engine.obs import logger
from acestep.engine.sa3_helpers import (
    import_loader_helpers,
    import_stream_helpers,
    require_sa3_vendor,
)

# Models whose SAME decoder is too slow to full-decode per render tick
# (SAME-L: ~80 ms eager full at 60 s) and therefore use the windowed
# codec. SAME-S (small-music) full-decodes in ~11 ms flat — windowing
# would only add seam surface there.
WINDOWED_DECODE_MODELS = {"medium"}

# ---- Song-length conditioning -------------------------------------------
# What ``seconds_total`` tells the SA3 DiT (see
# ``prepare_sa3_conditioning``): upstream training labels every example
# with the FULL file length, so a label equal to the render window means
# "the whole song, ending here" and the model composes an outro into the
# last seconds of every loop (the DreamSampler "song ending" report).
# A label longer than the window is the training regime for "a slice of
# a longer track" — no ending at the loop boundary. Measured with
# ``scripts/sa3/tail_probe.py`` on sa3-medium (5090): under the legacy
# label every probed loop ended in a 17-24 dB fade over its last 2 s
# (30 s cuts and the full 57.6 s default track alike, sa3_denoise 0.9
# and 1.0); under a 180 s label the same loops stay flat. 120 s is NOT
# enough — it equals the model's training window, so it still reads as
# "the whole file", and the fade came back in every run.
SONG_SECONDS_ENV = "DEMON_SA3_SONG_SECONDS"
SONG_SCHEDULE_ENV = "DEMON_SA3_SONG_SCHEDULE"
OUTRO_PAD_ENV = "DEMON_SA3_OUTRO_PAD_S"
# Default song-length label, seconds. ``0`` in the env var disables the
# label (legacy: label = render duration + 6 s outro pad).
DEFAULT_SONG_SECONDS: float = 180.0
# The conditioner's ``max_val`` (model_config ``seconds_total`` number
# conditioner); labels above it saturate the Fourier features.
SONG_SECONDS_MAX = 384.0
# Upstream ``generate()`` pads the render window by 6 s of outro headroom
# because the model fades past the label; the pad is silence. Under the
# song-length label there is no outro to make room for, but the render
# still gets a few seconds past the loop: the source anchor is TILED
# (the loop wrapped around onto its own start) into that extra window
# instead of zero-padded, so the anchor never hard-stops inside the
# render. Measured (tail_probe, sa3_denoise 0.9): with the anchor ending
# at the window's last frame the model still resolved some full-length
# loops into a fade, label or not; wrapped, the tail stays flat. Only
# the loop itself is playable (``playable_duration_s``). The wrap is
# also what the TRT clamp now costs a 60 s loop on the 646-latent
# engine (~57 s, was ~54 s under the 6 s pad); ``DEMON_SA3_OUTRO_PAD_S=0``
# trades the wrap for the full 60 s.
LEGACY_OUTRO_PAD_S = 6.0
DEFAULT_LOOP_WRAP_S = 3.0


def song_seconds_setting(env: Mapping[str, str] | None = None) -> float | None:
    """The configured song-length label, or None when disabled."""
    env = os.environ if env is None else env
    raw = env.get(SONG_SECONDS_ENV)
    if raw is None or raw.strip() == "":
        value = DEFAULT_SONG_SECONDS
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{SONG_SECONDS_ENV} must be a number of seconds (0 disables), "
                f"got {raw!r}"
            ) from exc
    if value <= 0:
        return None
    return min(value, SONG_SECONDS_MAX)


def song_schedule_from_window(env: Mapping[str, str] | None = None) -> bool:
    """``DEMON_SA3_SONG_SCHEDULE``: ``song`` (default, training-consistent:
    the dist-shift schedule follows the label) or ``window`` (the
    schedule follows the render length)."""
    env = os.environ if env is None else env
    mode = env.get(SONG_SCHEDULE_ENV, "song").strip().lower()
    if mode not in ("song", "window"):
        raise ValueError(
            f"{SONG_SCHEDULE_ENV} must be song|window, got {mode!r}"
        )
    return mode == "window"


def outro_pad_setting(
    song_seconds: float | None, env: Mapping[str, str] | None = None,
) -> float:
    """Extra render window past the loop, seconds: the wrapped-loop
    headroom (:data:`DEFAULT_LOOP_WRAP_S`) under a song-length label,
    the upstream 6 s silent outro pad otherwise; ``DEMON_SA3_OUTRO_PAD_S``
    overrides either (``0`` = window exactly the loop)."""
    env = os.environ if env is None else env
    raw = env.get(OUTRO_PAD_ENV)
    if raw is not None and raw.strip() != "":
        try:
            pad = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{OUTRO_PAD_ENV} must be a number of seconds, got {raw!r}"
            ) from exc
        return max(0.0, pad)
    return DEFAULT_LOOP_WRAP_S if song_seconds is not None else LEGACY_OUTRO_PAD_S


def label_seconds_for(duration_s: float, song_seconds: float | None) -> float:
    """The ``seconds_total`` label for a render of ``duration_s``: the song
    length when it is longer than the render, else the render itself
    (legacy semantics — a loop longer than the label is still "the whole
    song")."""
    duration_s = float(duration_s)
    if song_seconds is None or float(song_seconds) <= duration_s:
        return duration_s
    return float(song_seconds)


class SA3Context:
    """Loaded SA3 model + family-private operations. See module docstring."""

    def __init__(
        self,
        model_id: str = "small-music",
        *,
        device: str = "cuda",
        model_half: bool = True,
    ):
        require_sa3_vendor()
        loader = import_loader_helpers()
        self._helpers = import_stream_helpers()

        self.model_id = model_id
        ckpt = loader.checkpoint_dir(model_id)
        logger.info("sa3_model_load_start model_id={} dir={}", model_id, ckpt)
        self.sam = loader.load_local_model(ckpt, device=device, model_half=model_half)
        self.device = torch.device(self.sam.device)
        self.dtype = next(self.sam.model.model.parameters()).dtype
        self.sample_rate = int(self.sam.model.sample_rate)          # 44100
        self.downsampling_ratio = int(
            self.sam.model.pretransform.downsampling_ratio          # 4096
        )
        self.latent_channels = int(self.sam.model.io_channels)      # 256
        # Song-length conditioning, resolved once per process (env-driven
        # so a pod can A/B it without a code change).
        self.song_seconds = song_seconds_setting()
        self.schedule_from_window = song_schedule_from_window()
        self.outro_pad_s = outro_pad_setting(self.song_seconds)
        logger.info(
            "sa3_model_loaded model_id={} latent_rate_hz={:.4f} dtype={} "
            "song_seconds={} schedule={} outro_pad_s={:.1f}",
            model_id, self.sample_rate / self.downsampling_ratio, self.dtype,
            self.song_seconds, "window" if self.schedule_from_window else "song",
            self.outro_pad_s,
        )

    @property
    def dit(self):
        """The per-step callable (``DiTWrapper``): ``dit(x_bct, t, **cond)``."""
        return self.sam.model.model

    @property
    def latent_rate_hz(self) -> float:
        return self.sample_rate / self.downsampling_ratio

    # ---- TRT-or-eager component selection ----------------------------------

    def make_dit(
        self, *, latent_frames: int, seconds_total: float, backend: str = "eager",
        prefer_refittable: bool = False,
    ):
        """The per-step velocity callable for one session: with
        ``backend="tensorrt"``, the built TRT engine when one covers
        this session's latent window (medium: ~11-17 ms/step vs ~54 ms
        eager, per-step cos >= 0.9998 vs eager on real conditioning,
        ``scripts/sa3/sa3_trt_dit_cond_parity.py``); the torch
        ``DiTWrapper`` otherwise. The TRT wrapper is per-session (fixed
        L + duration, own execution context); the deserialized engine
        is shared.

        ``backend`` is the resolved acceleration value the session
        creator threads through from the serving layer's accel param
        (compile is already normalized to eager there: SA3 has no
        torch.compile path). ``prefer_refittable`` is the LoRA-session
        preference (notes/SA3_LORA_PLAN.md D6b): pick a refit-built
        engine when one covers the window, and avoid fp8 (whose refit
        story is unproven) otherwise."""
        from acestep.engine.sa3_trt import SA3TRTDit, find_dit_engine

        if backend != "tensorrt":
            logger.info(
                "sa3_dit_eager model_id={} latent_frames={} reason=backend_{}",
                self.model_id, latent_frames, backend,
            )
            return self.dit

        engine_path = find_dit_engine(
            self.model_id, int(latent_frames),
            want_refittable=bool(prefer_refittable),
        )
        if engine_path is None:
            logger.info(
                "sa3_dit_eager model_id={} latent_frames={} reason=no_trt_engine",
                self.model_id, latent_frames,
            )
            return self.dit
        return SA3TRTDit(
            engine_path,
            latent_frames=int(latent_frames),
            seconds_total=float(seconds_total),
        )

    def make_codec(self, *, backend: str = "eager"):
        """The family codec for one session: SAME-S full decode for
        small (measured ~11 ms flat, so windowing buys nothing; eager
        only, ``backend`` has no TRT flavor to select), the SAME-L
        windowed codec for medium (full decode is ~80 ms per call — too
        slow to run per render tick), whose per-window decode runs the
        built TRT engine when ``backend="tensorrt"`` and eager
        otherwise."""
        if self.model_id in WINDOWED_DECODE_MODELS:
            return SA3SAMEWindowCodec(self, use_trt=(backend == "tensorrt"))
        return SA3SAMECodec(self)

    def cond_seconds_total(self, duration_s: float) -> float:
        """The ``seconds_total`` label a render of ``duration_s`` is
        conditioned with (see :func:`label_seconds_for`). Every consumer
        of the label — the eager cond capture and the TRT DiT's seconds
        scalar — must go through here so they can't disagree."""
        return label_seconds_for(duration_s, self.song_seconds)

    def window_latent_frames(self, duration_s: float) -> int:
        """Latent frames of the render window for ``duration_s`` (the
        requested length + :attr:`outro_pad_s`, aligned the way
        ``prepare_sa3_conditioning`` sizes it). Pure arithmetic on the
        model config — no model call."""
        audio_samples = self.sam._adapt_sample_size(
            [{"seconds_total": float(duration_s)}],
            self._helpers.SA3_DEFAULT_SAMPLE_SIZE,
            self.outro_pad_s,
        )
        return int(audio_samples) // self.downsampling_ratio

    def clamp_duration_for_trt(
        self, duration_s: float, *, backend: str = "eager",
    ) -> float:
        """Clamp a requested duration so its (padded, aligned) latent
        window fits a built TRT DiT engine — landing on the fast path
        instead of silently falling back to the ~5x-slower eager DiT.
        No-op for models without engines (small) or durations already
        inside. No-op unless ``backend="tensorrt"`` (see
        :meth:`make_dit`) — the eager DiT has no length cap worth
        truncating the source for."""
        from acestep.engine.sa3_trt import max_dit_engine_latents

        if backend != "tensorrt":
            return duration_s
        max_l = max_dit_engine_latents(self.model_id)
        if max_l is None or self.window_latent_frames(duration_s) <= max_l:
            return duration_s
        # Walk down in 0.1 s steps against the SAME window arithmetic the
        # capture uses, so alignment rounding can't push the clamped
        # window one frame past the engine.
        cap = math.floor(duration_s * 10.0) / 10.0
        while cap > 0 and self.window_latent_frames(cap) > max_l:
            cap = round(cap - 0.1, 1)
        logger.warning(
            "sa3_duration_clamped_for_trt model_id={} requested_s={:.1f} cap_s={:.1f}",
            self.model_id, duration_s, cap,
        )
        return cap

    # ---- per-prompt (outside the hot loop) ---------------------------------

    def prepare_cond(self, *, prompt: str, duration: float, steps: int):
        """Capture the DiT kwargs + schedule inputs for one prompt and
        fixed render duration (the spike's ``SA3Conditioning``), labelled
        with :meth:`cond_seconds_total` and padded by :attr:`outro_pad_s`."""
        return self._helpers.prepare_sa3_conditioning(
            self.sam, prompt=prompt, duration=duration, steps=steps,
            duration_padding_sec=self.outro_pad_s,
            song_seconds_total=self.song_seconds,
            schedule_from_window=self.schedule_from_window,
        )

    def make_schedule_builder(
        self, cond, steps: int,
    ) -> Callable[[float], torch.Tensor]:
        """``denoise -> (steps+1,)`` schedule closure over the captured
        ``sched_args``, parameterized by step count so a
        ``steps_override`` pipeline rebuild gets a matching builder.
        Mirrors the spike's ``SA3StreamPipeline.from_sched_args``."""
        prepared = dict(cond.sched_args)
        esl = prepared.get("effective_seq_len")
        if torch.is_tensor(esl):
            prepared["effective_seq_len"] = esl.detach().cpu()

        def _builder(denoise: float) -> torch.Tensor:
            import stable_audio_3.inference.sampling as sampling

            from .sa3_denoise_mapping import map_denoise_to_entry_sigma

            schedule = sampling.build_schedule(
                steps=int(steps),
                sigma_max=map_denoise_to_entry_sigma(float(denoise)),
                dist_shift=prepared["dist_shift"],
                effective_seq_len=prepared["effective_seq_len"],
                fallback_seq_len=prepared["fallback_seq_len"],
                include_endpoint=True,
                device="cpu",
            )
            if schedule.dim() == 2:
                schedule = schedule[0]
            return schedule.detach().float().cpu()

        return _builder

    def encode_source(self, audio_input, audio_sample_size: int) -> torch.Tensor:
        """SAME-encode an audio source for audio-to-audio streaming.
        Returns the native ``[1, 256, T]`` latent. Under the song-length
        label the source is tiled to fill the render window (see
        :meth:`tile_loop`); otherwise upstream's ``prepare_audio``
        zero-pads it."""
        if self.song_seconds is not None:
            audio_input = self.tile_loop(audio_input, audio_sample_size)
        return self._helpers.encode_sa3_source(
            self.sam, audio_input, audio_sample_size,
        )

    def tile_loop(self, audio_input, audio_sample_size: int):
        """Wrap a ``(sample_rate, waveform[C, N])`` loop around onto its
        own start until it covers ``audio_sample_size`` model-rate
        samples (plus one, so the resample can't leave a zero frame).
        The anchor then continues past the playable loop end the way
        the loop itself does when it cycles — no hard stop for the
        model to read as a song ending. A source already covering the
        window is returned untouched (``prepare_audio`` crops it)."""
        sr, wav = audio_input
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        n = int(wav.shape[-1])
        target = math.ceil(int(audio_sample_size) * float(sr) / self.sample_rate) + 1
        if n <= 0 or n >= target:
            return audio_input
        reps = math.ceil(target / n)
        tiled = wav.repeat(*([1] * (wav.dim() - 1)), reps)[..., :target]
        return sr, tiled


class SA3SAMECodec:
    """The SA3 family codec: SAME latent -> 44.1 kHz stereo audio.

    v1 decodes the FULL latent per fresh generation rather than a
    window: SAME-S decode is ~11 ms flat out to 60 s (measured), so
    windowed decode buys nothing for small-music and full decode
    sidesteps the chunk-phase (``chunk_midpoint_shift``) window
    artifacts entirely. The windowed path
    (``decode_sa3_latent_window`` + slice alignment) stays available in
    the spike helpers for larger models. The 44.1→48 kHz delivery
    resample is NOT here — it sits in the backend at the decode
    boundary (round_3 decision 2).
    """

    def __init__(self, context: SA3Context):
        self._context = context
        self._helpers = context._helpers

    def decode_full(
        self, latent_bct: torch.Tensor, *, decode_seed: int | None = None,
    ) -> torch.Tensor:
        """Native ``[1, 256, T]`` latent -> ``[C, N]`` float audio at
        44.1 kHz, clamped to [-1, 1].

        ``decode_seed`` pins the decode RNG (``sa3_decode_rng``) so the
        SAME decoder's inference-time noise (bottleneck renoise +
        decoder mask_noise) is reproducible: same latent + same seed →
        bit-identical audio. ``None`` keeps the legacy unseeded draw.
        """
        with self._helpers.sa3_decode_rng(decode_seed, device=latent_bct.device):
            audio = self._helpers.decode_sa3_latent(self._context.sam, latent_bct)
        return audio[0]


class SA3SAMEWindowCodec:
    """The SAME-L (medium) family codec: windowed latent decode.

    Per render-tick decode of a small latent window around the playhead
    (the same geometry the spike validated:
    ``resolve_sa3_decode_window`` with 2 s context, ``slice_align=1`` —
    SAME-L's sliding-window attention needs no chunk-phase snapping).
    Two execution paths, identical interface:

    * **TRT** when ``use_trt`` and the built window engine exists
      (``same_l_decode_window_<plugin_tag>_t*``): ~9-10 ms per ~1 s
      window, latent
      scaled by ``pretransform.scale`` before the call (spike
      ``scale_mode="pretransform"``, rel_rms ~8e-3 vs eager full).
    * **Eager** fallback: the spike's ``decode_sa3_latent_window``
      (deterministic), ~16 ms per window — still per-tick viable.

    ``decode_full`` (legacy full-buffer path only) stays eager.
    """

    #: Decode context on each side of the kept region. The spike swept
    #: ctx1/ctx2/ctx4; 2 s is the validated default (window T ~50 ≤ the
    #: engine's 96-frame max, parity rel_rms ~8e-3).
    context_sec = 2.0

    def __init__(self, context: SA3Context, *, use_trt: bool = True):
        from acestep.engine.sa3_trt import (
            SameLWindowTRTDecoder,
            find_same_l_window_engine,
        )

        self._context = context
        self._helpers = context._helpers
        self._scale = float(getattr(context.sam.model.pretransform, "scale", 1.0))
        found = find_same_l_window_engine() if use_trt else None
        if found is None:
            logger.info(
                "sa3_same_l_window_decode mode=eager reason={}",
                "no_trt_engine" if use_trt else "backend",
            )
            self._trt = None
            self._min_t = self._max_t = 0
        else:
            path, self._min_t, self._max_t = found
            self._trt = SameLWindowTRTDecoder(path)

    def _decode_window_eager(self, latent_bct, start: int, num: int) -> torch.Tensor:
        result = self._helpers.decode_sa3_latent_window(
            self._context.sam, latent_bct,
            target_start_sample=int(start),
            target_num_samples=int(num),
            context_sec=self.context_sec,
            chunked=False,
            deterministic=True,
        )
        return result.audio_ct

    def decode_window(self, latent_bct: torch.Tensor, start: int, num: int) -> torch.Tensor:
        """``[1, 256, T]`` latent -> ``[C, num]`` float 44.1 kHz audio
        covering samples ``[start, start+num)`` of the full decode."""
        if self._trt is None:
            return self._decode_window_eager(latent_bct, start, num)
        ds = self._context.downsampling_ratio
        window = self._helpers.resolve_sa3_decode_window(
            latent_bct,
            target_start_sample=int(start),
            target_num_samples=int(num),
            context_sec=self.context_sec,
            sample_rate=self._context.sample_rate,
            downsampling_ratio=ds,
            slice_align_latents=1,
        )
        slice_start, slice_end = window.slice_start, window.slice_end
        crop_start = window.crop_start
        total_t = latent_bct.shape[-1]
        # Engine floor: near the song edges the resolved slice can drop
        # under the profile minimum — grow it (right first, then left,
        # shifting the crop offset) rather than fall off the TRT path.
        need = self._min_t - (slice_end - slice_start)
        if need > 0:
            grow_r = min(need, total_t - slice_end)
            slice_end += grow_r
            need -= grow_r
        if need > 0:
            grow_l = min(need, slice_start)
            slice_start -= grow_l
            crop_start += grow_l * ds
            need -= grow_l
        if need > 0 or (slice_end - slice_start) > self._max_t:
            # Latent shorter than the engine minimum, or a window the
            # profile can't serve: decode eagerly rather than guess.
            return self._decode_window_eager(latent_bct, start, num)
        latent_window = latent_bct[..., slice_start:slice_end].contiguous()
        audio_ct = self._trt.decode(latent_window * self._scale)
        out = audio_ct[:, crop_start:crop_start + int(num)]
        if out.shape[-1] < int(num):
            out = torch.nn.functional.pad(out, (0, int(num) - out.shape[-1]))
        return out

    def decode_full(
        self, latent_bct: torch.Tensor, *, decode_seed: int | None = None,
    ) -> torch.Tensor:
        """Eager full decode (legacy full-buffer mode only; the hot path
        never calls this for windowed-codec families). ``decode_seed``
        pins the decode RNG exactly as on :class:`SA3SAMECodec`."""
        with self._helpers.sa3_decode_rng(decode_seed, device=latent_bct.device):
            audio = self._helpers.decode_sa3_latent(self._context.sam, latent_bct)
        return audio[0]
