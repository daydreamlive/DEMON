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

import os
from typing import Callable

import torch

from acestep.engine.obs import logger
from acestep.engine.sa3_helpers import (
    import_loader_helpers,
    import_stream_helpers,
    require_sa3_vendor,
)


def resolve_sa3_device() -> str:
    """Best available torch device for the SA3 eager path: cuda > mps >
    cpu, the same order upstream ``StableAudioModel.from_pretrained``
    uses. ``DEMON_SA3_DEVICE`` overrides for debugging (e.g. forcing
    ``cpu`` on an Apple Silicon machine to bisect an MPS kernel issue)."""
    override = os.environ.get("DEMON_SA3_DEVICE", "").strip()
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_model_half(device: str) -> bool:
    """fp16 on CUDA, fp32 elsewhere — upstream forces fp32 off-CUDA
    (``model.py`` sets ``model_half = False`` without CUDA), and MPS
    fp16 is unproven for this stack. ``DEMON_SA3_HALF=1|0`` overrides
    to experiment (e.g. fp16 on MPS for the ~2x memory/throughput win)."""
    override = os.environ.get("DEMON_SA3_HALF", "").strip()
    if override:
        return override not in ("0", "false", "no")
    return device == "cuda"

# Models whose SAME decoder is too slow to full-decode per render tick
# (SAME-L: ~80 ms eager full at 60 s) and therefore use the windowed
# codec. SAME-S (small-music) full-decodes in ~11 ms flat — windowing
# would only add seam surface there.
WINDOWED_DECODE_MODELS = {"medium"}

# Session-geometry ceilings applied on non-CUDA devices (MPS / CPU),
# where the eager path runs ~20-50x slower than the fleet GPUs. See
# clamp_duration_for_device / max_depth_for_device for the measured
# numbers behind these values; both have env overrides.
NONCUDA_MAX_DURATION_S = 24.0
NONCUDA_MAX_DEPTH = 1


class SA3Context:
    """Loaded SA3 model + family-private operations. See module docstring."""

    def __init__(
        self,
        model_id: str = "small-music",
        *,
        device: str | None = None,
        model_half: bool | None = None,
    ):
        require_sa3_vendor()
        loader = import_loader_helpers()
        self._helpers = import_stream_helpers()

        if device is None:
            device = resolve_sa3_device()
        if model_half is None:
            model_half = _resolve_model_half(device)

        self.model_id = model_id
        ckpt = loader.checkpoint_dir(model_id)
        logger.info(
            "sa3_model_load_start model_id={} dir={} device={} model_half={}",
            model_id, ckpt, device, model_half,
        )
        self.sam = loader.load_local_model(ckpt, device=device, model_half=model_half)
        self.device = torch.device(self.sam.device)
        self.dtype = next(self.sam.model.model.parameters()).dtype
        self.sample_rate = int(self.sam.model.sample_rate)          # 44100
        self.downsampling_ratio = int(
            self.sam.model.pretransform.downsampling_ratio          # 4096
        )
        self.latent_channels = int(self.sam.model.io_channels)      # 256
        logger.info(
            "sa3_model_loaded model_id={} latent_rate_hz={:.4f} dtype={}",
            model_id, self.sample_rate / self.downsampling_ratio, self.dtype,
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

    def clamp_duration_for_trt(
        self, duration_s: float, *, padding_s: float = 6.0, backend: str = "eager",
    ) -> float:
        """Clamp a requested duration so its padded latent window fits a
        built TRT DiT engine — landing on the fast path instead of
        silently falling back to the ~5x-slower eager DiT. No-op for
        models without engines (small) or durations already inside.
        No-op unless ``backend="tensorrt"`` (see :meth:`make_dit`) —
        the eager DiT has no length cap worth truncating the source
        for."""
        from acestep.engine.sa3_trt import trt_duration_cap_s

        if backend != "tensorrt":
            return duration_s
        cap = trt_duration_cap_s(self.model_id, padding_s=padding_s)
        if cap is None or duration_s <= cap:
            return duration_s
        logger.warning(
            "sa3_duration_clamped_for_trt model_id={} requested_s={:.1f} cap_s={:.1f}",
            self.model_id, duration_s, cap,
        )
        return cap

    def clamp_duration_for_device(self, duration_s: float) -> float:
        """Cap the diffusion window on non-CUDA devices so ticks stay
        interactive. The eager DiT tick scales linearly with the latent
        window; measured on an M1 Pro (fp32/MPS, depth 1): 57.6 s window
        = 416 ms/tick + 1.05 s per full decode (a fresh generation lands
        every ~14 s — the playhead outruns it and the client keeps
        playing the raw source), 24 s = 227 ms/tick + 0.6 s decode
        (~2.5 s knob-to-new-audio at 8 steps). No-op on CUDA.
        ``DEMON_SA3_MAX_DURATION_S`` overrides; 0 disables the cap."""
        if self.device.type == "cuda":
            return duration_s
        cap = float(
            os.environ.get("DEMON_SA3_MAX_DURATION_S", "")
            or NONCUDA_MAX_DURATION_S
        )
        if cap <= 0 or duration_s <= cap:
            return duration_s
        logger.warning(
            "sa3_duration_clamped_for_device device={} requested_s={:.1f} "
            "cap_s={:.1f} (override: DEMON_SA3_MAX_DURATION_S)",
            self.device.type, duration_s, cap,
        )
        return cap

    def max_depth_for_device(self) -> int | None:
        """Pipeline-depth ceiling for this device, or None for no extra
        cap (CUDA). On a saturated MPS/CPU device the batched tick cost
        is LINEAR in depth (measured M1 Pro: depth 4 = 4.2x depth 1 at
        57.6 s), so extra slots buy no throughput and multiply the
        knob-to-audio latency — depth 1 is strictly better there.
        ``DEMON_SA3_MAX_DEPTH`` overrides."""
        if self.device.type == "cuda":
            return None
        return max(1, int(
            os.environ.get("DEMON_SA3_MAX_DEPTH", "")
            or NONCUDA_MAX_DEPTH
        ))

    # ---- per-prompt (outside the hot loop) ---------------------------------

    def prepare_cond(self, *, prompt: str, duration: float, steps: int):
        """Capture the DiT kwargs + schedule inputs for one prompt and
        fixed duration (the spike's ``SA3Conditioning``)."""
        return self._helpers.prepare_sa3_conditioning(
            self.sam, prompt=prompt, duration=duration, steps=steps,
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
        Returns the native ``[1, 256, T]`` latent."""
        return self._helpers.encode_sa3_source(
            self.sam, audio_input, audio_sample_size,
        )


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
