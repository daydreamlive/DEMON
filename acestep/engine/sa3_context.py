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

Requires the vendored ``stable_audio_3`` source (see
``sa3_helpers.sa3_vendor_dir``); construction fails with a clear
ImportError when it's absent.
"""

from __future__ import annotations

from typing import Callable

import torch

from acestep.engine.obs import logger
from acestep.engine.sa3_helpers import import_loader_helpers, import_stream_helpers


class SA3Context:
    """Loaded SA3 model + family-private operations. See module docstring."""

    def __init__(
        self,
        model_id: str = "small-music",
        *,
        device: str = "cuda",
        model_half: bool = True,
    ):
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

            schedule = sampling.build_schedule(
                steps=int(steps),
                sigma_max=float(denoise),
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

    def decode_full(self, latent_bct: torch.Tensor) -> torch.Tensor:
        """Native ``[1, 256, T]`` latent -> ``[C, N]`` float audio at
        44.1 kHz, clamped to [-1, 1]."""
        audio = self._helpers.decode_sa3_latent(self._context.sam, latent_bct)
        return audio[0]
