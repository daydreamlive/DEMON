"""MiniMaxContext: the loaded MiniMax-Music3 stack, once per process.

The peer of :class:`~acestep.engine.sa3_context.SA3Context`. Owns the
renderer (DiT), the decoder (DAV), the condition encoder, and — only
when it is actually needed — the 8.58B autoregressive stage.

The AR stage is treated differently from every other model in this
repo, and deliberately. It is 17 GB, it is needed only when *capturing*
a composition, and it is never touched inside a tick. Keeping it
resident would cost more VRAM than the renderer it feeds. So the
default policy parks it in host memory and pages it onto the GPU for
the seconds a capture takes. On a 32 GB card that is the difference
between a session that fits and one that does not.

A capture is a first-class artifact here: ``prepare_cond`` will happily
load one off disk instead of computing it. That matters for more than
convenience — the AR stage needs a newer ``transformers`` than the rest
of this repo pins, so being able to stream from a saved capture keeps
the renderer usable on the version DEMON actually ships.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

import torch

from acestep.engine.minimax_helpers import (
    minimax_capture_dir,
    resolve_model_dir,
)
from acestep.engine.obs import logger

# 25 AR frames/s in, 86.133 latent frames/s out.
AR_FRAME_RATE_HZ = 25.0
COND_DIM = 2048


class MiniMaxCodec:
    """DAV decoder behind the codec contract.

    The decoder is deterministic — no sampling, no injected noise
    anywhere in its forward — which spares this family the whole
    decode-reproducibility problem SA3 had to solve with seeded RNG
    forks. Repeated decodes of one latent are bit-identical, so
    overlapping renders simply agree.
    """

    def __init__(self, dav, *, device, dtype):
        self.dav = dav
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def decode_full(self, latent_bct: torch.Tensor) -> torch.Tensor:
        """``[1, 128, T]`` latent -> ``[2, N]`` audio at 44.1 kHz."""
        latent = latent_bct.to(device=self.device, dtype=self.dtype)
        audio = self.dav(latent)
        if audio.ndim == 3:
            audio = audio[0]
        return audio.float()


class MiniMaxContext:
    """Process-cached MiniMax-Music3 stack. See module docstring."""

    sample_rate = 44100
    downsampling_ratio = 512
    latent_channels = 128

    def __init__(
        self,
        model_dir=None,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        ar_policy: str = "offload",
    ):
        if ar_policy not in ("resident", "offload", "absent"):
            raise ValueError(
                f"ar_policy must be resident|offload|absent, got {ar_policy!r}"
            )
        self.root = Path(resolve_model_dir(model_dir))
        self.device = torch.device(device)
        # The acoustic stage supports fp32 and bf16 only; there is no
        # fp16 path in this checkpoint and forcing one produces garbage.
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                f"MiniMax supports float32 or bfloat16, not {dtype}"
            )
        self.dtype = dtype
        self.ar_policy = ar_policy

        from acestep.engine.minimax_dit import (
            MiniMaxConditionEncoder,
            MiniMaxDAV,
            MiniMaxDiT,
        )

        logger.info("minimax_load_start root={} dtype={}", self.root, dtype)
        self._dit = MiniMaxDiT.from_pretrained(
            self.root / "transformer", dtype=dtype, device=self.device,
        )
        self._dav = MiniMaxDAV.from_pretrained(
            self.root / "vocoder", dtype=dtype, device=self.device,
        )
        self._cond_encoder = MiniMaxConditionEncoder.from_pretrained(
            self.root / "condition_encoder", dtype=dtype, device=self.device,
        )
        logger.info("minimax_load_done")

        self._ar = None
        self._tokenizer = None
        self._ar_lock = threading.Lock()

    @property
    def latent_rate_hz(self) -> float:
        return float(self.sample_rate) / float(self.downsampling_ratio)

    # ---- accel seams ---------------------------------------------------------

    def make_dit(self, *, latent_frames: int, backend: str = "eager"):
        """Return the renderer, TRT-accelerated when an engine covers
        this shape. Degrades to eager LOUDLY rather than silently, so a
        five-times-slower session is never a surprise."""
        if backend == "tensorrt":
            try:
                from acestep.engine.minimax_trt import find_dit_engine

                engine = find_dit_engine(latent_frames)
                if engine is not None:
                    logger.info(
                        "minimax_dit_trt engine={} frames={}",
                        engine, latent_frames,
                    )
                    return engine
                logger.warning(
                    "minimax_dit_eager reason=no_engine_for_frames frames={}",
                    latent_frames,
                )
            except ImportError:
                logger.warning("minimax_dit_eager reason=trt_module_absent")
        elif backend == "compile":
            logger.warning("minimax_dit_eager reason=compile_unsupported")
        return self._dit

    def make_codec(self, *, backend: str = "eager") -> MiniMaxCodec:
        # The decoder is ~2% of the render budget on a 5090; there is no
        # headroom worth an engine here yet.
        if backend == "tensorrt":
            logger.info("minimax_codec_eager reason=no_engine_needed")
        return MiniMaxCodec(self._dav, device=self.device, dtype=self.dtype)

    # ---- conditioning --------------------------------------------------------

    def make_schedule_builder(self, cond: dict, steps: int) -> Callable:
        """``denoise -> (steps+1,)`` schedule in DEMON convention.

        MiniMax's own sampler walks a uniform grid, so this is a plain
        descending ramp; the family's warp lives in the adapter's
        ``shift_alpha`` on top of it.
        """
        def _build(denoise: float) -> torch.Tensor:
            return torch.linspace(float(denoise), 0.0, int(steps) + 1)

        return _build

    def capture_path(self, name: str) -> Path:
        return minimax_capture_dir() / f"{name}.safetensors"

    def load_capture(self, path) -> dict:
        """Load a saved conditioning capture."""
        from safetensors.torch import load_file

        data = load_file(str(path))
        cond = data["encoder_hidden_states"]
        # Captures are stored without the batch dim (L, 2048); the DiT
        # wants (1, L, 2048).
        if cond.ndim == 2:
            cond = cond.unsqueeze(0)
        if cond.ndim != 3 or cond.shape[-1] != COND_DIM:
            raise ValueError(
                f"capture {path} has encoder_hidden_states "
                f"{tuple(cond.shape)}, expected [1, T, {COND_DIM}]"
            )
        return {
            "encoder_hidden_states": cond.to(
                device=self.device, dtype=self.dtype
            )
        }

    def save_capture(self, cond: dict, path) -> None:
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"encoder_hidden_states": cond["encoder_hidden_states"].cpu()},
            str(path),
        )

    def prepare_cond(
        self,
        *,
        prompt: str,
        duration_s: float,
        lyrics: str = "",
        capture: Optional[str] = None,
    ) -> dict:
        """The composition for this session.

        With ``capture`` this is a disk read. Without it, this runs the
        autoregressive stage — seconds of an 8.58B LM — and is why
        ``set_prompt`` on this family is not a per-tick operation.
        """
        if capture is not None:
            return self.load_capture(capture)
        frame_hiddens = self._run_ar(
            # The tokenizer rejects an empty lyric outright; upstream's
            # own convention for "no singing" is the tag, not "".
            prompt=prompt, lyrics=lyrics or "[instrumental]",
            duration_s=duration_s,
        )
        with torch.no_grad():
            cond = self._cond_encoder(
                frame_hiddens.to(device=self.device, dtype=self.dtype)
            )
        return {"encoder_hidden_states": cond}

    def _ensure_ar(self):
        if self.ar_policy == "absent":
            raise RuntimeError(
                "MiniMax autoregressive stage is disabled for this context "
                "(ar_policy='absent'); stream from a saved capture, or "
                "construct the context with ar_policy='offload'"
            )
        if self._ar is not None:
            return
        from acestep.engine.minimax_ar import MiniMaxAR

        # Loaded onto the CPU under 'offload' and paged in per capture.
        target = self.device if self.ar_policy == "resident" else "cpu"
        logger.info("minimax_ar_load policy={} target={}", self.ar_policy, target)
        self._ar = MiniMaxAR.from_pretrained(
            self.root, dtype=torch.bfloat16, device=target,
        )

    def _run_ar(self, *, prompt: str, lyrics: str, duration_s: float):
        """Fused per-frame hidden states ``[1, F, 8*4096]``."""
        frames = int(round(duration_s * AR_FRAME_RATE_HZ))
        with self._ar_lock:
            self._ensure_ar()
            paged = self.ar_policy == "offload"
            try:
                if paged:
                    self._ar.to(self.device)
                with torch.no_grad():
                    hidden = self._ar.generate_frame_hiddens(
                        prompt=prompt, lyrics=lyrics, frames=frames,
                    )
                return hidden.detach().to("cpu")
            finally:
                if paged:
                    self._ar.to("cpu")
                    torch.cuda.empty_cache()

    def close(self) -> None:
        self._ar = None
        self._tokenizer = None


# Process-wide cache: the load is tens of seconds and the weights are
# immutable, so every session after the first is warm. Held across the
# load, not just around the dict lookup, or two concurrent creates each
# pay for a copy.
_CONTEXTS: dict = {}
_CONTEXTS_LOCK = threading.Lock()


def get_minimax_context(
    model_dir=None,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    ar_policy: str = "offload",
) -> MiniMaxContext:
    key = (str(model_dir or ""), device, str(dtype), ar_policy)
    with _CONTEXTS_LOCK:
        ctx = _CONTEXTS.get(key)
        if ctx is None:
            ctx = MiniMaxContext(
                model_dir, device=device, dtype=dtype, ar_policy=ar_policy,
            )
            _CONTEXTS[key] = ctx
        return ctx
