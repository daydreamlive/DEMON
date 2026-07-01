"""SA3 streaming helpers: conditioning, source encode, and SAME windowed
decode.

These are the load-bearing pieces of the SA3 production path — the
:class:`~acestep.engine.sa3_adapter.SA3Adapter` and
:class:`~acestep.engine.sa3_context.SA3Context` call them directly. They
were originally prototyped in ``scripts/sa3/sa3_stream_pipeline.py``
alongside the standalone ``SA3StreamPipeline`` ringbuffer; the runtime
helpers now live in the ``acestep`` package so production no longer
imports them out of ``scripts/`` over a front-injected ``sys.path``. The
spike module re-exports these names for its own (test/demo-only)
``SA3StreamPipeline``.

Imports of the vendored ``stable_audio_3`` package are lazy (inside the
functions that need it) — by the time these run the caller has already
loaded the SAME model from the vendor tree, so the path is set.

Latents use SA3/SAME's native layout throughout: ``[B, C, T]``.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def stack_sa3_cond_bundles(bundles: list[dict]) -> dict:
    """Stack per-slot SA3 conditioning bundles for one batched DiT call.

    Tensor values are concatenated on dim 0. The variable-length cross-attn
    tensors are padded to the batch max length; their masks keep padding inert.
    Scalars, booleans, and ``None`` are passed through from the first row.
    """
    if not bundles:
        raise ValueError("stack_sa3_cond_bundles requires at least one bundle")

    keys = bundles[0].keys()
    lens = [
        b["cross_attn_cond"].shape[1]
        for b in bundles
        if torch.is_tensor(b.get("cross_attn_cond"))
    ]
    max_l = max(lens) if lens else 0
    out = {}
    for key in keys:
        vals = [b[key] for b in bundles]
        first = vals[0]
        if not torch.is_tensor(first):
            out[key] = first
            continue
        if first.ndim == 0:
            out[key] = first
            continue
        if key == "cross_attn_cond":
            vals = [
                F.pad(v, (0, 0, 0, max_l - v.shape[1]))
                if v.shape[1] < max_l else v
                for v in vals
            ]
        elif key == "cross_attn_mask":
            vals = [
                F.pad(v, (0, max_l - v.shape[1]), value=0)
                if v.shape[1] < max_l else v
                for v in vals
            ]
        out[key] = torch.cat(vals, dim=0)
    return out


@dataclass
class SA3Conditioning:
    """Precomputed SA3 conditioning and schedule metadata for one prompt."""

    cond_bundle: dict
    sched_args: dict
    latent_frames: int
    audio_sample_size: int


@dataclass(frozen=True)
class SA3DecodeWindow:
    """Resolved SAME latent-window decode geometry."""

    target_start_sample: int
    target_num_samples: int
    latent_start: int
    latent_end: int
    slice_start: int
    slice_end: int
    crop_start: int
    context_latents: int
    slice_align_latents: int


@dataclass(frozen=True)
class SA3WindowDecodeResult:
    """One SAME-native window decode result."""

    audio_ct: torch.Tensor
    window: SA3DecodeWindow
    decode_ms: float


def prepare_sa3_conditioning(
    sam,
    *,
    prompt: str,
    duration: float,
    steps: int,
    sample_size: int = 5292032,
    duration_padding_sec: float = 6.0,
    cfg_scale: float = 1.0,
    apg_scale: float = 1.0,
    dist_shift=None,
) -> SA3Conditioning:
    """Build SA3 conditioning without running the sampler.

    This mirrors ``StableAudioModel.generate`` up to the call into
    ``sample_diffusion`` and returns the exact DiT kwargs plus the schedule
    inputs needed by the streaming pipeline.
    """
    from stable_audio_3.data.utils import (
        compute_effective_seq_len_from_conditioning,
        create_padding_mask_from_lengths,
    )

    device = str(sam.device)
    conditioning, _negative = sam._build_conditioning_dicts(
        prompt, None, duration, batch_size=1,
    )
    audio_sample_size = sam._adapt_sample_size(
        conditioning, sample_size, duration_padding_sec,
    )
    downsampling_ratio = sam.model.pretransform.downsampling_ratio
    latent_frames = audio_sample_size // downsampling_ratio

    conditioning_tensors = sam.model.conditioner(conditioning, device)

    mask = torch.zeros((1, 1, latent_frames), device=device)
    inpaint_input = torch.zeros(
        (1, sam.model.io_channels, latent_frames), device=device,
    )
    conditioning_tensors["inpaint_mask"] = [mask]
    conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
    conditioning_inputs = sam.model.get_conditioning_inputs(conditioning_tensors)

    model_dtype = next(sam.model.model.parameters()).dtype
    conditioning_inputs = {
        k: v.type(model_dtype) if torch.is_tensor(v) else v
        for k, v in conditioning_inputs.items()
    }

    effective_seq_len = compute_effective_seq_len_from_conditioning(
        conditioning, sam.model.sample_rate, downsampling_ratio, device,
    )
    headroom_tokens = int(duration_padding_sec * sam.model.sample_rate / downsampling_ratio)
    valid_lengths = (effective_seq_len + headroom_tokens).clamp(
        max=latent_frames,
    ).long()
    padding_mask = create_padding_mask_from_lengths(valid_lengths, latent_frames)

    cond_bundle = {
        **conditioning_inputs,
        "cfg_scale": cfg_scale,
        "batch_cfg": True,
        "rescale_cfg": True,
        "padding_mask": padding_mask,
        "apg_scale": apg_scale,
    }

    sched_args = {
        "steps": steps,
        "dist_shift": dist_shift if dist_shift is not None else sam.model.sampling_dist_shift,
        "effective_seq_len": effective_seq_len.detach().cpu()
        if torch.is_tensor(effective_seq_len) else effective_seq_len,
        "fallback_seq_len": latent_frames,
    }
    return SA3Conditioning(
        cond_bundle=cond_bundle,
        sched_args=sched_args,
        latent_frames=latent_frames,
        audio_sample_size=audio_sample_size,
    )


def encode_sa3_source(sam, audio_input, audio_sample_size: int) -> torch.Tensor:
    """Encode an audio source through SAME for SA3 audio-to-audio streaming."""
    encoded, _ = sam._encode_audio_input(audio_input, audio_sample_size, inpaint_mask=None)
    return encoded


def decode_sa3_latent(sam, latent_bct: torch.Tensor) -> torch.Tensor:
    """Decode a native SA3 latent ``[1, 256, T]`` with SAME."""
    pt_dtype = next(sam.model.pretransform.parameters()).dtype
    return sam.model.pretransform.decode(latent_bct.to(pt_dtype)).float().clamp(-1, 1)


def infer_sa3_decode_slice_alignment(sam) -> int:
    """Infer the latent-index phase needed by SAME manual window slicing.

    SA3 small's decoder uses non-sliding transformer chunks, so slicing from an
    arbitrary latent index can shift the decoder's internal chunk phase versus a
    full decode. SA3 medium currently uses sliding-window attention in its SAME
    decoder, so it does not require that chunk-phase snapping.
    """
    decoder = getattr(sam.model.pretransform.model, "decoder", None)
    layers = getattr(decoder, "layers", [])
    align = 1
    for layer in layers:
        if not hasattr(layer, "type") or getattr(layer, "type") != "decoder":
            continue
        if getattr(layer, "sliding_window_latents", None) is not None:
            continue
        chunk_size = int(getattr(layer, "chunk_size", 1) or 1)
        stride = int(getattr(layer, "stride", 1) or 1)
        # chunk_midpoint_shift offsets the chunk's sampling phase but
        # leaves the alignment PERIOD (chunk_size/stride) unchanged, so
        # it doesn't affect the lcm here.
        phase = max(1, chunk_size // max(stride, 1))
        align = math.lcm(align, phase)
    return max(1, align)


def resolve_sa3_decode_window(
    latent_bct: torch.Tensor,
    *,
    target_start_sample: int,
    target_num_samples: int,
    context_sec: float,
    sample_rate: int,
    downsampling_ratio: int,
    slice_align_latents: int = 1,
) -> SA3DecodeWindow:
    """Resolve sample-domain playback window to an aligned latent slice."""
    if latent_bct.ndim != 3:
        raise ValueError(f"SA3 latent must be [B,C,T], got {tuple(latent_bct.shape)}")
    if target_start_sample < 0:
        raise ValueError("target_start_sample must be non-negative")
    if target_num_samples < 1:
        raise ValueError("target_num_samples must be positive")

    target_end_sample = target_start_sample + target_num_samples
    latent_start = target_start_sample // downsampling_ratio
    latent_end = int(math.ceil(target_end_sample / downsampling_ratio))
    context_latents = int(math.ceil(context_sec * sample_rate / downsampling_ratio))

    slice_start = max(0, latent_start - context_latents)
    align = max(1, int(slice_align_latents))
    if align > 1:
        slice_start = (slice_start // align) * align
    slice_end = min(latent_bct.shape[-1], latent_end + context_latents)
    crop_start = target_start_sample - slice_start * downsampling_ratio

    return SA3DecodeWindow(
        target_start_sample=target_start_sample,
        target_num_samples=target_num_samples,
        latent_start=latent_start,
        latent_end=latent_end,
        slice_start=slice_start,
        slice_end=slice_end,
        crop_start=crop_start,
        context_latents=context_latents,
        slice_align_latents=align,
    )


@contextmanager
def sa3_decode_noise_mode(sam, *, enabled: bool):
    """Temporarily toggle SAME decode-time noise sources.

    This is useful for deterministic streaming/window parity. SAME small and
    medium both include bottleneck decode noise; medium also has substantial
    decoder token ``mask_noise``.
    """
    changed: list[tuple[object, str, object]] = []
    pretransform = sam.model.pretransform.model
    bottleneck = getattr(pretransform, "bottleneck", None)
    if bottleneck is not None and hasattr(bottleneck, "noise_regularize"):
        changed.append((bottleneck, "noise_regularize", bottleneck.noise_regularize))
        bottleneck.noise_regularize = bool(enabled)
    if not enabled:
        for module in pretransform.decoder.modules():
            if hasattr(module, "mask_noise"):
                changed.append((module, "mask_noise", module.mask_noise))
                module.mask_noise = 0.0
    try:
        yield
    finally:
        for obj, attr, old in reversed(changed):
            setattr(obj, attr, old)


@torch.no_grad()
def decode_sa3_latent_window(
    sam,
    latent_bct: torch.Tensor,
    *,
    target_start_sample: int,
    target_num_samples: int,
    context_sec: float,
    chunked: bool = False,
    slice_align_latents: Optional[int] = None,
    deterministic: bool = False,
) -> SA3WindowDecodeResult:
    """Decode a SAME latent slice and return the requested center audio.

    This is the SA3 pipeline-level replacement for decoding the whole latent
    and then taking a playback slice. It slices in latent space, decodes the
    context-padded window, then crops in audio-sample space.
    """
    pretransform = sam.model.pretransform
    sample_rate = int(pretransform.model.sample_rate)
    downsampling_ratio = int(pretransform.downsampling_ratio)
    align = (
        infer_sa3_decode_slice_alignment(sam)
        if slice_align_latents is None
        else int(slice_align_latents)
    )
    window = resolve_sa3_decode_window(
        latent_bct,
        target_start_sample=target_start_sample,
        target_num_samples=target_num_samples,
        context_sec=context_sec,
        sample_rate=sample_rate,
        downsampling_ratio=downsampling_ratio,
        slice_align_latents=align,
    )
    latent_window = latent_bct[..., window.slice_start:window.slice_end].contiguous()
    pt_dtype = next(pretransform.parameters()).dtype

    if latent_window.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with sa3_decode_noise_mode(sam, enabled=not deterministic):
        decoded = pretransform.decode(latent_window.to(pt_dtype), chunked=chunked).float().clamp(-1, 1)
    if latent_window.device.type == "cuda":
        torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - t0) * 1000.0

    audio_ct = decoded[0, :, window.crop_start:window.crop_start + target_num_samples]
    if audio_ct.shape[-1] < target_num_samples:
        audio_ct = F.pad(audio_ct, (0, target_num_samples - audio_ct.shape[-1]))
    return SA3WindowDecodeResult(audio_ct=audio_ct, window=window, decode_ms=decode_ms)
