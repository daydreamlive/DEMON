"""VAE encode/decode nodes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Optional

from loguru import logger
import torch

from .base import BaseNode, NodeDefinition, NodeParam, NodePort, NodeRegistry
from .types import Audio, Curve, Latent, ModelHandle, VAEHandle


# -----------------------------------------------------------------------
# TRT VAE helpers (loaded once, reused across calls)
# -----------------------------------------------------------------------

_trt_vae_cache: dict[str, Any] = {}

# Shared polygraphy CUDA stream for all TRT engines in this process.
# Using torch.cuda.Stream causes a 14x performance degradation on
# Blackwell GPUs when multiple TRT engines coexist. Polygraphy's
# cuda.Stream (a thin wrapper around cudaStreamCreate) avoids this.
_trt_stream = None

def _get_trt_stream():
    """Get or create the shared polygraphy CUDA stream."""
    global _trt_stream
    if _trt_stream is None:
        from polygraphy import cuda as pg_cuda
        _trt_stream = pg_cuda.Stream()
    return _trt_stream


def _trt_available() -> bool:
    """Check if TensorRT is importable."""
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def _get_trt_vae(engine_path: str, device: torch.device):
    """Load or return cached TRT VAE engine + context + stream."""
    engine_path = os.path.abspath(engine_path)
    if engine_path in _trt_vae_cache:
        return _trt_vae_cache[engine_path]

    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes

    engine = engine_from_bytes(bytes_from_path(engine_path))
    ctx = engine.create_execution_context()
    logger.info("Loaded TRT VAE engine: %s", engine_path)

    entry = {"engine": engine, "context": ctx}
    _trt_vae_cache[engine_path] = entry
    return entry


def _evict_trt_vae(engine_path: str) -> bool:
    """Drop a cached TRT VAE engine and free its GPU buffers.

    Used by the profile manager when swapping to a different-duration
    engine: the old context, its workspace, and the cached output buffer
    must be released before the new engine is loaded so VRAM doesn't
    spike past the budget.

    Returns True if an entry was evicted, False if the path wasn't cached.
    """
    engine_path = os.path.abspath(engine_path)
    entry = _trt_vae_cache.pop(engine_path, None)
    if entry is None:
        return False
    # Clear refs in deterministic order: cached I/O buffers first (so
    # nothing still holds the device pointer), then the execution
    # context, then the engine. Polygraphy releases workspace when the
    # context's refcount hits zero.
    entry.pop("_decode_buf", None)
    entry.pop("_encode_buf", None)
    entry["context"] = None
    entry["engine"] = None
    torch.cuda.empty_cache()
    logger.info("Evicted TRT VAE engine: %s", engine_path)
    return True


def _trt_vae_decode(
    latents_bdt: torch.Tensor, engine_path: str, device: torch.device
) -> torch.Tensor:
    """Decode latents [B, D, T] -> audio [B, 2, samples] via TRT."""
    entry = _get_trt_vae(engine_path, device)
    ctx = entry["context"]
    stream = _get_trt_stream()

    # Ensure input is on GPU, fp32, contiguous
    lat = latents_bdt.to(device=device, dtype=torch.float32).contiguous()

    ctx.set_input_shape("latents", tuple(lat.shape))
    ctx.set_tensor_address("latents", lat.data_ptr())

    out_shape = tuple(ctx.get_tensor_shape("audio"))

    cached = entry.get("_decode_buf")
    if cached is not None and cached.shape == out_shape:
        audio_buf = cached
    else:
        audio_buf = torch.empty(out_shape, dtype=torch.float32, device=device)
        entry["_decode_buf"] = audio_buf

    ctx.set_tensor_address("audio", audio_buf.data_ptr())

    if not ctx.execute_async_v3(stream.ptr):
        raise RuntimeError("TRT VAE decode failed")
    stream.synchronize()

    return audio_buf.clone()


def _trt_vae_encode(
    audio_bct: torch.Tensor, engine_path: str, device: torch.device
) -> torch.Tensor:
    """Encode audio [B, 2, samples] -> latents [B, D, T] via TRT.

    The ONNX export produces moments [B, 128, T] (mean+logvar concatenated).
    We split and sample: latent = mean + exp(0.5 * logvar) * noise,
    matching the VAE's latent_dist.sample() behavior.
    """
    entry = _get_trt_vae(engine_path, device)
    ctx = entry["context"]
    stream = _get_trt_stream()

    inp = audio_bct.float().contiguous().to(device)

    # Release PyTorch's unused reserved VRAM before TRT encode.
    torch.cuda.empty_cache()

    ctx.set_input_shape("audio", tuple(inp.shape))
    ctx.set_tensor_address("audio", inp.data_ptr())

    out_shape = tuple(ctx.get_tensor_shape("moments"))
    moments_buf = torch.empty(out_shape, dtype=torch.float32, device=device)
    ctx.set_tensor_address("moments", moments_buf.data_ptr())

    if not ctx.execute_async_v3(stream.ptr):
        raise RuntimeError("TRT VAE encode failed")
    stream.synchronize()

    # Split moments into mean and logvar, sample
    mean, logvar = moments_buf.chunk(2, dim=1)  # [B, 64, T] each
    std = torch.exp(0.5 * logvar)
    latent = mean + std * torch.randn_like(mean)
    return latent


def _find_trt_engine(name: str) -> Optional[str]:
    """Search for a TRT engine file.

    Checks the central models directory (from acestep.paths) first,
    then falls back to relative paths for backward compatibility.
    """
    from acestep.paths import trt_engines_dir

    stem = name.replace(".engine", "")
    trt_dir = str(trt_engines_dir())
    candidates = [
        # Central models directory (preferred)
        os.path.join(trt_dir, stem, name),
        # CWD relative (legacy fallback)
        os.path.join("trt_engines", stem, name),
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.exists(p):
            return p
    return None


def _find_best_vae_engine(component: str) -> Optional[str]:
    """Return a TRT VAE engine path for *component* if one was preloaded.

    Only returns engines already in `_trt_vae_cache` (populated by
    `Session(vae_backend="tensorrt", trt_engines={...})`). There is no
    filesystem discovery: TRT is opt-in, never magic.

    Matches both ``vae_<action>_*`` and ``dreamvae_<action>_*`` for the
    same component, since the distilled DreamVAE engines are drop-in
    replacements for the standard decoder.

    Args:
        component: "vae_decode" or "vae_encode"
    """
    # component is "vae_decode" or "vae_encode" -> action is "decode"/"encode".
    action = component.split("_", 1)[-1]
    accepted = (f"vae_{action}_", f"dreamvae_{action}_")
    for cached_path in _trt_vae_cache:
        basename = os.path.basename(cached_path).lower()
        if basename.startswith(accepted):
            return cached_path
    return None



# -----------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------

@NodeRegistry.register
class VAEEncodeAudio(BaseNode):
    """Encode audio waveform to latent space.

    Uses TRT engine if available (vae_encode_fp16.engine), falls back
    to PyTorch VAE via the handler.
    """

    node_type_id: ClassVar[str] = "acestep.VAEEncodeAudio"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="VAE Encode Audio",
            category="vae",
            description="Encode audio waveform to latent representation.",
            inputs=(
                NodePort(name="vae", type="VAE"),
                NodePort(name="audio", type="AUDIO"),
            ),
            outputs=(
                NodePort(name="latent", type="LATENT"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        vae: VAEHandle = kwargs["vae"]
        audio: Audio = kwargs["audio"]
        handler = vae.handler
        device = torch.device(handler.device)
        dtype = handler.dtype

        waveform = audio.waveform
        # Ensure [B, C, samples]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        trt_path = _find_best_vae_engine("vae_encode") if _trt_available() else None
        if trt_path:
            logger.info("VAE encode via TRT")
            latents_bdt = _trt_vae_encode(waveform, trt_path, device)
            # [B, D, T] -> [B, T, D]
            latents = latents_bdt.transpose(1, 2).to(dtype)
        else:
            logger.info("VAE encode via PyTorch")
            with handler._load_model_context("vae"):
                latents = handler._encode_audio_to_latents(waveform)
            if latents.dim() == 2:
                latents = latents.unsqueeze(0)

        return {"latent": Latent(tensor=latents)}


@NodeRegistry.register
class VAEDecodeAudio(BaseNode):
    """Decode latents back to audio waveform.

    Uses TRT engine if available (vae_decode_fp16.engine), falls back
    to PyTorch VAE via the handler.
    """

    node_type_id: ClassVar[str] = "acestep.VAEDecodeAudio"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="VAE Decode Audio",
            category="vae",
            description="Decode latent representation to audio waveform.",
            inputs=(
                NodePort(name="vae", type="VAE"),
                NodePort(name="latent", type="LATENT"),
            ),
            outputs=(
                NodePort(name="audio", type="AUDIO"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        vae: VAEHandle = kwargs["vae"]
        latent: Latent = kwargs["latent"]
        handler = vae.handler
        device = torch.device(handler.device)

        # [B, T, D] -> [B, D, T]
        lat_bdt = latent.tensor.transpose(1, 2)

        trt_path = _find_best_vae_engine("vae_decode") if _trt_available() else None
        if trt_path:
            logger.info("VAE decode via TRT")
            waveform = _trt_vae_decode(lat_bdt, trt_path, device)
        else:
            logger.info("VAE decode via PyTorch (no TRT engine found)")
            with handler._load_model_context("vae"):
                waveform = handler.tiled_decode(lat_bdt)

        return {"audio": Audio(waveform=waveform, sample_rate=48000)}


@NodeRegistry.register
class StreamVAEDecode(BaseNode):
    """Windowed VAE decode for streaming playback.

    Wraps :class:`VAEDecodeAudio` with the overlap-margin windowing that
    the realtime demo needs. The finished latent is typically long
    (60 s); decoding a short interior window at ``t_start`` keeps tick
    latency low while the margins supply receptive-field context so the
    window edges match the full decode.

    Realtime skip gate
    ------------------
    When the host supplies a per-call ``playhead_seconds`` kwarg (scope
    injects this from ``AudioProcessingTrack._timestamp / 48000``), the
    node follows the playhead instead of the widget ``t_start`` and
    applies the same skip logic as
    ``demos/realtime_motion_graph_web/pipeline.py``:

      * Decode is skipped when the current latent is close to the last
        one (``mse < mse_skip_threshold``) **and** the playhead has not
        yet advanced past ``(vae_window - prefetch_seconds)`` since the
        last actual decode.
      * When skipping, the node returns ``{"audio": None}`` so scope's
        ``_route_outputs`` short-circuits — the downstream sink is not
        touched and the upstream ring buffer is not back-pressured.

    Both gates are opt-in: the classic, stateless decode path is
    preserved when ``follow_playhead`` is false or no playhead is
    provided.

    Node parameters:
        vae_window: Window length in seconds. ``<= 0`` decodes the full
            latent in one shot (parity with ``VAEDecodeAudio``).
        vae_overlap: Extra seconds of context on each side of the window.
            Trimmed from the returned Audio.
        t_start: Window start in seconds. Used only when
            ``follow_playhead`` is false or no playhead is provided.
        follow_playhead: If true and ``playhead_seconds`` is supplied in
            kwargs, drive ``t_start`` from the playhead and enable the
            skip gate.
        mse_skip_threshold: MSE threshold for the skip gate. Matches the
            demo's ``skip_threshold`` (default 1e-3). ``<= 0`` disables
            the MSE check but leaves the prefetch-window check active.
        prefetch_seconds: Playhead-advance cushion. Demo uses
            ``min(1.0, vae_window * 0.2)`` — we default to 1.0.
    """

    node_type_id: ClassVar[str] = "acestep.StreamVAEDecode"

    FRAMES_PER_SEC: ClassVar[int] = 25
    SAMPLES_PER_FRAME: ClassVar[int] = 1920  # 48_000 / 25

    def __init__(self):
        # Skip-gate state. Mirrors the demo's ``last_latent`` and
        # ``last_decode_pos`` locals, hoisted to instance scope because
        # Scope's graph runtime holds the node instance across ticks.
        self._last_latent_for_skip: Optional[torch.Tensor] = None
        self._last_decode_pos: Optional[float] = None

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Stream VAE Decode",
            category="vae",
            description="Windowed VAE decode for realtime streaming playback.",
            inputs=(
                NodePort(name="vae", type="VAE"),
                NodePort(name="latent", type="LATENT"),
            ),
            outputs=(
                NodePort(name="audio", type="AUDIO"),
            ),
            params=(
                NodeParam(
                    name="vae_window", type="number", default=5.0,
                    description=(
                        "Decode window (s); <=0 decodes full latent. "
                        "Positive values are clamped to [3, 30] to fit "
                        "the windowed VAE engine profile."
                    ),
                    min=0.0, max=30.0, step=0.1,
                ),
                NodeParam(
                    name="vae_overlap", type="number", default=0.5,
                    description="Extra context seconds on each window edge",
                    min=0.0, max=5.0, step=0.05,
                ),
                NodeParam(
                    name="t_start", type="number", default=0.0,
                    description="Window start (s); ignored when follow_playhead is on",
                    min=0.0, max=60.0, step=0.1,
                ),
                NodeParam(
                    name="follow_playhead", type="boolean", default=False,
                    description="Drive t_start from audio sink playhead + enable skip gate",
                ),
                NodeParam(
                    name="mse_skip_threshold", type="number", default=1e-3,
                    description="Skip decode when latent MSE < threshold",
                    min=0.0, max=1.0, step=1e-4,
                ),
                NodeParam(
                    name="prefetch_seconds", type="number", default=1.0,
                    description="Seconds of headroom before forcing a re-decode",
                    min=0.0, max=5.0, step=0.05,
                ),
                NodeParam(
                    name="playhead_seconds", type="any", default=None,
                    description="Ambient audio playhead injected by host",
                    hidden=True,
                ),
                NodeParam(
                    name="cyclic", type="boolean", default=False,
                    description=(
                        "Treat the latent as cyclic: pull missing left/right "
                        "context from the opposite end of the latent instead "
                        "of clamping. Use only when the audio is intended to "
                        "loop, otherwise the song's tail will bleed into the "
                        "head's decode through the VAE receptive field."
                    ),
                ),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        from acestep.paths import WINDOWED_VAE_WINDOW_RANGE_S

        vae: VAEHandle = kwargs["vae"]
        latent: Latent = kwargs["latent"]
        window_s = float(kwargs.get("vae_window", 0.0))
        overlap_s = float(kwargs.get("vae_overlap", 0.5))

        # Hard-clamp positive windows to the engine-supported range.
        # ``<= 0`` is the disable sentinel and falls through to a full
        # decode, so we leave it untouched.
        if window_s > 0:
            lo, hi = WINDOWED_VAE_WINDOW_RANGE_S
            window_s = max(lo, min(hi, window_s))

        follow_playhead = bool(kwargs.get("follow_playhead", False))
        playhead_seconds = kwargs.get("playhead_seconds")
        mse_skip_threshold = float(kwargs.get("mse_skip_threshold", 1e-3))
        prefetch_seconds = float(kwargs.get("prefetch_seconds", 1.0))
        cyclic = bool(kwargs.get("cyclic", False))

        use_playhead = (
            follow_playhead
            and playhead_seconds is not None
            and window_s > 0
        )

        if use_playhead:
            t_start = float(playhead_seconds)
        else:
            t_start = float(kwargs.get("t_start", 0.0))

        # -- Skip gate (demo pipeline.py:215-225) --------------------
        # Only active when following the playhead. Skipping returns
        # ``{"audio": None}``; scope's _route_outputs drops None so no
        # chunk reaches the sink and upstream stays unblocked.
        if use_playhead:
            tensor = latent.tensor
            mse_ok_to_skip = True
            if mse_skip_threshold > 0.0 and self._last_latent_for_skip is not None:
                try:
                    mse = (tensor - self._last_latent_for_skip).pow(2).mean().item()
                except RuntimeError:
                    # Shape changed — don't skip; fall through to decode.
                    mse = float("inf")
                mse_ok_to_skip = mse < mse_skip_threshold

            within_window = (
                self._last_decode_pos is not None
                and abs(t_start - self._last_decode_pos) < (window_s - prefetch_seconds)
            )

            if mse_ok_to_skip and within_window:
                # Keep the reference latent fresh so a future divergence
                # compares against what we actually last decoded.
                return {"audio": None}

        if window_s <= 0:
            out = VAEDecodeAudio().execute(vae=vae, latent=latent)
            if use_playhead:
                self._last_latent_for_skip = latent.tensor.detach().clone()
                self._last_decode_pos = t_start
            return out

        tensor = latent.tensor
        T = tensor.shape[1]
        win_frames = int(window_s * self.FRAMES_PER_SEC)
        ovl_frames = int(overlap_s * self.FRAMES_PER_SEC)

        if T <= win_frames:
            out = VAEDecodeAudio().execute(vae=vae, latent=latent)
            if use_playhead:
                self._last_latent_for_skip = tensor.detach().clone()
                self._last_decode_pos = t_start
            return out

        # Playhead-follow clamp: the demo does
        # ``t_pos = min(t_pos, eff_dur - vae_window)`` to keep the
        # window inside the latent; frame-quantization below already
        # handles the floor. In cyclic mode we let the playhead reach
        # the end and rely on wrapped context for the right margin.
        if use_playhead and not cyclic:
            max_t = max(0.0, (T / self.FRAMES_PER_SEC) - window_s)
            t_start = min(t_start, max_t)

        keep_start = max(0, int(t_start * self.FRAMES_PER_SEC))
        keep_end = min(T, keep_start + win_frames)
        keep_start = max(0, keep_end - win_frames)

        if cyclic:
            # Pull missing context from the opposite end so boundary
            # frames see a full receptive field, instead of clamping to
            # zero context (which makes frame 0 / frame T-1 sound less
            # denoised than every interior frame).
            decode_start = keep_start - ovl_frames
            decode_end = keep_end + ovl_frames
            pieces = []
            if decode_start < 0:
                pieces.append(tensor[:, T + decode_start:, :])
                pieces.append(tensor[:, :decode_end, :])
            elif decode_end > T:
                pieces.append(tensor[:, decode_start:, :])
                pieces.append(tensor[:, :decode_end - T, :])
            else:
                pieces.append(tensor[:, decode_start:decode_end, :])
            chunk_lat = Latent(tensor=torch.cat(pieces, dim=1).contiguous())
            pre_margin_frames = ovl_frames
        else:
            decode_start = max(0, keep_start - ovl_frames)
            decode_end = min(T, keep_end + ovl_frames)
            chunk_lat = Latent(
                tensor=tensor[:, decode_start:decode_end, :].contiguous()
            )
            pre_margin_frames = keep_start - decode_start

        chunk_audio = VAEDecodeAudio().execute(vae=vae, latent=chunk_lat)["audio"]

        pre_margin = pre_margin_frames * self.SAMPLES_PER_FRAME
        keep_samples = (keep_end - keep_start) * self.SAMPLES_PER_FRAME
        trimmed = chunk_audio.waveform[:, :, pre_margin:pre_margin + keep_samples]

        if use_playhead:
            self._last_latent_for_skip = tensor.detach().clone()
            self._last_decode_pos = keep_start / self.FRAMES_PER_SEC

        return {
            "audio": Audio(
                waveform=trimmed,
                sample_rate=chunk_audio.sample_rate,
                start_sample=keep_start * self.SAMPLES_PER_FRAME,
            ),
        }


@NodeRegistry.register
class EmptyLatent(BaseNode):
    """Create a silence-based latent of a given duration.

    Node parameters:
        duration: Duration in seconds.
    """

    node_type_id: ClassVar[str] = "acestep.EmptyLatent"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Empty ACE-Step Latent",
            category="vae",
            description="Create an empty (silence) latent for a given duration.",
            inputs=(
                NodePort(name="model", type="MODEL"),
            ),
            outputs=(
                NodePort(name="latent", type="LATENT"),
            ),
            params=(
                NodeParam(
                    name="duration", type="number", default=60.0,
                    description="Duration (s)",
                    min=1.0, max=600.0, step=1.0,
                ),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        model: ModelHandle = kwargs["model"]
        handler = model.handler
        duration = kwargs.get("duration", 60.0)

        handler._ensure_silence_latent_on_device()
        silence = handler.silence_latent  # [1, T_full, D]

        T = int(duration * 25)  # 25 fps latent rate
        # Take first T frames from the silence latent (tiled if needed)
        if silence.dim() == 3:
            latent = silence[:, :T, :].clone()
            if latent.shape[1] < T:
                reps = (T + latent.shape[1] - 1) // latent.shape[1]
                latent = latent.repeat(1, reps, 1)[:, :T, :]
        else:
            # Fallback: treat as [1, D] single frame
            latent = silence.unsqueeze(0).expand(1, T, -1).clone()

        return {"latent": Latent(tensor=latent)}


@NodeRegistry.register
class LatentBlend(BaseNode):
    """Blend two latents by weighted interpolation.

    Supports scalar or per-frame (CURVE) blend factor.
    Useful for timbre strength control (blend reference with silence)
    or mixing any two latent representations.

    Node parameters:
        alpha: Blend factor (0.0 = all A, 1.0 = all B).
               Ignored if a blend_curve input is connected.
    """

    node_type_id: ClassVar[str] = "acestep.LatentBlend"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Latent Blend",
            category="vae",
            description="Blend two latents with scalar or per-frame factor.",
            inputs=(
                NodePort(name="latent_a", type="LATENT"),
                NodePort(name="latent_b", type="LATENT"),
                NodePort(
                    name="blend_curve",
                    type="CURVE",
                    required=False,
                    description="Per-frame blend factor (overrides scalar alpha).",
                ),
            ),
            outputs=(
                NodePort(name="latent", type="LATENT"),
            ),
            params=(
                NodeParam(
                    name="alpha", type="number", default=0.5,
                    description="Blend factor (0 = all A, 1 = all B). Overridden by blend_curve.",
                    min=0.0, max=1.0, step=0.01,
                ),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        latent_a: Latent = kwargs["latent_a"]
        latent_b: Latent = kwargs["latent_b"]
        blend_curve: Optional[Curve] = kwargs.get("blend_curve")

        a = latent_a.tensor
        b = latent_b.tensor

        alpha = kwargs.get("alpha", 0.5)
        if blend_curve is not None:
            alpha = blend_curve.tensor.to(device=a.device, dtype=a.dtype)
            if alpha.ndim == 1:
                alpha = alpha.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            elif alpha.ndim == 2:
                alpha = alpha.unsqueeze(-1)  # [B, T, 1]

        blended = (1.0 - alpha) * a + alpha * b
        return {"latent": Latent(tensor=blended)}
