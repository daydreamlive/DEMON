"""TensorRT runtimes for the YuE2 family: the flexible acoustic NAR engine
and the 37-frame VAE window decoder, plus the window-planning math.

Engines (builder: ``python -m acestep.engine.trt.yue2_build``) live in
``DEMON_YUE2_TRT_DIR``:

* ``flexible_song/velocity.trt``: the NAR transformer only, BF16
  strongly typed, with the conditioning (AR-prefix keys/values, rotary
  cos/sin, latent position embedding) as engine INPUTS, so a new song
  never rebuilds weights. Optimization profiles (:data:`FLEX_PROFILES`):
  profile 0 = state frames 1000..2500 (40-100 s), profile 1 = 250..1000
  (10-40 s, short songs), both batch 1..4 and conditioning tokens up to
  4000. Engines built before the short profile have profile 0 only; the
  runtime reads the profiles from the engine, so both kinds load. The
  engine broadcasts ONE KV tensor over the batch, so one execution
  serves rows of one bundle.
* ``vae_fp32_t37.trt``: the VAE decoder over a fixed 37-frame latent
  window, FP32 with TF32 off; the 5 core frames 16..21 (200 ms) are
  kept, the 16-frame margins cover the decoder's receptive field.

``tensorrt`` imports stay inside functions: the module imports on hosts
without TRT, and engine discovery then reports nothing.
"""

from __future__ import annotations

import math
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

from acestep.engine.obs import logger

#: The flexible NAR engine's long profile (profile 0) as the builder
#: writes it: frames, conditioning tokens. At run time :class:`TRTVelocity`
#: reads every profile from the loaded engine; these are the fallback
#: when it cannot.
FLEX_FRAMES = (1000, 2500)
FLEX_COND_TOKENS = (1001, 4000)
FLEX_MAX_BATCH = 4
#: The short-song profile (profile 1): 10-40 s songs.
FLEX_SHORT_FRAMES = (250, 1000)
FLEX_SHORT_COND_TOKENS = (251, 4000)


@dataclass(frozen=True)
class FlexProfile:
    """One optimization profile of the flexible NAR engine: inclusive
    bounds on the song's latent frames and its conditioning length."""

    frames: Tuple[int, int]
    cond_tokens: Tuple[int, int]
    max_batch: int = FLEX_MAX_BATCH

    def fits(self, frames: int, cond_tokens: int) -> bool:
        return (self.frames[0] <= frames <= self.frames[1]
                and self.cond_tokens[0] <= cond_tokens <= self.cond_tokens[1])


#: The profiles the builder writes, in engine order (index = profile index).
FLEX_PROFILES = (FlexProfile(FLEX_FRAMES, FLEX_COND_TOKENS),
                 FlexProfile(FLEX_SHORT_FRAMES, FLEX_SHORT_COND_TOKENS))

#: VAE window engine geometry, in latent frames (one frame = 1920 samples).
VAE_CTX_FRAMES = 37
VAE_CORE_FRAMES = 5
VAE_MARGIN_FRAMES = 16
SAMPLES_PER_FRAME = 1920

FLEX_ENGINE_DIR = "flexible_song"
VAE_ENGINE_FILE = "vae_fp32_t37.trt"


def profile_index(profiles, frames: int, cond_tokens: int) -> Optional[int]:
    """Index of the first profile that holds one song's geometry, or None."""
    for index, profile in enumerate(profiles):
        if profile.fits(int(frames), int(cond_tokens)):
            return index
    return None


def frames_floor(profiles, max_frames: int) -> int:
    """The shortest song (frames) worth asking the semantic stage for so
    that a song of at most ``max_frames`` lands on TensorRT: the largest
    profile floor not above the budget (1000 when the budget allows it,
    as before the short profile existed), else 1 (eager below every
    profile)."""
    floors = [p.frames[0] for p in profiles if p.frames[0] <= int(max_frames)]
    return max(floors) if floors else 1


def engine_profiles(engine) -> Tuple[FlexProfile, ...]:
    """The flexible NAR engine's optimization profiles, read from the
    engine (``state`` is ``[batch, frames, 64]``, ``keys`` is ``[layers,
    tokens, ...]``). Falls back to the builder's long profile, with a
    warning, when the engine cannot say."""
    try:
        out = []
        for index in range(int(getattr(engine, "num_optimization_profiles", 1))):
            lo, _opt, hi = engine.get_tensor_profile_shape("state", index)
            klo, _kopt, khi = engine.get_tensor_profile_shape("keys", index)
            out.append(FlexProfile((int(lo[1]), int(hi[1])), (int(klo[1]), int(khi[1])), int(hi[0])))
        return tuple(out)
    except Exception as exc:  # noqa: BLE001 - an older runtime or a foreign engine
        logger.warning("yue2_nar_trt_profile_unknown error={!r} using=builder_constants", exc)
        return FLEX_PROFILES[:1]


def find_flexible_engine(trt_dir: Optional[Path]) -> Optional[Path]:
    """``flexible_song/`` directly under the TRT dir or under ``nar_trt/``
    (the spike's runtime layout); None when neither holds an engine."""
    if trt_dir is None:
        return None
    for cand in (trt_dir / FLEX_ENGINE_DIR, trt_dir / "nar_trt" / FLEX_ENGINE_DIR):
        if (cand / "velocity.trt").is_file():
            return cand
    return None


def find_vae_window_engine(trt_dir: Optional[Path]) -> Optional[Path]:
    """``vae_fp32_t37.trt`` directly under the TRT dir or under
    ``trt_engines/`` (the spike's layout); None when absent."""
    if trt_dir is None:
        return None
    for cand in (trt_dir / VAE_ENGINE_FILE, trt_dir / "trt_engines" / VAE_ENGINE_FILE):
        if cand.is_file():
            return cand
    return None


def _deserialize(path: Path):
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT could not deserialize {path}")
    return engine


class _BoundExecution:
    """One execution context bound to one bundle at one batch size.

    ``state`` / ``raw_time`` are staging inputs; the conditioning
    tensors are the bundle's own (bound by address, so the bundle must
    outlive this object; the owner keys it weakly by bundle).

    The staging and output buffers are ordinary tensors even when the
    first bind happens under ``torch.inference_mode`` (the create-time
    anchor solve): the ring later writes them from the runner thread,
    outside inference mode, where an inference tensor refuses in-place
    updates."""

    def __init__(self, engine, bundle, batch: int, device, profile: int = 0):
        with torch.inference_mode(False):
            self._bind(engine, bundle, batch, device, profile)

    def _bind(self, engine, bundle, batch: int, device, profile: int):
        nar = bundle.nar
        self.state = torch.empty(batch, bundle.frames, 64, device=device, dtype=torch.bfloat16)
        self.raw_time = torch.empty(batch, device=device, dtype=torch.bfloat16)
        tensors = {
            "state": self.state, "raw_time": self.raw_time,
            "keys": bundle.keys, "values": bundle.values,
            "cos": nar.cos, "sin": nar.sin, "position": nar.pos_emb,
        }
        self.context = engine.create_execution_context()
        if profile:
            # A new context starts on profile 0; select before any shape.
            if not self.context.set_optimization_profile_async(
                    profile, torch.cuda.current_stream().cuda_stream):
                raise RuntimeError(f"yue2 NAR engine has no optimization profile {profile}")
        self.profile = profile
        for name, value in tensors.items():
            if not self.context.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(f"yue2 NAR input {name} {tuple(value.shape)} outside the engine profile")
            self.context.set_tensor_address(name, value.data_ptr())
        self.output = torch.empty(
            tuple(self.context.get_tensor_shape("velocity")), device=device, dtype=torch.bfloat16,
        )
        self.context.set_tensor_address("velocity", self.output.data_ptr())
        # Keep the bound conditioning alive as long as the context.
        self._tensors = tensors

    def __call__(self, state: torch.Tensor, raw_times: List[float]) -> torch.Tensor:
        self.state.copy_(state)
        # Python floats straight to bf16, as the spike's solver built them.
        self.raw_time.copy_(torch.tensor(raw_times, dtype=torch.bfloat16))
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("yue2 NAR TensorRT execution failed")
        return self.output


class TRTVelocity:
    """Velocity backend over the flexible NAR engine, with a per-bundle
    execution-context cache and an eager fallback.

    Each bundle binds on the first engine profile that holds its
    geometry (read from the engine, :func:`engine_profiles`). A bundle
    outside every profile (too short, too long, or a conditioning prefix
    past its token bound), or one whose bind the engine refuses anyway,
    runs on ``fallback`` and is logged once. Contexts are cached per
    (bundle, batch), LRU-bounded at
    ``max_bundles`` bundles and dropped when their bundle is collected.
    Returns the engine's persistent output buffer (see
    ``yue2_velocity.midpoint_velocity``).
    """

    def __init__(self, engine_dir: Path, fallback: Callable, *, max_bundles: int = 3,
                 device: str = "cuda"):
        self.engine_dir = Path(engine_dir)
        self.engine = _deserialize(self.engine_dir / "velocity.trt")
        self.profiles = engine_profiles(self.engine)
        self.fallback = fallback
        self.max_bundles = int(max_bundles)
        self.device = torch.device(device)
        self._contexts: "OrderedDict[int, Tuple[weakref.ref, dict]]" = OrderedDict()
        self._eager_logged: set = set()
        self._bind_failed: "weakref.WeakSet" = weakref.WeakSet()

    def profile_for(self, bundle) -> Optional[int]:
        return profile_index(self.profiles, bundle.frames, bundle.cond_tokens)

    def covers(self, bundle) -> bool:
        return self.profile_for(bundle) is not None

    def frames_floor(self, max_frames: int) -> int:
        return frames_floor(self.profiles, max_frames)

    def path_for(self, bundle) -> str:
        """"trt" or "eager": the path a batch-1 call on ``bundle`` takes."""
        return "trt" if self.covers(bundle) and bundle not in self._bind_failed else "eager"

    def __call__(self, bundle, state: torch.Tensor, raw_times: List[float]) -> torch.Tensor:
        profile = self.profile_for(bundle)
        if (profile is None or bundle in self._bind_failed
                or state.shape[0] > self.profiles[profile].max_batch):
            self._log_eager(bundle, state.shape[0])
            return self.fallback(bundle, state, raw_times)
        try:
            bound = self._bound(bundle, state.shape[0], profile)
        except RuntimeError as exc:  # the engine refused a shape: eager for this bundle
            self._bind_failed.add(bundle)
            self._log_eager(bundle, state.shape[0], error=exc)
            return self.fallback(bundle, state, raw_times)
        return bound(state, raw_times)

    def _log_eager(self, bundle, batch: int, error=None) -> None:
        if id(bundle) in self._eager_logged:
            return
        self._eager_logged.add(id(bundle))
        logger.warning(
            "yue2_nar_trt_skipped frames={} cond_tokens={} batch={} profiles={} error={!r} using=eager",
            bundle.frames, bundle.cond_tokens, batch, self.profiles, error,
        )

    def _bound(self, bundle, batch: int, profile: int = 0) -> _BoundExecution:
        key = id(bundle)
        entry = self._contexts.get(key)
        if entry is None or entry[0]() is not bundle:
            ref = weakref.ref(bundle, lambda _r, k=key: self._contexts.pop(k, None))
            entry = (ref, {})
            self._contexts[key] = entry
            while len(self._contexts) > self.max_bundles:
                self._contexts.popitem(last=False)
        self._contexts.move_to_end(key)
        by_batch = entry[1]
        if batch not in by_batch:
            by_batch[batch] = _BoundExecution(self.engine, bundle, batch, self.device, profile)
        return by_batch[batch]

    def release(self, bundle) -> None:
        """Drop the contexts bound to ``bundle`` (before freeing its KV)."""
        self._contexts.pop(id(bundle), None)


# ---- VAE window decode -------------------------------------------------------


def window_plan(start_frame: int, n_frames: int, total_frames: int) -> List[Tuple[int, int, int]]:
    """How to decode latent frames ``[start, start+n)`` with fixed
    37-frame windows: ``[(ctx_start, core_offset, core_len), ...]``.

    Each window keeps up to 5 core frames. Interior windows put the core
    at offset 16 (16-frame margins both sides); a window that would run
    past a song boundary is clamped into ``[0, T-37]`` and its core moves
    inside it, so a boundary window sees the same edge padding the full
    decode sees. Requires ``T >= 37``.
    """
    if total_frames < VAE_CTX_FRAMES:
        raise ValueError(f"song of {total_frames} frames is shorter than the {VAE_CTX_FRAMES}-frame window")
    start = max(0, min(int(start_frame), total_frames))
    end = min(total_frames, start + max(0, int(n_frames)))
    plan = []
    frame = start
    while frame < end:
        core_len = min(VAE_CORE_FRAMES, end - frame)
        ctx = min(max(0, frame - VAE_MARGIN_FRAMES), total_frames - VAE_CTX_FRAMES)
        plan.append((ctx, frame - ctx, core_len))
        frame += core_len
    return plan


def decode_span(decode_ctx: Callable, latent_btc: torch.Tensor, start_frame: int,
                n_frames: int) -> torch.Tensor:
    """Audio ``[2, n_frames*1920]`` for latent frames ``[start, start+n)``
    through a fixed-window decoder ``decode_ctx([1,64,37]) -> [1,2,S]``.
    Samples past the song's decoded end are zero (the full decode is
    ``1920*T - 64`` samples long)."""
    total = latent_btc.shape[1]
    pieces = []
    for ctx, offset, core_len in window_plan(start_frame, n_frames, total):
        window = latent_btc[0, ctx:ctx + VAE_CTX_FRAMES].T[None].float().contiguous()
        audio = decode_ctx(window)[0]
        lo = offset * SAMPLES_PER_FRAME
        piece = audio[:, lo:lo + core_len * SAMPLES_PER_FRAME]
        if piece.shape[-1] < core_len * SAMPLES_PER_FRAME:
            piece = torch.nn.functional.pad(piece, (0, core_len * SAMPLES_PER_FRAME - piece.shape[-1]))
        pieces.append(piece.clone())
    want = int(n_frames) * SAMPLES_PER_FRAME
    out = torch.cat(pieces, dim=-1) if pieces else latent_btc.new_zeros(2, 0, dtype=torch.float32)
    if out.shape[-1] < want:
        out = torch.nn.functional.pad(out, (0, want - out.shape[-1]))
    return out


class TRTWindowVAE:
    """The 37-frame FP32 VAE window engine as ``decode_ctx``."""

    def __init__(self, path: Path, device: str = "cuda"):
        self.engine = _deserialize(Path(path))
        self.context = self.engine.create_execution_context()
        self.input = torch.zeros(1, 64, VAE_CTX_FRAMES, device=device, dtype=torch.float32)
        self.output = torch.empty(
            tuple(self.context.get_tensor_shape("audio")), device=device, dtype=torch.float32,
        )
        self.context.set_tensor_address("latent", self.input.data_ptr())
        self.context.set_tensor_address("audio", self.output.data_ptr())

    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        self.input.copy_(window)
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("yue2 VAE TensorRT execution failed")
        return self.output


def frames_for_seconds(seconds: float, rate_hz: float = 25.0) -> int:
    """Whole latent frames covering ``seconds``, rounded up to whole
    5-frame window cores."""
    frames = max(1, math.ceil(seconds * rate_hz - 1e-9))
    return VAE_CORE_FRAMES * math.ceil(frames / VAE_CORE_FRAMES)
