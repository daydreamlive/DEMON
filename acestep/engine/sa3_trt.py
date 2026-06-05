"""TensorRT runtimes for the SA3 family: the medium DiT engine and the
SAME-L windowed decoder, wrapped for the production streaming stack.

The engines are the spike-built artifacts under
``<MODELS_DIR>/sa3/trt_engines/`` (build scripts:
``scripts/sa3/sa3_build_medium_dit_trt.py`` and
``sa3_build_same_l_window_trt.py``; both wrap Stability's OFFICIAL ONNX
exports from ``stabilityai/stable-audio-3-optimized``, so the graphs are
upstream's, not a hand export). Facts the wrappers encode, measured by
the spike benchmarks (``sa3_bench_medium_dit_trt.py``,
``sa3_medium_window_trt_benchmark.py``, 5090):

* **The DiT engine is BATCH-1** — every profile fixes dim 0 at 1 — with
  raw-conditioning inputs: ``x(1,256,L)``, ``t(1,)``,
  ``t5_hidden(1,256,768)``, ``t5_mask(1,256)``, ``seconds_total(1,)``,
  ``local_add_cond(1,257,L)`` → ``velocity``; fp32 IO, BF16 internals.
  The conditioner tail (padding_embedding + seconds_total Linear) is
  baked into the graph as constants, so it consumes the RAW T5Gemma
  hidden states — the prompt block of the torch ``cond_bundle``'s
  ``cross_attn_cond`` (768-dim; its trailing seconds_total token must
  be stripped) — plus the raw requested-duration scalar. ~11 ms/step at
  L≈324, ~17 ms at L=646 (eager torch: ~54 ms). The ring buffer's
  batched tick therefore LOOPS slots through the engine
  (:attr:`SA3TRTDit.trt_batch1`, consumed by
  :class:`~acestep.engine.sa3_adapter.SA3Adapter`).
* **The SAME-L window decoder** decodes ``latent(1,256,T)`` for T within
  its profile (built t32_56_96); sliding-window attention, so no
  chunk-phase snapping (``slice_align_latents=1``). The latent must be
  multiplied by ``pretransform.scale`` before the call (the spike's
  ``scale_mode="pretransform"``: rel_rms ~8e-3 vs eager full decode;
  the raw mode is wrong). Requires the ``samel::diff_attn_swa`` plugin
  (vendored ``optimized/tensorRT/scripts``; triton kernel) registered
  BEFORE deserialization. ~9-10 ms per ~1 s window at 2 s context.

Deserialized engines are process-cached (the DiT file is 2.8 GB, the
decoder 1.2 GB); every wrapper creates its own execution context so
concurrent sessions never share mutable TRT state. ``tensorrt`` imports
stay inside functions: this module must be importable on hosts without
TRT (engine discovery then simply reports nothing).
"""

from __future__ import annotations

import math
import re
import sys
import threading
from pathlib import Path
from typing import Optional

import torch

from acestep import paths
from acestep.engine.obs import logger
from acestep.engine.sa3_helpers import sa3_vendor_dir

IO_CHANNELS = 256
T5_TOKENS = 256
COND_DIM = 768
SAMPLES_PER_LATENT = 4096
SA3_SAMPLE_RATE = 44100

# Per-family DiT engine name prefixes: which engines can serve which
# model_id's weights. Only medium has built engines today; small runs
# real-time eager and has none.
DIT_ENGINE_PREFIX = {"medium": "sa3_m_dit"}

_DIT_DIR_RE = re.compile(r"^(?P<prefix>.+_dit)_l(?P<lo>\d+)_(?P<opt>\d+)_(?P<hi>\d+)$")
_SAME_L_DIR_RE = re.compile(r"^same_l_decode_window_t(?P<lo>\d+)_(?P<opt>\d+)_(?P<hi>\d+)$")

# Deserialized-engine process cache. Engines are immutable post-load and
# support multiple execution contexts, so sharing one deserialization
# across sessions is safe; the per-wrapper state is the context+buffers.
_ENGINE_CACHE: dict = {}
_ENGINE_CACHE_LOCK = threading.Lock()
_SAME_PLUGIN_REGISTERED = False


def trt_engines_dir() -> Path:
    return paths.models_dir() / "sa3" / "trt_engines"


def _deserialize_engine(path: Path):
    import tensorrt as trt

    with _ENGINE_CACHE_LOCK:
        engine = _ENGINE_CACHE.get(str(path))
        if engine is None:
            logger.info(
                "sa3_trt_engine_load path={} size_gb={:.1f}",
                path, path.stat().st_size / 1e9,
            )
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(path.read_bytes())
            if engine is None:
                raise RuntimeError(f"failed to deserialize TRT engine {path}")
            _ENGINE_CACHE[str(path)] = engine
        return engine


def _register_same_plugin() -> None:
    """Register ``samel::diff_attn_swa`` (idempotent). Must precede
    SAME-L engine deserialization or TRT can't resolve the node."""
    global _SAME_PLUGIN_REGISTERED
    if _SAME_PLUGIN_REGISTERED:
        return
    plugin_dir = sa3_vendor_dir() / "optimized" / "tensorRT" / "scripts"
    if not (plugin_dir / "diff_attn_nocast_plugin.py").is_file():
        raise ImportError(
            f"SAME-L TRT plugin not found at {plugin_dir}; the vendored "
            "stable_audio_3 tree must include optimized/tensorRT/scripts"
        )
    if str(plugin_dir) not in sys.path:
        sys.path.insert(0, str(plugin_dir))
    import diff_attn_nocast_plugin  # noqa: F401  (registers on import)

    _SAME_PLUGIN_REGISTERED = True


# ---------------------------------------------------------------------------
# Engine discovery
# ---------------------------------------------------------------------------


def find_dit_engine(model_id: str, latent_frames: int) -> Optional[Path]:
    """Smallest-profile built DiT engine covering ``latent_frames`` for
    ``model_id``'s weights, or None (caller falls back to eager)."""
    prefix = DIT_ENGINE_PREFIX.get(model_id)
    base = trt_engines_dir()
    if prefix is None or not base.is_dir():
        return None
    best = None
    for sub in base.iterdir():
        m = _DIT_DIR_RE.match(sub.name)
        if not m or m.group("prefix") != prefix:
            continue
        lo, hi = int(m.group("lo")), int(m.group("hi"))
        f = sub / f"{sub.name}.trt"
        if lo <= latent_frames <= hi and f.is_file():
            if best is None or hi < best[0]:
                best = (hi, f)
    return best[1] if best else None


def max_dit_engine_latents(model_id: str) -> Optional[int]:
    """Largest latent-frame count any built DiT engine for ``model_id``
    can serve, or None when no engine exists. Used by the session create
    path to clamp the requested duration onto the TRT fast path instead
    of silently landing on the ~5x-slower eager DiT."""
    prefix = DIT_ENGINE_PREFIX.get(model_id)
    base = trt_engines_dir()
    if prefix is None or not base.is_dir():
        return None
    his = [
        int(m.group("hi"))
        for sub in base.iterdir()
        if (m := _DIT_DIR_RE.match(sub.name))
        and m.group("prefix") == prefix
        and (sub / f"{sub.name}.trt").is_file()
    ]
    return max(his) if his else None


def find_same_l_window_engine() -> Optional[tuple]:
    """``(path, min_t, max_t)`` of the built SAME-L window decoder, or
    None (caller falls back to eager windowed decode)."""
    base = trt_engines_dir()
    if not base.is_dir():
        return None
    for sub in base.iterdir():
        m = _SAME_L_DIR_RE.match(sub.name)
        if not m:
            continue
        f = sub / f"{sub.name}.trt"
        if f.is_file():
            return f, int(m.group("lo")), int(m.group("hi"))
    return None


def _trt_dtype_to_torch(trt_mod, dtype):
    return {
        trt_mod.DataType.FLOAT: torch.float32,
        trt_mod.DataType.HALF: torch.float16,
        trt_mod.DataType.BF16: torch.bfloat16,
        trt_mod.DataType.INT32: torch.int32,
        trt_mod.DataType.INT64: torch.int64,
        trt_mod.DataType.BOOL: torch.bool,
        trt_mod.DataType.INT8: torch.int8,
        trt_mod.DataType.UINT8: torch.uint8,
    }[dtype]


# ---------------------------------------------------------------------------
# DiT engine wrapper
# ---------------------------------------------------------------------------


class SA3TRTDit:
    """One session's TRT DiT: fixed L and duration, per-slot stepping.

    ``trt_batch1`` tells :class:`~acestep.engine.sa3_adapter.SA3Adapter`
    to loop ring-buffer slots through :meth:`step_bundle` instead of one
    stacked torch forward. Input buffers are persistent and bound once
    (L and duration are fixed for the session lifetime); the cond bundle
    is re-staged only when its identity changes (per-prompt swap, or the
    old/new alternation while in-flight slots drain after one).

    The private stream is a default-constructed (blocking) torch stream,
    so it synchronizes with the legacy default stream the input ``copy_``
    calls run on — the same ordering contract the spike runner used.
    """

    trt_batch1 = True

    def __init__(self, engine_path: Path, *, latent_frames: int, seconds_total: float):
        engine = _deserialize_engine(engine_path)
        self._ctx = engine.create_execution_context()
        self._stream = torch.cuda.Stream()
        L = int(latent_frames)
        self._L = L

        self._ctx.set_input_shape("x", (1, IO_CHANNELS, L))
        self._ctx.set_input_shape("t", (1,))
        self._ctx.set_input_shape("t5_hidden", (1, T5_TOKENS, COND_DIM))
        self._ctx.set_input_shape("t5_mask", (1, T5_TOKENS))
        self._ctx.set_input_shape("seconds_total", (1,))
        self._ctx.set_input_shape("local_add_cond", (1, 257, L))
        out_shape = tuple(self._ctx.get_tensor_shape("velocity"))

        dev = torch.device("cuda")
        self._x = torch.zeros(1, IO_CHANNELS, L, dtype=torch.float32, device=dev)
        self._t = torch.zeros(1, dtype=torch.float32, device=dev)
        self._t5_hidden = torch.zeros(1, T5_TOKENS, COND_DIM, dtype=torch.float32, device=dev)
        self._t5_mask = torch.zeros(1, T5_TOKENS, dtype=torch.float32, device=dev)
        self._seconds = torch.full((1,), float(seconds_total), dtype=torch.float32, device=dev)
        self._local_add = torch.zeros(1, 257, L, dtype=torch.float32, device=dev)
        self._velocity = torch.empty(out_shape, dtype=torch.float32, device=dev)

        for name, buf in (
            ("x", self._x), ("t", self._t), ("t5_hidden", self._t5_hidden),
            ("t5_mask", self._t5_mask), ("seconds_total", self._seconds),
            ("local_add_cond", self._local_add), ("velocity", self._velocity),
        ):
            self._ctx.set_tensor_address(name, buf.data_ptr())

        self._bundle_key = None
        logger.info(
            "sa3_trt_dit_ready engine={} L={} seconds_total={:.1f}",
            engine_path.parent.name, L, seconds_total,
        )

    def _stage_bundle(self, bundle: dict) -> None:
        """Copy the torch cond bundle's raw pieces into the bound input
        buffers. ``cross_attn_cond`` is the ``cross_attention_cond_ids``
        concat ``["prompt", "seconds_total"]``: 256 max-length-padded
        T5Gemma tokens plus one trailing seconds token. The engine
        rebuilds the seconds token internally from its ``seconds_total``
        scalar input (the baked conditioner tail), so only the prompt
        block is staged here. ``local_add_cond`` is the same (1,257,L)
        concat the torch DiT consumes (zeros for the streaming cover
        task)."""
        if self._bundle_key == id(bundle):
            return
        ca = bundle["cross_attn_cond"]
        mask = bundle["cross_attn_mask"].reshape(1, -1)
        if ca.shape[-1] != COND_DIM:
            raise ValueError(
                f"cond cross_attn dim {ca.shape[-1]} != engine COND_DIM "
                f"{COND_DIM}; this engine does not match the loaded model"
            )
        n_tok = ca.shape[1]
        if n_tok == T5_TOKENS + 1:
            # Drop the trailing seconds_total token — the engine appends
            # its own from the bound seconds scalar.
            ca = ca[:, :T5_TOKENS]
            mask = mask[:, :T5_TOKENS]
            n_tok = T5_TOKENS
        if n_tok > T5_TOKENS:
            raise ValueError(
                f"cond has {n_tok} tokens > engine max {T5_TOKENS}"
            )
        self._t5_hidden.zero_()
        self._t5_hidden[:, :n_tok].copy_(ca.float())
        self._t5_mask.zero_()
        self._t5_mask[:, :n_tok].copy_(mask.float())
        lac = bundle.get("local_add_cond")
        if lac is None:
            self._local_add.zero_()
        else:
            if lac.shape[-1] != self._L:
                raise ValueError(
                    f"local_add_cond L {lac.shape[-1]} != engine-bound L {self._L}"
                )
            self._local_add.copy_(lac.float())
        self._bundle_key = id(bundle)

    @torch.no_grad()
    def step_bundle(self, x_1ct: torch.Tensor, t: float, bundle: dict) -> torch.Tensor:
        """One velocity forward: SA3-native ``[1, 256, L]`` in and out.
        Returns the persistent output buffer — the caller must consume
        (copy/cast) it before the next step overwrites it."""
        if x_1ct.shape[-1] != self._L:
            raise ValueError(
                f"x latent frames {x_1ct.shape[-1]} != engine-bound L {self._L}"
            )
        self._stage_bundle(bundle)
        self._x.copy_(x_1ct.float())
        self._t[0] = float(t)
        with torch.cuda.stream(self._stream):
            ok = self._ctx.execute_async_v3(self._stream.cuda_stream)
        if not ok:
            raise RuntimeError("SA3 TRT DiT step failed")
        self._stream.synchronize()
        return self._velocity


# ---------------------------------------------------------------------------
# SAME-L window decoder wrapper
# ---------------------------------------------------------------------------


class SameLWindowTRTDecoder:
    """The spike's SAME-L window decoder, productionized: latent
    ``[1, 256, T]`` (already pretransform-scaled by the caller) →
    ``[C, T*4096]`` float audio at 44.1 kHz."""

    def __init__(self, engine_path: Path):
        import tensorrt as trt

        _register_same_plugin()
        engine = _deserialize_engine(engine_path)
        self._ctx = engine.create_execution_context()
        self._stream = torch.cuda.Stream()
        self._in_dtype = _trt_dtype_to_torch(trt, engine.get_tensor_dtype("latent"))
        names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
        self._out_name = "pcm" if "pcm" in names else "audio"
        self._out_dtype = _trt_dtype_to_torch(trt, engine.get_tensor_dtype(self._out_name))
        self._out_buf: Optional[torch.Tensor] = None
        logger.info("sa3_trt_same_l_ready engine={}", engine_path.parent.name)

    @torch.no_grad()
    def decode(self, latent_1ct: torch.Tensor) -> torch.Tensor:
        lat = latent_1ct.to(device="cuda", dtype=self._in_dtype).contiguous()
        if not self._ctx.set_input_shape("latent", tuple(lat.shape)):
            raise RuntimeError(f"TRT rejected latent shape {tuple(lat.shape)}")
        out_shape = tuple(self._ctx.get_tensor_shape(self._out_name))
        if self._out_buf is None or tuple(self._out_buf.shape) != out_shape:
            self._out_buf = torch.empty(out_shape, dtype=self._out_dtype, device="cuda")
        self._ctx.set_tensor_address("latent", lat.data_ptr())
        self._ctx.set_tensor_address(self._out_name, self._out_buf.data_ptr())
        with torch.cuda.stream(self._stream):
            ok = self._ctx.execute_async_v3(self._stream.cuda_stream)
        if not ok:
            raise RuntimeError("SA3 TRT SAME-L decode failed")
        self._stream.synchronize()
        out = self._out_buf
        if self._out_name == "pcm":
            # PCM-baked engine flavor: (1, N, 2) int scaled to int16 range.
            return (out[0].to(torch.float32).T / 32767.0).clamp(-1, 1)
        return out[0].float().clamp(-1, 1)


def trt_duration_cap_s(model_id: str, *, padding_s: float) -> Optional[float]:
    """Largest requested duration whose padded latent window still fits
    a built DiT engine for ``model_id`` (None = no engine)."""
    max_l = max_dit_engine_latents(model_id)
    if max_l is None:
        return None
    total_s = max_l * SAMPLES_PER_LATENT / SA3_SAMPLE_RATE
    return math.floor((total_s - padding_s) * 10.0) / 10.0
