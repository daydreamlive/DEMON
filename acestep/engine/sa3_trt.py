"""TensorRT runtimes for the SA3 family: the medium DiT engine and the
SAME-L windowed decoder, wrapped for the production streaming stack.

The engines live under ``<MODELS_DIR>/sa3/trt_engines/`` (builder:
``python -m acestep.engine.trt.sa3_build``, which holds the same shape
as the ACE-Step builder and wraps Stability's OFFICIAL ONNX exports
from ``stabilityai/stable-audio-3-optimized``, so the graphs are
upstream's, not a hand export). Facts the wrappers encode, measured by
the spike benchmarks (``sa3_bench_medium_dit_trt.py``,
``sa3_medium_window_trt_benchmark.py``, 5090):

* **The DiT engine is BATCH-1** — every profile fixes dim 0 at 1 — with
  raw-conditioning inputs: ``x(1,256,L)``, ``t(1,)``,
  ``t5_hidden(1,256,768)``, ``t5_mask(1,256)``, ``seconds_total(1,)``,
  ``local_add_cond(1,257,L)`` → ``velocity``; fp32 IO, fp16-mixed
  internals (upstream's pre-surgered ``dit_fp16mixed.onnx`` built
  STRONGLY_TYPED — per-step cos ≥ 0.9998 vs eager on real cond; the
  early BF16-flag builds drifted to cos 0.8-0.97 and are retired).
  The engine has no ``padding_mask`` input: every latent frame is
  treated as valid, so at durations whose padded window has masked
  tail frames the eager path's masked attention differs slightly
  (cos ~0.991-0.999/step; see ``sa3_trt_dit_cond_parity.py``).
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
import os
import re
import sys
import threading
from pathlib import Path
from typing import Optional

import torch

from acestep import paths
from acestep.engine.obs import logger
from acestep.engine.sa3_helpers import SA3_SAME_L_PLUGIN_REVISION, sa3_vendor_dir

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
# The fp8-trunk DiT engine: same ranged-profile naming as the fp16mixed
# engine with an ``_fp8`` marker (``sa3_m_dit_fp8_l{min}_{opt}_{max}``).
# It runs ~1.8x faster per step than fp16mixed at compounded-euler cos
# ~0.976 vs it; under the production pingpong sampler it yields an
# equally-good but different take (the chaotic early steps amplify any
# non-bit-identical perturbation). Built by ``acestep.engine.trt.sa3_build``
# alongside the fp16mixed engine; discovery prefers it under the
# ``tensorrt`` accel path when one covers the window, falling back to
# fp16mixed otherwise. No env opt-in: selection rides the standard accel
# param like every other engine.
_DIT_FP8_DIR_RE = re.compile(
    r"^(?P<prefix>.+_dit)_fp8_l(?P<lo>\d+)_(?P<opt>\d+)_(?P<hi>\d+)$"
)
# Refittable fp16mixed DiT engines (BuilderFlag.REFIT; built by
# ``sa3_build --refit``): ``sa3_m_dit_refit_l{min}_{opt}_{max}``. The
# LoRA refit path (notes/SA3_LORA_PLAN.md D6b) mutates these in place,
# so they are EXCLUDED from the shared deserialization cache below —
# every consumer gets an exclusively-owned instance that dies with its
# session (which is also the base-weight rollback guarantee: a mutated
# engine can never be handed to a later session).
_DIT_REFIT_DIR_RE = re.compile(
    r"^(?P<prefix>.+_dit)_refit_l(?P<lo>\d+)_(?P<opt>\d+)_(?P<hi>\d+)$"
)
# fp16mixed DiT engines WITH the activation-steering input
# (``sa3_build --steer``; graph surgery in
# acestep.engine.trt.sa3_steering_onnx): ``sa3_m_dit_steer_l{min}_{opt}_{max}``.
# Same precision and graph as the plain fp16mixed engine plus one additive
# input; selected when the session asks for steering (it has packs).
_DIT_STEER_DIR_RE = re.compile(
    r"^(?P<prefix>.+_dit)_steer_l(?P<lo>\d+)_(?P<opt>\d+)_(?P<hi>\d+)$"
)
# fp16mixed DiT engines with the steering input at every block's
# cross-attention OUTPUT (``sa3_build --steer-cross-attn``; input
# ``steering_xattn``): ``sa3_m_dit_steerxa_l{min}_{opt}_{max}``. TADA's
# hook point (arXiv 2602.11910); selected when the session's packs target
# ``cross_attn_output``.
_DIT_STEERXA_DIR_RE = re.compile(
    r"^(?P<prefix>.+_dit)_steerxa_l(?P<lo>\d+)_(?P<opt>\d+)_(?P<hi>\d+)$"
)
#: Engine steering input name -> the steering layout hook it implements.
STEERING_INPUT_HOOKS = {
    "steering": "post_block_residual",
    "steering_xattn": "cross_attn_output",
}
_SAME_L_DIR_RE = re.compile(
    r"^same_l_decode_window_(?P<tag>[a-z0-9_]+)_t"
    r"(?P<lo>\d+)_(?P<opt>\d+)_(?P<hi>\d+)$"
)

# Deserialized-engine process cache. Engines are immutable post-load and
# support multiple execution contexts, so sharing one deserialization
# across sessions is safe; the per-wrapper state is the context+buffers.
# The IMMUTABILITY ASSUMPTION is the cache's contract: refittable
# engines (see _DIT_REFIT_DIR_RE) must never enter it — they are
# deserialized per-session with exclusive ownership instead.
_ENGINE_CACHE: dict = {}
_ENGINE_CACHE_LOCK = threading.Lock()
_SAME_PLUGIN_REGISTERED = False


def same_l_plugin_build_tag() -> str:
    """Identity of the plugin implementation compiled into a SAME-L engine.

    The upstream plugin is part of the serialized TensorRT engine, so changing
    the vendored source revision or its selected AOT backend requires a new
    engine even when the ONNX graph is unchanged.
    """
    requested_plugin = os.environ.get("SA3_SWA_PLUGIN", "aot").strip().lower()
    plugin = "jit" if requested_plugin == "jit" else "aot"
    if plugin == "jit":
        implementation = "jit"
    else:
        requested_backend = os.environ.get("SA3_SWA_AOT", "mma").strip().lower()
        implementation = "mma" if requested_backend == "mma" else "ptx"
    return f"{plugin}_{implementation}_v{SA3_SAME_L_PLUGIN_REVISION[:12]}"


def trt_engines_dir() -> Path:
    return paths.models_dir() / "sa3" / "trt_engines"


def is_refittable_engine_path(path: Path) -> bool:
    """Whether ``path`` names a refit-built DiT engine (by the naming
    marker ``sa3_build --refit`` stamps)."""
    return _DIT_REFIT_DIR_RE.match(Path(path).parent.name) is not None


def _deserialize_engine(path: Path, *, exclusive: bool = False):
    import tensorrt as trt

    exclusive = exclusive or is_refittable_engine_path(path)
    if exclusive:
        # Refittable (or explicitly exclusive) engines bypass the cache:
        # in-place refit would otherwise mutate every consumer and
        # outlive the session that made it. The load cost (~seconds for
        # the 2.8 GB DiT) is the accepted price of exclusive ownership.
        logger.info(
            "sa3_trt_engine_load path={} size_gb={:.1f} ownership=exclusive",
            path, path.stat().st_size / 1e9,
        )
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize TRT engine {path}")
        return engine

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
            "stable_audio_3 tree must include optimized/tensorRT/scripts. "
            "Run `uv run demon-setup` to fetch the pinned SA3 source."
        )
    # Append, not prepend: this dir also exposes bare modules (sa3_trt,
    # sa3_trt_core, ...) that would shadow if placed ahead of DEMON's path.
    if str(plugin_dir) not in sys.path:
        sys.path.append(str(plugin_dir))
    import diff_attn_nocast_plugin  # noqa: F401  (registers on import)

    _SAME_PLUGIN_REGISTERED = True


# ---------------------------------------------------------------------------
# Engine discovery
# ---------------------------------------------------------------------------


def find_dit_engine(
    model_id: str, latent_frames: int, *, want_refittable: bool = False,
    want_steering: "bool | str" = False,
) -> Optional[Path]:
    """Smallest-profile built DiT engine covering ``latent_frames`` for
    ``model_id``'s weights, or None (caller falls back to eager).

    ``want_refittable`` is the LoRA-session preference (plan D6b):

    * A covering refit-built engine wins when one exists — LoRA strength
      changes then run as in-place refits instead of the eager-DiT swap.
    * Otherwise selection prefers **fp16mixed over fp8** (D6c: FP8 refit
      on Stability's graph is its own unproven investigation; until then
      a LoRA session must not land on an engine the refit path can't
      serve at higher fidelity than the fp16mixed one), logged, and the
      interim eager swap covers actual enables.

    Without it, selection is unchanged: fp8 preferred when covering,
    else fp16mixed; refit-built engines are ignored (their refit
    support costs a little optimization freedom, and non-LoRA sessions
    shouldn't pay it).

    ``want_steering`` (the session has steering packs): a covering
    steering-input engine (``sa3_m_dit_steer_*``) wins over fp8 and the
    plain fp16mixed engine, since only it can apply steering on TRT. The
    LoRA preference above still takes precedence (refit engines carry no
    steering input; steering then no-ops on TRT, logged). Steering
    engines are never chosen otherwise.

    ``want_steering`` may also name the hook point the session's packs
    target: ``"post_block_residual"`` (same as True) or
    ``"cross_attn_output"``, which selects a covering
    ``sa3_m_dit_steerxa_*`` engine instead.
    """
    if want_steering is True:
        want_steering = "post_block_residual"
    prefix = DIT_ENGINE_PREFIX.get(model_id)
    base = trt_engines_dir()
    if prefix is None or not base.is_dir():
        return None
    best = None        # smallest-covering fp16mixed engine
    best_fp8 = None    # smallest-covering fp8 engine
    best_refit = None  # smallest-covering refit-built engine
    best_steer = None  # smallest-covering steering-input engine
    best_steerxa = None  # smallest-covering cross-attn steering engine
    for sub in base.iterdir():
        f = sub / f"{sub.name}.trt"
        if not f.is_file():
            continue
        mx = _DIT_STEERXA_DIR_RE.match(sub.name)
        if mx and mx.group("prefix") == prefix:
            lo, hi = int(mx.group("lo")), int(mx.group("hi"))
            if lo <= latent_frames <= hi and (best_steerxa is None or hi < best_steerxa[0]):
                best_steerxa = (hi, f)
            continue
        ms = _DIT_STEER_DIR_RE.match(sub.name)
        if ms and ms.group("prefix") == prefix:
            lo, hi = int(ms.group("lo")), int(ms.group("hi"))
            if lo <= latent_frames <= hi and (best_steer is None or hi < best_steer[0]):
                best_steer = (hi, f)
            continue
        mr = _DIT_REFIT_DIR_RE.match(sub.name)
        if mr and mr.group("prefix") == prefix:
            lo, hi = int(mr.group("lo")), int(mr.group("hi"))
            if lo <= latent_frames <= hi and (best_refit is None or hi < best_refit[0]):
                best_refit = (hi, f)
            continue
        mf = _DIT_FP8_DIR_RE.match(sub.name)
        if mf and mf.group("prefix") == prefix:
            lo, hi = int(mf.group("lo")), int(mf.group("hi"))
            if lo <= latent_frames <= hi and (best_fp8 is None or hi < best_fp8[0]):
                best_fp8 = (hi, f)
            continue
        m = _DIT_DIR_RE.match(sub.name)
        if m and m.group("prefix") == prefix:
            lo, hi = int(m.group("lo")), int(m.group("hi"))
            if lo <= latent_frames <= hi and (best is None or hi < best[0]):
                best = (hi, f)
    if want_refittable:
        if best_refit is not None:
            logger.info(
                "sa3_dit_refit_selected engine={} latent_frames={}",
                best_refit[1].parent.name, latent_frames,
            )
            return best_refit[1]
        if best_fp8 is not None and best is not None:
            logger.info(
                "sa3_dit_fp8_skipped_for_lora engine={} using={} "
                "reason=fp8_refit_unproven",
                best_fp8[1].parent.name, best[1].parent.name,
            )
        return best[1] if best else (best_fp8[1] if best_fp8 else None)
    if want_steering == "cross_attn_output":
        if best_steerxa is not None:
            logger.info(
                "sa3_dit_steer_selected engine={} hook=cross_attn_output "
                "latent_frames={}",
                best_steerxa[1].parent.name, latent_frames,
            )
            return best_steerxa[1]
        logger.warning(
            "sa3_dit_steer_unavailable latent_frames={} hook=cross_attn_output "
            "reason=no_covering_steerxa_engine (build: python -m "
            "acestep.engine.trt.sa3_build --dit --min-latents 1 --opt-latents "
            "646 --max-latents 646 --steer-cross-attn); steering will no-op on TRT",
            latent_frames,
        )
        want_steering = False
    if want_steering and best_steer is not None:
        logger.info(
            "sa3_dit_steer_selected engine={} latent_frames={}",
            best_steer[1].parent.name, latent_frames,
        )
        return best_steer[1]
    if want_steering:
        logger.warning(
            "sa3_dit_steer_unavailable latent_frames={} reason=no_covering_"
            "steer_engine (build: python -m acestep.engine.trt.sa3_build "
            "--all --dit-only --steer); steering will no-op on TRT",
            latent_frames,
        )
    # fp8 is ~1.8x faster; prefer it when one covers the window.
    if best_fp8 is not None:
        logger.info(
            "sa3_dit_fp8_selected engine={} latent_frames={}",
            best_fp8[1].parent.name, latent_frames,
        )
        return best_fp8[1]
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
    # Both the fp16mixed and fp8 engines can serve this model_id (see
    # find_dit_engine), so the cap must consider either — an fp8-only
    # install would otherwise report no cap and skip the clamp.
    his = []
    for sub in base.iterdir():
        m = (
            _DIT_REFIT_DIR_RE.match(sub.name)
            or _DIT_STEER_DIR_RE.match(sub.name)
            or _DIT_STEERXA_DIR_RE.match(sub.name)
            or _DIT_DIR_RE.match(sub.name)
            or _DIT_FP8_DIR_RE.match(sub.name)
        )
        if (
            m
            and m.group("prefix") == prefix
            and (sub / f"{sub.name}.trt").is_file()
        ):
            his.append(int(m.group("hi")))
    return max(his) if his else None


def find_same_l_window_engine() -> Optional[tuple]:
    """``(path, min_t, max_t)`` of the built SAME-L window decoder, or
    None (caller falls back to eager windowed decode)."""
    base = trt_engines_dir()
    if not base.is_dir():
        return None
    expected_tag = same_l_plugin_build_tag()
    for sub in base.iterdir():
        m = _SAME_L_DIR_RE.match(sub.name)
        if not m or m.group("tag") != expected_tag:
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

    The engine runs on a private, non-default torch stream. Torch
    streams are created ``cudaStreamNonBlocking``, so they do NOT
    implicitly order after work on the caller's current stream (where
    the input ``copy_`` calls run); :meth:`step_bundle` therefore issues
    an explicit ``wait_stream`` before the launch so the engine can't
    read a half-written input buffer. This is the same wait_stream
    ordering contract ``acestep.engine.trt.runtime`` documents.
    """

    trt_batch1 = True

    def __init__(self, engine_path: Path, *, latent_frames: int, seconds_total: float):
        engine = _deserialize_engine(engine_path)
        # Exposed for the LoRA refit mirror: on a refit-built engine
        # (exclusively owned, never cached) the mirror attaches an
        # IRefitter to this same instance. None of the wrapper's own
        # state cares about refits — buffers and context stay valid
        # across refit_cuda_engine().
        self.engine = engine
        self.engine_path = Path(engine_path)
        self.refittable = is_refittable_engine_path(engine_path)
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
        # Activation-steering input (``sa3_m_dit_steer_*``: ``steering``
        # at the post-block residual; ``sa3_m_dit_steerxa_*``:
        # ``steering_xattn`` at the cross-attention output): static
        # [1, num_blocks, hidden]. ``steering_shape`` / ``steering_hook``
        # advertise it to SA3Adapter.steering_layout; None on engines
        # without one.
        io_names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
        self.steering_shape: Optional[tuple] = None
        self.steering_hook: Optional[str] = None
        self._steering_input: Optional[str] = None
        self._steering = None
        self._steering_dirty = False
        present = [n for n in STEERING_INPUT_HOOKS if n in io_names]
        if len(present) > 1:
            raise RuntimeError(f"SA3 engine carries several steering inputs {present}")
        if present:
            name = present[0]
            s_shape = tuple(engine.get_tensor_shape(name))
            if len(s_shape) != 3 or s_shape[0] != 1:
                raise RuntimeError(f"unexpected SA3 {name} input shape {s_shape}")
            self.steering_shape = (int(s_shape[1]), int(s_shape[2]))
            self.steering_hook = STEERING_INPUT_HOOKS[name]
            self._steering_input = name
        out_shape = tuple(self._ctx.get_tensor_shape("velocity"))

        dev = torch.device("cuda")
        self._x = torch.zeros(1, IO_CHANNELS, L, dtype=torch.float32, device=dev)
        self._t = torch.zeros(1, dtype=torch.float32, device=dev)
        self._t5_hidden = torch.zeros(1, T5_TOKENS, COND_DIM, dtype=torch.float32, device=dev)
        self._t5_mask = torch.zeros(1, T5_TOKENS, dtype=torch.float32, device=dev)
        self._seconds = torch.full((1,), float(seconds_total), dtype=torch.float32, device=dev)
        self._local_add = torch.zeros(1, 257, L, dtype=torch.float32, device=dev)
        self._velocity = torch.empty(out_shape, dtype=torch.float32, device=dev)
        if self.steering_shape is not None:
            self._steering = torch.zeros(
                1, *self.steering_shape, dtype=torch.float32, device=dev,
            )
            self._ctx.set_tensor_address(
                self._steering_input, self._steering.data_ptr(),
            )

        for name, buf in (
            ("x", self._x), ("t", self._t), ("t5_hidden", self._t5_hidden),
            ("t5_mask", self._t5_mask), ("seconds_total", self._seconds),
            ("local_add_cond", self._local_add), ("velocity", self._velocity),
        ):
            self._ctx.set_tensor_address(name, buf.data_ptr())

        # Strong ref to the currently-staged bundle (NOT its id()): an
        # id() key can stale-hit after the old bundle is GC'd and a fresh
        # one is allocated at the same address, silently skipping a
        # re-stage and running the previous prompt's conditioning. Holding
        # the object keeps its identity unique for as long as it's the key.
        self._staged_bundle = None
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
        if self._staged_bundle is bundle:
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
        self._staged_bundle = bundle

    @torch.no_grad()
    def step_bundle(
        self, x_1ct: torch.Tensor, t: float, bundle: dict,
        steering: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One velocity forward: SA3-native ``[1, 256, L]`` in and out.
        Returns the persistent output buffer — the caller must consume
        (copy/cast) it before the next step overwrites it.

        ``steering`` is ``[1, num_blocks, hidden]`` (None = zero shift);
        ignored by engines without the steering input. The bound buffer
        is only rewritten when steering is active or was last tick, so
        the no-steering path costs nothing."""
        if x_1ct.shape[-1] != self._L:
            raise ValueError(
                f"x latent frames {x_1ct.shape[-1]} != engine-bound L {self._L}"
            )
        self._stage_bundle(bundle)
        if self._steering is not None:
            if steering is not None:
                self._steering.copy_(steering.reshape(self._steering.shape))
                self._steering_dirty = True
            elif self._steering_dirty:
                self._steering.zero_()
                self._steering_dirty = False
        self._x.copy_(x_1ct.float())
        self._t[0] = float(t)
        # The copies above (and _stage_bundle's) ran on the caller's
        # current stream; a non-blocking torch stream would otherwise race
        # them, so make the launch wait for that stream first. Captured
        # BEFORE the context switch — inside it current_stream() is already
        # self._stream, which would make the wait a no-op.
        caller_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._stream):
            self._stream.wait_stream(caller_stream)
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
        # ``lat`` was produced on the caller's current stream; a non-blocking
        # torch stream would race it. Capture the caller stream before the
        # context switch (see SA3TRTDit.step_bundle).
        caller_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._stream):
            self._stream.wait_stream(caller_stream)
            ok = self._ctx.execute_async_v3(self._stream.cuda_stream)
        if not ok:
            raise RuntimeError("SA3 TRT SAME-L decode failed")
        self._stream.synchronize()
        out = self._out_buf
        if self._out_name == "pcm":
            # PCM-baked engine flavor: (1, N, 2) int scaled to int16 range.
            return (out[0].to(torch.float32).T / 32767.0).clamp(-1, 1)
        return out[0].float().clamp(-1, 1)
