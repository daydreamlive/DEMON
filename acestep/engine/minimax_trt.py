"""TensorRT runtime for the MiniMax-Music3 flow-matching DiT.

The peer of :mod:`acestep.engine.sa3_trt`, and a much smaller one: this
family needs no plugins. The DiT is self-attention only (no
cross-attention, no KV cache, no custom ops), so a stock ONNX export
compiles with the generic builder in
:mod:`acestep.engine.trt.minimax_build` and nothing has to be
registered before deserialization.

Engines live under ``<MODELS_DIR>/minimax/trt_engines/`` (override with
``DEMON_MINIMAX_TRT_DIR``) in the house layout
``<dir>/<name>/<name>.trt``. The name carries the whole contract::

    minimax_dit_{precision}_b{min_b}_{max_b}_l{min_l}_{opt_l}_{max_l}

``precision`` is part of the *name*, not a sidecar field, deliberately.
An engine whose precision the caller did not ask for must be
undiscoverable rather than loadable-and-wrong: the fp16 and fp32
engines have byte-identical IO signatures (fp32 in, fp32 out; the
half-precision trunk is entirely internal), so nothing downstream could
tell them apart at runtime if discovery let the wrong one through.

IO contract, identical across precisions:

* ``hidden_states``          ``(B, 128, L)``   fp32
* ``timestep``               ``(B,)``          fp32, MiniMax time: 0 noise -> 1 data
* ``encoder_hidden_states``  ``(B, L, 2048)``  fp32
* ``velocity``               ``(B, 128, L)``   fp32

**The engines are batch-1.** Not a policy choice: ``torch.export``
refuses to keep dim 0 symbolic through the ``matmul`` decomposition,
which guards ``batch != 1`` when it folds a 3-D x 2-D matmul (fires at
``proj_out``). A batch-dynamic export is therefore only possible with
``min_batch >= 2``, which cannot serve production. So the streaming
path loops ring-buffer slots through :meth:`MiniMaxTRTDit.step_bundle`
(:attr:`MiniMaxTRTDit.trt_batch1`, consumed by
:class:`~acestep.engine.minimax_adapter.MiniMaxAdapter`), exactly as
SA3 does. ``b2_4`` engines exist only for benchmarking and are excluded
from discovery.

Three things in here are scar tissue, not style:

* Buffers are allocated at the **engine-declared** dtype
  (``engine.get_tensor_dtype``), never at a dtype this module assumed.
  TRT reads the buffer as raw bytes through the declared dtype, so a
  fp32 buffer bound to a half input is silently garbage, not an error.
* Deserialization goes through polygraphy's ``engine_from_bytes``,
  never ``trt.Runtime().deserialize_cuda_engine``, which corrupts
  process-global TRT state on Blackwell.
* Execution rides the **shared** polygraphy stream. A per-wrapper
  ``torch.cuda.Stream`` costs 14x on Blackwell once several engines
  coexist in one process. The shared stream is wrapped in a
  ``torch.cuda.ExternalStream`` purely so the launch can
  ``wait_stream`` on the caller: the input ``copy_`` calls run on the
  caller's stream and would otherwise race the engine.

``tensorrt`` and ``polygraphy`` imports stay inside functions: this
module must import on hosts without TRT, where discovery simply reports
nothing and :meth:`MiniMaxContext.make_dit` falls back to eager.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Optional

import torch

from acestep.engine.minimax_helpers import minimax_root
from acestep.engine.obs import logger

MINIMAX_LATENT_CHANNELS = 128
MINIMAX_COND_DIM = 2048

#: Supported precisions, in discovery-preference order. fp16 first: it is
#: the faster engine and the one the parity gate is written against; fp32
#: is the known-good control, kept discoverable so a host can pin it.
PRECISIONS = ("fp16", "fp32")

_DIT_DIR_RE = re.compile(
    r"^minimax_dit_(?P<precision>fp16|fp32)"
    r"_b(?P<blo>\d+)_(?P<bhi>\d+)"
    r"_l(?P<lo>\d+)_(?P<opt>\d+)_(?P<hi>\d+)$"
)

# Deserialized-engine process cache. Engines are immutable post-load and
# support multiple execution contexts, so one deserialization can back
# many sessions; the per-wrapper state is the context plus its buffers.
# (No refit path exists for this family yet; when one lands, refittable
# engines must bypass this cache the way SA3's do.)
_ENGINE_CACHE: dict = {}
_ENGINE_CACHE_LOCK = threading.Lock()


def trt_engines_dir() -> Path:
    """Where MiniMax engines live. ``DEMON_MINIMAX_TRT_DIR`` overrides:
    these engines are 5-10 GB each and the models drive is not always
    the roomy one."""
    override = os.environ.get("DEMON_MINIMAX_TRT_DIR")
    if override:
        return Path(override)
    return minimax_root() / "trt_engines"


def engine_dir_name(
    *,
    precision: str,
    min_batch: int,
    max_batch: int,
    min_latents: int,
    opt_latents: int,
    max_latents: int,
) -> str:
    """The single place the engine directory name is spelled. The
    builder writes it and :func:`find_dit_engine_path` parses it back."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    return (
        f"minimax_dit_{precision}_b{min_batch}_{max_batch}"
        f"_l{min_latents}_{opt_latents}_{max_latents}"
    )


def list_dit_engines() -> list[dict]:
    """Every built DiT engine, as parsed name fields plus ``path``.
    Unparseable directories are ignored, which is what makes the
    precision tag a hard gate rather than a hint."""
    base = trt_engines_dir()
    if not base.is_dir():
        return []
    found = []
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        match = _DIT_DIR_RE.match(sub.name)
        if not match:
            continue
        engine_file = sub / f"{sub.name}.trt"
        if not engine_file.is_file():
            continue
        found.append({
            "name": sub.name,
            "path": engine_file,
            "precision": match.group("precision"),
            "min_batch": int(match.group("blo")),
            "max_batch": int(match.group("bhi")),
            "min_latents": int(match.group("lo")),
            "opt_latents": int(match.group("opt")),
            "max_latents": int(match.group("hi")),
        })
    return found


def find_dit_engine_path(
    latent_frames: int,
    *,
    precision: Optional[str] = None,
    batch: int = 1,
) -> Optional[Path]:
    """Smallest-profile engine covering ``latent_frames`` at ``batch``,
    or None.

    ``precision`` pins one recipe. ``None`` means "the operator did not
    say", which resolves to ``DEMON_MINIMAX_TRT_PRECISION`` if set and
    otherwise walks :data:`PRECISIONS` in order: fp16, then the fp32
    control. A pinned precision NEVER falls through to the other one:
    an operator who asked for the control and silently got the fast
    engine has no control.
    """
    if precision is None:
        precision = os.environ.get("DEMON_MINIMAX_TRT_PRECISION") or None
    if precision is not None and precision not in PRECISIONS:
        raise ValueError(
            f"precision must be one of {PRECISIONS} or None, got {precision!r}"
        )
    wanted = (precision,) if precision is not None else PRECISIONS

    engines = list_dit_engines()
    for want in wanted:
        best = None
        for entry in engines:
            if entry["precision"] != want:
                continue
            if not entry["min_batch"] <= batch <= entry["max_batch"]:
                continue
            if not entry["min_latents"] <= latent_frames <= entry["max_latents"]:
                continue
            # Smallest covering max_latents wins: a tighter profile means
            # less activation workspace and tactics tuned nearer the shape.
            if best is None or entry["max_latents"] < best["max_latents"]:
                best = entry
        if best is not None:
            return best["path"]
    return None


def find_dit_engine(
    latent_frames: int,
    *,
    precision: Optional[str] = None,
) -> Optional["MiniMaxTRTDit"]:
    """The DiT replacement :meth:`MiniMaxContext.make_dit` asks for: a
    ready-to-step wrapper, or None when no engine covers this shape (the
    caller then logs and falls back to eager).

    Returns the runtime object rather than a path because that is what
    the seam consumes: ``make_dit`` hands its result straight to
    :class:`~acestep.engine.minimax_adapter.MiniMaxAdapter` as ``dit``.
    """
    path = find_dit_engine_path(latent_frames, precision=precision, batch=1)
    if path is None:
        return None
    return MiniMaxTRTDit(path, latent_frames=latent_frames)


def _deserialize_engine(path: Path):
    """Load (and process-cache) a serialized engine.

    polygraphy's ``engine_from_bytes``, not ``trt.Runtime``: the latter
    corrupts process-global TRT state on Blackwell when several engines
    are deserialized in one process.
    """
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes

    key = str(path)
    with _ENGINE_CACHE_LOCK:
        engine = _ENGINE_CACHE.get(key)
        if engine is None:
            logger.info(
                "minimax_trt_engine_load path={} size_gb={:.2f}",
                path, path.stat().st_size / 1e9,
            )
            engine = engine_from_bytes(bytes_from_path(str(path)))
            if engine is None:
                raise RuntimeError(f"failed to deserialize TRT engine {path}")
            _ENGINE_CACHE[key] = engine
        return engine


def _trt_dtype_to_torch(name: str, dtype) -> torch.dtype:
    import tensorrt as trt

    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.INT8: torch.int8,
        trt.DataType.UINT8: torch.uint8,
    }
    if hasattr(trt.DataType, "BF16"):
        mapping[trt.DataType.BF16] = torch.bfloat16
    try:
        return mapping[dtype]
    except KeyError:
        raise RuntimeError(
            f"TRT tensor {name!r} has dtype {dtype} with no torch equivalent; "
            "allocating it as anything else would silently produce garbage"
        ) from None


class MiniMaxTRTDit:
    """One session's TRT DiT: fixed latent length, per-slot stepping.

    ``trt_batch1`` tells
    :class:`~acestep.engine.minimax_adapter.MiniMaxAdapter` to loop the
    ring buffer's slots through :meth:`step_bundle` instead of issuing
    one stacked torch forward.

    Input buffers are persistent and bound once; the latent length is
    fixed for the session lifetime. The conditioning is re-staged only
    when its identity changes, which for this family means a prompt
    swap: MiniMax conditioning is a *captured* artifact (see
    :class:`~acestep.engine.minimax_context.MiniMaxContext`), so it is
    the same 5.6 MB tensor for thousands of consecutive steps and
    re-uploading it per step would be pure waste.
    """

    trt_batch1 = True

    def __init__(self, engine_path, *, latent_frames: int, device="cuda"):
        engine_path = Path(engine_path)
        match = _DIT_DIR_RE.match(engine_path.parent.name)
        if match is None:
            raise ValueError(
                f"{engine_path.parent.name!r} is not a MiniMax DiT engine "
                "directory name; the precision tag is part of the contract "
                "and an unparseable name cannot be trusted to be the recipe "
                "the caller asked for"
            )
        self.engine_path = engine_path
        self.precision = match.group("precision")
        self.name = engine_path.parent.name

        engine = _deserialize_engine(engine_path)
        self.engine = engine
        self._ctx = engine.create_execution_context()
        if self._ctx is None:
            # TRT returns None (rather than raising) when the execution
            # context's device memory cannot be allocated.
            raise RuntimeError(
                f"could not create a TRT execution context for {self.name}: "
                "this is CUDA OOM; free VRAM or build a tighter latent "
                "profile"
            )

        from acestep.nodes.vae_nodes import _get_trt_stream

        self._pg_stream = _get_trt_stream()
        # A torch view of the shared polygraphy stream, used only for
        # wait_stream/synchronize. The engine still launches on the raw
        # polygraphy pointer.
        self._stream = torch.cuda.ExternalStream(self._pg_stream.ptr)

        self._device = torch.device(device)
        L = int(latent_frames)
        self._L = L

        shapes = {
            "hidden_states": (1, MINIMAX_LATENT_CHANNELS, L),
            "timestep": (1,),
            "encoder_hidden_states": (1, L, MINIMAX_COND_DIM),
        }
        for tname, shape in shapes.items():
            if not self._ctx.set_input_shape(tname, shape):
                raise RuntimeError(
                    f"engine {self.name} rejected {tname} shape {shape}; "
                    f"latent_frames={L} is outside its profile"
                )
        missing = self._ctx.infer_shapes()
        if missing:
            raise RuntimeError(
                f"engine {self.name} shapes insufficiently specified: {missing}"
            )
        out_shape = tuple(self._ctx.get_tensor_shape("velocity"))

        # Engine-declared dtypes, never assumed ones. See the module
        # docstring: a mismatched buffer is garbage, not an error.
        dt = {
            tname: _trt_dtype_to_torch(tname, engine.get_tensor_dtype(tname))
            for tname in ("hidden_states", "timestep", "encoder_hidden_states", "velocity")
        }
        self._io_dtypes = dt
        dev = self._device
        self._x = torch.zeros(shapes["hidden_states"], dtype=dt["hidden_states"], device=dev)
        self._t = torch.zeros(1, dtype=dt["timestep"], device=dev)
        self._cond = torch.zeros(
            shapes["encoder_hidden_states"], dtype=dt["encoder_hidden_states"], device=dev,
        )
        self._velocity = torch.empty(out_shape, dtype=dt["velocity"], device=dev)

        for tname, buf in (
            ("hidden_states", self._x),
            ("timestep", self._t),
            ("encoder_hidden_states", self._cond),
            ("velocity", self._velocity),
        ):
            if not self._ctx.set_tensor_address(tname, buf.data_ptr()):
                raise RuntimeError(f"engine {self.name} rejected address for {tname}")

        # Strong ref to the staged bundle, not its id(). An id() key can
        # stale-hit after the old bundle is collected and a new one lands
        # at the same address, which would silently keep rendering the
        # previous prompt's composition.
        self._staged_bundle = None

        logger.info(
            "minimax_trt_dit_ready engine={} precision={} L={} io_dtypes={}",
            self.name, self.precision, L,
            {k: str(v).replace("torch.", "") for k, v in dt.items()},
        )

    def __repr__(self) -> str:
        return f"<MiniMaxTRTDit {self.name} L={self._L}>"

    @property
    def latent_frames(self) -> int:
        return self._L

    def _stage_bundle(self, bundle: dict) -> None:
        if self._staged_bundle is bundle:
            return
        cond = bundle.get("encoder_hidden_states")
        if cond is None:
            raise ValueError("minimax cond bundle is missing encoder_hidden_states")
        if cond.ndim == 2:
            cond = cond.unsqueeze(0)
        if cond.ndim != 3 or cond.shape[-1] != MINIMAX_COND_DIM:
            raise ValueError(
                "minimax encoder_hidden_states must be "
                f"[1, T, {MINIMAX_COND_DIM}], got {tuple(cond.shape)}"
            )
        if cond.shape[1] != self._L:
            raise ValueError(
                f"cond latent frames {cond.shape[1]} != engine-bound L {self._L}"
            )
        self._cond.copy_(cond[:1])
        self._staged_bundle = bundle

    @torch.no_grad()
    def step_bundle(self, x_1ct: torch.Tensor, t: float, bundle: dict) -> torch.Tensor:
        """One velocity forward: MiniMax-native ``[1, 128, L]`` in and out.

        ``t`` is MiniMax time (0 noise -> 1 data); the DEMON-convention
        conversion lives in the adapter, not here.

        Returns the persistent output buffer; the caller must consume
        (copy or cast) it before the next step overwrites it. The adapter
        does exactly that (``copy=True``).
        """
        if x_1ct.shape[-1] != self._L:
            raise ValueError(
                f"x latent frames {x_1ct.shape[-1]} != engine-bound L {self._L}"
            )
        if x_1ct.shape[0] != 1:
            raise ValueError(
                f"engine {self.name} is batch-1; got batch {x_1ct.shape[0]}. "
                "The adapter must loop slots (trt_batch1)."
            )
        self._stage_bundle(bundle)
        self._x.copy_(x_1ct)
        self._t.fill_(float(t))

        # The copies above ran on the caller's current stream; the shared
        # polygraphy stream is a different, non-blocking stream and would
        # otherwise be free to read a half-written buffer. Captured before
        # the wait so the ordering is caller -> engine.
        caller_stream = torch.cuda.current_stream()
        self._stream.wait_stream(caller_stream)
        if not self._ctx.execute_async_v3(self._pg_stream.ptr):
            raise RuntimeError(f"minimax TRT DiT step failed on {self.name}")
        self._stream.synchronize()
        return self._velocity

    # The eager module is callable as ``dit(x, t, cond)``; keeping the
    # same shape here means benchmarks and one-off probes can drive
    # either without a branch. Production goes through step_bundle.
    @torch.no_grad()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.step_bundle(
            hidden_states,
            float(timestep.reshape(-1)[0]),
            {"encoder_hidden_states": encoder_hidden_states},
        )
