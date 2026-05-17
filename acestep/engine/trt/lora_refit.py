"""TRT subclass of :class:`LoRAManagerBase`: writeback via IRefitter.

The base class owns catalog, lifecycle, prewarm, and delta math. This
file is just the TRT-specific writeback path: pre-allocated CPU
buffers per refittable weight, named in the engine's ``decoder.``
prefix space.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Set

from loguru import logger
import numpy as np
import torch

from acestep.engine.lora import (
    LoRAManagerBase,
    LoRAState,           # re-exported for back-compat
    LoRADescriptor,      # re-exported for back-compat
    _LoRAEntry,          # re-exported for tests
)

# numpy dtype -> torch dtype
_NP_TO_TORCH = {
    np.float32: torch.float32,
    np.float16: torch.float16,
}

# FP8 E4M3FN saturation bound. Mirrors fp8_onnx.FP8_E4M3_MAX so the refit
# uses the same per-output-channel scales the patcher baked into the engine.
_FP8_E4M3_MAX = 448.0
_FP8_ABSMAX_FLOOR = 1e-12


class _ShardedSafeTensors:
    """Lazy multi-shard safetensors reader keyed by tensor name.

    Quacks like the single-file ``safetensors.safe_open(...)`` handle
    used in the legacy path — exposes ``get_tensor(name)`` returning a
    torch tensor. The HuggingFace sharded checkpoint format stores each
    tensor in exactly one shard, with a sidecar ``index.json`` mapping
    name -> shard filename. We open shard handles on demand and cache
    them so repeat reads don't reopen the file. Memory is mmap-backed
    inside safetensors so the resident cost is bounded by what's
    actually touched.
    """

    def __init__(self, index_path: "Path") -> None:
        import json
        from pathlib import Path as _Path

        self._index_path = _Path(index_path)
        idx = json.loads(self._index_path.read_text(encoding="utf-8"))
        self._weight_map: Dict[str, str] = idx["weight_map"]
        self._shard_dir = self._index_path.parent
        self._handles: Dict[str, object] = {}

    def get_tensor(self, name: str):
        shard = self._weight_map.get(name)
        if shard is None:
            raise KeyError(name)
        handle = self._handles.get(shard)
        if handle is None:
            from safetensors import safe_open
            handle = safe_open(str(self._shard_dir / shard), framework="pt")
            self._handles[shard] = handle
        return handle.get_tensor(name)


def _open_checkpoint(path):
    """Open a HF checkpoint that may be single-file or sharded.

    Accepts:
      - a file path to ``model.safetensors`` (single file)
      - a file path to ``model.safetensors.index.json`` (sharded index)
      - a directory containing either of the above

    Returns an object exposing ``get_tensor(name)``.
    """
    from pathlib import Path as _Path
    from safetensors import safe_open

    p = _Path(path)
    if p.is_dir():
        idx = p / "model.safetensors.index.json"
        if idx.is_file():
            return _ShardedSafeTensors(idx)
        single = p / "model.safetensors"
        if single.is_file():
            return safe_open(str(single), framework="pt")
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json in {p}"
        )
    if p.name == "model.safetensors.index.json":
        return _ShardedSafeTensors(p)
    return safe_open(str(p), framework="pt")


# Backward-compat alias for tests / external code that imported the old
# name.  The old type held only (lora_id, path, strength, deltas); the
# new one is a superset, so kw-only construction with the old fields
# still works.
_ActiveLoRA = _LoRAEntry


class TRTLoRAManager(LoRAManagerBase):
    """LoRA writeback into a TRT engine via IRefitter."""

    def __init__(
        self,
        engine,
        decoder: torch.nn.Module,
        device: torch.device = torch.device("cuda"),
        trt_weight_prefix: str = "decoder.",
        checkpoint_path: Optional[str] = None,
        engine_path: Optional[str] = None,
    ):
        import tensorrt as trt

        if device.type != "cuda":
            raise RuntimeError(
                f"TRTLoRAManager requires a CUDA device; got {device}. "
                "Refit composes base + deltas on a GPU scratch and "
                "needs a real device to allocate against."
            )

        self._engine = engine
        self._device = device
        self._trt_prefix = trt_weight_prefix
        self._trt = trt
        self._trt_logger = trt.Logger(trt.Logger.WARNING)

        # Per-param transpose flag: True if the engine slot stores the
        # weight in ONNX MatMul's ``[in_dim, out_dim]`` orientation (dynamo
        # output) instead of torch nn.Linear's ``[out_dim, in_dim]``.
        # Populated from a sidecar refit manifest emitted by
        # ``rename_val_initializers_to_fqn`` next to the ONNX, then copied
        # next to the engine by the build pipeline. Absent manifest means
        # all weights use torch orientation (legacy torchscript path).
        self._transpose_for_engine: Dict[str, bool] = {}
        # FP8 enrichment from the refit manifest's ``fp8`` field
        # (manifest version >= 2). Lets us re-submit fused FP8 scale
        # initializers TRT considers "missing" on multi-weight refits
        # but won't expose via get_named_weights.
        self._manifest_activation_scales: list[dict] = []
        self._manifest_weight_scale_names: list[dict] = []
        if engine_path is not None:
            manifest_path = Path(str(engine_path) + ".refit_manifest.json")
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    transposed = manifest.get("weights_transposed", [])
                    prefix = trt_weight_prefix
                    for fqn in transposed:
                        # Manifest entries are engine-namespace ("decoder.X");
                        # strip the prefix to match the param-name keys used
                        # by the manager elsewhere.
                        if fqn.startswith(prefix):
                            self._transpose_for_engine[fqn[len(prefix):]] = True
                    fp8_block = manifest.get("fp8", {})
                    self._manifest_activation_scales = list(
                        fp8_block.get("activation_scales", [])
                    )
                    self._manifest_weight_scale_names = list(
                        fp8_block.get("weight_scale_names", [])
                    )
                    logger.info(
                        "Refit manifest loaded: {} transposed-layout weights, "
                        "{} fp8 activation scales, {} fp8 weight scales",
                        len(self._transpose_for_engine),
                        len(self._manifest_activation_scales),
                        len(self._manifest_weight_scale_names),
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to read refit manifest at {}: {}. LoRA refit "
                        "will assume torch [out, in] orientation; for dynamo-"
                        "built engines this produces wrong outputs under LoRA.",
                        manifest_path, exc,
                    )

        # TRT dtype -> numpy dtype (storage) + torch dtype (base buffer).
        # Numpy float32/float16 round-trip natively; BF16 and FP8 don't
        # have numpy dtypes so we stage their storage bytes via uint16
        # / uint8 views and wrap them in trt.Weights(trt_dtype, ptr,
        # count) at refit time (the numpy-overload of
        # set_named_weights mistypes them as UINT16 / UINT8 which TRT
        # rejects with "refit weights data type Float must equal to
        # weights prototype data type BFloat16" / "UINT8 weights
        # cannot be refitted").
        _trt_to_np = {trt.float32: np.float32, trt.float16: np.float16}
        _trt_to_torch = {trt.float32: torch.float32, trt.float16: torch.float16}
        if hasattr(trt, "bfloat16"):
            _trt_to_np[trt.bfloat16] = np.uint16
            _trt_to_torch[trt.bfloat16] = torch.bfloat16
        _trt_fp8 = getattr(trt, "fp8", None)
        if _trt_fp8 is not None:
            _trt_to_np[_trt_fp8] = np.uint8
            # FP8 base lives in fp32 for delta math; the FP8 storage
            # buffer is a separate uint8 tensor sized to match.
            _trt_to_torch[_trt_fp8] = torch.float32

        refitter = trt.Refitter(engine, self._trt_logger)
        if not hasattr(refitter, "get_all_weights"):
            raise RuntimeError("TRT engine refitting requires TensorRT 10.0+")

        all_trt_names = list(refitter.get_all_weights())
        if not all_trt_names:
            raise RuntimeError(
                "Engine has no refittable weights. Rebuild with refit=True."
            )

        # FP8 engines built with SmoothQuant (--smoothquant-alpha > 0)
        # mutate the base weight (w *= s[in]) BEFORE FP8 quantization, so
        # the engine slot no longer matches the torch base. The per-input
        # SmoothQuant ``s`` factor isn't stored in the refit manifest and
        # would need to be recomputed from the activation absmax JSON to
        # do this correctly. Bail loud rather than silently scramble
        # LoRA outputs — rebuild with ``--smoothquant-alpha 0`` for the
        # refit-capable engine path, or extend this code path with full
        # SmoothQuant-aware refit.
        if any(n.endswith("_sq_inv_s") for n in all_trt_names):
            raise RuntimeError(
                "TRT engine was built with SmoothQuant (FP8 + alpha>0); "
                "LoRA refit can't reconstruct the per-input-channel scale "
                "without it being persisted. Rebuild the FP8 engine with "
                "--smoothquant-alpha 0 for refit, or extend TRTLoRAManager."
            )

        self._refitter = refitter
        self._fp8_dtype = _trt_fp8

        decoder_params = dict(decoder.named_parameters()) if decoder is not None else {}
        checkpoint_file = None
        if not decoder_params and checkpoint_path:
            checkpoint_file = _open_checkpoint(checkpoint_path)
            logger.info(
                "Loading base weights from checkpoint: {} ({})",
                checkpoint_path,
                "sharded" if isinstance(checkpoint_file, _ShardedSafeTensors)
                else "single file",
            )

        has_prototype = hasattr(refitter, "get_weights_prototype")

        # Mapping + per-weight buffers/dtypes. _base_weights and
        # _param_dtype are what LoRAManagerBase looks at.
        self._param_to_trt: Dict[str, str] = {}
        self._base_weights: Dict[str, torch.Tensor] = {}   # native dtype, CPU
        self._refit_bufs: Dict[str, torch.Tensor] = {}     # pre-alloc output
        self._np_dtype: Dict[str, np.dtype] = {}
        self._param_dtype: Dict[str, torch.dtype] = {}
        # Element count per param (orientation-independent) for the
        # base-class shape sanity check in _compute_deltas.
        self._param_numel: Dict[str, int] = {}
        # TRT dtype per param (for the trt.Weights(dtype, ptr, count)
        # wrapper used at refit time — necessary for BF16 and FP8 since
        # neither has a numpy dtype that TRT's set_named_weights(name,
        # ndarray) overload can infer correctly).
        self._trt_dtype: Dict[str, int] = {}
        # FP8-specific state. For an FP8 engine slot:
        #   _is_fp8[param]      = True
        #   _base_weights[param] = fp32 base in engine orientation (not fp8)
        #   _fp8_scale[param]    = 1D fp32 scale on the LAST axis of base
        #   _refit_bufs[param]   = uint8 buffer sized to fp8 storage
        # Per-refit fp32 accumulation runs on a shared GPU scratch
        # (self._scratch_acc); the final fp8 bytes get D2H'd into
        # _refit_bufs and pushed via the HOST refit path.
        self._is_fp8: Dict[str, bool] = {}
        self._fp8_scale: Dict[str, torch.Tensor] = {}

        matched = 0
        matched_fp8 = 0
        for trt_name in all_trt_names:
            if not trt_name.startswith(trt_weight_prefix):
                continue
            param_name = trt_name[len(trt_weight_prefix):]

            trt_dtype = None
            np_dt = np.float32
            torch_dt = torch.float32
            if has_prototype:
                try:
                    proto = refitter.get_weights_prototype(trt_name)
                    trt_dtype = proto.dtype
                    np_dt = _trt_to_np.get(proto.dtype, np.float32)
                    torch_dt = _trt_to_torch.get(proto.dtype, torch.float32)
                except Exception:
                    pass
            is_fp8 = (
                _trt_fp8 is not None
                and trt_dtype is not None
                and trt_dtype == _trt_fp8
            )

            raw_w = None
            if param_name in decoder_params:
                raw_w = decoder_params[param_name].data
            elif checkpoint_file is not None:
                try:
                    raw_w = checkpoint_file.get_tensor(trt_name)
                except Exception:
                    pass

            if raw_w is None:
                continue

            base = raw_w.to(dtype=torch_dt).cpu().contiguous()
            # If the engine stored this weight transposed (dynamo MatMul
            # layout), keep our base + refit buffer in the same
            # orientation so the bytes we hand to ``set_named_weights``
            # match the slot's memory layout. Deltas arrive in torch
            # [out, in] from ``_compute_deltas`` and are transposed
            # on the fly inside ``_apply_to_engine``.
            if self._transpose_for_engine.get(param_name, False) and base.dim() == 2:
                base = base.transpose(0, 1).contiguous()
            # Pin the base for fast async H2D into the GPU scratch on
            # every refit. Try/except so a host with a tight lockable-
            # RAM ulimit (or already-locked pages) doesn't hard-fail
            # boot — we just lose H2D overlap.
            try:
                base = base.pin_memory()
            except RuntimeError as exc:
                logger.warning(
                    "pin_memory() failed for {}: {}. Falling back to "
                    "unpinned base; H2D will be synchronous.",
                    param_name, exc,
                )

            self._param_to_trt[param_name] = trt_name
            self._np_dtype[param_name] = np_dt
            self._param_dtype[param_name] = torch_dt
            self._trt_dtype[param_name] = trt_dtype
            self._param_numel[param_name] = base.numel()

            if is_fp8:
                # Compute the FP8 per-output-channel scale from the base
                # weight. The engine was built using fp8_onnx.py's
                # _quantize_weight_e4m3fn against this same base, so the
                # scale we derive here matches what's already in the
                # engine slot's sibling _fp8_scale initializer. We
                # intentionally do NOT refit the scale: LoRA deltas are
                # small relative to the base, saturating at +/-448 is
                # rare, and keeping the scale fixed means the engine's
                # DequantizeLinear output stays numerically stable.
                if base.dim() < 2:
                    raise RuntimeError(
                        f"FP8 refit expects 2D weights; {param_name} has "
                        f"dim={base.dim()}"
                    )
                reduce_axes = tuple(range(base.dim() - 1))
                absmax = base.abs().amax(dim=reduce_axes)
                scale = absmax.clamp(min=_FP8_ABSMAX_FLOOR) / _FP8_E4M3_MAX
                scale = scale.contiguous()

                self._is_fp8[param_name] = True
                self._fp8_scale[param_name] = scale
                self._base_weights[param_name] = base
                # set_named_weights wants raw fp8 storage bytes. Stage
                # them as a uint8 tensor view-compatible with what we'll
                # produce from torch.float8_e4m3fn -> view(uint8). The
                # refit path D2Hs its GPU fp8 output into this buffer
                # and then pushes via HOST location. Not pinned —
                # pinning only matters for true async D2H and we sync
                # before TRT reads the bytes.
                self._refit_bufs[param_name] = torch.empty(
                    base.shape, dtype=torch.uint8,
                )
                matched_fp8 += 1
            else:
                self._is_fp8[param_name] = False
                self._base_weights[param_name] = base
                self._refit_bufs[param_name] = torch.empty_like(base)
            matched += 1

        logger.info(
            "TRT LoRA manager ready: {}/{} engine weights mapped "
            "(prefix='{}', fp8={})",
            matched, len(all_trt_names), trt_weight_prefix, matched_fp8,
        )
        if matched == 0:
            logger.warning(
                "No engine weights matched! TRT names sample: {}",
                all_trt_names[:5],
            )

        # Single GPU scratch used as the per-param fp32 accumulator
        # (base + Σ s·delta) during refit. Sized to the largest
        # refittable param's element count. The downstream fp8 / native-
        # dtype output tensors are allocated per iteration from the
        # caching allocator and only need to outlive each per-param
        # D2H into the host refit buffer.
        self._scratch_acc: Optional[torch.Tensor] = None
        if self._base_weights:
            max_numel = max(b.numel() for b in self._base_weights.values())
            self._scratch_acc = torch.empty(
                max_numel, dtype=torch.float32, device=self._device,
            )
            logger.info(
                "TRT LoRA refit scratch: {:.1f} MB on {} (max param numel={})",
                max_numel * 4 / 1e6, self._device, max_numel,
            )

        # ------------------------------------------------------------------
        # Co-refit set: scale initializers TRT considers "missing" any
        # time we touch a LoRA-target weight, because the FP8 graph
        # fused them into the consumer MatMul tactic. From the TRT
        # docs on get_missing_weights:
        #   "if some Weights have been set, but the engine was optimized
        #    in a way that combines weights, any unsupplied Weights in
        #    the combination are considered missing."
        # IRefitter::get_named_weights can't read these back from the
        # engine post-deserialize (returns an "Assertion iter !=
        # mCpuInputs.end() failed" internal error), so we source them
        # from the refit manifest's ``fp8`` field — populated by the
        # FP8 patcher at build time. Two sources:
        #   * Per-tensor activation Q-DQ scales: scalar bf16, value
        #     persisted directly in manifest as activation_amax/FP8_MAX.
        #   * Per-output-channel weight scales: 1D bf16, recomputed
        #     here from the torch base weight (same formula the FP8
        #     patcher used, see fp8_onnx._quantize_weight_e4m3fn).
        # Each entry is (name, numpy bytes, TRT dtype); the buffer is
        # owned by the manager and stays alive across refits.
        # Static co-refit entries: activation Q-DQ scales. These don't
        # change with LoRA, so we build them once and re-submit each refit.
        self._co_refit_static: list[tuple[str, np.ndarray, object]] = []
        # Dynamic co-refit entries: per-output-channel weight scales. These
        # MUST be recomputed from ``acc`` each refit. If we kept the
        # base-derived scale fixed, the LoRA-modified weight value would
        # overshoot the channel's representable range; torch's
        # float8_e4m3fn cast turns any value above 448 into the FN-NaN
        # bit pattern (verified: 1145 -> NaN, not 448), poisoning the
        # whole engine output.
        # Indexed by param_key so ``_apply_to_engine`` can update the
        # underlying ndarray bytes in place per refit.
        self._weight_scale_co_refit: dict[
            str, tuple[str, torch.Tensor, np.ndarray, object]
        ] = {}
        trt_bf16 = getattr(trt, "bfloat16", None)
        if self._manifest_activation_scales and trt_bf16 is not None:
            for rec in self._manifest_activation_scales:
                scale_name = rec["scale_init"]
                # Scalar bf16: pack one float as 2 bytes via torch.
                t = torch.tensor(rec["scale"], dtype=torch.bfloat16)
                arr = t.view(torch.uint16).contiguous().numpy().copy()
                self._co_refit_static.append((scale_name, arr, trt_bf16))
        if self._manifest_weight_scale_names and trt_bf16 is not None:
            for rec in self._manifest_weight_scale_names:
                weight_engine_name = rec["weight"]
                scale_name = rec["scale_init"]
                # Find our cached base. The manifest stores engine-namespace
                # names ("decoder.layers.X.y.weight"); our keys are stripped
                # to "layers.X.y.weight".
                param_key = weight_engine_name[len(trt_weight_prefix):] \
                    if weight_engine_name.startswith(trt_weight_prefix) \
                    else weight_engine_name
                base = self._base_weights.get(param_key)
                if base is None or base.dim() < 2:
                    # Mismatched manifest vs engine; skip silently. The
                    # subsequent refit will fail loud if this matters.
                    continue
                # Pre-allocate the bf16 scale tensor + uint16 ndarray view
                # the refitter will keep alive. We initialize from base
                # so the construction-time dry-run refit submits a valid
                # initial scale; ``_apply_to_engine`` rewrites it from
                # ``acc`` on every subsequent refit.
                reduce_axes = tuple(range(base.dim() - 1))
                absmax = base.abs().amax(dim=reduce_axes)
                scale_fp32 = absmax.clamp(min=_FP8_ABSMAX_FLOOR) / _FP8_E4M3_MAX
                scale_bf16 = scale_fp32.to(torch.bfloat16).contiguous()
                arr = scale_bf16.view(torch.uint16).numpy().copy()
                # ``arr`` is the bytes the refitter will dereference; we
                # mutate it in place per refit so the same pointer stays
                # valid across calls.
                self._weight_scale_co_refit[param_key] = (
                    scale_name, scale_bf16, arr, trt_bf16,
                )
        if self._co_refit_static or self._weight_scale_co_refit:
            logger.info(
                "TRT LoRA manager co-refit set: {} fused initializers "
                "(activation_scales={}, weight_scales={} dynamic)",
                len(self._co_refit_static) + len(self._weight_scale_co_refit),
                len(self._co_refit_static),
                len(self._weight_scale_co_refit),
            )

        # Initialize the base class so ``self._loras`` exists for the
        # initial refit below.
        super().__init__()

        # FP8-only init refit: when the engine has fused FP8 scale slots
        # (co-refit set non-empty), push them once at construction so
        # TRT sees a fully-satisfied initial weight state. Walks every
        # _param_to_trt entry too because TRT treats the FP8 fusion as
        # a single missing-set. SKIPPED on non-FP8 engines (2B bf16-
        # hybrid, XL bf16-mixed): the build-time inlined weights are
        # already correct and rewriting bf16 LayerNorm bytes via the
        # typed-Weights path subtly perturbs the engine output (root
        # cause not fully understood, but stash-confirmed on 2B).
        # The first LoRA enable will trigger the normal refit path,
        # which on FP8 also re-supplies the co-refit set.
        if self._co_refit_static or self._weight_scale_co_refit:
            all_param_names = set(self._param_to_trt.keys())
            if all_param_names:
                self._apply_to_engine(all_param_names)

    def _delta_compute_device(self) -> torch.device:
        return self._device

    def _delta_storage_device(self) -> torch.device:
        # Refit composes on GPU; deltas need to live there so
        # ``acc.add_(delta, alpha=...)`` is a single-device op.
        return self._device

    # ------------------------------------------------------------------
    # Engine writeback (IRefitter): GPU compose -> D2H -> HOST batch refit
    # ------------------------------------------------------------------

    def _apply_to_engine(
        self,
        param_names: Set[str],
        *,
        _refit_ok_required: bool = True,
    ) -> None:
        """GPU compose + D2H into host refit buffers + batch HOST refit.

        Per param: H2D pinned base → shared fp32 GPU scratch, accumulate
        GPU-resident enabled deltas, FP8 quantize on GPU (or cast to
        native dtype for non-FP8 slots), D2H the result bytes into the
        CPU ``_refit_bufs`` slot. After the loop, one batch HOST refit
        pushes every touched weight plus the full static and dynamic
        co-refit sets and calls ``refit_cuda_engine()`` once.

        FP8-fused engines have a missing-set that spans the entire
        engine: every fp8 weight scale, plus several
        ``cross_attn_norm.weight`` slots, is considered missing any
        time we touch one fused weight. That forces the single-batch
        refit topology — a per-param ``refit_cuda_engine()`` can never
        satisfy the missing set on the first iteration.

        A strength-0 ENABLED entry is skipped explicitly: the add_ would
        be a math-no-op but still walks the full weight, wasting cycles
        for slider-driven UIs that leave placeholders at 0.
        """
        refitter = self._refitter
        param_list = list(param_names)
        count = 0

        for param_name in param_list:
            trt_name = self._param_to_trt.get(param_name)
            if trt_name is None:
                continue

            transpose_delta = self._transpose_for_engine.get(param_name, False)
            is_fp8 = self._is_fp8.get(param_name, False)
            base_cpu = self._base_weights[param_name]
            shape = base_cpu.shape
            numel = base_cpu.numel()

            # Slice the shared fp32 scratch to this param's size and
            # view it back to the base's N-D shape. Reused every
            # iteration — peak VRAM is one max-param fp32 buffer.
            acc = self._scratch_acc[:numel].view(shape)
            # H2D + dtype cast in one shot. non_blocking is real here
            # because base_cpu is pinned (see __init__).
            acc.copy_(base_cpu, non_blocking=True)

            for entry in self._loras.values():
                if entry.state != LoRAState.ENABLED:
                    continue
                if entry.strength == 0.0:
                    continue
                if entry.deltas and param_name in entry.deltas:
                    delta = entry.deltas[param_name]
                    if transpose_delta and delta.dim() == 2:
                        delta = delta.transpose(0, 1).contiguous()
                    if delta.dtype != torch.float32:
                        delta = delta.to(torch.float32)
                    acc.add_(delta, alpha=entry.strength)

            cpu_buf = self._refit_bufs[param_name]
            if is_fp8:
                # Per-channel scale is recomputed from ``acc``, not
                # held fixed from base: torch's float8_e4m3fn cast
                # produces the FN-NaN bit pattern above 448, so a
                # stale base-derived scale + a LoRA bump past base
                # range = engine outputs NaN.
                reduce_axes = tuple(range(acc.dim() - 1))
                absmax_acc = acc.abs().amax(dim=reduce_axes)
                scale = (
                    absmax_acc.clamp(min=_FP8_ABSMAX_FLOOR)
                    / _FP8_E4M3_MAX
                )
                bcast = (1,) * (acc.dim() - 1) + (scale.shape[0],)
                scaled = (acc / scale.view(bcast)).clamp_(
                    -_FP8_E4M3_MAX, _FP8_E4M3_MAX,
                )
                fp8 = scaled.to(torch.float8_e4m3fn).contiguous()
                # D2H into the pinned host refit buffer. Cross-device
                # copy_ handles the transfer; non_blocking=True is fine
                # because the destination is pinned and we synchronize
                # implicitly before reading via numpy below.
                cpu_buf.copy_(fp8.view(torch.uint8), non_blocking=True)

                rec = self._weight_scale_co_refit.get(param_name)
                if rec is not None:
                    _, scale_bf16_buf, scale_arr, _ = rec
                    # D2H a 1D scale (tens of KB) — negligible.
                    scale_bf16_buf.copy_(scale.to(torch.bfloat16).cpu())
                    scale_arr[:] = scale_bf16_buf.view(torch.uint16).numpy()
            else:
                native_dt = self._param_dtype[param_name]
                cast = acc.to(native_dt).contiguous()
                cpu_buf.copy_(cast, non_blocking=True)
            count += 1

        # All GPU work issued; force completion before TRT reads the
        # host pointers below. One sync covers every param's D2H.
        if count > 0:
            torch.cuda.synchronize(self._device)

        # ----- HOST refit batch: push every touched param + the full -----
        # ----- static and dynamic co-refit sets, then commit once.   -----
        for param_name in param_list:
            trt_name = self._param_to_trt.get(param_name)
            if trt_name is None:
                continue
            buf = self._refit_bufs[param_name]
            if buf.dtype == torch.bfloat16:
                arr = buf.view(torch.uint16).numpy()
            else:
                arr = buf.numpy()
            weights = self._trt.Weights(
                self._trt_dtype[param_name],
                int(arr.ctypes.data),
                int(arr.size),
            )
            ok = refitter.set_named_weights(trt_name, weights)
            if not ok:
                proto_desc = "unknown"
                if hasattr(refitter, "get_weights_prototype"):
                    try:
                        proto = refitter.get_weights_prototype(trt_name)
                        proto_desc = (
                            f"dtype={proto.dtype}, size={proto.size}"
                        )
                    except Exception:
                        pass
                raise RuntimeError(
                    f"TRT rejected refit weights for {trt_name}: "
                    f"buf dtype={buf.dtype}, arr dtype={arr.dtype} "
                    f"size={arr.size}; engine prototype {proto_desc}, "
                    f"fp8={self._is_fp8.get(param_name, False)}"
                )

        for name, arr, dtype in getattr(self, "_co_refit_static", ()):
            weights = self._trt.Weights(
                dtype, int(arr.ctypes.data), int(arr.size),
            )
            if not refitter.set_named_weights(name, weights):
                raise RuntimeError(
                    f"TRT rejected co-refit weight {name}: "
                    f"dtype={dtype}, size={arr.size}"
                )
        for param_key, (name, _scale_buf, arr, dtype) in getattr(
            self, "_weight_scale_co_refit", {}
        ).items():
            weights = self._trt.Weights(
                dtype, int(arr.ctypes.data), int(arr.size),
            )
            if not refitter.set_named_weights(name, weights):
                raise RuntimeError(
                    f"TRT rejected co-refit weight scale {name} "
                    f"(param={param_key}): dtype={dtype}, size={arr.size}"
                )

        if count > 0:
            ok = refitter.refit_cuda_engine()
            if not ok:
                missing = refitter.get_missing_weights()
                if _refit_ok_required:
                    raise RuntimeError(
                        f"TRT refit failed. Missing weights: {missing}"
                    )


# Re-export the lifecycle public symbols so existing imports
# (`from acestep.engine.trt.lora_refit import TRTLoRAManager, LoRAState, ...`)
# keep working without churn.
__all__ = [
    "TRTLoRAManager",
    "LoRAState",
    "LoRADescriptor",
    "_LoRAEntry",
    "_ActiveLoRA",
]
