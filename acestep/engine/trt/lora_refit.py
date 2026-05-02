"""Dynamic LoRA application to TRT engines via weight refitting.

Uses TensorRT's IRefitter API to modify engine weights at runtime,
enabling LoRA application/removal without rebuilding the engine.

Requirements:
  - Engine built with trt.BuilderFlag.REFIT
  - ONNX exported with do_constant_folding=False (preserves weight names)

Performance notes:
  - Base weights and deltas are stored in the engine's native dtype
    (typically fp16 for mixed-precision engines) to avoid per-refit
    dtype conversion.
  - Pre-allocated refit buffers eliminate memory allocation during
    strength adjustment.
  - The refitter object is cached across calls.
  - numpy views are zero-copy from contiguous CPU tensors.

Thread safety: refit must not run concurrently with inference.
Apply/remove LoRAs between generations, not during denoising steps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from loguru import logger
import numpy as np
import torch
from safetensors.torch import load_file

# numpy dtype -> torch dtype
_NP_TO_TORCH = {
    np.float32: torch.float32,
    np.float16: torch.float16,
}


@dataclass
class _ActiveLoRA:
    """Internal state for one applied LoRA."""
    lora_id: int
    path: str
    strength: float
    # Precomputed full-rank deltas (B @ A) WITHOUT strength multiplied.
    # Stored in the engine's native dtype for zero-copy refit.
    # Keyed by decoder param name (e.g., "layers.0.self_attn.q_proj.weight").
    deltas: Dict[str, torch.Tensor]


class TRTLoRAManager:
    """Manages dynamic LoRA application to a TRT engine via weight refitting."""

    def __init__(
        self,
        engine,
        decoder: torch.nn.Module,
        device: torch.device = torch.device("cuda"),
        trt_weight_prefix: str = "decoder.",
        checkpoint_path: Optional[str] = None,
    ):
        import tensorrt as trt

        self._engine = engine
        self._device = device
        self._trt_prefix = trt_weight_prefix
        self._trt = trt
        self._trt_logger = trt.Logger(trt.Logger.WARNING)

        # TRT dtype -> numpy dtype
        _trt_to_np = {trt.float32: np.float32, trt.float16: np.float16}
        if hasattr(trt, "bfloat16"):
            _trt_to_np[trt.bfloat16] = np.float32

        # Query refittable weight names
        refitter = trt.Refitter(engine, self._trt_logger)
        if not hasattr(refitter, "get_all_weights"):
            raise RuntimeError("TRT engine refitting requires TensorRT 10.0+")

        all_trt_names = list(refitter.get_all_weights())
        if not all_trt_names:
            raise RuntimeError(
                "Engine has no refittable weights. Rebuild with refit=True."
            )

        # Cache refitter for reuse
        self._refitter = refitter

        # Resolve base weight source
        decoder_params = dict(decoder.named_parameters()) if decoder is not None else {}
        checkpoint_file = None
        if not decoder_params and checkpoint_path:
            from safetensors import safe_open
            logger.info("Loading base weights from checkpoint: %s", checkpoint_path)
            checkpoint_file = safe_open(checkpoint_path, framework="pt")

        has_prototype = hasattr(refitter, "get_weights_prototype")

        # Build mapping and cache base weights + refit buffers
        self._param_to_trt: Dict[str, str] = {}
        self._base_weights: Dict[str, torch.Tensor] = {}  # native dtype, CPU
        self._refit_bufs: Dict[str, torch.Tensor] = {}    # pre-alloc output
        self._np_dtype: Dict[str, np.dtype] = {}

        matched = 0
        for trt_name in all_trt_names:
            if not trt_name.startswith(trt_weight_prefix):
                continue
            param_name = trt_name[len(trt_weight_prefix):]

            # Detect engine dtype for this weight
            np_dt = np.float32
            if has_prototype:
                try:
                    proto = refitter.get_weights_prototype(trt_name)
                    np_dt = _trt_to_np.get(proto.dtype, np.float32)
                except Exception:
                    pass
            torch_dt = _NP_TO_TORCH.get(np_dt, torch.float32)

            # Load base weight
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

            # Store base weight in engine's native dtype (e.g., fp16)
            base = raw_w.to(dtype=torch_dt).cpu().contiguous()
            self._param_to_trt[param_name] = trt_name
            self._base_weights[param_name] = base
            self._refit_bufs[param_name] = torch.empty_like(base)
            self._np_dtype[param_name] = np_dt
            matched += 1

        logger.info(
            "TRT LoRA manager ready: %d/%d engine weights mapped (prefix='%s')",
            matched, len(all_trt_names), trt_weight_prefix,
        )
        if matched == 0:
            logger.warning(
                "No engine weights matched! TRT names sample: %s",
                all_trt_names[:5],
            )

        self._active_loras: List[_ActiveLoRA] = []
        self._next_id = 0
        self._ever_dirty: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_lora(self, lora_path: str, strength: float = 1.0) -> int:
        """Apply a LoRA. Returns ID for removal/strength adjustment."""
        t0 = time.perf_counter()

        raw = load_file(lora_path)
        pairs: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, tensor in raw.items():
            parts = key.replace("base_model.model.", "")
            if ".lora_A.weight" in parts:
                param_name = parts.replace(".lora_A.weight", ".weight")
                pairs.setdefault(param_name, {})["A"] = tensor
            elif ".lora_B.weight" in parts:
                param_name = parts.replace(".lora_B.weight", ".weight")
                pairs.setdefault(param_name, {})["B"] = tensor

        deltas: Dict[str, torch.Tensor] = {}
        skipped = 0
        for param_name, ab in pairs.items():
            if "A" not in ab or "B" not in ab:
                continue
            if param_name not in self._param_to_trt:
                skipped += 1
                continue
            # Compute B @ A on GPU in fp32, then convert to engine dtype on CPU
            A = ab["A"].to(device=self._device, dtype=torch.float32)
            B = ab["B"].to(device=self._device, dtype=torch.float32)
            target_dt = self._base_weights[param_name].dtype
            deltas[param_name] = (B @ A).to(dtype=target_dt).cpu().contiguous()

        lora_id = self._next_id
        self._next_id += 1
        self._active_loras.append(_ActiveLoRA(
            lora_id=lora_id, path=lora_path, strength=strength, deltas=deltas,
        ))

        # Strength-0 contributes nothing to any param's recomputed value,
        # so the refit would be a mathematical no-op (engine already has
        # the correct weights from the prior state). Skip it. The LoRA's
        # deltas are still cached and will participate in the next refit
        # once strength != 0.
        if strength != 0.0:
            self._refit_weights(set(deltas.keys()))

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Applied TRT LoRA %d: %s (%d params, %d skipped, "
            "scale=%.2f) in %.1fms",
            lora_id, Path(lora_path).name, len(deltas), skipped,
            strength, elapsed,
        )
        return lora_id

    def remove_lora(self, lora_id: int = -1) -> bool:
        """Remove a LoRA by ID. Default (-1) removes the most recent."""
        if not self._active_loras:
            return False
        if lora_id == -1:
            removed = self._active_loras.pop()
        else:
            idx = next(
                (i for i, l in enumerate(self._active_loras)
                 if l.lora_id == lora_id), None,
            )
            if idx is None:
                return False
            removed = self._active_loras.pop(idx)

        self._refit_weights(set(removed.deltas.keys()))
        logger.info(
            "Removed TRT LoRA %d: %s (%d params)",
            removed.lora_id, Path(removed.path).name, len(removed.deltas),
        )
        return True

    def set_lora_strength(self, lora_id: int, strength: float) -> None:
        """Adjust strength of an active LoRA and refit."""
        for lora in self._active_loras:
            if lora.lora_id == lora_id:
                old = lora.strength
                if old == strength:
                    # Slider sometimes lands back on the previous value;
                    # skip the full refit + GPU re-upload in that case.
                    return
                lora.strength = strength
                self._refit_weights(set(lora.deltas.keys()))
                logger.info(
                    "TRT LoRA %d strength: %.3f -> %.3f (%d params)",
                    lora_id, old, strength, len(lora.deltas),
                )
                return
        raise ValueError(f"LoRA {lora_id} not found in active stack")

    def remove_all(self) -> None:
        """Remove all LoRAs and restore engine to base weights."""
        if not self._active_loras:
            return
        all_params: Set[str] = set()
        for lora in self._active_loras:
            all_params.update(lora.deltas.keys())
        self._active_loras.clear()
        self._refit_weights(all_params)
        logger.info("All TRT LoRAs removed, engine restored to base weights")

    @property
    def has_active_loras(self) -> bool:
        return len(self._active_loras) > 0

    @property
    def active_lora_count(self) -> int:
        return len(self._active_loras)

    @property
    def active_lora_ids(self) -> List[int]:
        return [l.lora_id for l in self._active_loras]

    @property
    def refittable_param_count(self) -> int:
        return len(self._param_to_trt)

    # ------------------------------------------------------------------
    # Internal refit
    # ------------------------------------------------------------------

    def _refit_weights(self, param_names: Set[str]) -> None:
        """Refit engine weights. Uses pre-allocated buffers and in-place
        ops to avoid memory allocation. All math is in the engine's
        native dtype (typically fp16) for zero-copy numpy handoff."""
        if not param_names:
            return

        t0 = time.perf_counter()
        refitter = self._refitter
        count = 0

        for param_name in param_names:
            trt_name = self._param_to_trt.get(param_name)
            if trt_name is None:
                continue

            # Copy base into pre-allocated buffer (no allocation)
            buf = self._refit_bufs[param_name]
            buf.copy_(self._base_weights[param_name])

            # Accumulate LoRA contributions in-place (native dtype). Skip
            # strength-0 LoRAs; their delta contributes nothing but the
            # add_ traverses the full weight, which is wasteful when the
            # caller leaves placeholders in the stack at strength 0
            # (a common pattern for slider-driven UIs).
            for lora in self._active_loras:
                if lora.strength == 0.0:
                    continue
                if param_name in lora.deltas:
                    buf.add_(lora.deltas[param_name], alpha=lora.strength)

            # Zero-copy numpy view (contiguous CPU tensor, matching dtype)
            refitter.set_named_weights(trt_name, buf.numpy())
            count += 1
            self._ever_dirty.add(param_name)

        if count > 0:
            ok = refitter.refit_cuda_engine()
            if not ok:
                missing = refitter.get_missing_weights()
                raise RuntimeError(
                    f"TRT refit failed. Missing weights: {missing}"
                )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Refitted %d weights in %.1fms", count, elapsed)
