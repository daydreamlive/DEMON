"""Dynamic LoRA application to TRT engines via weight refitting.

Uses TensorRT's IRefitter API to modify engine weights at runtime,
enabling LoRA application/removal without rebuilding the engine.

Lifecycle (per LoRA):
  REGISTERED   - in the library catalog, no CPU RAM cost (only path + label)
  MATERIALIZING - background prewarm in flight (worker computing B@A)
  MATERIALIZED - deltas in CPU RAM, NOT yet contributing to engine state
  ENABLED      - deltas applied to the engine; contributes at current strength

Strength is orthogonal to the lifecycle: an ENABLED LoRA can sit at
strength 0 indefinitely (placeholder pattern for slider-driven UIs).
``set_lora_strength`` is rejected on non-ENABLED LoRAs to keep the
materialization cost explicit at the call site rather than buried under
a slider event.

Library:
  ``register_library(directory)`` discovers ``*.safetensors`` in a flat
  directory and registers each as a REGISTERED entry whose id is the
  filename stem. The catalog backs an "infinite library" workflow where
  hundreds of LoRAs cost nothing until enabled.

Threading:
  - register / enable / disable / set_strength / refit run on the
    inference-owning thread; refit and inference are mutually exclusive.
  - prewarm runs on a background ThreadPoolExecutor; the worker only
    fills the entry's deltas dict and never touches the engine.
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger
import numpy as np
import torch
from safetensors.torch import load_file

# numpy dtype -> torch dtype
_NP_TO_TORCH = {
    np.float32: torch.float32,
    np.float16: torch.float16,
}


class LoRAState(Enum):
    REGISTERED = "registered"
    MATERIALIZING = "materializing"
    MATERIALIZED = "materialized"
    ENABLED = "enabled"


@dataclass
class LoRADescriptor:
    """Read-only public view of a LoRA in the library."""
    id: str
    path: str
    name: str
    state: str
    strength: float
    materialized_bytes: int


@dataclass
class _LoRAEntry:
    """Internal mutable state for one library entry.

    ``strength`` is preserved across enable/disable cycles so flipping a
    UI toggle off and back on doesn't reset the slider.
    """
    lora_id: str
    path: str
    name: str = ""
    state: LoRAState = LoRAState.REGISTERED
    strength: float = 0.0
    deltas: Optional[Dict[str, torch.Tensor]] = None
    future: Optional[concurrent.futures.Future] = None
    materialized_bytes: int = 0


# Backward-compat alias for tests / external code that imported the old
# name.  The old type held only (lora_id, path, strength, deltas); the
# new one is a superset, so kw-only construction with the old fields
# still works.
_ActiveLoRA = _LoRAEntry


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

        # Library + lifecycle state.  Insertion order is preserved so
        # ``remove_lora(-1)`` can pop the most-recently-registered entry,
        # matching the legacy stack-style API.
        self._loras: Dict[str, _LoRAEntry] = {}
        self._ever_dirty: Set[str] = set()
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    # ------------------------------------------------------------------
    # Library: catalog without RAM cost
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(path: str) -> str:
        """Filename-stem id. Two files with the same stem collide; the
        registrar treats this as identity, so that's fine for a flat
        library directory but means a caller can't register two distinct
        LoRAs that happen to share a stem."""
        return Path(path).stem

    def register_lora(
        self, path: str, name: Optional[str] = None,
    ) -> str:
        """Add a LoRA to the catalog without materializing deltas.

        Idempotent: re-registering the same id (filename stem) returns
        the existing id and leaves any in-flight prewarm / enabled
        state alone.  The existing entry's name is NOT overwritten on
        re-register; pass an explicit ``name`` only on first registration.
        """
        lora_id = self._make_id(path)
        if lora_id in self._loras:
            existing = self._loras[lora_id]
            if existing.path != str(path):
                logger.warning(
                    "LoRA id %r already registered to %s; ignoring re-register from %s",
                    lora_id, existing.path, path,
                )
            return lora_id
        self._loras[lora_id] = _LoRAEntry(
            lora_id=lora_id, path=str(path),
            name=name if name is not None else lora_id,
        )
        logger.info("Registered TRT LoRA: %s (path=%s)", lora_id, path)
        return lora_id

    def register_library(
        self, directory: Optional[Path] = None,
    ) -> List[str]:
        """Discover and register every ``*.safetensors`` in ``directory``.

        Defaults to :func:`acestep.paths.loras_dir`.  Returns the list
        of registered ids in directory order (sorted by filename).
        Missing directory returns an empty list.
        """
        from acestep.paths import discover_loras, loras_dir
        d = directory if directory is not None else loras_dir()
        files = discover_loras(d)
        ids: List[str] = []
        for p in files:
            try:
                ids.append(self.register_lora(str(p)))
            except Exception as e:
                logger.warning("Failed to register %s: %s", p, e)
        if files:
            logger.info(
                "Registered library: %d LoRAs from %s", len(ids), d,
            )
        return ids

    # ------------------------------------------------------------------
    # Lifecycle: REGISTERED <-> MATERIALIZING <-> MATERIALIZED <-> ENABLED
    # ------------------------------------------------------------------

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Lazy-init single-worker pool for prewarm.  Single worker
        because the materialization runs (B @ A) on the same CUDA device
        as inference; oversubscribing would just block on the GPU."""
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="lora_prewarm",
            )
        return self._executor

    def prewarm_lora(self, lora_id: str) -> concurrent.futures.Future:
        """Kick off background materialization of ``lora_id``.

        Returns a Future that resolves to ``None`` once deltas are in
        CPU RAM.  Subsequent ``enable_lora`` will skip the materialization
        step.  Calling on a MATERIALIZED / ENABLED entry returns an
        already-completed future; calling on a MATERIALIZING entry
        returns the in-flight future.
        """
        entry = self._require_entry(lora_id)
        if entry.state in (LoRAState.MATERIALIZED, LoRAState.ENABLED):
            f: concurrent.futures.Future = concurrent.futures.Future()
            f.set_result(None)
            return f
        if entry.state == LoRAState.MATERIALIZING:
            assert entry.future is not None
            return entry.future
        # REGISTERED -> MATERIALIZING
        entry.state = LoRAState.MATERIALIZING
        entry.future = self._get_executor().submit(
            self._materialize_worker, entry,
        )
        logger.info("Prewarming TRT LoRA: %s", lora_id)
        return entry.future

    def _materialize_worker(self, entry: _LoRAEntry) -> None:
        """Worker-thread body.  Loads safetensors, computes deltas,
        writes them to the entry.  Engine state untouched.

        If the entry was concurrently disabled (state changed away from
        MATERIALIZING), the result is dropped to avoid resurrecting the
        deltas after disable cleared them.
        """
        t0 = time.perf_counter()
        try:
            deltas, bytes_ = self._compute_deltas(entry.path)
        except Exception:
            if entry.state == LoRAState.MATERIALIZING:
                entry.state = LoRAState.REGISTERED
                entry.future = None
            raise
        if entry.state == LoRAState.MATERIALIZING:
            entry.deltas = deltas
            entry.materialized_bytes = bytes_
            entry.state = LoRAState.MATERIALIZED
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "Materialized TRT LoRA %s (%d params, %.1f MB) in %.1fms",
                entry.lora_id, len(deltas), bytes_ / 1e6, elapsed,
            )
        # else: entry was disabled mid-prewarm; drop the deltas silently.

    def _compute_deltas(
        self, lora_path: str,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Load LoRA from disk and compute full-rank deltas (B @ A).

        Pure compute; safe to call from a worker thread.  Returns
        (deltas_dict, total_bytes_in_cpu_ram).
        """
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
        total_bytes = 0
        skipped = 0
        for param_name, ab in pairs.items():
            if "A" not in ab or "B" not in ab:
                continue
            if param_name not in self._param_to_trt:
                skipped += 1
                continue
            A = ab["A"].to(device=self._device, dtype=torch.float32)
            B = ab["B"].to(device=self._device, dtype=torch.float32)
            target_dt = self._base_weights[param_name].dtype
            d = (B @ A).to(dtype=target_dt).cpu().contiguous()
            deltas[param_name] = d
            total_bytes += d.numel() * d.element_size()
        if skipped:
            logger.debug(
                "_compute_deltas(%s): %d params skipped (not in engine)",
                Path(lora_path).name, skipped,
            )
        return deltas, total_bytes

    def enable_lora(
        self, lora_id: str, strength: Optional[float] = None,
    ) -> None:
        """Promote a LoRA to ENABLED.  Synchronous; materializes if needed.

        ``strength``, when provided, overrides the entry's stored strength
        BEFORE the refit fires.  Pass it whenever you know the target
        strength up-front: it lets a one-shot caller atomically transition
        REGISTERED -> ENABLED-at-S without an intermediate refit at 0
        followed by a second refit at S, which the streaming pipeline can
        observe as a one-tick "missing LoRA" glitch in the first decode
        window.

        Refits the engine weights iff the resulting strength is non-zero
        (a strength-0 enable is a placeholder for a slider that hasn't
        ramped yet, and the refit would be a no-op).
        """
        entry = self._require_entry(lora_id)
        if strength is not None:
            entry.strength = float(strength)
        if entry.state == LoRAState.ENABLED:
            return
        if entry.state == LoRAState.MATERIALIZING:
            assert entry.future is not None
            entry.future.result()
            # state should now be MATERIALIZED (or REGISTERED on failure)
        if entry.state == LoRAState.REGISTERED:
            t0 = time.perf_counter()
            deltas, bytes_ = self._compute_deltas(entry.path)
            entry.deltas = deltas
            entry.materialized_bytes = bytes_
            entry.state = LoRAState.MATERIALIZED
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "Materialized TRT LoRA %s inline (%d params, %.1f MB) in %.1fms",
                lora_id, len(deltas), bytes_ / 1e6, elapsed,
            )
        # MATERIALIZED -> ENABLED
        entry.state = LoRAState.ENABLED
        entry.future = None
        if entry.strength != 0.0 and entry.deltas:
            self._refit_weights(set(entry.deltas.keys()))
        logger.info(
            "Enabled TRT LoRA %s (%d params, %.1f MB, strength=%.2f)",
            lora_id, len(entry.deltas or {}),
            entry.materialized_bytes / 1e6, entry.strength,
        )

    def disable_lora(self, lora_id: str) -> None:
        """Drop deltas from CPU RAM and refit if the LoRA was contributing.

        Strength is preserved on the entry so re-enable returns to the
        same slider position.
        """
        entry = self._require_entry(lora_id)
        if entry.state == LoRAState.REGISTERED:
            return

        # If a prewarm is in flight, wait it out so the worker can't
        # later resurrect the deltas we're about to drop.  Worker checks
        # state before assigning, so transitions during the wait are
        # benign.
        if entry.state == LoRAState.MATERIALIZING and entry.future is not None:
            try:
                entry.future.result()
            except Exception:
                pass

        was_contributing = (
            entry.state == LoRAState.ENABLED and entry.strength != 0.0
        )
        affected_params: Set[str] = (
            set(entry.deltas.keys()) if entry.deltas else set()
        )

        entry.state = LoRAState.REGISTERED
        entry.deltas = None
        entry.materialized_bytes = 0
        entry.future = None

        if was_contributing:
            self._refit_weights(affected_params)
        logger.info(
            "Disabled TRT LoRA %s (was_contributing=%s)",
            lora_id, was_contributing,
        )

    def set_lora_strength(self, lora_id: str, strength: float) -> None:
        """Adjust the strength of an ENABLED LoRA.

        Raises ValueError if the LoRA is not enabled.  Auto-enable was
        considered and rejected: it would hide materialization cost
        (hundreds of ms) behind a slider event.  The UI is responsible
        for calling ``enable_lora`` before a slider becomes interactive.
        """
        entry = self._require_entry(lora_id)
        if entry.state != LoRAState.ENABLED:
            raise ValueError(
                f"LoRA {lora_id!r} is not enabled (state={entry.state.value}). "
                "Call enable_lora() first."
            )
        if entry.strength == strength:
            # Slider sometimes lands back on the previous value;
            # skip the full refit + GPU re-upload in that case.
            return
        old = entry.strength
        entry.strength = strength
        if entry.deltas:
            self._refit_weights(set(entry.deltas.keys()))
        logger.info(
            "TRT LoRA %s strength: %.3f -> %.3f (%d params)",
            lora_id, old, strength, len(entry.deltas or {}),
        )

    def remove_lora(self, lora_id=-1) -> bool:
        """Drop a LoRA from the catalog entirely.

        Disables first if enabled.  Default ``-1`` removes the most
        recently registered entry, preserving the legacy stack-pop API
        used by ``acestep.nodes.lora_nodes.RemoveLoRA``.
        """
        if lora_id == -1:
            if not self._loras:
                return False
            lora_id = next(reversed(self._loras))
        if lora_id not in self._loras:
            return False
        self.disable_lora(lora_id)
        del self._loras[lora_id]
        logger.info("Removed TRT LoRA %s from catalog", lora_id)
        return True

    def remove_all(self) -> None:
        """Remove every LoRA from the catalog and restore engine to base."""
        for lid in list(self._loras.keys()):
            self.remove_lora(lid)

    # ------------------------------------------------------------------
    # Backward-compat one-shot API
    # ------------------------------------------------------------------

    def apply_lora(self, lora_path: str, strength: float = 1.0) -> str:
        """Register, set strength, and enable a LoRA in one call.

        Idempotent on path: calling twice is the same as registering
        once and setting the second-call's strength.  Kept for
        ``workflows/``, ``acestep.nodes.lora_nodes``, and demo bootstrap
        paths that haven't migrated to explicit register/enable.
        """
        lora_id = self.register_lora(lora_path)
        self._loras[lora_id].strength = float(strength)
        self.enable_lora(lora_id)
        return lora_id

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_loras(self) -> List[LoRADescriptor]:
        return [
            LoRADescriptor(
                id=e.lora_id, path=e.path, name=e.name,
                state=e.state.value, strength=e.strength,
                materialized_bytes=e.materialized_bytes,
            )
            for e in self._loras.values()
        ]

    def get_lora(self, lora_id: str) -> LoRADescriptor:
        e = self._require_entry(lora_id)
        return LoRADescriptor(
            id=e.lora_id, path=e.path, name=e.name,
            state=e.state.value, strength=e.strength,
            materialized_bytes=e.materialized_bytes,
        )

    def _require_entry(self, lora_id: str) -> _LoRAEntry:
        if lora_id not in self._loras:
            raise ValueError(f"LoRA {lora_id!r} not registered")
        return self._loras[lora_id]

    @property
    def has_active_loras(self) -> bool:
        return any(e.state == LoRAState.ENABLED for e in self._loras.values())

    @property
    def active_lora_count(self) -> int:
        return sum(
            1 for e in self._loras.values() if e.state == LoRAState.ENABLED
        )

    @property
    def active_lora_ids(self) -> List[str]:
        return [
            e.lora_id for e in self._loras.values()
            if e.state == LoRAState.ENABLED
        ]

    @property
    def total_materialized_bytes(self) -> int:
        return sum(e.materialized_bytes for e in self._loras.values())

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

            # Accumulate ENABLED LoRA contributions in-place (native
            # dtype). Skip strength-0 contributions; they're a no-op
            # but the add_ traverses the full weight, which is wasteful
            # for slider-driven UIs that leave placeholders at 0.
            for entry in self._loras.values():
                if entry.state != LoRAState.ENABLED:
                    continue
                if entry.strength == 0.0:
                    continue
                if entry.deltas and param_name in entry.deltas:
                    buf.add_(entry.deltas[param_name], alpha=entry.strength)

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
