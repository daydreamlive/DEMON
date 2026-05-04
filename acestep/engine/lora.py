"""Backend-agnostic LoRA manager.

Owns the catalog, lifecycle (REGISTERED -> MATERIALIZING -> MATERIALIZED
-> ENABLED), background prewarm, and delta math (B @ A) for dynamic LoRA
application. Subclasses plug in *where the live weights live*: a TRT
engine via IRefitter, or a PyTorch decoder's parameters directly.

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

import abc
import concurrent.futures
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger
import torch
from safetensors.torch import load_file


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


class LoRAManagerBase(abc.ABC):
    """Catalog + lifecycle + delta math; subclasses do the engine writeback.

    Subclasses must:

    - Populate ``self._base_weights`` (dict: param_name -> CPU/GPU tensor in
      the live weight's native dtype) and ``self._param_dtype`` (dict:
      param_name -> torch.dtype) during __init__, then call
      ``super().__init__()`` to set up the lifecycle bookkeeping.
    - Implement :meth:`_apply_to_engine`, which writes
      ``base + sum_over_enabled(strength * delta)`` to the live weights
      for the supplied ``param_names``.

    Param naming is decoder-relative (matches ``decoder.named_parameters()``).
    The TRT subclass adds the ``decoder.`` prefix only at the refitter
    boundary.
    """

    def __init__(self) -> None:
        # Subclasses set _base_weights and _param_dtype before super().__init__.
        if not hasattr(self, "_base_weights"):
            self._base_weights: Dict[str, torch.Tensor] = {}
        if not hasattr(self, "_param_dtype"):
            self._param_dtype: Dict[str, torch.dtype] = {}

        # Library + lifecycle state.  Insertion order is preserved so
        # ``remove_lora(-1)`` can pop the most-recently-registered entry,
        # matching the legacy stack-style API.
        self._loras: Dict[str, _LoRAEntry] = {}
        self._ever_dirty: Set[str] = set()
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _apply_to_engine(self, param_names: Set[str]) -> None:
        """Write ``base + Σ strength·delta`` to the live weights.

        ``param_names`` is the set of decoder-relative param names that
        need updating. Subclasses iterate, accumulate enabled-LoRA
        contributions in a buffer, and push to the live weights.
        """

    # ------------------------------------------------------------------
    # Lifecycle transition hooks (default: no-ops)
    #
    # The base lifecycle (enable_lora / disable_lora) calls these around
    # state transitions so subclasses can attach side effects without
    # overriding the lifecycle proper. The motivating case is the eager
    # backend's "promote materialized deltas to GPU on enable, drop the
    # GPU mirror on disable" pattern: deltas live cheaply in CPU RAM
    # while merely MATERIALIZED, but the active set lives in VRAM so
    # slider-driven refits stay zero-copy on the device.
    # ------------------------------------------------------------------

    def _on_enabled(self, entry: "_LoRAEntry") -> None:
        """Hook: ``entry`` just transitioned to ENABLED. Called BEFORE
        the contributing-strength refit fires so subclasses can stage
        runtime data the refit will read."""

    def _on_disabled(self, entry: "_LoRAEntry") -> None:
        """Hook: ``entry`` just transitioned away from ENABLED. Called
        AFTER ``entry.deltas`` is cleared but BEFORE the rollback refit
        fires (so subclasses can drop their mirrors first)."""

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
        logger.info("Registered LoRA: %s (path=%s)", lora_id, path)
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
        entry.state = LoRAState.MATERIALIZING
        entry.future = self._get_executor().submit(
            self._materialize_worker, entry,
        )
        logger.info("Prewarming LoRA: %s", lora_id)
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
                "Materialized LoRA %s (%d params, %.1f MB) in %.1fms",
                entry.lora_id, len(deltas), bytes_ / 1e6, elapsed,
            )

    def _compute_deltas(
        self, lora_path: str,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Load LoRA from disk and compute full-rank deltas (B @ A).

        Pure compute; safe to call from a worker thread.  Returns
        (deltas_dict, total_bytes_in_cpu_ram). Each delta is stored in
        the live weight's native dtype on CPU. Subclasses pull deltas
        through to whatever device they need.
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

        device = self._delta_compute_device()
        deltas: Dict[str, torch.Tensor] = {}
        total_bytes = 0
        skipped = 0
        for param_name, ab in pairs.items():
            if "A" not in ab or "B" not in ab:
                continue
            if param_name not in self._param_dtype:
                skipped += 1
                continue
            A = ab["A"].to(device=device, dtype=torch.float32)
            B = ab["B"].to(device=device, dtype=torch.float32)
            target_dt = self._param_dtype[param_name]
            d = (B @ A).to(dtype=target_dt).to(
                device=self._delta_storage_device(),
            ).contiguous()
            deltas[param_name] = d
            total_bytes += d.numel() * d.element_size()
        if skipped:
            logger.debug(
                "_compute_deltas(%s): %d params skipped (not in engine)",
                Path(lora_path).name, skipped,
            )
        return deltas, total_bytes

    def _delta_compute_device(self) -> torch.device:
        """Where the (B @ A) matmul runs. GPU when available."""
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _delta_storage_device(self) -> torch.device:
        """Where deltas live in RAM after materialization.

        TRT keeps them on CPU because refit buffers and numpy views are
        CPU. Eager keeps them on the decoder's device so refits don't
        pay an H2D transfer per slider tick.
        """
        return torch.device("cpu")

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
        if entry.state == LoRAState.REGISTERED:
            t0 = time.perf_counter()
            deltas, bytes_ = self._compute_deltas(entry.path)
            entry.deltas = deltas
            entry.materialized_bytes = bytes_
            entry.state = LoRAState.MATERIALIZED
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "Materialized LoRA %s inline (%d params, %.1f MB) in %.1fms",
                lora_id, len(deltas), bytes_ / 1e6, elapsed,
            )
        entry.state = LoRAState.ENABLED
        entry.future = None
        # Subclass hook: stage backend-specific runtime state (e.g. the
        # eager backend's CPU->GPU delta promotion) BEFORE the refit so
        # the refit can read it.
        self._on_enabled(entry)
        if entry.strength != 0.0 and entry.deltas:
            self._refit_weights(set(entry.deltas.keys()))
        logger.info(
            "Enabled LoRA %s (%d params, %.1f MB, strength=%.2f)",
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

        # Subclass hook: drop backend-specific runtime state (e.g. the
        # eager backend's GPU delta mirror). Runs before the rollback
        # refit so the composing code never sees a stale mirror.
        self._on_disabled(entry)

        if was_contributing:
            self._refit_weights(affected_params)
        logger.info(
            "Disabled LoRA %s (was_contributing=%s)",
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
            return
        old = entry.strength
        entry.strength = strength
        if entry.deltas:
            self._refit_weights(set(entry.deltas.keys()))
        logger.info(
            "LoRA %s strength: %.3f -> %.3f (%d params)",
            lora_id, old, strength, len(entry.deltas or {}),
        )

    def remove_lora(self, lora_id=-1) -> bool:
        """Drop a LoRA from the catalog entirely.

        Disables first if enabled.  Default ``-1`` removes the most
        recently registered entry, preserving the legacy stack-pop API.
        """
        if lora_id == -1:
            if not self._loras:
                return False
            lora_id = next(reversed(self._loras))
        if lora_id not in self._loras:
            return False
        self.disable_lora(lora_id)
        del self._loras[lora_id]
        logger.info("Removed LoRA %s from catalog", lora_id)
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
        once and setting the second-call's strength.
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
        return len(self._param_dtype)

    # ------------------------------------------------------------------
    # Refit driver (subclass writeback)
    # ------------------------------------------------------------------

    def _refit_weights(self, param_names: Set[str]) -> None:
        """Time + log the engine writeback. Subclass does the actual work.

        Splitting time/log out of ``_apply_to_engine`` keeps the subclass
        method narrow and lets the TRT path reuse pre-allocated refit
        buffers without bookkeeping noise.
        """
        if not param_names:
            return
        t0 = time.perf_counter()
        self._apply_to_engine(param_names)
        for name in param_names:
            self._ever_dirty.add(name)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Refitted %d weights in %.1fms", len(param_names), elapsed,
        )


class EagerLoRAManager(LoRAManagerBase):
    """LoRA writeback into a PyTorch decoder's parameters in place.

    Storage strategy ("hybrid"):

    - ``MATERIALIZED`` deltas live on **CPU**, matching the TRT backend.
      A library workflow that prewarms dozens of LoRAs costs system
      RAM, not VRAM.
    - On the ENABLED transition, the entry's deltas are mirrored into a
      per-id **GPU** buffer (``self._gpu_deltas[id]``). Refits read from
      that mirror, so slider-driven strength updates run zero-copy on
      the device. The mirror is freed on disable.
    - The base-weight snapshot is also lazy: nothing is cloned at
      construction. The first refit clones the live params it's about
      to overwrite into ``self._base_weights``, so a session that never
      enables a LoRA pays no extra VRAM.

    All writes use ``param.data.copy_`` / ``.add_`` so the live
    ``nn.Parameter`` identity stays the same — required for safety
    under ``torch.compile``: the compiled graph reads params as tensor
    inputs (not constants), and storage-level mutations are visible to
    the next forward without retracing.
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        device: Optional[torch.device] = None,
    ):
        if decoder is None:
            raise ValueError("EagerLoRAManager requires a decoder module")

        # If the decoder has been wrapped by ``torch.compile``, the live
        # parameters live on the original module; named_parameters()
        # routes through the OptimizedModule wrapper transparently, so
        # this works for both compiled and eager decoders.
        named = list(decoder.named_parameters())
        if not named:
            raise RuntimeError(
                "Decoder has no parameters; cannot init EagerLoRAManager. "
                "(If running with skip_decoder=True, use the TRT path.)"
            )

        # Resolve device: caller-provided wins, else infer from the first
        # parameter. We tolerate a meta-device decoder by deferring to CPU.
        first_param = named[0][1]
        inferred = first_param.device
        self._device = device if device is not None else (
            inferred if inferred.type != "meta" else torch.device("cpu")
        )

        self._decoder_params: Dict[str, torch.nn.Parameter] = {}
        # Populated lazily on first refit (see _ensure_base_snapshot).
        self._base_weights: Dict[str, torch.Tensor] = {}
        self._param_dtype: Dict[str, torch.dtype] = {}
        # Per-id GPU mirror of MATERIALIZED CPU deltas, populated on
        # _on_enabled and dropped on _on_disabled.
        self._gpu_deltas: Dict[str, Dict[str, torch.Tensor]] = {}

        for name, param in named:
            self._decoder_params[name] = param
            self._param_dtype[name] = param.dtype

        logger.info(
            "Eager LoRA manager ready: %d decoder params indexed on %s "
            "(base snapshot deferred until first enable)",
            len(self._param_dtype), self._device,
        )

        super().__init__()

    def _delta_compute_device(self) -> torch.device:
        return self._device

    # _delta_storage_device falls back to base default (CPU): MATERIALIZED
    # deltas live in system RAM. The hot, GPU-resident mirror is only
    # populated for ENABLED entries via _on_enabled.

    def _on_enabled(self, entry: "_LoRAEntry") -> None:
        """Promote CPU deltas to a GPU mirror so refits stay zero-copy.

        Pays one H2D transfer at enable time; subsequent refits (slider
        ticks, disable rollbacks) read straight from the device mirror.
        """
        if not entry.deltas:
            return
        self._gpu_deltas[entry.lora_id] = {
            name: d.to(device=self._device, non_blocking=True).contiguous()
            for name, d in entry.deltas.items()
        }

    def _on_disabled(self, entry: "_LoRAEntry") -> None:
        """Drop the GPU mirror; the CPU MATERIALIZED copy was already
        cleared by the lifecycle. Frees VRAM equal to the LoRA's delta
        footprint."""
        self._gpu_deltas.pop(entry.lora_id, None)

    def _ensure_base_snapshot(self, param_names: Set[str]) -> None:
        """Lazily clone live params into ``_base_weights`` for the
        requested set.

        Called on the refit hot path. The clone happens at most once
        per param: after first capture, subsequent refits reuse the
        snapshot to recompose ``base + Σ s·d``. A session that only
        ever sits at strength 0 (placeholder pattern) never triggers a
        snapshot because no refit fires.
        """
        for name in param_names:
            if name in self._base_weights:
                continue
            param = self._decoder_params.get(name)
            if param is None:
                continue
            self._base_weights[name] = param.data.detach().clone()

    def _apply_to_engine(self, param_names: Set[str]) -> None:
        """Compose ``base + Σ strength·delta`` directly into param.data.

        Reads deltas from ``self._gpu_deltas`` (the on-device mirror),
        not from ``entry.deltas`` (which live on CPU). The mirror is
        guaranteed populated for every ENABLED entry by ``_on_enabled``.
        """
        self._ensure_base_snapshot(param_names)

        for param_name in param_names:
            param = self._decoder_params.get(param_name)
            if param is None:
                continue
            param.data.copy_(self._base_weights[param_name])

            for entry in self._loras.values():
                if entry.state != LoRAState.ENABLED:
                    continue
                if entry.strength == 0.0:
                    continue
                gpu = self._gpu_deltas.get(entry.lora_id)
                if gpu and param_name in gpu:
                    param.data.add_(gpu[param_name], alpha=entry.strength)
