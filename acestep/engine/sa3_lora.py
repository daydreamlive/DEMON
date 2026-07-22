"""SA3LoRAManager: DEMON's LoRA lifecycle over the vendored
``stable_audio_3`` parametrization engine.

Design (notes/SA3_LORA_PLAN.md, D1/D4/D5): the manager conforms to the
public manager surface the backend facade and session code consume
(register / prewarm / enable / disable / set_strength / list /
descriptors) by subclassing the shared catalog layer
(:class:`~acestep.engine.lora.LoRACatalogBase`), but its weight
semantics are NOT the ACE delta merge — application is the vendored
package's own math (``LoRAParametrization`` registered via
``torch.nn.utils.parametrize``, live ``lora_strength`` buffer,
``alpha/rank`` scaling, DoRA/BoRA magnitude renormalization), so every
adapter variant renders exactly as the trainer defined it.

Key mechanics, each validated by the Phase 0.5 GPU prototype
(``scripts/sa3/lora_derisk_phase05.py``):

* **Manager-owned slots.** Upstream's ``remove_lora_by_index``
  physically deletes ``ParametrizationList`` entries (shifting later
  positions) while its state-dict loading addresses physical indices —
  so after removing a middle adapter, index-remapped ``load_state_dict``
  can hit the wrong slot. This manager therefore owns the authoritative
  id → ``lora_index`` mapping (a monotonic counter, never reused) and
  installs weights by **copying tensors directly into the
  parametrization objects it just created**, selected by their
  ``lora_index`` attribute — immune to physical shifts.
* **Registration is per-module, driven by the file's own key set**,
  using the exact vendored constructors (``from_linear`` /
  ``from_conv1d`` + ``parametrize.register_parametrization``, the same
  calls upstream ``apply_lora`` makes). This is equivalent to
  upstream's ``add_lora(include/exclude)`` + ``load_state_dict``
  pathway because the trainer saves adapter tensors for exactly the
  modules its filters parametrized — and it never leaves zero-weight
  parametrizations behind to tax the forward pass.
* **Transactional enable.** ENABLED commits only after registration,
  weight install, and the strength-buffer write all succeed; any
  failure rolls the just-allocated index back out of both roots before
  re-raising (bit-identical restore, prototype check 2).
* **Teardown.** The SA3 torch model is process-cached across sessions
  (``sa3_session.get_sa3_context``), so :meth:`close` must hand the
  next session a pristine model: it removes every index this manager
  ever allocated (coping with a partially-failed prior enable) and
  shuts the prewarm executor down.
* **`-xs` variants are rejected at materialize** with a clear error:
  their apply-time CPU SVD measured 70.3 s on medium (prototype check
  3), far beyond the stall pre-coverage envelope; support waits on the
  checkpoint-versioned SVD disk cache work item.

Application roots are upstream's exact ones for ``diffusion_cond``
models: ``sam.model.model`` (the DiT tree) and ``sam.model.conditioner``
— never ``sam.model`` wholesale (which would also sweep the
pretransform and double-traverse the conditioner).

Threading contract (same as ACE's managers): register / enable /
disable / set_strength run on the inference-owning thread (the
session's pending-drain rendezvous); prewarm materializes on a
background executor and touches only staged CPU state, never the model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from loguru import logger
import torch

from acestep.engine.lora import (
    LoRACatalogBase,
    LoRAState,
    _LoRAEntry,
)

# Adapter tensor names the vendored trainer saves, by adapter family.
_SA3_PARAM_NAMES = {
    "lora_A", "lora_B", "magnitude", "magnitude_r", "magnitude_c", "M_xs",
}
_SA3_KEY_MARKER = ".parametrizations.weight."
_ACE_KEY_MARKERS = (".lora_A.weight", ".lora_B.weight")


def _vendored_lora():
    """Import the vendored LoRA modules through the managed sys.path
    setup SA3Context already uses — never a pip dependency."""
    from acestep.engine.sa3_helpers import ensure_sa3_paths

    ensure_sa3_paths()
    from stable_audio_3.models.lora import model as lora_model  # noqa: PLC0415
    from stable_audio_3.models.lora import utils as lora_utils  # noqa: PLC0415

    return lora_model, lora_utils


@dataclass
class _StagedSA3File:
    """CPU-staged, validated payload for one materialized file."""

    adapter_type: str
    rank: int
    alpha: float
    # (root_index, module_fqn) -> {param_name: cpu tensor}. root_index
    # 0 = DiT root, 1 = conditioner root.
    weights: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = field(
        default_factory=dict,
    )
    touches_conditioner: bool = False
    total_bytes: int = 0


class SA3LoRAManager(LoRACatalogBase):
    """SA3 LoRA lifecycle over the vendored parametrization engine.

    ``model_root`` / ``conditioner_root`` are the live application
    roots (``sam.model.model`` / ``sam.model.conditioner``);
    ``checkpoint_id`` is the runtime model id ("medium" /
    "small-music" / "small-sfx") the lineage check compares against.
    """

    def __init__(
        self,
        *,
        model_root: torch.nn.Module,
        conditioner_root: Optional[torch.nn.Module] = None,
        checkpoint_id: str = "",
    ) -> None:
        if model_root is None:
            raise ValueError("SA3LoRAManager requires a model_root module")
        self._model_root = model_root
        self._conditioner_root = conditioner_root
        self._checkpoint_id = str(checkpoint_id)

        # Authoritative id -> lora_index mapping for ENABLED adapters.
        # Indices come from a monotonic counter and are never reused, so
        # a stale index can never alias a live adapter (D4).
        self._index_by_id: Dict[str, int] = {}
        self._index_counter = 0
        # Every index ever allocated — the close() sweep target, so a
        # partially-failed enable whose rollback also failed still gets
        # stripped from the process-cached model at teardown.
        self._allocated_indices: set[int] = set()
        # Staged payloads for MATERIALIZED/ENABLED entries, keyed by id.
        # Written by the prewarm worker (CPU only), read on the
        # inference thread — same single-worker + GIL discipline as the
        # ACE managers' entry.deltas.
        self._staged: Dict[str, _StagedSA3File] = {}

        super().__init__()
        logger.info(
            "SA3 LoRA manager ready: checkpoint_id={} conditioner={}",
            self._checkpoint_id, conditioner_root is not None,
        )

    # ------------------------------------------------------------------
    # Root / module helpers
    # ------------------------------------------------------------------

    def _roots(self) -> list:
        roots = [self._model_root]
        if self._conditioner_root is not None:
            roots.append(self._conditioner_root)
        return roots

    def _resolve_module(self, fqn: str):
        """Locate ``fqn`` under the application roots. Returns
        ``(root_index, module)`` or ``(None, None)`` on a miss."""
        for ri, root in enumerate(self._roots()):
            mod = root
            found = True
            for part in fqn.split("."):
                if not hasattr(mod, part):
                    found = False
                    break
                mod = getattr(mod, part)
            if found and isinstance(mod, torch.nn.Module):
                return ri, mod
        return None, None

    # ------------------------------------------------------------------
    # Materialize (background executor; GPU-free)
    # ------------------------------------------------------------------

    def _materialize_worker(self, entry: _LoRAEntry) -> None:
        t0 = time.perf_counter()
        try:
            staged = self._stage_file(entry.path)
        except Exception:
            if entry.state == LoRAState.MATERIALIZING:
                entry.state = LoRAState.REGISTERED
                entry.future = None
            raise
        if entry.state == LoRAState.MATERIALIZING:
            self._staged[entry.lora_id] = staged
            entry.materialized_bytes = staged.total_bytes
            entry.state = LoRAState.MATERIALIZED
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "Materialized SA3 LoRA {} ({} modules, {:.1f} MB, "
                "adapter={}, rank={}) in {:.1f}ms",
                entry.lora_id, len(staged.weights),
                staged.total_bytes / 1e6, staged.adapter_type,
                staged.rank, elapsed,
            )

    def _stage_file(self, lora_path: str) -> _StagedSA3File:
        """Read + validate one file into a CPU-staged payload.

        Hard validation at this boundary (D3): wrong-family key layouts,
        ``-xs`` variants, lineage mismatches, and
        no-module-matches-the-model all raise typed errors — a bad file
        must never become a silent no-op enable.
        """
        _, lora_utils = _vendored_lora()
        name = Path(lora_path).name

        if not str(lora_path).endswith(".safetensors"):
            # Upstream's .ckpt path is torch.load with pickles — an RCE
            # vector we never expose at runtime.
            raise RuntimeError(
                f"SA3 LoRA {name}: only .safetensors files are accepted "
                f"(.ckpt is a pickle and is never loaded at runtime; "
                f"convert with stable_audio_3's "
                f"convert_lora_ckpt_to_safetensors)"
            )

        sd, config = lora_utils.load_lora_checkpoint(lora_path)

        # Family validation from the actual key layout.
        if not any(_SA3_KEY_MARKER in k for k in sd):
            ace_like = any(
                m in k for k in sd for m in _ACE_KEY_MARKERS
            )
            hint = (
                " The key layout matches the ACE-Step (PEFT) format; "
                "ACE LoRAs cannot load on an SA3 model."
                if ace_like else ""
            )
            raise RuntimeError(
                f"LoRA {name} is not an SA3 LoRA: none of its "
                f"{len(sd)} tensors use the parametrization key layout "
                f"(…​.parametrizations.weight.<idx>.<param>).{hint}"
            )

        adapter_type = lora_utils.resolve_adapter_type(
            config.get("adapter_type", "lora"), sd,
        )
        if adapter_type.endswith("-xs"):
            raise RuntimeError(
                f"SA3 LoRA {name}: adapter type {adapter_type!r} is not "
                f"yet supported — -xs variants need SVD bases of the "
                f"base weights at apply time (~70s CPU on medium, "
                f"measured), which requires the planned per-checkpoint "
                f"SVD disk cache. Use a non-xs export of this adapter."
            )

        rank = int(config.get("rank") or lora_utils.infer_global_rank(sd))
        alpha = float(config.get("alpha") or rank)

        # Lineage check (the SA3 analog of ACE's "2B LoRA on XL engine"
        # message). Permissive when the file doesn't say or the spelling
        # is unrecognized, per the seam contract.
        from acestep.lora_metadata import canonical_sa3_lineage

        lineage = canonical_sa3_lineage(config.get("base_model"))
        if lineage is not None and self._checkpoint_id and (
            lineage != self._checkpoint_id
        ):
            raise RuntimeError(
                f"SA3 LoRA {name} was trained against the "
                f"{config.get('base_model')!r} base model "
                f"(runtime lineage {lineage!r}) but the loaded "
                f"checkpoint is {self._checkpoint_id!r}. Load it on a "
                f"matching session."
            )

        # DoRA magnitude tensors may be saved 2D; squeeze to the 1D the
        # live parametrization params use (upstream loader parity).
        lora_utils.prepare_dora_state_dict(sd)

        # Group tensors by module FQN and resolve against the live tree.
        by_fqn: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, tensor in sd.items():
            marker_at = key.find(_SA3_KEY_MARKER)
            if marker_at < 0:
                continue
            fqn = key[:marker_at]
            tail = key[marker_at + len(_SA3_KEY_MARKER):]
            parts = tail.split(".", 1)
            if len(parts) != 2 or parts[1] not in _SA3_PARAM_NAMES:
                continue
            by_fqn.setdefault(fqn, {})[parts[1]] = tensor

        pin = torch.cuda.is_available()
        staged = _StagedSA3File(
            adapter_type=adapter_type, rank=rank, alpha=alpha,
        )
        unmatched = 0
        shape_mismatch = 0
        first_mismatch: Optional[tuple] = None
        for fqn, params in by_fqn.items():
            ri, mod = self._resolve_module(fqn)
            if ri is None or not isinstance(
                mod, (torch.nn.Linear, torch.nn.Conv1d),
            ):
                unmatched += 1
                continue
            weight = getattr(mod, "weight", None)
            if weight is None:
                unmatched += 1
                continue
            w2 = weight.view(weight.shape[0], -1)
            fan_out, fan_in = int(w2.shape[0]), int(w2.shape[1])
            a = params.get("lora_A")
            b = params.get("lora_B")
            bad = (
                (a is not None and tuple(a.shape) != (rank, fan_in))
                or (b is not None and tuple(b.shape) != (fan_out, rank))
            )
            if bad:
                shape_mismatch += 1
                if first_mismatch is None:
                    got = tuple(a.shape) if a is not None else tuple(b.shape)
                    first_mismatch = (fqn, got, (fan_out, fan_in))
                continue
            cpu_params = {}
            for pname, t in params.items():
                t = t.detach().contiguous()
                if pin:
                    t = t.pin_memory()
                cpu_params[pname] = t
                staged.total_bytes += t.numel() * t.element_size()
            staged.weights[(ri, fqn)] = cpu_params
            if ri == 1:
                staged.touches_conditioner = True

        if not staged.weights:
            detail = ""
            if first_mismatch is not None:
                fqn, got, fans = first_mismatch
                detail = (
                    f" E.g. {fqn!r}: adapter shape {got} does not fit "
                    f"the base weight (fan_out, fan_in)={fans}."
                )
            raise RuntimeError(
                f"SA3 LoRA {name} does not fit the loaded "
                f"{self._checkpoint_id or 'model'}: "
                f"{unmatched} module(s) missing from the model tree, "
                f"{shape_mismatch} with mismatched shapes.{detail} "
                f"Common cause: a LoRA trained for a different SA3 "
                f"checkpoint size."
            )
        if unmatched or shape_mismatch:
            logger.warning(
                "SA3 LoRA {}: {} modules staged, {} unmatched, {} "
                "shape-mismatched (partial overlap; the rest applies)",
                name, len(staged.weights), unmatched, shape_mismatch,
            )
        return staged

    # ------------------------------------------------------------------
    # Enable / disable / strength (inference thread)
    # ------------------------------------------------------------------

    def touches_conditioner(self, lora_id: str) -> bool:
        """Whether ``lora_id``'s staged payload targets the conditioner
        (drives the backend's cond-bundle rebuild, D5). False for
        entries that aren't materialized."""
        staged = self._staged.get(lora_id)
        return bool(staged is not None and staged.touches_conditioner)

    def enable_lora(
        self, lora_id: str, strength: Optional[float] = None,
    ) -> None:
        """Promote a LoRA to ENABLED — transactional.

        Registers parametrizations on the staged modules, installs the
        staged weights by direct copy, writes the strength buffer, and
        only then commits ENABLED state + the id→index mapping. On any
        failure the just-allocated index is removed from both roots
        (bit-identical restore, Phase 0.5 check 2) before re-raising —
        the loud-failure path, never a half-applied adapter.

        Atomic enable-at-strength holds structurally: everything here
        runs on the inference thread inside the pending-drain
        rendezvous, so no forward pass can observe the intermediate
        state, and the strength buffer is written before commit.
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
            staged = self._stage_file(entry.path)
            self._staged[lora_id] = staged
            entry.materialized_bytes = staged.total_bytes
            entry.state = LoRAState.MATERIALIZED
            logger.info(
                "Materialized SA3 LoRA {} inline in {:.1f}ms",
                lora_id, (time.perf_counter() - t0) * 1000,
            )
        staged = self._staged[lora_id]

        lora_model, _ = _vendored_lora()
        from functools import partial

        import torch.nn.utils.parametrize as parametrize

        idx = self._index_counter
        self._index_counter += 1
        self._allocated_indices.add(idx)

        t0 = time.perf_counter()
        try:
            # 1. Register parametrizations on exactly the staged
            #    modules, via the vendored constructors, keeping a
            #    direct handle on each created object so the install
            #    step below cannot depend on physical list positions.
            created: list = []
            for (ri, fqn), params in staged.weights.items():
                _ri, mod = self._resolve_module(fqn)
                if mod is None:
                    raise RuntimeError(
                        f"SA3 LoRA {lora_id}: staged module {fqn!r} "
                        f"vanished from the model tree"
                    )
                if isinstance(mod, torch.nn.Conv1d):
                    ctor = lora_model.LoRAParametrization.from_conv1d
                else:
                    ctor = lora_model.LoRAParametrization.from_linear
                param_fn = partial(
                    ctor,
                    rank=staged.rank, lora_alpha=staged.alpha,
                    adapter_type=staged.adapter_type, lora_index=idx,
                )
                p_obj = param_fn(mod)
                parametrize.register_parametrization(
                    mod, "weight", p_obj, unsafe=True,
                )
                created.append((p_obj, params))
            # 2. Direct-copy install into the objects just created (D4:
            #    bypasses index-remapped load_state_dict entirely, so
            #    enable is immune to prior removals' physical shifts).
            with torch.no_grad():
                for p_obj, params in created:
                    for pname, tensor in params.items():
                        target = getattr(p_obj, pname, None)
                        if target is None:
                            logger.warning(
                                "SA3 LoRA {}: adapter param {} missing "
                                "on live parametrization; skipped",
                                lora_id, pname,
                            )
                            continue
                        target.copy_(tensor.to(
                            device=target.device, dtype=target.dtype,
                            non_blocking=True,
                        ))
            # 3. Strength buffer — before commit, so the first forward
            #    after the rendezvous already sees the target strength.
            for root in self._roots():
                lora_model.set_lora_strength(
                    root, float(entry.strength), lora_index=idx,
                )
        except Exception:
            self._remove_index(idx)
            raise

        # 4. Commit.
        self._index_by_id[lora_id] = idx
        entry.state = LoRAState.ENABLED
        entry.future = None
        logger.info(
            "Enabled SA3 LoRA {} (index={}, {} modules, adapter={}, "
            "strength={:.2f}, conditioner={}) in {:.1f}ms",
            lora_id, idx, len(staged.weights), staged.adapter_type,
            entry.strength, staged.touches_conditioner,
            (time.perf_counter() - t0) * 1000,
        )

    def _remove_index(self, idx: int) -> None:
        """Strip every parametrization carrying ``idx`` from both roots
        (upstream removal; restores the original weight when a module's
        last adapter goes)."""
        lora_model, _ = _vendored_lora()
        for root in self._roots():
            lora_model.remove_lora_by_index(root, idx)

    def disable_lora(self, lora_id: str) -> None:
        """Remove the adapter from the live model, drop staged CPU
        tensors, and return the freed VRAM to the driver. Strength is
        preserved on the entry so re-enable returns to the same slider
        position (ACE convention)."""
        entry = self._require_entry(lora_id)
        if entry.state == LoRAState.REGISTERED:
            return
        if entry.state == LoRAState.MATERIALIZING and entry.future is not None:
            try:
                entry.future.result()
            except Exception:
                pass

        was_enabled = entry.state == LoRAState.ENABLED
        idx = self._index_by_id.pop(lora_id, None)
        entry.state = LoRAState.REGISTERED
        entry.materialized_bytes = 0
        entry.future = None
        self._staged.pop(lora_id, None)

        if was_enabled and idx is not None:
            self._remove_index(idx)
            # Same fragmentation rationale as the ACE managers: hand the
            # freed adapter tensors back to the driver at the actual
            # free event. The gc.collect() first is load-bearing here:
            # torch's parametrize injects a per-module class whose type
            # object sits in reference cycles, so the removed
            # ParametrizationList (and the adapter parameters it holds)
            # waits on the cycle collector — measured on medium, the
            # ~38 MB rank-8 footprint stays "allocated" until then.
            if torch.cuda.is_available():
                import gc

                gc.collect()
                torch.cuda.empty_cache()
        logger.info(
            "Disabled SA3 LoRA {} (was_enabled={})", lora_id, was_enabled,
        )

    def set_lora_strength(self, lora_id: str, strength: float) -> None:
        """Live strength update: a buffer write on every parametrization
        of this adapter — no recompute, no refit (upstream's realtime
        mechanism). Raises for non-enabled ids, matching ACE."""
        entry = self._require_entry(lora_id)
        if entry.state != LoRAState.ENABLED:
            raise ValueError(
                f"LoRA {lora_id!r} is not enabled (state={entry.state.value}). "
                "Call enable_lora() first."
            )
        if entry.strength == strength:
            return
        old = entry.strength
        entry.strength = float(strength)
        idx = self._index_by_id[lora_id]
        lora_model, _ = _vendored_lora()
        for root in self._roots():
            lora_model.set_lora_strength(
                root, float(strength), lora_index=idx,
            )
        logger.info(
            "SA3 LoRA {} strength: {:.3f} -> {:.3f}", lora_id, old, strength,
        )

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Return the process-cached model to pristine state: strip
        every index this manager ever allocated (not just the currently
        enabled ones — a partially-failed enable whose rollback also
        failed still gets swept), drop staged payloads, and shut down
        the prewarm executor. Idempotent. Phase 0.5 check 4 validated
        that this restore is bitwise-complete."""
        for idx in sorted(self._allocated_indices):
            try:
                self._remove_index(idx)
            except Exception as e:
                logger.warning(
                    "SA3 LoRA teardown: removing index {} raised: {}",
                    idx, e,
                )
        self._allocated_indices.clear()
        self._index_by_id.clear()
        self._staged.clear()
        for entry in self._loras.values():
            if entry.state != LoRAState.REGISTERED:
                entry.state = LoRAState.REGISTERED
            entry.materialized_bytes = 0
            entry.future = None
        self._loras.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        if torch.cuda.is_available():
            import gc

            gc.collect()  # break the injected-class cycles (see disable)
            torch.cuda.empty_cache()
