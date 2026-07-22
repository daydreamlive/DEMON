"""SA3TRTRefitMirror: LoRA weights onto a refittable SA3 DiT engine.

The Phase-2 endgame of notes/SA3_LORA_PLAN.md (D6b): instead of the
interim eager-DiT swap, a LoRA-enabled TRT session runs a refit-built
engine (``sa3_m_dit_refit_l*``, exclusively owned — see
``acestep.engine.sa3_trt._deserialize_engine``) and mirrors the eager
parametrization state onto it after every LoRA mutation.

Composition is **merged-weight by construction**: the source of truth
stays the eager modules that :class:`~acestep.engine.sa3_lora.SA3LoRAManager`
parametrizes — reading ``module.weight`` evaluates the vendored
parametrization chain (alpha/rank scaling, DoRA/BoRA magnitude
renormalization, live strength buffers), so the mirrored value is exact
for every adapter variant with zero re-implemented math. The mirror
D2Hs each parametrized module's merged weight into a staging buffer,
pushes it under the ONNX initializer name the manifest maps it to
(transposing when the graph stores the MatMul ``[in, out]``
orientation), and commits one ``refit_cuda_engine()``.

The manifest (``gen_sa3_refit_manifest.py``) is generated offline by
shape + value fingerprinting over Stability's ONNX and validated by a
bit-identity refit (every mapped weight refit with its own base value
must leave the engine output bit-identical) — a wrong manifest cannot
ship silently.

Sync scope: modules that currently carry parametrizations (their merged
value), plus previously-synced modules whose adapters were since
removed (their restored base goes back — the dirty set). Untouched
weights already hold their build-time values, and on session close the
engine is simply dropped (exclusive ownership — the rollback
guarantee), so no full base-restore pass ever runs.

Sync stalls are full refits (hundreds of ms class). They are always
announced before they run: enables/disables ride ``has_pending_refit``
via the session pending queues, and knob-driven strength changes are
detected in ``rebuild_imminent`` and stashed (never applied
mid-tick unannounced) — see ``SA3Backend`` (plan D6b.5).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

from loguru import logger
import torch

from acestep.engine.trt.refit_core import (
    commit_refit,
    np_view_for_push,
    set_typed_weights,
)

MANIFEST_VERSION = 1
# Shared (per-ONNX) manifest filename; the per-engine sidecar
# ``<engine>.refit_manifest.json`` wins when present.
SHARED_MANIFEST_NAME = "sa3_m_dit_refit_manifest.json"


def find_refit_manifest(engine_path: Path) -> Optional[Path]:
    """Manifest lookup: per-engine sidecar first, then the shared
    per-ONNX manifest in the engines root."""
    engine_path = Path(engine_path)
    sidecar = Path(str(engine_path) + ".refit_manifest.json")
    if sidecar.is_file():
        return sidecar
    shared = engine_path.parent.parent / SHARED_MANIFEST_NAME
    if shared.is_file():
        return shared
    return None


class SA3TRTRefitMirror:
    """See module docstring. ``engine`` is the exclusively-owned
    refittable engine; ``model_root`` the live DiT tree
    (``sam.model.model``) the SA3LoRAManager parametrizes."""

    def __init__(
        self,
        engine,
        model_root: torch.nn.Module,
        manifest_path: Path,
        *,
        _refitter=None,
        _trt=None,
    ) -> None:
        # _refitter/_trt are test injection points: the dirty-set and
        # staging logic is CPU-testable against a recording fake, while
        # production always constructs the real IRefitter here.
        if _trt is None:
            import tensorrt as _trt  # noqa: PLC0415

        trt = _trt
        self._trt = trt
        self._model_root = model_root
        self._refitter = _refitter if _refitter is not None else (
            trt.Refitter(engine, trt.Logger(trt.Logger.WARNING))
        )
        if not hasattr(self._refitter, "get_all_weights"):
            raise RuntimeError("TRT engine refitting requires TensorRT 10.0+")
        refittable_names = set(self._refitter.get_all_weights())
        if not refittable_names:
            raise RuntimeError(
                "engine has no refittable weights; was it built with "
                "sa3_build --refit?"
            )

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("version") != MANIFEST_VERSION:
            raise RuntimeError(
                f"refit manifest {manifest_path} has version "
                f"{manifest.get('version')}, expected {MANIFEST_VERSION}"
            )
        # fqn -> (initializer_name, transposed)
        self._map: Dict[str, tuple] = {}
        missing = 0
        for fqn, rec in manifest.get("weights", {}).items():
            init = rec["initializer"]
            if init not in refittable_names:
                missing += 1
                continue
            self._map[fqn] = (init, bool(rec.get("transposed", False)))
        if not self._map:
            raise RuntimeError(
                f"refit manifest {manifest_path} maps no weight that this "
                f"engine exposes as refittable ({missing} mapped to "
                f"non-refittable names)"
            )
        if missing:
            logger.warning(
                "sa3_refit_manifest_partial mapped={} not_refittable={}",
                len(self._map), missing,
            )
        # Staging buffers, allocated lazily per synced fqn (pinned when
        # possible) and reused across syncs; each must stay alive until
        # commit (TRT dereferences the host pointers there).
        self._staging: Dict[str, torch.Tensor] = {}
        self._dtype_cache: Dict[str, tuple] = {}
        # Fqns whose engine slot currently holds a LoRA-merged value. A
        # later sync that finds such a module UNparametrized (adapter
        # disabled) must push its base weight back — skipping it would
        # leave the stale merged value in the engine.
        self._dirty: set = set()
        logger.info(
            "sa3_trt_refit_mirror_ready mapped_weights={} manifest={}",
            len(self._map), manifest_path,
        )

    def _resolve_module(self, fqn: str):
        mod = self._model_root
        for part in fqn.split("."):
            if not hasattr(mod, part):
                return None
            mod = getattr(mod, part)
        return mod

    def _prototype(self, init_name: str):
        """(trt dtype, torch staging dtype) for one initializer."""
        cached = self._dtype_cache.get(init_name)
        if cached is not None:
            return cached
        trt = self._trt
        torch_dt = torch.float32
        trt_dt = trt.float32
        try:
            proto = self._refitter.get_weights_prototype(init_name)
            trt_dt = proto.dtype
            torch_dt = {
                trt.float32: torch.float32,
                trt.float16: torch.float16,
            }.get(proto.dtype, torch.float32)
            if hasattr(trt, "bfloat16") and proto.dtype == trt.bfloat16:
                torch_dt = torch.bfloat16
        except Exception:
            pass
        self._dtype_cache[init_name] = (trt_dt, torch_dt)
        return trt_dt, torch_dt

    @torch.no_grad()
    def sync(self, *, reason: str = "") -> int:
        """Push the merged weight of every currently-parametrized mapped
        module and commit one refit. Returns the number of weights
        pushed (0 = nothing parametrized; no commit issued)."""
        import torch.nn.utils.parametrize as parametrize

        t0 = time.perf_counter()
        to_push: list = []       # (fqn, init_name)
        now_clean: list = []     # dirty fqns whose base value goes back
        with parametrize.cached():
            for fqn, (init_name, transposed) in self._map.items():
                mod = self._resolve_module(fqn)
                if mod is None:
                    continue
                parametrized = bool(getattr(mod, "parametrizations", None))
                if not parametrized and fqn not in self._dirty:
                    continue  # engine already holds this base value
                # Parametrized: reading .weight evaluates the vendored
                # chain (the merged value). Unparametrized-but-dirty:
                # .weight IS the restored base — push it back.
                merged = mod.weight.detach()
                if transposed and merged.dim() == 2:
                    merged = merged.transpose(0, 1)
                trt_dt, torch_dt = self._prototype(init_name)
                buf = self._staging.get(fqn)
                if buf is None or buf.shape != merged.shape or buf.dtype != torch_dt:
                    buf = torch.empty(
                        merged.shape, dtype=torch_dt, device="cpu",
                    )
                    try:
                        buf = buf.pin_memory()
                    except RuntimeError:
                        pass
                    self._staging[fqn] = buf
                buf.copy_(merged.to(torch_dt), non_blocking=True)
                to_push.append((fqn, init_name))
                if parametrized:
                    self._dirty.add(fqn)
                else:
                    now_clean.append(fqn)
        if not to_push:
            return 0
        # All D2H copies must land before TRT reads the host pointers.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for fqn, init_name in to_push:
            trt_dt, _torch_dt = self._prototype(init_name)
            set_typed_weights(
                self._refitter, self._trt, init_name,
                np_view_for_push(self._staging[fqn]), trt_dt,
                context=f"sa3 refit mirror fqn={fqn}",
            )
        commit_refit(self._refitter)
        self._dirty.difference_update(now_clean)
        logger.info(
            "sa3_trt_refit_sync pushed={} restored={} reason={} refit_ms={:.1f}",
            len(to_push), len(now_clean), reason or "-",
            (time.perf_counter() - t0) * 1000,
        )
        return len(to_push)
