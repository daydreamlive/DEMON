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

fp8 engines (manifest version 2, ``acestep.engine.trt.sa3_fp8_refit``):
Stability's fp8 graph quantizes fp32 weight initializers with baked
scales. The mirror composes ``checkpoint fp32 base + (merged - original)``
from the eager parametrization (the eager model runs fp16; pushing its
fp16 merged weight would not reproduce the build-time quantization even
at strength 0), then pushes fp32 when the engine slot is fp32 (TRT
re-quantizes with its baked scale) or quantizes itself under the FIXED
baked scale when the slot is fp8. Scale initializers the refit reports
missing are re-submitted at their baked values.

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
from typing import Callable, Dict, Optional

from loguru import logger
import torch

from acestep.engine.trt.refit_core import (
    commit_refit,
    np_view_for_push,
    set_typed_weights,
)

MANIFEST_VERSION = 1
FP8_MANIFEST_VERSION = 2
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
        base_weights: Optional[Callable[[str], torch.Tensor]] = None,
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
        self._engine = engine
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
        version = manifest.get("version")
        if version not in (MANIFEST_VERSION, FP8_MANIFEST_VERSION):
            raise RuntimeError(
                f"refit manifest {manifest_path} has version "
                f"{version}, expected {MANIFEST_VERSION} or {FP8_MANIFEST_VERSION}"
            )
        # fp8 (v2): fqn -> (scale_init, axis); baked scale values by
        # initializer name; fp32 checkpoint base reader.
        self._fp8 = version == FP8_MANIFEST_VERSION
        self._fp8_meta: Dict[str, tuple] = {}
        self._scales: Dict[str, torch.Tensor] = {}
        self._scale_push: Dict[str, tuple] = {}
        self._base_cache: Dict[str, torch.Tensor] = {}
        self._base_weights = base_weights
        if self._fp8:
            from acestep.engine.trt.sa3_fp8_refit import (  # noqa: PLC0415
                checkpoint_base_reader,
                load_fp8_scales,
            )

            self._scales = load_fp8_scales(manifest_path, manifest)
            for fqn, rec in manifest.get("weights", {}).items():
                self._fp8_meta[fqn] = (rec["scale_init"], rec.get("axis"))
            if self._base_weights is None:
                ckpt = manifest.get("base_checkpoint")
                if not ckpt or not Path(ckpt).is_file():
                    raise RuntimeError(
                        f"fp8 refit manifest {manifest_path} needs the fp32 "
                        f"base: pass base_weights= or record base_checkpoint"
                    )
                self._base_weights = checkpoint_base_reader(ckpt)
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
            if getattr(trt, "fp8", None) is not None and proto.dtype == trt.fp8:
                torch_dt = torch.uint8  # E4M3 storage bytes
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
                if self._fp8:
                    merged = self._fp8_merged(fqn, mod, parametrized)
                else:
                    merged = mod.weight.detach()
                if transposed and merged.dim() == 2:
                    merged = merged.transpose(0, 1)
                trt_dt, torch_dt = self._prototype(init_name)
                if torch_dt == torch.uint8:
                    from acestep.engine.trt.sa3_fp8_refit import (  # noqa: PLC0415
                        quantize_e4m3_fixed,
                    )

                    scale_init, axis = self._fp8_meta[fqn]
                    merged = quantize_e4m3_fixed(
                        merged, self._scales[scale_init], axis,
                    )
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
        co_pushed = self._push_missing_scales() if self._fp8 else 0
        commit_refit(self._refitter)
        # Any CUDA graph captured over this engine predates the new
        # weights; SA3TRTDit re-captures on its next step.
        from acestep.engine.sa3_trt import note_engine_refit  # noqa: PLC0415

        note_engine_refit(self._engine)
        self._dirty.difference_update(now_clean)
        logger.info(
            "sa3_trt_refit_sync pushed={} restored={} co_scales={} reason={} "
            "refit_ms={:.1f}",
            len(to_push), len(now_clean), co_pushed, reason or "-",
            (time.perf_counter() - t0) * 1000,
        )
        return len(to_push)

    # ------------------------------------------------------------------
    # fp8 (manifest v2)
    # ------------------------------------------------------------------

    def _fp8_base(self, fqn: str) -> torch.Tensor:
        base = self._base_cache.get(fqn)
        if base is None:
            base = self._base_weights(fqn).to(torch.float32).contiguous()
            self._base_cache[fqn] = base
        return base

    def _fp8_merged(self, fqn: str, mod, parametrized: bool) -> torch.Tensor:
        """fp32 checkpoint base + the eager parametrization's delta
        (merged - original, so DoRA/BoRA and live strength are exact up
        to the eager dtype). Unparametrized: the base itself, bit for bit."""
        base = self._fp8_base(fqn)
        if not parametrized:
            return base
        merged = mod.weight.detach()
        original = mod.parametrizations.weight.original.detach()
        delta = merged.to(torch.float32) - original.to(torch.float32)
        if delta.shape != base.shape:
            raise RuntimeError(
                f"fp8 refit {fqn}: eager shape {tuple(delta.shape)} != base "
                f"{tuple(base.shape)}"
            )
        return base.to(delta.device, non_blocking=True) + delta

    def _push_missing_scales(self) -> int:
        """Re-submit baked scale initializers the refit reports missing
        (TRT may count a fused weight's scale as part of its refit set)."""
        if not hasattr(self._refitter, "get_missing_weights"):
            return 0
        n = 0
        for name in list(self._refitter.get_missing_weights()):
            if name not in self._scales:
                continue
            trt_dt, torch_dt = self._prototype(name)
            rec = self._scale_push.get(name)
            if rec is None:
                dt = torch.float32 if torch_dt == torch.uint8 else torch_dt
                buf = self._scales[name].to(dt).contiguous()
                rec = (buf, np_view_for_push(buf))
                self._scale_push[name] = rec
            set_typed_weights(
                self._refitter, self._trt, name, rec[1], trt_dt,
                context="sa3 fp8 co-refit baked scale",
            )
            n += 1
        return n
