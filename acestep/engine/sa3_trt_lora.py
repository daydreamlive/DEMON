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
baked scale when the slot is fp8.

Missing-set completion (both manifest versions): an engine built with
``BuilderFlag.REFIT`` marks EVERY initializer refittable (the fp8 medium
engine exposes 1673 names), and TensorRT's ``get_missing_weights`` counts
any unsupplied weight it fused with a pushed one as missing (on the fp8
graph: every baked scale touched, all 24 ``cross_attend_norm.gamma`` and
``padding_embedding``). After the mapped pushes, every name the refitter
still reports missing is supplied at its build-time base value, on every
commit: baked scales from the fp8 sidecar, anything else from the fp32
base checkpoint (manifest ``base_checkpoint``), else the live eager
parameter/buffer, cast to the slot dtype. The staged buffers are pinned
and reused. A name with no resolvable base raises (a refit that silently
drops a fused weight would corrupt the engine).

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
# The fp8 graph's shared manifest: an fp8 refit engine must never fall
# back to the fp16mixed (v1) shared manifest, nor the reverse.
FP8_SHARED_MANIFEST_NAME = "sa3_m_dit_fp8_refit_manifest.json"


def find_refit_manifest(engine_path: Path) -> Optional[Path]:
    """Manifest lookup: per-engine sidecar first, then the shared
    per-ONNX manifest of the engine's graph (fp8 or fp16mixed) in the
    engines root."""
    from acestep.engine.sa3_trt import is_fp8_refit_engine_path  # noqa: PLC0415

    engine_path = Path(engine_path)
    sidecar = Path(str(engine_path) + ".refit_manifest.json")
    if sidecar.is_file():
        return sidecar
    name = (
        FP8_SHARED_MANIFEST_NAME if is_fp8_refit_engine_path(engine_path)
        else SHARED_MANIFEST_NAME
    )
    shared = engine_path.parent.parent / name
    if shared.is_file():
        return shared
    return None


def make_refit_mirror(dit, model_root: torch.nn.Module, **kwargs):
    """The session-create hook: an :class:`SA3TRTRefitMirror` over
    ``dit`` (an :class:`~acestep.engine.sa3_trt.SA3TRTDit` on a
    refit-built engine), or None with a logged reason, in which case the
    interim eager-DiT swap covers LoRA. The engine's precision (fp8 refit
    vs fp16mixed refit, by name) must match the manifest version: an fp8
    engine is only ever driven through the v2 (fixed baked scale) path.
    ``kwargs`` pass through to the mirror (test injection)."""
    from acestep.engine.sa3_trt import is_fp8_refit_engine_path  # noqa: PLC0415

    engine_path = Path(dit.engine_path)
    fp8 = is_fp8_refit_engine_path(engine_path)
    manifest = find_refit_manifest(engine_path)
    if manifest is None:
        logger.warning(
            "sa3_refit_manifest_missing engine={} — LoRA will use "
            "the eager-DiT swap; generate the manifest with "
            "scripts/sa3/gen_sa3_refit_manifest.py{}",
            engine_path.parent.name, " --fp8" if fp8 else "",
        )
        return None
    try:
        mirror = SA3TRTRefitMirror(
            dit.engine, model_root, manifest, expect_fp8=fp8, **kwargs,
        )
    except Exception as exc:
        logger.warning(
            "sa3_refit_mirror_init_failed engine={} error={} — LoRA will "
            "use the eager-DiT swap", engine_path.parent.name, exc,
        )
        return None
    logger.info(
        "sa3_refit_mirror_engaged engine={} precision={} manifest={}",
        engine_path.parent.name, "fp8" if fp8 else "fp16mixed", manifest.name,
    )
    return mirror


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
        expect_fp8: Optional[bool] = None,
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
        if expect_fp8 is not None and (version == FP8_MANIFEST_VERSION) != bool(expect_fp8):
            raise RuntimeError(
                f"refit manifest {manifest_path} is version {version} but the "
                f"engine is {'fp8' if expect_fp8 else 'fp16mixed'} refit "
                f"(fp8 engines need version {FP8_MANIFEST_VERSION})"
            )
        # fp8 (v2): fqn -> (scale_init, axis); baked scale values by
        # initializer name; fp32 checkpoint base reader.
        self._fp8 = version == FP8_MANIFEST_VERSION
        self._fp8_meta: Dict[str, tuple] = {}
        self._scales: Dict[str, torch.Tensor] = {}
        # Missing-set completion: name -> (buf, ndarray view, trt dtype),
        # staged once (pinned) and re-pushed on every commit.
        self._extra_push: Dict[str, tuple] = {}
        self._ckpt_handle = None
        self._ckpt_keys: Optional[list] = None
        self._graph_prefix = manifest.get("graph_prefix", "dit.")
        self._module_prefix = manifest.get("module_prefix", "model.")
        ckpt = manifest.get("base_checkpoint")
        self._base_checkpoint = ckpt if ckpt and Path(ckpt).is_file() else None
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
            "sa3_trt_refit_mirror_ready mapped_weights={} precision={} manifest={}",
            len(self._map), "fp8" if self._fp8 else "fp16mixed", manifest_path,
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
        co_pushed = self._push_missing()
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

    def _push_missing(self) -> int:
        """Supply every name the refitter reports missing after the mapped
        pushes, at its build-time base value (see module docstring).
        Loops until the missing set is empty: supplying one fused weight
        may pull in another. Raises on a name with no resolvable base."""
        if not hasattr(self._refitter, "get_missing_weights"):
            return 0
        supplied: set = set()
        while True:
            missing = [
                n for n in self._refitter.get_missing_weights() if n not in supplied
            ]
            if not missing:
                break
            unresolved = []
            for name in missing:
                rec = self._extra_record(name)
                if rec is None:
                    unresolved.append(name)
                    continue
                _buf, arr, trt_dt = rec
                set_typed_weights(
                    self._refitter, self._trt, name, arr, trt_dt,
                    context="sa3 refit missing-set base value",
                )
                supplied.add(name)
            if unresolved:
                raise RuntimeError(
                    f"TRT refit reports {len(unresolved)} missing weights with no "
                    f"resolvable base value (first: {unresolved[:5]}); record "
                    f"base_checkpoint in the refit manifest"
                )
        return len(supplied)

    def _extra_record(self, name: str):
        rec = self._extra_push.get(name)
        if rec is not None:
            return rec
        base = self._extra_base(name)
        if base is None:
            return None
        trt_dt, torch_dt = self._prototype(name)
        if torch_dt == torch.uint8:
            if name not in self._scales:
                raise RuntimeError(
                    f"refit missing-set weight {name} has an fp8 slot; only "
                    f"mapped Linear weights are quantized by the mirror"
                )
            torch_dt = torch.float32
        buf = base.detach().to(device="cpu", dtype=torch_dt).contiguous()
        try:
            buf = buf.pin_memory()
        except RuntimeError:
            pass
        rec = (buf, np_view_for_push(buf), trt_dt)
        self._extra_push[name] = rec
        return rec

    def _extra_base(self, name: str) -> Optional[torch.Tensor]:
        """Build-time base value of a non-mapped initializer: fp8 sidecar
        scale, else fp32 checkpoint, else the live eager tensor."""
        if name in self._scales:
            return self._scales[name]
        for fqn, (init_name, transposed) in self._map.items():
            if init_name != name:
                continue
            # A mapped weight fused with a pushed one but not pushed this
            # sync: it holds its base (parametrized modules are pushed).
            if self._fp8:
                w = self._fp8_base(fqn)
            else:
                mod = self._resolve_module(fqn)
                w = None if mod is None else mod.weight.detach()
            if w is not None and transposed and w.dim() == 2:
                w = w.transpose(0, 1)
            return w
        gp = self._graph_prefix
        stripped = name[len(gp):] if gp and name.startswith(gp) else name
        rel = self._module_prefix + stripped     # relative to model_root
        if self._base_checkpoint is not None:
            t = self._ckpt_lookup(rel, stripped)
            if t is not None:
                return t.to(torch.float32)
        t = self._eager_lookup(rel, stripped)
        if t is not None:
            if t.dtype != torch.float32:
                logger.warning(
                    "sa3_refit_missing_from_eager name={} dtype={} (no base "
                    "checkpoint; build-time bits may differ)", name, t.dtype,
                )
            return t
        return None

    def _ckpt_lookup(self, rel: str, stripped: str) -> Optional[torch.Tensor]:
        if self._ckpt_handle is None:
            from safetensors import safe_open  # noqa: PLC0415

            self._ckpt_handle = safe_open(str(self._base_checkpoint), framework="pt")
            self._ckpt_keys = list(self._ckpt_handle.keys())
        keys = self._ckpt_keys
        # model_root is the checkpoint's ``model.`` subtree (see
        # sa3_fp8_refit.checkpoint_base_reader).
        for k in ("model." + rel, rel):
            if k in keys:
                return self._ckpt_handle.get_tensor(k)
        hits = [k for k in keys if k.endswith("." + stripped)]
        if len(hits) == 1:
            return self._ckpt_handle.get_tensor(hits[0])
        return None

    def _eager_lookup(self, rel: str, stripped: str) -> Optional[torch.Tensor]:
        named = dict(self._model_root.named_parameters())
        named.update(dict(self._model_root.named_buffers()))
        if rel in named:
            return named[rel].detach()
        hits = [k for k in named if k == stripped or k.endswith("." + stripped)]
        if len(hits) == 1:
            return named[hits[0]].detach()
        return None
