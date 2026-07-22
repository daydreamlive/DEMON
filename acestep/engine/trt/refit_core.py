"""Shared TRT refit primitives (notes/SA3_LORA_PLAN.md D6b).

The pieces of in-place ``IRefitter`` writeback that are family-agnostic:
typed ``trt.Weights`` construction for dtypes numpy cannot represent
(BF16 / FP8 storage travels as uint16/uint8 views, and the ndarray
overload of ``set_named_weights`` would mistype them — TRT then rejects
the refit with a dtype-mismatch error), the loud set/commit error
shapes, and the missing-weights diagnostics.

Consumers:

* :class:`acestep.engine.trt.lora_refit.TRTLoRAManager` (ACE) — its
  delta-composition and FP8 co-refit machinery stay family-private; the
  typed push + commit go through here (byte-identical behavior, guarded
  by its existing unit tests).
* :class:`acestep.engine.sa3_trt_lora.SA3TRTRefitMirror` — composes
  MERGED weights from the parametrized torch modules and pushes them
  through the same primitives.
"""

from __future__ import annotations

from typing import Optional

import torch


def np_view_for_push(buf: torch.Tensor):
    """A CPU numpy view of ``buf`` suitable for a typed weights push.

    BF16 has no numpy dtype; its storage travels as a uint16 view (the
    ``trt.Weights(dtype, ptr, count)`` wrapper re-types it). FP8
    storage is expected pre-viewed as uint8 by the caller. The returned
    ndarray aliases ``buf``'s memory — the caller must keep ``buf``
    alive until the refit commits.
    """
    if buf.dtype == torch.bfloat16:
        return buf.view(torch.uint16).numpy()
    return buf.numpy()


def set_typed_weights(
    refitter,
    trt_mod,
    name: str,
    arr,
    trt_dtype,
    *,
    context: str = "",
) -> None:
    """``set_named_weights`` with an explicit TRT dtype, raising with
    full diagnostics on rejection. ``arr`` must stay alive until
    :func:`commit_refit` returns (TRT dereferences the pointer then)."""
    weights = trt_mod.Weights(trt_dtype, int(arr.ctypes.data), int(arr.size))
    if refitter.set_named_weights(name, weights):
        return
    proto_desc = "unknown"
    if hasattr(refitter, "get_weights_prototype"):
        try:
            proto = refitter.get_weights_prototype(name)
            proto_desc = f"dtype={proto.dtype}, size={proto.size}"
        except Exception:
            pass
    raise RuntimeError(
        f"TRT rejected refit weights for {name}: "
        f"arr dtype={getattr(arr, 'dtype', '?')} size={arr.size}; "
        f"engine prototype {proto_desc}"
        + (f"; {context}" if context else "")
    )


def commit_refit(refitter, *, required: bool = True) -> bool:
    """``refit_cuda_engine()`` with the missing-weights diagnostic.
    Returns the TRT result; raises when ``required`` and it failed."""
    ok = refitter.refit_cuda_engine()
    if not ok and required:
        missing: Optional[list] = None
        try:
            missing = refitter.get_missing_weights()
        except Exception:
            pass
        raise RuntimeError(f"TRT refit failed. Missing weights: {missing}")
    return bool(ok)
