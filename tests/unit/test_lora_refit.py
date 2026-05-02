"""Unit tests for the strength-0 short-circuits in TRTLoRAManager.

These don't build a real TRT engine; they bypass __init__ and wire up
the manager with a fake refitter so we can assert what would have been
sent to the engine.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.engine.trt.lora_refit import TRTLoRAManager, _ActiveLoRA


def _make_mgr():
    """Hand-build a TRTLoRAManager with two fake refittable fp16 params
    (``q.weight`` and ``k.weight``, both 8x16 zeros)."""
    mgr = TRTLoRAManager.__new__(TRTLoRAManager)
    mgr._engine = None
    mgr._device = torch.device("cpu")
    mgr._trt_prefix = "decoder."

    refitter = MagicMock()
    refitter.refit_cuda_engine.return_value = True
    mgr._refitter = refitter

    base_q = torch.zeros(8, 16, dtype=torch.float16)
    base_k = torch.zeros(8, 16, dtype=torch.float16)
    mgr._param_to_trt = {
        "q.weight": "decoder.q.weight",
        "k.weight": "decoder.k.weight",
    }
    mgr._base_weights = {"q.weight": base_q, "k.weight": base_k}
    mgr._refit_bufs = {
        "q.weight": torch.empty_like(base_q),
        "k.weight": torch.empty_like(base_k),
    }
    mgr._np_dtype = {"q.weight": np.float16, "k.weight": np.float16}
    mgr._active_loras = []
    mgr._next_id = 0
    mgr._ever_dirty = set()
    return mgr, refitter


def test_strength_zero_lora_in_stack_does_not_leak_into_output():
    """A LoRA sitting in the stack at strength 0 must not affect the
    refitted weights, even when its delta is non-trivial."""
    mgr, refitter = _make_mgr()
    mgr._active_loras.append(_ActiveLoRA(
        lora_id=0, path="zero.safetensors", strength=0.0,
        deltas={"q.weight": torch.ones(8, 16, dtype=torch.float16)},
    ))

    mgr._refit_weights({"q.weight"})

    refitter.set_named_weights.assert_called_once()
    arr = refitter.set_named_weights.call_args[0][1]
    np.testing.assert_array_equal(arr, np.zeros((8, 16), dtype=np.float16))


def test_strength_zero_lora_skipped_when_other_active_present():
    """Stack of [lora_zero (s=0), lora_active (s=0.5)] on the same param:
    refitted weight must equal base + delta_active * 0.5; lora_zero's
    delta must not contribute."""
    mgr, refitter = _make_mgr()
    delta_zero = torch.ones(8, 16, dtype=torch.float16)
    delta_active = torch.full((8, 16), 2.0, dtype=torch.float16)
    mgr._active_loras.append(_ActiveLoRA(
        lora_id=0, path="zero.safetensors", strength=0.0,
        deltas={"q.weight": delta_zero},
    ))
    mgr._active_loras.append(_ActiveLoRA(
        lora_id=1, path="active.safetensors", strength=0.5,
        deltas={"q.weight": delta_active},
    ))

    mgr._refit_weights({"q.weight"})

    arr = refitter.set_named_weights.call_args[0][1]
    expected = (delta_active.float() * 0.5).to(torch.float16).numpy()
    np.testing.assert_allclose(arr, expected, rtol=1e-3)


def test_set_lora_strength_same_value_is_noop():
    """set_lora_strength to the LoRA's current value must short-circuit
    before triggering a refit (no GPU re-upload, no buffer recompute)."""
    mgr, refitter = _make_mgr()
    mgr._active_loras.append(_ActiveLoRA(
        lora_id=0, path="x.safetensors", strength=0.5,
        deltas={"q.weight": torch.zeros(8, 16, dtype=torch.float16)},
    ))

    mgr.set_lora_strength(0, 0.5)

    refitter.set_named_weights.assert_not_called()
    refitter.refit_cuda_engine.assert_not_called()


def test_set_lora_strength_changed_value_does_refit():
    """Sanity check the inverse: a real strength change still refits."""
    mgr, refitter = _make_mgr()
    mgr._active_loras.append(_ActiveLoRA(
        lora_id=0, path="x.safetensors", strength=0.5,
        deltas={"q.weight": torch.ones(8, 16, dtype=torch.float16)},
    ))

    mgr.set_lora_strength(0, 0.7)

    refitter.set_named_weights.assert_called_once()
    refitter.refit_cuda_engine.assert_called_once()


def test_set_lora_strength_unknown_id_raises():
    mgr, _ = _make_mgr()
    with pytest.raises(ValueError, match="not found"):
        mgr.set_lora_strength(999, 0.5)
