"""Unit tests for ``TRTLoRAManager``.

These don't build a real TRT engine; they bypass __init__ and wire up
the manager with a fake refitter so we can assert what would have been
sent to the engine.
"""
import ctypes
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.engine.trt.lora_refit import (
    TRTLoRAManager, _LoRAEntry, LoRAState,
)


@pytest.fixture(autouse=True)
def _no_cuda_sync(monkeypatch):
    """_refit_weights issues torch.cuda.synchronize(self._device) after the
    D2H batch; the fixture manager lives on CPU (no real engine), where that
    call raises. No-op it — there is nothing in flight to synchronize."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **kw: None)


def _snapshot_weights(np_dtype, ptr, count):
    """Stand-in for tensorrt.Weights: copy ``count`` elements of ``np_dtype``
    out of the raw host pointer so set_named_weights receives comparable
    values instead of an address. The fixture stores numpy dtypes in
    ``_trt_dtype`` (production stores trt dtypes; only this fake reads them).
    Snapshotting matters: the host refit buffers are reused across refits."""
    nbytes = count * np.dtype(np_dtype).itemsize
    raw = (ctypes.c_char * nbytes).from_address(ptr)
    return np.frombuffer(bytes(raw), dtype=np_dtype).copy()


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
    mgr._param_dtype = {"q.weight": torch.float16, "k.weight": torch.float16}
    # torch orientation, no fp8 (matches __init__ defaults when no refit
    # manifest sits next to the engine)
    mgr._transpose_for_engine = {}
    mgr._is_fp8 = {}
    # shared fp32 accumulation scratch, sized to the largest param
    mgr._scratch_acc = torch.zeros(8 * 16, dtype=torch.float32)
    mgr._trt = types.SimpleNamespace(Weights=_snapshot_weights)
    mgr._trt_dtype = {"q.weight": np.float16, "k.weight": np.float16}
    mgr._loras = {}
    mgr._ever_dirty = set()
    mgr._executor = None
    return mgr, refitter


def _add_enabled_entry(mgr, lora_id, deltas, strength):
    """Insert a fully-materialized ENABLED entry without going through
    enable_lora() (which would try to read a real file from disk)."""
    bytes_ = sum(d.numel() * d.element_size() for d in deltas.values())
    mgr._loras[lora_id] = _LoRAEntry(
        lora_id=lora_id, path=f"{lora_id}.safetensors", name=lora_id,
        state=LoRAState.ENABLED, strength=strength,
        deltas=deltas, materialized_bytes=bytes_,
    )


# ----------------------------------------------------------------------
# Existing strength-0 invariants (refit math) - preserved verbatim under
# the new lifecycle so the b20fba9 short-circuits stay covered.
# ----------------------------------------------------------------------


def test_strength_zero_lora_in_stack_does_not_leak_into_output():
    """A LoRA sitting in the stack at strength 0 must not affect the
    refitted weights, even when its delta is non-trivial."""
    mgr, refitter = _make_mgr()
    _add_enabled_entry(
        mgr, "zero",
        deltas={"q.weight": torch.ones(8, 16, dtype=torch.float16)},
        strength=0.0,
    )

    mgr._refit_weights({"q.weight"})

    refitter.set_named_weights.assert_called_once()
    arr = refitter.set_named_weights.call_args[0][1].reshape(8, 16)
    np.testing.assert_array_equal(arr, np.zeros((8, 16), dtype=np.float16))


def test_strength_zero_lora_skipped_when_other_active_present():
    """Stack of [lora_zero (s=0), lora_active (s=0.5)] on the same param:
    refitted weight must equal base + delta_active * 0.5; lora_zero's
    delta must not contribute."""
    mgr, refitter = _make_mgr()
    delta_zero = torch.ones(8, 16, dtype=torch.float16)
    delta_active = torch.full((8, 16), 2.0, dtype=torch.float16)
    _add_enabled_entry(mgr, "zero", deltas={"q.weight": delta_zero}, strength=0.0)
    _add_enabled_entry(mgr, "active", deltas={"q.weight": delta_active}, strength=0.5)

    mgr._refit_weights({"q.weight"})

    arr = refitter.set_named_weights.call_args[0][1].reshape(8, 16)
    expected = (delta_active.float() * 0.5).to(torch.float16).numpy()
    np.testing.assert_allclose(arr, expected, rtol=1e-3)


def test_set_lora_strength_same_value_is_noop():
    """set_lora_strength to the LoRA's current value must short-circuit
    before triggering a refit (no GPU re-upload, no buffer recompute)."""
    mgr, refitter = _make_mgr()
    _add_enabled_entry(
        mgr, "x",
        deltas={"q.weight": torch.zeros(8, 16, dtype=torch.float16)},
        strength=0.5,
    )

    mgr.set_lora_strength("x", 0.5)

    refitter.set_named_weights.assert_not_called()
    refitter.refit_cuda_engine.assert_not_called()


def test_set_lora_strength_changed_value_does_refit():
    """Sanity check the inverse: a real strength change still refits."""
    mgr, refitter = _make_mgr()
    _add_enabled_entry(
        mgr, "x",
        deltas={"q.weight": torch.ones(8, 16, dtype=torch.float16)},
        strength=0.5,
    )

    mgr.set_lora_strength("x", 0.7)

    refitter.set_named_weights.assert_called_once()
    refitter.refit_cuda_engine.assert_called_once()


def test_set_lora_strength_unknown_id_raises():
    mgr, _ = _make_mgr()
    with pytest.raises(ValueError, match="not registered"):
        mgr.set_lora_strength("missing", 0.5)


# ----------------------------------------------------------------------
# Lifecycle: register / enable / disable / prewarm
# ----------------------------------------------------------------------


def test_register_does_not_materialize():
    """register_lora is the cheap catalog op: no deltas in RAM, no refit,
    no disk read.  This is the property that lets the library hold N
    placeholders at near-zero cost."""
    mgr, refitter = _make_mgr()

    lid = mgr.register_lora("/tmp/death.safetensors", name="Death")

    assert lid == "death"
    desc = mgr.get_lora("death")
    assert desc.state == "registered"
    assert desc.strength == 0.0
    assert desc.materialized_bytes == 0
    assert mgr.total_materialized_bytes == 0
    refitter.set_named_weights.assert_not_called()
    refitter.refit_cuda_engine.assert_not_called()


def test_register_is_idempotent_on_path():
    """Calling register_lora twice with the same filename returns the
    same id and does not duplicate the entry."""
    mgr, _ = _make_mgr()
    a = mgr.register_lora("/tmp/x.safetensors")
    b = mgr.register_lora("/tmp/x.safetensors")
    assert a == b
    assert len(mgr.list_loras()) == 1


def test_set_strength_on_non_enabled_errors():
    """set_lora_strength is rejected on REGISTERED / MATERIALIZED entries
    so materialization cost can't be hidden behind a slider event."""
    mgr, refitter = _make_mgr()
    mgr.register_lora("/tmp/x.safetensors")

    with pytest.raises(ValueError, match="not enabled"):
        mgr.set_lora_strength("x", 0.5)

    refitter.set_named_weights.assert_not_called()


def test_enable_with_strength_refits_once_at_target(monkeypatch):
    """enable_lora(id, strength=S) must apply the LoRA at S in a single
    refit, not enable-at-0 + later set_strength-to-S. The streaming
    pipeline relies on this: an enable-at-0 followed by ramp-to-S means
    the first decode window sees base weights and produces a glitch.
    """
    mgr, refitter = _make_mgr()
    delta = torch.full((8, 16), 4.0, dtype=torch.float16)
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: ({"q.weight": delta}, delta.numel() * 2),
    )

    mgr.register_lora("/tmp/x.safetensors")
    mgr.enable_lora("x", strength=0.5)

    # Exactly one refit, with the LoRA contributing at 0.5.
    refitter.refit_cuda_engine.assert_called_once()
    arr = refitter.set_named_weights.call_args_list[0][0][1].reshape(8, 16)
    expected = (delta.float() * 0.5).to(torch.float16).numpy()
    np.testing.assert_allclose(arr, expected, rtol=1e-3)
    assert mgr.get_lora("x").strength == 0.5


def test_enable_with_strength_zero_is_still_placeholder(monkeypatch):
    """enable_lora(id, strength=0.0) is the explicit placeholder form:
    the deltas materialize but no refit fires."""
    mgr, refitter = _make_mgr()
    canned = {"q.weight": torch.ones(8, 16, dtype=torch.float16)}
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: (dict(canned), canned["q.weight"].numel() * 2),
    )

    mgr.register_lora("/tmp/x.safetensors")
    mgr.enable_lora("x", strength=0.0)

    assert mgr.get_lora("x").state == "enabled"
    assert mgr.get_lora("x").strength == 0.0
    refitter.refit_cuda_engine.assert_not_called()


def test_enable_then_disable_refits_only_when_contributing(monkeypatch):
    """Enable with non-zero strength refits.  Disable while contributing
    refits back.  Strength-0 enable + disable: both no-op refits."""
    mgr, refitter = _make_mgr()

    # Stub _compute_deltas so we don't need a real safetensors file.
    canned = {"q.weight": torch.full((8, 16), 3.0, dtype=torch.float16)}
    bytes_ = canned["q.weight"].numel() * canned["q.weight"].element_size()
    monkeypatch.setattr(mgr, "_compute_deltas", lambda p: (dict(canned), bytes_))

    mgr.register_lora("/tmp/x.safetensors")

    # Enable with strength 0: materializes but no refit (math no-op).
    mgr.enable_lora("x")
    assert mgr.get_lora("x").state == "enabled"
    assert mgr.get_lora("x").materialized_bytes == bytes_
    refitter.refit_cuda_engine.assert_not_called()

    # Bump strength: refit fires.
    mgr.set_lora_strength("x", 0.5)
    assert refitter.refit_cuda_engine.call_count == 1

    # Disable: deltas freed, contributing -> refits back to base.
    mgr.disable_lora("x")
    assert mgr.get_lora("x").state == "registered"
    assert mgr.get_lora("x").materialized_bytes == 0
    assert mgr.total_materialized_bytes == 0
    assert refitter.refit_cuda_engine.call_count == 2


def test_disable_strength_zero_skips_refit(monkeypatch):
    """A LoRA that was enabled but never had non-zero strength shouldn't
    refit on disable — there's nothing to undo."""
    mgr, refitter = _make_mgr()
    canned = {"q.weight": torch.ones(8, 16, dtype=torch.float16)}
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: (dict(canned), canned["q.weight"].numel() * 2),
    )

    mgr.register_lora("/tmp/x.safetensors")
    mgr.enable_lora("x")  # strength 0: no refit
    mgr.disable_lora("x")  # was_contributing=False: no refit

    refitter.refit_cuda_engine.assert_not_called()


def test_re_enable_restores_last_strength(monkeypatch):
    """Disable preserves strength so re-enabling returns to the same
    slider position (and the resulting refit picks up the strength)."""
    mgr, refitter = _make_mgr()
    canned = {"q.weight": torch.ones(8, 16, dtype=torch.float16)}
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: (dict(canned), canned["q.weight"].numel() * 2),
    )

    mgr.register_lora("/tmp/x.safetensors")
    mgr.enable_lora("x")
    mgr.set_lora_strength("x", 0.7)
    mgr.disable_lora("x")
    assert mgr.get_lora("x").strength == 0.7

    refitter.refit_cuda_engine.reset_mock()
    refitter.set_named_weights.reset_mock()
    mgr.enable_lora("x")

    # Re-enable with strength 0.7 should refit and apply the contribution.
    refitter.refit_cuda_engine.assert_called_once()
    arr = refitter.set_named_weights.call_args_list[0][0][1].reshape(8, 16)
    expected = (canned["q.weight"].float() * 0.7).to(torch.float16).numpy()
    np.testing.assert_allclose(arr, expected, rtol=1e-3)


def test_prewarm_completes_then_enable_skips_inline_materialization(monkeypatch):
    """After prewarm resolves the entry should be MATERIALIZED, and
    enable_lora must not call _compute_deltas again."""
    mgr, refitter = _make_mgr()
    canned = {"q.weight": torch.ones(8, 16, dtype=torch.float16)}
    bytes_ = canned["q.weight"].numel() * 2
    call_count = {"n": 0}

    def fake_compute(path):
        call_count["n"] += 1
        return dict(canned), bytes_

    monkeypatch.setattr(mgr, "_compute_deltas", fake_compute)

    mgr.register_lora("/tmp/x.safetensors")
    f = mgr.prewarm_lora("x")
    f.result(timeout=5)
    assert mgr.get_lora("x").state == "materialized"
    assert call_count["n"] == 1

    mgr.enable_lora("x")
    assert mgr.get_lora("x").state == "enabled"
    # No second call to _compute_deltas — prewarm already paid the cost.
    assert call_count["n"] == 1


def test_apply_lora_backward_compat(monkeypatch):
    """The legacy apply_lora wrapper (register + set_strength + enable)
    keeps working for callers that haven't migrated."""
    mgr, refitter = _make_mgr()
    canned = {"q.weight": torch.ones(8, 16, dtype=torch.float16)}
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: (dict(canned), canned["q.weight"].numel() * 2),
    )

    lid = mgr.apply_lora("/tmp/foo.safetensors", strength=0.5)

    assert lid == "foo"
    desc = mgr.get_lora("foo")
    assert desc.state == "enabled"
    assert desc.strength == 0.5
    refitter.refit_cuda_engine.assert_called_once()


def test_apply_lora_strength_zero_is_placeholder(monkeypatch):
    """The placeholder pattern (apply at strength 0, ramp later) still
    materializes deltas but skips the refit — same as before."""
    mgr, refitter = _make_mgr()
    canned = {"q.weight": torch.ones(8, 16, dtype=torch.float16)}
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: (dict(canned), canned["q.weight"].numel() * 2),
    )

    mgr.apply_lora("/tmp/foo.safetensors", strength=0.0)

    assert mgr.get_lora("foo").state == "enabled"
    refitter.refit_cuda_engine.assert_not_called()


def test_register_library_scans_and_registers(tmp_path, monkeypatch):
    """register_library should pick up every .safetensors in the dir as
    a REGISTERED entry (no materialization) and skip non-matching files."""
    (tmp_path / "a.safetensors").write_bytes(b"")
    (tmp_path / "b.safetensors").write_bytes(b"")
    (tmp_path / "ignore.txt").write_text("not a lora")

    mgr, refitter = _make_mgr()
    ids = mgr.register_library(tmp_path)

    assert sorted(ids) == ["a", "b"]
    assert all(d.state == "registered" for d in mgr.list_loras())
    assert mgr.total_materialized_bytes == 0
    refitter.set_named_weights.assert_not_called()


def test_register_library_missing_dir_returns_empty():
    mgr, _ = _make_mgr()
    ids = mgr.register_library(Path("/no/such/path/123abc"))
    assert ids == []
    assert mgr.list_loras() == []


def test_remove_lora_full_lifecycle(monkeypatch):
    """remove_lora should disable + drop from catalog.  remove_lora(-1)
    pops the most recently registered entry."""
    mgr, refitter = _make_mgr()
    canned = {"q.weight": torch.ones(8, 16, dtype=torch.float16)}
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: (dict(canned), canned["q.weight"].numel() * 2),
    )

    mgr.register_lora("/tmp/a.safetensors")
    mgr.register_lora("/tmp/b.safetensors")
    mgr.enable_lora("a")
    mgr.set_lora_strength("a", 0.5)

    assert mgr.remove_lora(-1) is True  # pops "b" (most recent)
    assert {d.id for d in mgr.list_loras()} == {"a"}

    assert mgr.remove_lora("a") is True  # disables (refit) + drops
    assert mgr.list_loras() == []

    assert mgr.remove_lora("a") is False  # already gone
    assert mgr.remove_lora(-1) is False  # empty
