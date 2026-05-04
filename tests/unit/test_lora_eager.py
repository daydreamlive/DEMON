"""Unit tests for ``EagerLoRAManager``.

Builds a tiny ``nn.Module`` decoder, wires the manager onto it, and
asserts that enable / disable / set_strength push the right deltas
straight into ``param.data``. No TRT, no safetensors on disk — every
test stubs ``_compute_deltas`` with hand-built deltas the same way the
TRT tests do.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.engine.lora import EagerLoRAManager, LoRAState


def _make_decoder() -> nn.Module:
    """Decoder with two named params (``q.weight``, ``k.weight``),
    both 8x16 fp32 zeros so identity-base assertions are simple."""
    decoder = nn.Module()
    decoder.q = nn.Linear(16, 8, bias=False)
    decoder.k = nn.Linear(16, 8, bias=False)
    with torch.no_grad():
        decoder.q.weight.zero_()
        decoder.k.weight.zero_()
    return decoder


def _make_mgr():
    decoder = _make_decoder()
    mgr = EagerLoRAManager(decoder=decoder, device=torch.device("cpu"))
    return mgr, decoder


def test_init_indexes_params_without_snapshotting():
    """Construction records dtypes + param refs but does NOT clone weights.

    The whole point of the lazy snapshot is that a session that never
    enables a LoRA pays no extra VRAM. So _base_weights stays empty
    until the first refit hits.
    """
    mgr, decoder = _make_mgr()
    assert mgr.refittable_param_count == 2
    assert mgr._param_dtype == {
        "q.weight": torch.float32, "k.weight": torch.float32,
    }
    assert mgr._base_weights == {}


def test_first_enable_triggers_lazy_snapshot(monkeypatch):
    """The first refit clones the affected param into _base_weights."""
    mgr, decoder = _make_mgr()
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: ({"q.weight": torch.full((8, 16), 4.0)}, 8 * 16 * 4),
    )

    mgr.register_lora("/tmp/x.safetensors")
    assert mgr._base_weights == {}  # still empty

    mgr.enable_lora("x", strength=0.5)

    # Snapshot of q.weight captured (k.weight not touched: not in delta set).
    assert "q.weight" in mgr._base_weights
    assert torch.equal(mgr._base_weights["q.weight"], torch.zeros(8, 16))
    assert "k.weight" not in mgr._base_weights


def test_strength_zero_enable_does_not_snapshot(monkeypatch):
    """Placeholder enable (strength=0) doesn't trigger a refit, so the
    base snapshot must NOT be populated. The whole point of the
    placeholder pattern is "zero VRAM cost until you actually move the
    slider"."""
    mgr, decoder = _make_mgr()
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: ({"q.weight": torch.full((8, 16), 4.0)}, 8 * 16 * 4),
    )

    mgr.register_lora("/tmp/x.safetensors")
    mgr.enable_lora("x", strength=0.0)

    assert mgr.get_lora("x").state == "enabled"
    assert mgr._base_weights == {}  # no refit fired -> no snapshot


def test_enable_promotes_deltas_to_gpu_mirror(monkeypatch):
    """On enable, the manager keeps a GPU-resident mirror of the deltas
    so subsequent refits don't H2D-copy per slider tick.

    This test runs on CPU (no CUDA needed) because EagerLoRAManager
    promotes to ``self._device`` which is CPU here — the SHAPE of the
    promotion (mirror exists, holds the right tensors) is what the
    test asserts; the device transfer itself is a no-op cast on CPU.
    """
    mgr, decoder = _make_mgr()
    delta = torch.full((8, 16), 4.0)
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: ({"q.weight": delta.clone()}, delta.numel() * 4),
    )

    mgr.register_lora("/tmp/x.safetensors")
    assert "x" not in mgr._gpu_deltas

    mgr.enable_lora("x", strength=0.5)

    assert "x" in mgr._gpu_deltas
    assert torch.equal(mgr._gpu_deltas["x"]["q.weight"], delta)


def test_disable_drops_gpu_mirror(monkeypatch):
    """The whole reason for the hybrid is that disable frees VRAM."""
    mgr, decoder = _make_mgr()
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: ({"q.weight": torch.full((8, 16), 4.0)}, 8 * 16 * 4),
    )

    mgr.register_lora("/tmp/x.safetensors")
    mgr.enable_lora("x", strength=0.5)
    assert "x" in mgr._gpu_deltas

    mgr.disable_lora("x")
    assert "x" not in mgr._gpu_deltas


def test_prewarm_does_not_allocate_gpu_mirror(monkeypatch):
    """Prewarm pre-computes deltas in CPU RAM (the MATERIALIZED state),
    but must NOT reserve VRAM until the LoRA is actually enabled. This
    is the property that makes the library workflow ("register 100,
    prewarm 20, enable 3") translate cleanly to the eager backend."""
    mgr, decoder = _make_mgr()
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: ({"q.weight": torch.full((8, 16), 4.0)}, 8 * 16 * 4),
    )

    mgr.register_lora("/tmp/x.safetensors")
    f = mgr.prewarm_lora("x")
    f.result(timeout=5)

    assert mgr.get_lora("x").state == "materialized"
    assert mgr._gpu_deltas == {}  # nothing on GPU yet
    assert mgr._base_weights == {}  # snapshot still deferred


def test_enable_with_strength_writes_into_param_data(monkeypatch):
    """enable_lora(s=0.5) must end with param.data == base + 0.5 * delta.

    The point of the eager path: writeback lands directly in the live
    parameter, not in a side buffer.
    """
    mgr, decoder = _make_mgr()
    delta = torch.full((8, 16), 4.0)
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: ({"q.weight": delta.clone()}, delta.numel() * 4),
    )

    mgr.register_lora("/tmp/x.safetensors")
    mgr.enable_lora("x", strength=0.5)

    expected = torch.full((8, 16), 2.0)
    assert torch.allclose(decoder.q.weight.data, expected)
    # Untouched param stays at base.
    assert torch.allclose(decoder.k.weight.data, torch.zeros(8, 16))


def test_disable_restores_base_weights(monkeypatch):
    """disable_lora must reset param.data back to the snapshot."""
    mgr, decoder = _make_mgr()
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: ({"q.weight": torch.full((8, 16), 4.0)}, 8 * 16 * 4),
    )

    mgr.register_lora("/tmp/x.safetensors")
    mgr.enable_lora("x", strength=0.5)
    assert decoder.q.weight.data.abs().sum() > 0  # non-zero now

    mgr.disable_lora("x")
    assert torch.allclose(decoder.q.weight.data, torch.zeros(8, 16))


def test_strength_change_updates_param_data(monkeypatch):
    """set_lora_strength must move the live weight to base + new_s * delta."""
    mgr, decoder = _make_mgr()
    delta = torch.full((8, 16), 4.0)
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: ({"q.weight": delta.clone()}, delta.numel() * 4),
    )

    mgr.register_lora("/tmp/x.safetensors")
    mgr.enable_lora("x", strength=0.5)
    mgr.set_lora_strength("x", 0.25)

    expected = torch.full((8, 16), 1.0)  # 0.25 * 4.0
    assert torch.allclose(decoder.q.weight.data, expected)


def test_two_loras_compose_additively(monkeypatch):
    """Two ENABLED entries on the same param compose: param == base + Σ s·d."""
    mgr, decoder = _make_mgr()

    deltas = {
        "a": {"q.weight": torch.full((8, 16), 2.0)},
        "b": {"q.weight": torch.full((8, 16), 3.0)},
    }
    paths = {"/tmp/a.safetensors": "a", "/tmp/b.safetensors": "b"}

    def fake_compute(p):
        d = deltas[paths[p]]
        return {k: v.clone() for k, v in d.items()}, sum(
            v.numel() * 4 for v in d.values()
        )

    monkeypatch.setattr(mgr, "_compute_deltas", fake_compute)

    mgr.register_lora("/tmp/a.safetensors")
    mgr.register_lora("/tmp/b.safetensors")
    mgr.enable_lora("a", strength=0.5)
    mgr.enable_lora("b", strength=0.5)

    # 0.5 * 2.0 + 0.5 * 3.0 == 2.5
    expected = torch.full((8, 16), 2.5)
    assert torch.allclose(decoder.q.weight.data, expected)

    # Disabling one rolls back its contribution but leaves the other.
    mgr.disable_lora("a")
    expected = torch.full((8, 16), 1.5)  # only b at 0.5 * 3.0
    assert torch.allclose(decoder.q.weight.data, expected)


def test_enable_strength_zero_does_not_mutate_param(monkeypatch):
    """Strength-0 ENABLED placeholder must leave param.data at base.

    Same invariant as the TRT path's b20fba9 short-circuit: a slider
    parked at zero shouldn't pay the writeback cost.
    """
    mgr, decoder = _make_mgr()
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: ({"q.weight": torch.full((8, 16), 99.0)}, 8 * 16 * 4),
    )

    mgr.register_lora("/tmp/x.safetensors")
    mgr.enable_lora("x", strength=0.0)

    assert mgr.get_lora("x").state == "enabled"
    # Live param untouched.
    assert torch.allclose(decoder.q.weight.data, torch.zeros(8, 16))


def test_decoder_with_no_params_raises():
    decoder = nn.Module()
    with pytest.raises(RuntimeError, match="has no parameters"):
        EagerLoRAManager(decoder=decoder)


def test_apply_lora_one_shot_lifecycle(monkeypatch):
    """apply_lora is the legacy register+enable shorthand."""
    mgr, decoder = _make_mgr()
    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: ({"q.weight": torch.full((8, 16), 2.0)}, 8 * 16 * 4),
    )

    lid = mgr.apply_lora("/tmp/foo.safetensors", strength=0.5)
    assert lid == "foo"
    assert mgr.get_lora(lid).state == "enabled"
    assert torch.allclose(decoder.q.weight.data, torch.full((8, 16), 1.0))

    assert mgr.remove_lora(lid)
    assert torch.allclose(decoder.q.weight.data, torch.zeros(8, 16))


def test_param_dtype_preserved(monkeypatch):
    """Writeback must not silently upcast a fp16 param to fp32."""
    decoder = nn.Module()
    decoder.q = nn.Linear(16, 8, bias=False)
    with torch.no_grad():
        decoder.q.weight.data = decoder.q.weight.data.to(torch.float16).zero_()

    mgr = EagerLoRAManager(decoder=decoder, device=torch.device("cpu"))
    assert mgr._param_dtype["q.weight"] == torch.float16

    monkeypatch.setattr(
        mgr, "_compute_deltas",
        lambda p: (
            {"q.weight": torch.full((8, 16), 2.0, dtype=torch.float16)},
            8 * 16 * 2,
        ),
    )
    mgr.apply_lora("/tmp/foo.safetensors", strength=0.5)
    assert decoder.q.weight.data.dtype == torch.float16
