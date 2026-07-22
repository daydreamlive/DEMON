"""Unit tests for :mod:`acestep.engine.sa3_lora` (SA3LoRAManager).

CPU-only, but against the REAL vendored ``stable_audio_3`` LoRA code
(parametrize registration, strength buffers, upstream removal) on a
tiny module tree — the manager's whole value is wrapping that engine's
exact math and sharp edges, so stubbing it out would test nothing.
Skips when the managed vendor checkout isn't on disk (CI without
models), same convention as the checkpoint-gated tests.

Covers the plan's Phase 1 test contract: enable-at-strength atomicity,
rollback on failure, middle-slot removal/re-enable, strength targeting
after churn, the conditioner-dirty flag, ``-xs`` rejection, wrong-family
rejection, lineage validation, and teardown completeness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _vendored_or_skip():
    from acestep.engine.sa3_helpers import ensure_sa3_paths

    ensure_sa3_paths()
    lora_utils = pytest.importorskip(
        "stable_audio_3.models.lora.utils",
        reason="SA3 vendored source not on disk",
    )
    from stable_audio_3.models.lora import model as lora_model

    return lora_model, lora_utils


def _make_manager():
    """Tiny two-root tree mirroring the SA3 layout: a 'DiT' root with a
    Linear + Conv1d, and a conditioner root with one Linear under the
    conditioners.* prefix (the seconds_total shape)."""
    from acestep.engine.sa3_lora import SA3LoRAManager

    torch.manual_seed(7)
    model = nn.Module()
    blk = nn.Module()
    blk.lin = nn.Linear(16, 8, bias=False)
    model.blk = blk
    model.conv = nn.Conv1d(4, 6, 3, bias=False)

    cond = nn.Module()
    conds = nn.Module()
    seconds = nn.Module()
    seconds.lin = nn.Linear(5, 8, bias=False)
    conds.seconds_total = seconds
    cond.conditioners = conds

    mgr = SA3LoRAManager(
        model_root=model, conditioner_root=cond, checkpoint_id="medium",
    )
    return mgr, model, cond


def _write_lora(
    tmp_path: Path,
    stem: str,
    entries: dict,
    *,
    rank: int = 4,
    alpha: float = 8.0,
    adapter_type: str = "lora",
    base_model: str | None = "medium-base",
    seed: int = 0,
) -> Path:
    """Trainer-faithful file: fp16 tensors, parametrization keys at
    index 0, lora_config in the safetensors header (the vendored
    ``save_lora_safetensors`` does the fp16 cast + header embed).

    ``entries``: {fqn: (fan_out, fan_in)} — random A/B pairs are built
    per module from ``seed``.
    """
    _, lora_utils = _vendored_or_skip()
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for fqn, (fan_out, fan_in) in entries.items():
        prefix = f"{fqn}.parametrizations.weight.0"
        sd[f"{prefix}.lora_A"] = torch.randn(rank, fan_in, generator=g)
        sd[f"{prefix}.lora_B"] = torch.randn(fan_out, rank, generator=g)
    config = {"rank": rank, "alpha": alpha, "adapter_type": adapter_type}
    if base_model is not None:
        config["base_model"] = base_model
    p = tmp_path / f"{stem}.safetensors"
    lora_utils.save_lora_safetensors(sd, config, p)
    return p


def _adapters_on(mod):
    """(lora_index, scaling, strength, A, B) per parametrization, in
    physical order."""
    out = []
    plist = getattr(getattr(mod, "parametrizations", None), "weight", None)
    for p in plist or []:
        out.append((
            p.lora_index, p.scaling, float(p.lora_strength),
            p.lora_A.detach(), p.lora_B.detach(),
        ))
    return out


def _expected_weight(base: torch.Tensor, mod) -> torch.Tensor:
    """Replay the vendored lora_forward chain from the LIVE adapter
    tensors (physical order, same op order)."""
    w = base.clone()
    for _idx, scaling, strength, A, B in _adapters_on(mod):
        delta = (B @ A).view(w.shape)
        w = w + (scaling * strength * delta).to(w.dtype)
    return w


# ---------------------------------------------------------------------------
# Enable / strength / disable basics
# ---------------------------------------------------------------------------


def test_enable_at_strength_applies_scaled_delta(tmp_path):
    mgr, model, _cond = _make_manager()
    base = model.blk.lin.weight.detach().clone()
    p = _write_lora(tmp_path, "styleA", {"blk.lin": (8, 16)})

    lid = mgr.register_lora(str(p))
    mgr.enable_lora(lid, strength=0.5)

    assert mgr.get_lora(lid).state == "enabled"
    assert mgr.get_lora(lid).strength == 0.5
    adapters = _adapters_on(model.blk.lin)
    assert len(adapters) == 1
    _idx, scaling, strength, A, B = adapters[0]
    assert scaling == pytest.approx(8.0 / 4.0)  # alpha/rank, not ACE's raw B@A
    assert strength == 0.5
    # Non-trivial delta actually landed in the effective weight.
    assert torch.allclose(
        model.blk.lin.weight, _expected_weight(base, model.blk.lin),
    )
    assert not torch.allclose(model.blk.lin.weight, base)


def test_conv1d_target_applies(tmp_path):
    mgr, model, _cond = _make_manager()
    base = model.conv.weight.detach().clone()
    # Conv1d weight [6, 4, 3] flattens to (6, 12).
    p = _write_lora(tmp_path, "convy", {"conv": (6, 12)})
    lid = mgr.register_lora(str(p))
    mgr.enable_lora(lid, strength=1.0)
    assert torch.allclose(
        model.conv.weight, _expected_weight(base, model.conv),
    )
    assert not torch.allclose(model.conv.weight, base)


def test_set_strength_is_live_buffer_write(tmp_path):
    mgr, model, _cond = _make_manager()
    base = model.blk.lin.weight.detach().clone()
    p = _write_lora(tmp_path, "s", {"blk.lin": (8, 16)})
    lid = mgr.register_lora(str(p))
    mgr.enable_lora(lid, strength=1.0)
    w_at_1 = model.blk.lin.weight.detach().clone()

    mgr.set_lora_strength(lid, 0.25)
    assert float(_adapters_on(model.blk.lin)[0][2]) == 0.25
    w_at_quarter = model.blk.lin.weight.detach()
    assert torch.allclose(
        w_at_quarter - base, (w_at_1 - base) * 0.25, atol=1e-6,
    )


def test_set_strength_requires_enabled(tmp_path):
    mgr, _model, _cond = _make_manager()
    p = _write_lora(tmp_path, "s2", {"blk.lin": (8, 16)})
    lid = mgr.register_lora(str(p))
    with pytest.raises(ValueError, match="not enabled"):
        mgr.set_lora_strength(lid, 0.5)


def test_disable_restores_base_bitwise_and_frees_state(tmp_path):
    mgr, model, _cond = _make_manager()
    base = model.blk.lin.weight.detach().clone()
    p = _write_lora(tmp_path, "d", {"blk.lin": (8, 16)})
    lid = mgr.register_lora(str(p))
    mgr.enable_lora(lid, strength=1.0)

    mgr.disable_lora(lid)

    # Upstream removal restores the ORIGINAL tensor — bitwise.
    assert torch.equal(model.blk.lin.weight, base)
    assert not hasattr(model.blk.lin, "parametrizations")
    assert mgr.get_lora(lid).state == "registered"
    # Strength survives the cycle (slider-position convention).
    assert mgr.get_lora(lid).strength == 1.0
    mgr.enable_lora(lid)
    assert mgr.get_lora(lid).strength == 1.0


def test_prewarm_stages_without_touching_model(tmp_path):
    mgr, model, _cond = _make_manager()
    p = _write_lora(tmp_path, "pw", {"blk.lin": (8, 16)})
    lid = mgr.register_lora(str(p))

    f = mgr.prewarm_lora(lid)
    f.result(timeout=10)

    assert mgr.get_lora(lid).state == "materialized"
    assert mgr.get_lora(lid).materialized_bytes > 0
    assert not hasattr(model.blk.lin, "parametrizations")
    assert mgr.touches_conditioner(lid) is False


# ---------------------------------------------------------------------------
# Hard validation at materialize/enable (D3)
# ---------------------------------------------------------------------------


def test_ace_peft_file_fails_loudly(tmp_path):
    from safetensors.torch import save_file

    mgr, _model, _cond = _make_manager()
    p = tmp_path / "acefile.safetensors"
    save_file(
        {
            "base_model.model.q.lora_A.weight": torch.zeros(4, 16),
            "base_model.model.q.lora_B.weight": torch.zeros(8, 4),
        },
        str(p),
    )
    lid = mgr.register_lora(str(p))
    with pytest.raises(RuntimeError, match="not an SA3 LoRA") as ei:
        mgr.enable_lora(lid, strength=1.0)
    assert "ACE" in str(ei.value)


def test_xs_variant_rejected_with_clear_error(tmp_path):
    _, lora_utils = _vendored_or_skip()
    mgr, _model, _cond = _make_manager()
    sd = {
        "blk.lin.parametrizations.weight.0.M_xs": torch.zeros(4, 4),
    }
    p = tmp_path / "xsfile.safetensors"
    lora_utils.save_lora_safetensors(
        sd, {"rank": 4, "alpha": 4, "adapter_type": "lora-xs"}, p,
    )
    lid = mgr.register_lora(str(p))
    with pytest.raises(RuntimeError, match="not .*supported|not yet"):
        mgr.enable_lora(lid, strength=1.0)


def test_lineage_mismatch_fails_with_checkpoint_names(tmp_path):
    mgr, _model, _cond = _make_manager()  # checkpoint_id="medium"
    p = _write_lora(
        tmp_path, "wronglineage", {"blk.lin": (8, 16)},
        base_model="small-music-base",
    )
    lid = mgr.register_lora(str(p))
    with pytest.raises(RuntimeError, match="medium") as ei:
        mgr.enable_lora(lid, strength=1.0)
    assert "small-music" in str(ei.value)


def test_unknown_lineage_is_permissive(tmp_path):
    mgr, _model, _cond = _make_manager()
    p = _write_lora(
        tmp_path, "custom", {"blk.lin": (8, 16)},
        base_model="my-custom-finetune",
    )
    lid = mgr.register_lora(str(p))
    mgr.enable_lora(lid, strength=1.0)
    assert mgr.get_lora(lid).state == "enabled"


def test_shape_mismatch_everywhere_fails(tmp_path):
    mgr, _model, _cond = _make_manager()
    # fan_in 32 doesn't fit blk.lin's (8, 16).
    p = _write_lora(tmp_path, "wrongsize", {"blk.lin": (8, 32)})
    lid = mgr.register_lora(str(p))
    with pytest.raises(RuntimeError, match="does not fit"):
        mgr.enable_lora(lid, strength=1.0)


def test_ckpt_extension_rejected(tmp_path):
    mgr, _model, _cond = _make_manager()
    p = tmp_path / "old.ckpt"
    p.write_bytes(b"\x80\x04")  # never actually loaded
    lid = mgr.register_lora(str(p))
    with pytest.raises(RuntimeError, match="safetensors"):
        mgr.enable_lora(lid, strength=1.0)


# ---------------------------------------------------------------------------
# Transactional enable + rollback (D4)
# ---------------------------------------------------------------------------


def test_failed_enable_rolls_back_bitwise(tmp_path, monkeypatch):
    lora_model, _ = _vendored_or_skip()
    mgr, model, cond = _make_manager()
    pA = _write_lora(tmp_path, "okA", {"blk.lin": (8, 16)}, seed=1)
    lidA = mgr.register_lora(str(pA))
    mgr.enable_lora(lidA, strength=1.0)

    pre_params = {
        n: t.detach().clone()
        for root in (model, cond)
        for n, t in root.named_parameters()
    }
    pre_effective = model.blk.lin.weight.detach().clone()

    pB = _write_lora(
        tmp_path, "failB", {"blk.lin": (8, 16), "conv": (6, 12)}, seed=2,
    )
    lidB = mgr.register_lora(str(pB))

    # Fail AFTER registration + install, at the strength-buffer step —
    # the deepest point before commit.
    real_set = lora_model.set_lora_strength

    def _boom(*a, **k):
        raise RuntimeError("simulated failure at strength write")

    monkeypatch.setattr(lora_model, "set_lora_strength", _boom)
    with pytest.raises(RuntimeError, match="simulated failure"):
        mgr.enable_lora(lidB, strength=0.7)
    monkeypatch.setattr(lora_model, "set_lora_strength", real_set)

    # Bit-identical parameter set (names + values) vs pre-enable.
    now_params = {
        n: t.detach()
        for root in (model, cond)
        for n, t in root.named_parameters()
    }
    assert set(now_params) == set(pre_params)
    for n, t in now_params.items():
        assert torch.equal(t, pre_params[n]), n
    assert torch.equal(model.blk.lin.weight, pre_effective)
    # B is not enabled; A still is and stays targetable.
    assert mgr.get_lora(lidB).state == "materialized"
    assert mgr.get_lora(lidA).state == "enabled"
    mgr.set_lora_strength(lidA, 0.3)
    assert float(_adapters_on(model.blk.lin)[0][2]) == pytest.approx(0.3)
    # The failed transaction's index is never reused, and a retry works.
    mgr.enable_lora(lidB, strength=0.7)
    assert mgr.get_lora(lidB).state == "enabled"


# ---------------------------------------------------------------------------
# Slot churn (D4) — the remove_lora_by_index physical-shift hazard
# ---------------------------------------------------------------------------


def test_middle_slot_removal_then_enable_stays_id_accurate(tmp_path):
    mgr, model, _cond = _make_manager()
    base = model.blk.lin.weight.detach().clone()
    lids = []
    for i, stem in enumerate(("a", "b", "c")):
        p = _write_lora(tmp_path, stem, {"blk.lin": (8, 16)}, seed=10 + i)
        lid = mgr.register_lora(str(p))
        mgr.enable_lora(lid, strength=1.0)
        lids.append(lid)

    # Remove the MIDDLE adapter — physical positions shift under
    # upstream removal.
    mgr.disable_lora("b")
    assert len(_adapters_on(model.blk.lin)) == 2

    # Enable a fourth AFTER the shift (direct-copy install path).
    pD = _write_lora(tmp_path, "d", {"blk.lin": (8, 16)}, seed=99)
    mgr.enable_lora(mgr.register_lora(str(pD)), strength=1.0)
    assert len(_adapters_on(model.blk.lin)) == 3
    assert torch.allclose(
        model.blk.lin.weight, _expected_weight(base, model.blk.lin),
    )

    # Strength targeting by id lands on the right adapter objects.
    mgr.set_lora_strength("c", 0.5)
    mgr.set_lora_strength("d", 0.25)
    by_index = {
        idx: s for idx, _sc, s, _A, _B in _adapters_on(model.blk.lin)
    }
    assert by_index[mgr._index_by_id["a"]] == 1.0
    assert by_index[mgr._index_by_id["c"]] == 0.5
    assert by_index[mgr._index_by_id["d"]] == 0.25
    assert torch.allclose(
        model.blk.lin.weight, _expected_weight(base, model.blk.lin),
    )

    # Full teardown of the churned state restores base bitwise.
    for lid in ("a", "c", "d"):
        mgr.disable_lora(lid)
    assert torch.equal(model.blk.lin.weight, base)
    assert not hasattr(model.blk.lin, "parametrizations")


# ---------------------------------------------------------------------------
# Conditioner flag (D5) + teardown (close)
# ---------------------------------------------------------------------------


def test_touches_conditioner_flag(tmp_path):
    mgr, _model, cond = _make_manager()
    base = cond.conditioners.seconds_total.lin.weight.detach().clone()
    p = _write_lora(
        tmp_path, "condy",
        {"blk.lin": (8, 16), "conditioners.seconds_total.lin": (8, 5)},
    )
    lid = mgr.register_lora(str(p))
    mgr.prewarm_lora(lid).result(timeout=10)
    assert mgr.touches_conditioner(lid) is True

    mgr.enable_lora(lid, strength=1.0)
    assert not torch.allclose(
        cond.conditioners.seconds_total.lin.weight, base,
    )
    mgr.disable_lora(lid)
    assert torch.equal(cond.conditioners.seconds_total.lin.weight, base)
    # Staged payload dropped with the disable — flag is gone too.
    assert mgr.touches_conditioner(lid) is False


def test_close_returns_model_to_pristine(tmp_path):
    mgr, model, cond = _make_manager()
    pre = {
        n: t.detach().clone()
        for root in (model, cond)
        for n, t in root.named_parameters()
    }
    for i, stem in enumerate(("x", "y")):
        p = _write_lora(
            tmp_path, stem,
            {"blk.lin": (8, 16), "conditioners.seconds_total.lin": (8, 5)},
            seed=20 + i,
        )
        mgr.enable_lora(mgr.register_lora(str(p)), strength=1.0)

    mgr.close()

    now = {
        n: t.detach()
        for root in (model, cond)
        for n, t in root.named_parameters()
    }
    assert set(now) == set(pre)
    for n, t in now.items():
        assert torch.equal(t, pre[n]), n
    for root in (model, cond):
        for _n, mod in root.named_modules():
            assert not getattr(mod, "parametrizations", None)
    assert mgr.list_loras() == []
    mgr.close()  # idempotent
