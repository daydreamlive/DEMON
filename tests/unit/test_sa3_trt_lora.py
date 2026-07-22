"""SA3 TRT LoRA refit mirror (phase 2) — CPU tests.

The mirror's contract pieces that don't need an engine: manifest
consumption, merged-weight composition via the live parametrizations
(real vendored code on a tiny tree; skips without the vendor checkout),
transpose orientation, the dirty-set (a disabled adapter's BASE weights
must go back to the engine on the next sync), and single-commit
batching — all against a recording fake refitter. The real-engine
bit-identity gate lives in scripts/sa3/gen_sa3_refit_manifest.py
--validate-engine and runs on the GPU with a refit-built engine.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.engine.sa3_trt_lora import SA3TRTRefitMirror


class _FakeTrt(types.SimpleNamespace):
    def __init__(self):
        super().__init__(
            float32="F32", float16="F16",
            Weights=lambda dtype, ptr, size: (dtype, ptr, size),
        )


class _FakeRefitter:
    def __init__(self, names):
        self._names = list(names)
        self.sets: list = []
        self.commits = 0

    def get_all_weights(self):
        return list(self._names)

    def set_named_weights(self, name, weights):
        self.sets.append(name)
        return name in self._names

    def refit_cuda_engine(self):
        self.commits += 1
        return True


def _vendored_manager_or_skip(model_root):
    from acestep.engine.sa3_helpers import ensure_sa3_paths

    ensure_sa3_paths()
    pytest.importorskip(
        "stable_audio_3.models.lora.model",
        reason="SA3 vendored source not on disk",
    )
    from acestep.engine.sa3_lora import SA3LoRAManager

    return SA3LoRAManager(
        model_root=model_root, conditioner_root=None, checkpoint_id="medium",
    )


def _write_lora_file(tmp_path, fqn="blk.lin", fan=(8, 16), rank=4):
    from stable_audio_3.models.lora.utils import save_lora_safetensors

    g = torch.Generator().manual_seed(3)
    sd = {
        f"{fqn}.parametrizations.weight.0.lora_A":
            torch.randn(rank, fan[1], generator=g),
        f"{fqn}.parametrizations.weight.0.lora_B":
            torch.randn(fan[0], rank, generator=g),
    }
    p = tmp_path / "adapter.safetensors"
    save_lora_safetensors(
        sd, {"rank": rank, "alpha": rank, "adapter_type": "lora"}, p,
    )
    return p


def _mirror(tmp_path, model_root, *, transposed=True):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "weights": {
            "blk.lin": {
                "initializer": "onnx::MatMul_7", "transposed": transposed,
            },
        },
    }), encoding="utf-8")
    refitter = _FakeRefitter(["onnx::MatMul_7"])
    m = SA3TRTRefitMirror(
        object(), model_root, manifest,
        _refitter=refitter, _trt=_FakeTrt(),
    )
    return m, refitter


def _model():
    torch.manual_seed(11)
    model = nn.Module()
    blk = nn.Module()
    blk.lin = nn.Linear(16, 8, bias=False)
    model.blk = blk
    return model


def test_manifest_version_and_coverage_validated(tmp_path):
    model = _model()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 99, "weights": {}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="version"):
        SA3TRTRefitMirror(
            object(), model, bad,
            _refitter=_FakeRefitter(["x"]), _trt=_FakeTrt(),
        )
    unmapped = tmp_path / "unmapped.json"
    unmapped.write_text(json.dumps({
        "version": 1,
        "weights": {"blk.lin": {"initializer": "not_in_engine"}},
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="refittable"):
        SA3TRTRefitMirror(
            object(), model, unmapped,
            _refitter=_FakeRefitter(["other"]), _trt=_FakeTrt(),
        )


def test_sync_noop_without_parametrizations(tmp_path):
    model = _model()
    mirror, refitter = _mirror(tmp_path, model)
    assert mirror.sync(reason="test") == 0
    assert refitter.commits == 0
    assert refitter.sets == []


def test_sync_pushes_merged_weight_and_restores_base_on_disable(tmp_path):
    model = _model()
    base = model.blk.lin.weight.detach().clone()
    mgr = _vendored_manager_or_skip(model)
    lora_file = _write_lora_file(tmp_path)
    mirror, refitter = _mirror(tmp_path, model, transposed=True)

    lid = mgr.register_lora(str(lora_file))
    mgr.enable_lora(lid, strength=1.0)
    merged = model.blk.lin.weight.detach().clone()
    assert not torch.equal(merged, base)

    # Enable-sync: pushes the merged value (transposed orientation).
    assert mirror.sync(reason="enable") == 1
    assert refitter.commits == 1
    assert refitter.sets == ["onnx::MatMul_7"]
    staged = mirror._staging["blk.lin"]
    assert torch.equal(staged, merged.transpose(0, 1).to(staged.dtype))
    assert "blk.lin" in mirror._dirty

    # Strength change re-pushes.
    mgr.set_lora_strength(lid, 0.5)
    assert mirror.sync(reason="strength") == 1
    assert refitter.commits == 2
    half = model.blk.lin.weight.detach()
    assert torch.equal(
        mirror._staging["blk.lin"], half.transpose(0, 1),
    )

    # Disable: the module is unparametrized again, but the engine slot
    # still holds the merged value — the dirty-set forces a base push.
    mgr.disable_lora(lid)
    assert mirror.sync(reason="disable") == 1
    assert refitter.commits == 3
    assert torch.equal(
        mirror._staging["blk.lin"], base.transpose(0, 1),
    )
    assert "blk.lin" not in mirror._dirty

    # Nothing left to push.
    assert mirror.sync(reason="idle") == 0
    assert refitter.commits == 3


def test_untransposed_mapping_pushes_torch_orientation(tmp_path):
    model = _model()
    mgr = _vendored_manager_or_skip(model)
    lora_file = _write_lora_file(tmp_path)
    mirror, _refitter = _mirror(tmp_path, model, transposed=False)
    mgr.enable_lora(mgr.register_lora(str(lora_file)), strength=1.0)
    assert mirror.sync() == 1
    assert torch.equal(
        mirror._staging["blk.lin"], model.blk.lin.weight.detach(),
    )
