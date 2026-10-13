"""Live LoRA path on an fp8 refit engine (CPU, fake refitter).

The real SA3TRTRefitMirror (fp8, manifest v2) is constructed through the
session-create hook (make_refit_mirror) and driven by SA3Backend's LoRA
enable / strength / disable hooks, exactly as a /sa3 LoRA session drives
it: no eager swap, one refit commit per change, pushed values = fp32
checkpoint base + eager delta at the live strength.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

pytest.importorskip("onnx")

from test_sa3_fp8_refit import (  # noqa: E402
    A_INIT, A_SCALE, B_INIT, B_SCALE, _FakeRefitter, _FakeTrt, _model,
    _weights, _write_ckpt, _write_graph,
)
from tests.unit.test_sa3_backend import CTX, _backend, _knobs  # noqa: E402

from acestep.engine.sa3_trt_lora import (  # noqa: E402
    FP8_SHARED_MANIFEST_NAME,
    SHARED_MANIFEST_NAME,
    find_refit_manifest,
    make_refit_mirror,
)
from acestep.engine.trt.sa3_fp8_refit import (  # noqa: E402
    build_fp8_manifest,
    checkpoint_base_reader,
    quantize_e4m3_fixed,
    write_fp8_manifest,
)

FP8_ENGINE = "sa3_m_dit_fp8_refit_l1_140_140"
FP16_ENGINE = "sa3_m_dit_refit_l1_140_140"


@pytest.fixture(autouse=True)
def _no_engine_refit_hook(monkeypatch):
    import acestep.engine.sa3_trt as sa3_trt

    monkeypatch.setattr(sa3_trt, "note_engine_refit", lambda engine: None)


@pytest.fixture
def g(tmp_path):
    wa, wb = _weights()
    onnx_path = _write_graph(tmp_path, wa, wb)
    ckpt = _write_ckpt(tmp_path, wa, wb)
    return types.SimpleNamespace(onnx=onnx_path, ckpt=ckpt, wa=wa, wb=wb, dir=tmp_path)


def _engine_dir(root: Path, name: str) -> Path:
    d = root / "engines" / name
    d.mkdir(parents=True)
    f = d / f"{name}.trt"
    f.write_bytes(b"x")
    return f


def _fp8_sidecar(g, engine_path: Path) -> Path:
    manifest, scales = build_fp8_manifest(
        g.onnx, base_reader=checkpoint_base_reader(g.ckpt),
        base_checkpoint=str(g.ckpt),
    )
    return write_fp8_manifest(
        manifest, scales, Path(str(engine_path) + ".refit_manifest.json"),
    )


class _PerCommitRefitter(_FakeRefitter):
    """TensorRT re-reports fused weights as missing after every commit
    (D4b): the missing set counts only names pushed since the last
    refit_cuda_engine()."""

    def __init__(self, *a):
        super().__init__(*a)
        self._since = 0

    def get_missing_weights(self):
        pushed = set(self.sets[self._since:])
        return [m for m in self._missing if m not in pushed]

    def refit_cuda_engine(self):
        self._since = len(self.sets)
        return super().refit_cuda_engine()


class _Scaled(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", delta)
        self.register_buffer("strength", torch.tensor(1.0))

    def forward(self, w):
        return w + (self.strength * self.delta).to(w.dtype)


class _LoraMgr:
    """Minimal SA3LoRAManager stand-in that really parametrizes: enable
    registers W + strength * delta on blk.a, strength writes the buffer,
    disable removes the parametrization (base restored)."""

    def __init__(self, root, delta):
        self.root, self.delta = root, delta
        self.desc = types.SimpleNamespace(
            id="dt", name="dt", path="/x/dt.safetensors", state="registered",
            strength=0.0, materialized_bytes=0,
        )
        self.p = None

    def list_loras(self):
        return [self.desc]

    @property
    def has_active_loras(self):
        return self.desc.state == "enabled"

    def enable_lora(self, lora_id, strength=None):
        self.p = _Scaled(self.delta)
        parametrize.register_parametrization(self.root.model.blk.a, "weight", self.p)
        self.desc.state = "enabled"
        self.set_lora_strength(lora_id, 1.0 if strength is None else strength)

    def set_lora_strength(self, lora_id, strength):
        self.p.strength.fill_(float(strength))
        self.desc.strength = float(strength)

    def disable_lora(self, lora_id):
        parametrize.remove_parametrizations(
            self.root.model.blk.a, "weight", leave_parametrized=False,
        )
        self.desc.state = "registered"

    def touches_conditioner(self, lora_id):
        return False

    def close(self):
        pass


def _session(g, *, slot="F32"):
    """Session-create path: make_refit_mirror over an fp8 refit engine,
    then an SA3Backend wired with that mirror."""
    eng = _engine_dir(g.dir, FP8_ENGINE)
    manifest_path = _fp8_sidecar(g, eng)
    root = _model(g)
    proto = {A_INIT: slot, B_INIT: slot, A_SCALE: "F32", B_SCALE: "F32"}
    refitter = _PerCommitRefitter([A_INIT, B_INIT, A_SCALE, B_SCALE], proto, [A_SCALE])
    dit = types.SimpleNamespace(engine=object(), engine_path=eng, refittable=True)
    mirror = make_refit_mirror(dit, root, _refitter=refitter, _trt=_FakeTrt())
    assert mirror is not None and mirror._fp8
    delta = torch.full((4, 3), 0.25)
    mgr = _LoraMgr(root, delta)
    b = _backend(lora_manager=mgr, use_lora=True, refit_mirror=mirror)
    return types.SimpleNamespace(
        b=b, mirror=mirror, refitter=refitter, delta=delta, manifest=manifest_path,
    )


def test_fp8_engine_finds_fp8_shared_manifest_never_the_fp16_one(tmp_path):
    eng = _engine_dir(tmp_path, FP8_ENGINE)
    (tmp_path / "engines" / SHARED_MANIFEST_NAME).write_text("{}")
    assert find_refit_manifest(eng) is None
    (tmp_path / "engines" / FP8_SHARED_MANIFEST_NAME).write_text("{}")
    assert find_refit_manifest(eng).name == FP8_SHARED_MANIFEST_NAME
    fp16 = _engine_dir(tmp_path, FP16_ENGINE)
    assert find_refit_manifest(fp16).name == SHARED_MANIFEST_NAME


def test_manifest_precision_mismatch_falls_back_to_eager_swap(g):
    # An fp16mixed refit engine handed a v2 (fp8) manifest: no mirror.
    eng = _engine_dir(g.dir, FP16_ENGINE)
    _fp8_sidecar(g, eng)
    dit = types.SimpleNamespace(engine=object(), engine_path=eng, refittable=True)
    refitter = _FakeRefitter([A_INIT, B_INIT, A_SCALE, B_SCALE], {}, [])
    assert make_refit_mirror(dit, _model(g), _refitter=refitter, _trt=_FakeTrt()) is None


def test_missing_manifest_returns_none(tmp_path):
    eng = _engine_dir(tmp_path, FP8_ENGINE)
    dit = types.SimpleNamespace(engine=object(), engine_path=eng, refittable=True)
    assert make_refit_mirror(dit, nn.Module()) is None


def test_live_enable_strength_disable_through_backend(g):
    s = _session(g)
    b, mirror, refitter, delta = s.b, s.mirror, s.refitter, s.delta
    accel = b._dit_accel

    # enable at 1.0: one commit, base + full delta, no eager swap
    b.enable_lora("dt", strength=1.0)
    assert b.adapter.dit is accel
    assert refitter.commits == 1
    # fp32 base + the eager (fp16) delta: within fp16 rounding of +delta
    assert torch.allclose(mirror._staging["model.blk.a"], g.wa + delta, atol=1e-3)
    assert not torch.equal(mirror._staging["model.blk.a"], g.wa)

    # knob strength 1.0 -> 0.5: announced via rebuild_imminent, applied
    # by the same tick's produce, one more commit
    knobs = _knobs(b, **{"lora_str_dt": 0.5})
    assert b.rebuild_imminent(knobs) is True
    b.produce(knobs, CTX, "skip")
    assert refitter.commits == 2
    assert torch.allclose(mirror._staging["model.blk.a"], g.wa + 0.5 * delta, atol=1e-3)

    # 0.5 -> 0: the pushed value is the fp32 base bit for bit
    knobs = _knobs(b, **{"lora_str_dt": 0.0})
    assert b.rebuild_imminent(knobs) is True
    b.produce(knobs, CTX, "skip")
    assert refitter.commits == 3
    assert torch.equal(mirror._staging["model.blk.a"], g.wa)
    assert b.adapter.dit is accel

    # disable: dirty base goes back, still no swap
    b.disable_lora("dt")
    assert refitter.commits == 4
    assert torch.equal(mirror._staging["model.blk.a"], g.wa)
    assert b.adapter.dit is accel
    # every commit co-pushed the fused baked scale the refitter reported
    assert refitter.sets.count(A_SCALE) == 4


def test_live_fp8_slot_pushes_fixed_scale_bytes(g):
    s = _session(g, slot="FP8")
    s.b.enable_lora("dt", strength=1.0)
    staged = s.mirror._staging["model.blk.a"]
    assert staged.dtype == torch.uint8
    rec = json.loads(s.manifest.read_text(encoding="utf-8"))["weights"]["model.blk.a"]
    mod = s.b._lora_mgr.root.model.blk.a
    merged = g.wa + (mod.weight.float() - mod.parametrizations.weight.original.float())
    assert torch.equal(
        staged,
        quantize_e4m3_fixed(merged, s.mirror._scales[rec["scale_init"]], rec["axis"]),
    )
