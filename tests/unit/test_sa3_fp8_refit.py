"""fp8 refit manifest + mirror path for the SA3 fp8 DiT graph (CPU).

A tiny ONNX mirrors Stability's dit_fp8.onnx weight-quant pattern (fp32
initializer -> QuantizeLinear -> DequantizeLinear -> Transpose -> MatMul,
per-channel axis 0; plus a per-tensor [in, out] weight fed straight to
MatMul and an activation Q). The real engine gates (refittable inventory,
base-value bit identity, eager-LoRA parity) run on the GPU via
scripts/sa3/gen_sa3_refit_manifest.py --fp8 --validate-engine and
scripts/sa3/sa3_trt_dit_cond_parity.py --lora.
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

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from acestep.engine.trt.sa3_fp8_refit import (  # noqa: E402
    E4M3_MAX,
    build_fp8_manifest,
    checkpoint_base_reader,
    quantize_e4m3_fixed,
    refit_inventory,
    write_fp8_manifest,
)
from acestep.engine.sa3_trt_lora import SA3TRTRefitMirror  # noqa: E402

A_INIT, B_INIT = "dit.blk.a.weight", "dit.blk.b.weight"
A_SCALE, B_SCALE, X_SCALE = "a_wscale", "b_wscale", "x_ascale"


def _weights():
    g = torch.Generator().manual_seed(5)
    wa = torch.randn(4, 3, generator=g)          # torch [out=4, in=3]
    wb = torch.randn(2, 4, generator=g)          # torch [out=2, in=4]
    return wa, wb


def _fp8_zero(n):
    t = helper.make_tensor("z", TensorProto.FLOAT8E4M3FN, [n] if n > 1 else [], [0.0] * n)
    return t


def _write_graph(tmp_path, wa, wb) -> Path:
    sa = (wa.abs().amax(1) / E4M3_MAX).numpy()
    sb = (wb.abs().amax() / E4M3_MAX).reshape(()).numpy()
    inits = [
        numpy_helper.from_array(wa.numpy(), A_INIT),
        numpy_helper.from_array(wb.t().contiguous().numpy(), B_INIT),   # [in, out]
        numpy_helper.from_array(sa, A_SCALE),
        numpy_helper.from_array(sb, B_SCALE),
        numpy_helper.from_array(torch.tensor(0.05).numpy(), X_SCALE),
    ]
    za = _fp8_zero(4); za.name = "a_zp"
    zb = _fp8_zero(1); zb.name = "b_zp"
    zx = _fp8_zero(1); zx.name = "x_zp"
    inits += [za, zb, zx]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", X_SCALE, "x_zp"], ["xq"]),
        helper.make_node("DequantizeLinear", ["xq", X_SCALE, "x_zp"], ["xd"]),
        helper.make_node("QuantizeLinear", [A_INIT, A_SCALE, "a_zp"], ["aq"], axis=0),
        helper.make_node("DequantizeLinear", ["aq", A_SCALE, "a_zp"], ["ad"], axis=0),
        helper.make_node("Transpose", ["ad"], ["at"], perm=[1, 0]),
        helper.make_node("MatMul", ["xd", "at"], ["h"]),
        helper.make_node("QuantizeLinear", [B_INIT, B_SCALE, "b_zp"], ["bq"]),
        helper.make_node("DequantizeLinear", ["bq", B_SCALE, "b_zp"], ["bd"]),
        helper.make_node("MatMul", ["h", "bd"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny_fp8",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    path = tmp_path / "tiny_fp8.onnx"
    onnx.save_model(
        model, str(path), save_as_external_data=True,
        all_tensors_to_one_file=True, location="tiny_fp8.onnx.data",
        size_threshold=0,
    )
    return path


def _write_ckpt(tmp_path, wa, wb) -> Path:
    from safetensors.torch import save_file

    p = tmp_path / "model.safetensors"
    save_file({"model.model.blk.a.weight": wa.contiguous(),
               "model.model.blk.b.weight": wb.contiguous()}, str(p))
    return p


@pytest.fixture
def graph(tmp_path):
    wa, wb = _weights()
    onnx_path = _write_graph(tmp_path, wa, wb)
    ckpt = _write_ckpt(tmp_path, wa, wb)
    return types.SimpleNamespace(onnx=onnx_path, ckpt=ckpt, wa=wa, wb=wb, dir=tmp_path)


def test_manifest_maps_sites_orientation_and_scales(graph):
    manifest, scales = build_fp8_manifest(
        graph.onnx, base_reader=checkpoint_base_reader(graph.ckpt),
        base_checkpoint=str(graph.ckpt),
    )
    w = manifest["weights"]
    assert manifest["version"] == 2 and manifest["base_verified"]
    assert set(w) == {"model.blk.a", "model.blk.b"}
    a, b = w["model.blk.a"], w["model.blk.b"]
    assert (a["initializer"], a["transposed"], a["per_channel"], a["axis"]) == (A_INIT, False, True, 0)
    assert (b["initializer"], b["transposed"], b["per_channel"], b["axis"]) == (B_INIT, True, False, None)
    assert a["scale_init"] == A_SCALE and b["scale_init"] == B_SCALE
    assert manifest["activation_scales"] == [X_SCALE]
    assert torch.equal(scales[A_SCALE], graph.wa.abs().amax(1) / E4M3_MAX)
    assert scales[B_SCALE].numel() == 1 and scales[X_SCALE].item() == pytest.approx(0.05)


def test_manifest_refuses_a_base_that_differs_from_the_graph(graph):
    from safetensors.torch import save_file

    bad = graph.dir / "bad.safetensors"
    save_file({"model.model.blk.a.weight": graph.wa + 1e-3,
               "model.model.blk.b.weight": graph.wb}, str(bad))
    with pytest.raises(RuntimeError, match="differ from the base checkpoint"):
        build_fp8_manifest(graph.onnx, base_reader=checkpoint_base_reader(bad))


def test_quantize_fixed_scale_saturates_instead_of_nan():
    scale = torch.tensor([0.01, 0.02])
    w = torch.tensor([[1.0, 100.0, -100.0], [0.5, 0.0, -9.0]])
    q = quantize_e4m3_fixed(w, scale, axis=0).view(torch.float8_e4m3fn).float()
    assert not torch.isnan(q).any()
    assert q[0, 1] == E4M3_MAX and q[0, 2] == -E4M3_MAX and q[1, 2] == -E4M3_MAX
    assert q[0, 0] == torch.tensor(100.0).to(torch.float8_e4m3fn).float()
    # per-tensor (axis None) uses one scalar
    qt = quantize_e4m3_fixed(w, torch.tensor(0.5), axis=None).view(torch.float8_e4m3fn).float()
    assert qt[0, 1] == torch.tensor(200.0).to(torch.float8_e4m3fn).float()


class _AddDelta(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", delta)

    def forward(self, w):
        return w + self.delta.to(w.dtype)


class _FakeTrt(types.SimpleNamespace):
    def __init__(self):
        super().__init__(
            float32="F32", float16="F16", fp8="FP8",
            Weights=lambda dtype, ptr, size: (dtype, ptr, size),
        )


class _FakeRefitter:
    def __init__(self, names, proto, missing):
        self._names = list(names)
        self._proto = proto
        self._missing = list(missing)
        self.sets: list = []
        self.commits = 0

    def get_all_weights(self):
        return list(self._names)

    def get_weights_prototype(self, name):
        return types.SimpleNamespace(dtype=self._proto.get(name, "F32"))

    def get_missing_weights(self):
        return [m for m in self._missing if m not in self.sets]

    def set_named_weights(self, name, weights):
        self.sets.append(name)
        return name in self._names

    def refit_cuda_engine(self):
        self.commits += 1
        return True


def _model(graph):
    root = nn.Module()
    root.model = nn.Module()
    root.model.blk = nn.Module()
    root.model.blk.a = nn.Linear(3, 4, bias=False).half()
    root.model.blk.b = nn.Linear(4, 2, bias=False).half()
    with torch.no_grad():
        root.model.blk.a.weight.copy_(graph.wa.half())
        root.model.blk.b.weight.copy_(graph.wb.half())
    return root


def _mirror(graph, *, slot_dtype):
    manifest, scales = build_fp8_manifest(
        graph.onnx, base_reader=checkpoint_base_reader(graph.ckpt),
        base_checkpoint=str(graph.ckpt),
    )
    mpath = write_fp8_manifest(manifest, scales, graph.dir / "fp8_manifest.json")
    names = [A_INIT, B_INIT, A_SCALE, B_SCALE]
    proto = {A_INIT: slot_dtype, B_INIT: slot_dtype, A_SCALE: "F32", B_SCALE: "F32"}
    refitter = _FakeRefitter(names, proto, missing=[A_SCALE])
    root = _model(graph)
    mirror = SA3TRTRefitMirror(
        object(), root, mpath, _refitter=refitter, _trt=_FakeTrt(),
    )
    return mirror, refitter, root, scales


@pytest.fixture(autouse=True)
def _no_engine_refit_hook(monkeypatch):
    import acestep.engine.sa3_trt as sa3_trt

    monkeypatch.setattr(sa3_trt, "note_engine_refit", lambda engine: None)


def test_fp8_mirror_strength_zero_pushes_the_exact_fp32_base(graph):
    mirror, refitter, root, _ = _mirror(graph, slot_dtype="F32")
    parametrize.register_parametrization(root.model.blk.a, "weight", _AddDelta(torch.zeros(4, 3)))
    assert mirror.sync(reason="t") == 1
    staged = mirror._staging["model.blk.a"]
    assert staged.dtype == torch.float32
    # The eager weight is fp16; the pushed base must still be the fp32 checkpoint.
    assert torch.equal(staged, graph.wa)
    assert refitter.sets[:1] == [A_INIT]
    assert A_SCALE in refitter.sets          # reported missing -> co-pushed
    assert B_SCALE not in refitter.sets
    assert refitter.commits == 1


def test_fp8_mirror_float_slot_pushes_base_plus_eager_delta(graph):
    mirror, refitter, root, _ = _mirror(graph, slot_dtype="F32")
    delta = torch.full((4, 3), 0.25)
    parametrize.register_parametrization(root.model.blk.a, "weight", _AddDelta(delta))
    mirror.sync()
    mod = root.model.blk.a
    expect = graph.wa + (mod.weight.float() - mod.parametrizations.weight.original.float())
    assert torch.equal(mirror._staging["model.blk.a"], expect)
    # disable -> the dirty fqn's base goes back on the next sync
    parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)
    assert mirror.sync() == 1
    assert torch.equal(mirror._staging["model.blk.a"], graph.wa)


def test_fp8_mirror_fp8_slot_quantizes_under_the_baked_scale(graph):
    mirror, refitter, root, scales = _mirror(graph, slot_dtype="FP8")
    delta = graph.wa.abs().amax(1, keepdim=True).expand(4, 3) * 0.5   # pushes past the bound
    parametrize.register_parametrization(root.model.blk.a, "weight", _AddDelta(delta))
    mirror.sync()
    mod = root.model.blk.a
    merged = graph.wa + (mod.weight.float() - mod.parametrizations.weight.original.float())
    staged = mirror._staging["model.blk.a"]
    assert staged.dtype == torch.uint8
    assert torch.equal(staged, quantize_e4m3_fixed(merged, scales[A_SCALE], 0))
    assert not torch.isnan(staged.view(torch.float8_e4m3fn).float()).any()


def test_fp8_mirror_transposed_weight_goes_in_graph_orientation(graph):
    mirror, refitter, root, _ = _mirror(graph, slot_dtype="F32")
    parametrize.register_parametrization(root.model.blk.b, "weight", _AddDelta(torch.zeros(2, 4)))
    mirror.sync()
    assert torch.equal(mirror._staging["model.blk.b"], graph.wb.t())


def test_refit_inventory_reports_refittable_weights_and_scales(graph):
    manifest, _ = build_fp8_manifest(graph.onnx)
    refitter = _FakeRefitter([A_INIT, B_INIT, A_SCALE, "other"], {A_INIT: "FP8"}, [])
    inv = refit_inventory(refitter, manifest)
    assert inv["weights_mapped"] == 2 and inv["weights_refittable"] == 2
    assert inv["weight_scales"] == 2 and inv["weight_scales_refittable"] == 1
    assert inv["activation_scales"] == 1 and inv["activation_scales_refittable"] == 0
    assert inv["weight_prototype_dtypes"] == {"FP8": 1, "F32": 1}
    assert inv["other_refittable_sample"] == ["other"]


def test_manifest_json_roundtrip_has_scales_sidecar(graph):
    manifest, scales = build_fp8_manifest(graph.onnx)
    out = write_fp8_manifest(manifest, scales, graph.dir / "m.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert (graph.dir / data["scales_file"]).is_file()


# --- missing-set completion (D4b) -------------------------------------
# A REFIT-built engine marks every initializer refittable, and TensorRT
# reports fused non-LoRA weights (on medium fp8: 24 cross_attend_norm.gamma
# + padding_embedding) missing after the mapped pushes. The mirror must
# supply each at its build-time base value, in the slot dtype, every commit.

GAMMA = "dit.layers.0.cross_attend_norm.gamma"
PAD = "padding_embedding"


def _extras():
    g = torch.Generator().manual_seed(9)
    return torch.randn(3, generator=g) + 1.0, torch.randn(2, 3, generator=g)


def _mirror_with_extras(graph, *, extra_dtype, missing=None, ckpt_extras=True):
    from safetensors.torch import save_file

    gamma, pad = _extras()
    sd = {"model.model.blk.a.weight": graph.wa.contiguous(),
          "model.model.blk.b.weight": graph.wb.contiguous()}
    if ckpt_extras:
        sd["model.model.layers.0.cross_attend_norm.gamma"] = gamma
        sd["model.model.padding_embedding"] = pad
    ckpt = graph.dir / "model_extras.safetensors"
    save_file(sd, str(ckpt))
    manifest, scales = build_fp8_manifest(
        graph.onnx, base_reader=checkpoint_base_reader(ckpt), base_checkpoint=str(ckpt),
    )
    mpath = write_fp8_manifest(manifest, scales, graph.dir / "fp8_manifest_x.json")
    names = [A_INIT, B_INIT, A_SCALE, B_SCALE, GAMMA, PAD, "/Constant_3_output_0"]
    proto = {A_INIT: "F32", B_INIT: "F32", GAMMA: extra_dtype, PAD: extra_dtype}
    refitter = _FakeRefitter(
        names, proto, missing=[A_SCALE, GAMMA, PAD] if missing is None else missing,
    )
    root = _model(graph)
    mirror = SA3TRTRefitMirror(object(), root, mpath, _refitter=refitter, _trt=_FakeTrt())
    return mirror, refitter, root, gamma, pad


@pytest.mark.parametrize("slot,torch_dt", [("F32", torch.float32), ("F16", torch.float16)])
def test_fp8_mirror_supplies_every_missing_name_from_the_checkpoint(graph, slot, torch_dt):
    mirror, refitter, root, gamma, pad = _mirror_with_extras(graph, extra_dtype=slot)
    parametrize.register_parametrization(root.model.blk.a, "weight", _AddDelta(torch.zeros(4, 3)))
    assert mirror.sync(reason="t") == 1
    assert {A_SCALE, GAMMA, PAD} <= set(refitter.sets)
    assert refitter.get_missing_weights() == []
    assert "/Constant_3_output_0" not in refitter.sets    # refittable, not missing
    g_buf, _arr, g_dt = mirror._extra_push[GAMMA]
    p_buf, _arr, p_dt = mirror._extra_push[PAD]
    assert (g_dt, p_dt) == (slot, slot)
    assert g_buf.dtype == torch_dt and p_buf.dtype == torch_dt
    assert torch.equal(g_buf, gamma.to(torch_dt))
    assert torch.equal(p_buf, pad.to(torch_dt))
    # Every commit re-supplies them (fresh missing set), from the same buffers.
    refitter.sets.clear()
    parametrize.remove_parametrizations(root.model.blk.a, "weight", leave_parametrized=False)
    assert mirror.sync(reason="t2") == 1
    assert {GAMMA, PAD} <= set(refitter.sets)
    assert mirror._extra_push[GAMMA][0] is g_buf
    assert refitter.commits == 2


def test_fp8_mirror_missing_mapped_weight_goes_back_at_base_in_graph_orientation(graph):
    mirror, refitter, root, _g, _p = _mirror_with_extras(
        graph, extra_dtype="F32", missing=[B_INIT],
    )
    parametrize.register_parametrization(root.model.blk.a, "weight", _AddDelta(torch.ones(4, 3)))
    mirror.sync()
    assert B_INIT in refitter.sets
    assert torch.equal(mirror._extra_push[B_INIT][0], graph.wb.t().contiguous())


def test_fp8_mirror_unresolvable_missing_name_raises(graph):
    mirror, refitter, root, _g, _p = _mirror_with_extras(
        graph, extra_dtype="F32", ckpt_extras=False,
    )
    parametrize.register_parametrization(root.model.blk.a, "weight", _AddDelta(torch.zeros(4, 3)))
    with pytest.raises(RuntimeError, match="no resolvable base"):
        mirror.sync()
    assert refitter.commits == 0


@pytest.mark.parametrize("slot,torch_dt", [("F32", torch.float32), ("F16", torch.float16)])
def test_fp16mixed_mirror_supplies_missing_names_from_eager(tmp_path, slot, torch_dt):
    """v1 (fp16mixed) path: same latent gap; without a base checkpoint the
    missing base comes from the live eager parameter, cast to the slot."""
    model = nn.Module()
    model.blk = nn.Module()
    model.blk.lin = nn.Linear(3, 4, bias=False)
    model.blk.norm = nn.Module()
    model.blk.norm.gamma = nn.Parameter(torch.randn(3))
    manifest = tmp_path / "v1.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "weights": {"blk.lin": {"initializer": "onnx::MatMul_7", "transposed": False}},
    }), encoding="utf-8")
    name = "dit.blk.norm.gamma"
    refitter = _FakeRefitter(
        ["onnx::MatMul_7", name], {"onnx::MatMul_7": "F32", name: slot}, missing=[name],
    )
    mirror = SA3TRTRefitMirror(object(), model, manifest, _refitter=refitter, _trt=_FakeTrt())
    parametrize.register_parametrization(model.blk.lin, "weight", _AddDelta(torch.zeros(4, 3)))
    assert mirror.sync() == 1
    assert refitter.sets == ["onnx::MatMul_7", name]
    buf, _arr, dt = mirror._extra_push[name]
    assert dt == slot and torch.equal(buf, model.blk.norm.gamma.detach().to(torch_dt))
