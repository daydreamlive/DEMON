"""SA3 fp8 stock export driver: CPU tests with fake D1 modules and a tiny DiT."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sa3"))

import sa3_fp8_stock_export as ex  # noqa: E402


class Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Linear(4, 4)

    def forward(self, x, t, *, cross_attn_cond=None, cross_attn_cond_mask=None,
                local_add_cond=None, padding_mask=None, cfg_scale=1.0, **kw):
        v = self.transformer(x.transpose(1, 2)).transpose(1, 2) + t.view(-1, 1, 1)
        v = v + cross_attn_cond.mean() * cfg_scale
        if local_add_cond is not None:
            v = v + local_add_cond[:, :4]
        if padding_mask is not None:
            v = v * padding_mask.unsqueeze(1)
        return v


class FakeDiTWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()

    def forward(self, x, t, cross_attn_cond=None, cross_attn_mask=None, local_add_cond=None,
                global_cond=None, cfg_scale=1.0, batch_cfg=True, **kw):
        return self.model(x, t, cross_attn_cond=cross_attn_cond,
                          cross_attn_cond_mask=cross_attn_mask,
                          local_add_cond=local_add_cond, cfg_scale=cfg_scale, **kw)


def _bundle(L=12):
    return {"cross_attn_cond": torch.randn(1, 5, 8), "cross_attn_mask": torch.ones(1, 5, dtype=torch.bool),
            "padding_mask": torch.ones(1, L, dtype=torch.bool),
            "local_add_cond": torch.zeros(1, 6, L), "global_cond": None,
            "cfg_scale": 1.0, "batch_cfg": True, "rescale_cfg": True, "apg_scale": 1.0}


def test_export_spec_names_and_axes():
    exs, names, dyn = ex.export_spec(_bundle(), 4, 12)
    assert names == ["x", "t", "cross_attn_cond", "cross_attn_mask", "padding_mask", "local_add_cond"]
    assert exs["t"].shape == (1,) and exs["x"].shape == (1, 4, 12)
    assert dyn == {"x": {2: "latent_frames"}, "padding_mask": {1: "latent_frames"},
                   "local_add_cond": {2: "latent_frames"}, "velocity": {2: "latent_frames"}}
    with pytest.raises(ValueError):
        ex.export_spec(_bundle(L=11), 4, 12)


def test_wrapper_names_match_calibration_root_and_output_matches_eager():
    dit = FakeDiTWrapper().eval()
    b = _bundle()
    exs, names, _ = ex.export_spec(b, 4, 12)
    w = ex.make_export_wrapper(dit, names[2:], ex.bundle_constants(b), torch.float32)
    # Calibration captures on dit -> names "model.transformer"; the export
    # module must expose the same parameter names.
    assert set(dict(w.named_parameters())) == set(dict(dit.named_parameters()))
    assert {n for n, m in w.named_modules() if isinstance(m, torch.nn.Linear)} == {"model.transformer"}
    out = w(*[exs[n] for n in names])
    ref = dit(exs["x"], exs["t"], **{k: v for k, v in b.items() if v is not None})
    assert torch.allclose(out, ref)
    assert out.dtype == torch.float32


def test_bundle_constants_skip_tensors_and_none():
    c = ex.bundle_constants(_bundle())
    assert c == {"cfg_scale": 1.0, "batch_cfg": True, "rescale_cfg": True, "apg_scale": 1.0}


def _tiny_onnx(path: Path):
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, "latent_frames"])
    t = helper.make_tensor_value_info("t", TensorProto.FLOAT, [1])
    pm = helper.make_tensor_value_info("padding_mask", TensorProto.BOOL, [1, "latent_frames"])
    y = helper.make_tensor_value_info("velocity", TensorProto.FLOAT, [1, 4, "latent_frames"])
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["velocity"])], "g", [x, t, pm], [y])
    onnx.save(helper.make_model(g), str(path))


def test_onnx_profiles(tmp_path):
    p = tmp_path / "m.onnx"
    _tiny_onnx(p)
    prof = ex.onnx_profiles(p, (1, 140, 140))
    assert prof["x"] == ((1, 4, 1), (1, 4, 140), (1, 4, 140))
    assert prof["t"] == ((1,), (1,), (1,))
    assert prof["padding_mask"] == ((1, 1), (1, 140), (1, 140))


def _install(monkeypatch, module, **attrs):
    mod = types.ModuleType(module)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, module, mod)


def test_missing_d1_module_stops_clearly(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "acestep.engine.trt.sa3_fp8", None)
    args = ex.parse_args(["--graft", "--fp16-onnx", str(tmp_path / "a.onnx")])
    with pytest.raises(SystemExit) as e:
        ex.cmd_graft(args)
    assert "d1_graft" in str(e.value)


def test_graft_plumbing_and_count_check(monkeypatch, tmp_path):
    fp16 = tmp_path / "stock_fp16mixed.onnx"
    fp16.write_bytes(b"x")
    scales = tmp_path / "s.json"
    scales.write_text(json.dumps({"model.transformer": 3.5}))
    calls = []

    def graft_fp8(src, dst, act, **kw):
        calls.append((src, dst, act, kw))
        return {"linears_total": 224, "linears_fp8": 222, "unmatched": ["a", "b"], "scale_stats": {}}

    excl = ("*to_timestep_embed.*", "*transformer.global_cond_embedder.*")
    _install(monkeypatch, "acestep.engine.trt.sa3_fp8", graft_fp8=graft_fp8,
             HF_MEDIUM_EXCLUDE=excl)
    out = tmp_path / "stock_fp8.onnx"
    args = ex.parse_args(["--graft", "--fp16-onnx", str(fp16), "--fp8-onnx", str(out),
                          "--scales", str(scales)])
    assert ex.cmd_graft(args) == 0
    assert calls == [(fp16, out, {"model.transformer": 3.5}, {"exclude": excl})]
    side = json.loads((tmp_path / "stock_fp8.graft.json").read_text())
    assert side["linears_fp8"] == 222 and len(side["act_scales_sha256"]) == 64
    assert side["exclude"] == list(excl)
    args.expect_linears = 176
    with pytest.raises(SystemExit):
        ex.cmd_graft(args)


def test_build_plumbing_fixed_dirs(monkeypatch, tmp_path):
    fp16, fp8 = tmp_path / "a.onnx", tmp_path / "b.onnx"
    _tiny_onnx(fp16)
    _tiny_onnx(fp8)
    scales = tmp_path / "s.json"
    scales.write_text(json.dumps({"model.transformer": 3.5}))
    calls = []

    def build_engine_from_onnx(onnx_path, engine_dir, **kw):
        calls.append((onnx_path, engine_dir, kw))
        return engine_dir / f"{engine_dir.name}.trt"

    _install(monkeypatch, "acestep.engine.trt.sa3_build",
             build_engine_from_onnx=build_engine_from_onnx)
    args = ex.parse_args(["--build", "--fp16-onnx", str(fp16), "--fp8-onnx", str(fp8),
                          "--engine-root", str(tmp_path / "eng"),
                          "--scales", str(scales)])
    assert ex.cmd_build(args) == 0
    assert [c[1].name for c in calls] == ["sa3_m_dit_ours_fp16mixed_l1_140_140",
                                         "sa3_m_dit_ours_fp8_l1_140_140"]
    for _, _, kw in calls:
        assert kw["strongly_typed"] is True and kw["workspace_gb"] == 16.0
        assert kw["profiles"]["x"] == ((1, 4, 1), (1, 4, 140), (1, 4, 140))
    assert "_fp8_" in calls[1][1].name
    assert calls[0][2]["act_scales_path"] is None and calls[1][2]["act_scales_path"] == scales
    calls.clear()
    args.only = "fp8"
    ex.cmd_build(args)
    assert len(calls) == 1 and calls[0][2]["recipe"] == "demon_fp8_graft"


def test_real_d1_signatures_accept_driver_calls():
    """The real D1 modules expose what the driver calls, with these kwargs."""
    import inspect

    from acestep.engine.trt import sa3_build, sa3_export, sa3_fp8

    assert len(sa3_fp8.HF_MEDIUM_EXCLUDE) == 2 and sa3_fp8.HF_MEDIUM_FP8_LINEARS == ex.STOCK_LINEARS
    inspect.signature(sa3_fp8.graft_fp8).bind("a", "b", {}, exclude=())
    inspect.signature(sa3_fp8.capture_activation_scales).bind(None, lambda: None, min_features=128)
    inspect.signature(sa3_export.export_fp16mixed).bind(
        None, {}, input_names=[], output_names=["velocity"], dynamic_axes={}, out_onnx="o")
    inspect.signature(sa3_build.build_engine_from_onnx).bind(
        "o", "d", profiles={}, strongly_typed=True, workspace_gb=16.0, timing_cache=None,
        recipe="r", extra_metadata={}, act_scales_path=None)


def test_one_step_required():
    with pytest.raises(SystemExit):
        ex.parse_args([])
    with pytest.raises(SystemExit):
        ex.parse_args(["--export", "--build"])
